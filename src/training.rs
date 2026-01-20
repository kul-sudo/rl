use crate::consts::*;
use crate::env::{
    batch::BatchCollector,
    context::{Env, Perspective},
};
use crate::render::utils::Data;
use crate::rl::{
    actor::{Actor, ActorConfig},
    critic::{Critic, CriticConfig},
    gae::{compute_gae_fused, gae_custom},
    stochastic::gumbel_sample,
};
use burn::{
    backend::Autodiff,
    grad_clipping::GradientClippingConfig,
    module::Module,
    nn::loss::{MseLoss, Reduction},
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::ToElement,
    record::CompactRecorder,
    tensor::{
        Bool, Tensor,
        activation::log_softmax,
        backend::{AutodiffBackend, Backend},
        ops::{BoolTensor, FloatTensor},
    },
};
use burn_cubecl::{
    CubeBackend, CubeRuntime, FloatElement, IntElement,
    cubecl::{CubeElement, frontend::ScalarArgSettings},
    element::BoolElement,
};
use macroquad::prelude::*;
use std::{
    marker::PhantomData,
    path::Path,
    sync::mpsc::Sender,
    time::{Duration, Instant},
};

pub trait GaeBackend: Backend {
    fn fused_gae(
        rewards: FloatTensor<Self>,
        values: FloatTensor<Self>,
        dones: BoolTensor<Self>,
        bootstrap_values: FloatTensor<Self>,
        device: &Self::Device,
    ) -> FloatTensor<Self>;
}
pub trait GaeAutodiffBackend: GaeBackend + AutodiffBackend {}

const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(5);

impl<R: CubeRuntime, F, I, BT> GaeBackend for CubeBackend<R, F, I, BT>
where
    F: FloatElement + CubeElement + ScalarArgSettings,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_gae(
        rewards: FloatTensor<Self>,
        values: FloatTensor<Self>,
        dones: BoolTensor<Self>,
        bootstrap_values: FloatTensor<Self>,
        device: &Self::Device,
    ) -> FloatTensor<Self> {
        compute_gae_fused::<R, F>(rewards, values, dones, bootstrap_values, device)
    }
}

impl<B: GaeBackend> GaeBackend for Autodiff<B> {
    fn fused_gae(
        rewards: FloatTensor<Self>,
        values: FloatTensor<Self>,
        dones: BoolTensor<Self>,
        bootstrap_values: FloatTensor<Self>,
        device: &Self::Device,
    ) -> FloatTensor<Self> {
        let rewards_inner = Self::inner(rewards);
        let values_inner = Self::inner(values);
        let dones_inner = Self::bool_inner(dones);
        let bootstrap_inner = Self::inner(bootstrap_values);

        let output_inner = B::fused_gae(
            rewards_inner,
            values_inner,
            dones_inner,
            bootstrap_inner,
            device,
        );

        Self::from_inner(output_inner)
    }
}

fn gae_reference<B: Backend>(
    rewards: Tensor<B, 2>,
    values: Tensor<B, 2>,
    dones: Tensor<B, 2, Bool>,
    gamma: f32,
    lambda: f32,
) -> Tensor<B, 2> {
    let [batch_size, seq_len] = rewards.dims();
    let device = &rewards.device();

    let mut advantages = Vec::with_capacity(seq_len);
    let mut last_gae = Tensor::<B, 2>::zeros([batch_size, 1], device);
    let mut next_values = Tensor::<B, 2>::zeros([batch_size, 1], device);
    let zero_tensor = Tensor::<B, 2>::zeros([batch_size, 1], device);

    for t in (0..seq_len).rev() {
        let r = rewards.clone().slice([0..batch_size, t..t + 1]);
        let v = values.clone().slice([0..batch_size, t..t + 1]);
        let d = dones.clone().slice([0..batch_size, t..t + 1]);

        // If d is true (terminal), we reset future values/advantages to 0
        let current_next_val = next_values
            .clone()
            .mask_where(d.clone(), zero_tensor.clone());
        let current_last_gae = last_gae.clone().mask_where(d.clone(), zero_tensor.clone());

        // delta = r + gamma * next_val_masked - v
        let delta = r + (current_next_val * gamma) - v.clone();

        // gae = delta + gamma * lambda * last_gae_masked
        let gae = delta + (current_last_gae * (gamma * lambda));

        last_gae = gae.clone();
        next_values = v;

        advantages.push(gae);
    }

    advantages.reverse();
    Tensor::cat(advantages, 1)
}

impl<B: GaeBackend> GaeAutodiffBackend for Autodiff<B> {}

pub fn training<B: GaeAutodiffBackend, E: Env<B> + Clone>(
    mut env: E,
    data_tx: &Sender<Data<B, E>>,
    curiosity: &mut f32,
    device: &B::Device,
) {
    let mut pursuer: Actor<B> =
        ActorConfig::new(PURSUER_FACTORS, N_DIRECTIONS as usize).init(device);
    let mut p_critic: Critic<B> = CriticConfig::new(PURSUER_FACTORS).init(device);

    let mut target: Actor<B> = ActorConfig::new(TARGET_FACTORS, N_DIRECTIONS as usize).init(device);
    let mut t_critic: Critic<B> = CriticConfig::new(TARGET_FACTORS).init(device);

    // Actor optimizers
    let mut p_optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let mut t_optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    // Critic optimizers
    let mut pc_optimizer = AdamWConfig::new()
        // .with_cautious_weight_decay(true)
        // .with_weight_decay(1e-2)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let mut tc_optimizer = AdamWConfig::new()
        // .with_cautious_weight_decay(true)
        // .with_weight_decay(1e-2)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let mut checkpoint_timer = Instant::now();
    let update_agent = |actor: &mut Actor<B>,
                        critic: &mut Critic<B>,
                        actor_optimizer: &mut OptimizerAdaptor<AdamW, Actor<B>, B>,
                        critic_optimizer: &mut OptimizerAdaptor<AdamW, Critic<B>, B>,
                        rollout: BatchCollector<B>,
                        curiosity: f32,
                        next_state: Tensor<B, 2>| {
        let rollout = rollout.consume();

        let flat_values = critic.forward(rollout.states.clone());
        let values = flat_values.clone().reshape([1, BATCH_SIZE]);
        let bootstrap = critic.forward(next_state).detach();

        let advantage = gae_custom(
            rollout.rewards,
            values.detach(),
            rollout.dones,
            bootstrap,
            device,
        )
        .reshape([BATCH_SIZE, 1])
        .detach();

        let td_target = advantage.clone() + flat_values.clone().detach();
        // let ref_advantage = gae_reference(
        //     raw_rewards.clone(),
        //     values.clone(),
        //     dones.clone(),
        //     GAMMA,
        //     0.95, // LAMBDA
        // );
        //
        // let diff = (advantage.clone() - ref_advantage)
        //     .abs()
        //     .mean()
        //     .into_scalar();
        // println!("⚠️ GAE Mismatch detected! Mean Abs Error: {}", diff);

        let critic_loss = MseLoss::new().forward(flat_values, td_target, Reduction::Mean);
        *critic = critic_optimizer.step(
            3e-4,
            critic.clone(),
            GradientsParams::from_grads(critic_loss.backward(), critic),
        );

        let logits = actor.forward(rollout.states);
        let log_probs = log_softmax(logits, 1);
        let probs = log_probs.clone().exp();

        let entropy_loss = -(probs * log_probs.clone()).sum_dim(1).mean();
        let pick = log_probs.gather(1, rollout.actions);

        let (variance, mean) = advantage.clone().var_mean(0);
        let advantage_norm = advantage
            .clone()
            .sub(mean)
            .div(variance.add_scalar(1e-8).sqrt());

        let actor_loss = -(pick * advantage_norm).mean() - curiosity * entropy_loss;

        *actor = actor_optimizer.step(
            1e-4,
            actor.clone(),
            GradientsParams::from_grads(actor_loss.backward(), actor),
        );
    };

    loop {
        env.reset();

        let mut p_rollout = BatchCollector::new();
        let mut t_rollout = BatchCollector::new();
        let (mut p_state, _) = env.state_tensor(Perspective::Pursuer, device);
        let (mut t_state, _) = env.state_tensor(Perspective::Target, device);

        loop {
            let _ = data_tx.send(Data {
                env: env.clone(),
                curiosity: Some(*curiosity),
                _phantom: PhantomData,
            });

            let p_action = gumbel_sample(pursuer.forward(p_state.clone()));
            let t_action = gumbel_sample(target.forward(t_state.clone()));

            let (p_step, t_step) =
                env.step_simultaneous(p_action.clone(), t_action.clone(), device);

            p_rollout.push(p_state, p_action, p_step.clone());
            t_rollout.push(t_state, t_action, t_step.clone());

            p_state = p_step.next_state;
            t_state = t_step.next_state;

            if p_rollout.len() >= BATCH_SIZE {
                let p_bootstrap = p_state.clone();
                let t_bootstrap = t_state.clone();

                update_agent(
                    &mut pursuer,
                    &mut p_critic,
                    &mut p_optimizer,
                    &mut pc_optimizer,
                    p_rollout,
                    *curiosity,
                    p_bootstrap,
                );

                update_agent(
                    &mut target,
                    &mut t_critic,
                    &mut t_optimizer,
                    &mut tc_optimizer,
                    t_rollout,
                    *curiosity,
                    t_bootstrap,
                );

                p_rollout = BatchCollector::new();
                t_rollout = BatchCollector::new();
            }

            *curiosity = (*curiosity * CURIOSITY_DECAY).max(CURIOSITY_MIN);

            let p_done = p_step.done.clone().any().into_scalar().to_bool();
            let t_done = t_step.done.clone().any().into_scalar().to_bool();

            if p_done || t_done {
                break;
            }
        }

        if checkpoint_timer.elapsed() >= CHECKPOINT_INTERVAL {
            checkpoint_timer = Instant::now();

            pursuer
                .clone()
                .save_file(
                    Path::new(ARTIFACT_DIR).join("pursuer"),
                    &CompactRecorder::new(),
                )
                .unwrap();
            target
                .clone()
                .save_file(
                    Path::new(ARTIFACT_DIR).join("target"),
                    &CompactRecorder::new(),
                )
                .unwrap();
            p_critic
                .clone()
                .save_file(
                    Path::new(ARTIFACT_DIR).join("p_critic"),
                    &CompactRecorder::new(),
                )
                .unwrap();
            t_critic
                .clone()
                .save_file(
                    Path::new(ARTIFACT_DIR).join("t_critic"),
                    &CompactRecorder::new(),
                )
                .unwrap();
        }
    }
}
