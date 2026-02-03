use crate::consts::*;
use crate::env::{
    batch::BatchCollector,
    context::{Env, Perspective},
};
use crate::render::data::Data;
use crate::rl::{
    actor::{Actor, ActorConfig},
    critic::{Critic, CriticConfig},
    gae::{compute_gae_fused, gae_custom},
    stochastic::gumbel_sample,
};
use burn::{
    backend::Autodiff,
    grad_clipping::GradientClippingConfig,
    lr_scheduler::{
        LrScheduler,
        exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig},
    },
    module::{AutodiffModule, Module},
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
        terminated: BoolTensor<Self>,
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
        terminated: BoolTensor<Self>,
        bootstrap_values: FloatTensor<Self>,
        device: &Self::Device,
    ) -> FloatTensor<Self> {
        compute_gae_fused::<R, F>(rewards, values, dones, terminated, bootstrap_values, device)
    }
}

impl<B: GaeBackend> GaeBackend for Autodiff<B> {
    fn fused_gae(
        rewards: FloatTensor<Self>,
        values: FloatTensor<Self>,
        dones: BoolTensor<Self>,
        terminated: BoolTensor<Self>,
        bootstrap_values: FloatTensor<Self>,
        device: &Self::Device,
    ) -> FloatTensor<Self> {
        let rewards_inner = Self::inner(rewards);
        let values_inner = Self::inner(values);
        let dones_inner = Self::bool_inner(dones);
        let terminated_inner = Self::bool_inner(terminated);
        let bootstrap_inner = Self::inner(bootstrap_values);

        let output_inner = B::fused_gae(
            rewards_inner,
            values_inner,
            dones_inner,
            terminated_inner,
            bootstrap_inner,
            device,
        );

        Self::from_inner(output_inner)
    }
}

pub struct Learner<'a, B: GaeAutodiffBackend, M: AutodiffModule<B>> {
    pub model: &'a mut M,
    pub optimizer: &'a mut OptimizerAdaptor<AdamW, M, B>,
    pub scheduler: &'a mut ExponentialLrScheduler,
}

pub struct UpdateContext<'a, B: GaeAutodiffBackend> {
    pub actor_learner: Learner<'a, B, Actor<B>>,
    pub critic_learner: Learner<'a, B, Critic<B>>,
    pub rollout: BatchCollector<B>,
    pub next_state: Tensor<B, 2>,
    pub curiosity: f32,
    pub device: &'a B::Device,
}

fn update_agent<B: GaeAutodiffBackend>(context: UpdateContext<B>) {
    let UpdateContext {
        actor_learner:
            Learner {
                model: actor,
                optimizer: actor_optimizer,
                scheduler: actor_scheduler,
            },
        critic_learner:
            Learner {
                model: critic,
                optimizer: critic_optimizer,
                scheduler: critic_scheduler,
            },
        rollout,
        next_state,
        curiosity,
        device,
    } = context;

    let rollout = rollout.consume();
    let flat_values = critic.forward(rollout.states.clone());

    let values = flat_values.clone().reshape([1, BATCH_SIZE]);
    let bootstrap = critic.forward(next_state).detach();

    let (var, mean) = rollout.rewards.clone().var_mean(0);
    let normalized_rewards = rollout.rewards.sub(mean).div(var.sqrt().add_scalar(1e-8));

    let advantage = gae_custom(
        normalized_rewards,
        values.detach(),
        rollout.dones,
        rollout.terminated,
        bootstrap,
        device,
    )
    .reshape([BATCH_SIZE, 1])
    .detach();

    // let gpu_res = gae_custom(
    //     normalized_rewards.clone(),
    //     values.clone().detach(),
    //     rollout.dones.clone(),
    //     rollout.terminated.clone(),
    //     bootstrap.clone(),
    //     &device,
    // );
    //
    // let cpu_res = gae_cpu_reference(
    //     normalized_rewards,
    //     values.detach(),
    //     rollout.dones,
    //     rollout.terminated,
    //     bootstrap.clone(),
    //     GAMMA,
    // );

    // let diff = (gpu_res - cpu_res).abs().mean().into_scalar().to_f32();
    // dbg!(diff);

    let td_target = advantage.clone() + flat_values.clone().detach();

    let critic_lr = critic_scheduler.step();
    let critic_loss = MseLoss::new().forward(flat_values, td_target, Reduction::Mean);

    *critic = critic_optimizer.step(
        critic_lr,
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

    let actor_lr = actor_scheduler.step();
    let actor_loss = -(pick * advantage_norm).mean() - curiosity * entropy_loss;
    *actor = actor_optimizer.step(
        actor_lr,
        actor.clone(),
        GradientsParams::from_grads(actor_loss.backward(), actor),
    );
}

impl<B: GaeBackend> GaeAutodiffBackend for Autodiff<B> {}

fn gae_cpu_reference<B: Backend>(
    rewards: Tensor<B, 2>, // [BATCH_SIZE, 1]
    values: Tensor<B, 2>,  // [1, BATCH_SIZE]
    dones: Tensor<B, 2, Bool>,
    terminated: Tensor<B, 2, Bool>,
    bootstrap: Tensor<B, 2>, // [1, 1]
    gamma: f32,
) -> Tensor<B, 2> {
    let steps = BATCH_SIZE as usize;
    let mut advantages = vec![0.0; steps];

    let rewards_vec: Vec<f32> = rewards
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();
    let values_vec: Vec<f32> = values.into_data().convert::<f32>().to_vec().unwrap();
    let dones_vec: Vec<bool> = dones
        .into_data()
        .convert::<u8>()
        .to_vec()
        .unwrap()
        .into_iter()
        .map(|v: u8| v != 0_u8)
        .collect();

    let term_vec: Vec<bool> = terminated
        .into_data()
        .convert::<u8>()
        .to_vec()
        .unwrap()
        .into_iter()
        .map(|v: u8| v != 0_u8)
        .collect();

    let last_v = bootstrap.into_scalar().to_f32();

    let mut last_gae = 0.0;

    for t in (0..steps).rev() {
        let v_next = if t == steps - 1 {
            last_v
        } else {
            values_vec[t + 1]
        };

        let b_mask = if !term_vec[t] { 1.0 } else { 0.0 };
        let d_mask = if !dones_vec[t] { 1.0 } else { 0.0 };

        let delta = rewards_vec[t] + (gamma * v_next * b_mask) - values_vec[t];
        let advantage = delta + (gamma * 0.95 * d_mask * last_gae);

        advantages[t] = advantage;
        last_gae = advantage;
    }

    Tensor::<B, 1>::from_data(advantages.as_slice(), &rewards.device()).reshape([BATCH_SIZE, 1])
}

pub fn training<B: GaeAutodiffBackend, E: Env<B> + Clone>(
    mut env: E,
    data_tx: &Sender<Data<B, E>>,
    curiosity: &mut f32,
) {
    let device = B::Device::default();

    let mut pursuer: Actor<B> =
        ActorConfig::new(PURSUER_FACTORS, N_DIRECTIONS as usize).init(&device);
    let mut p_critic: Critic<B> = CriticConfig::new(PURSUER_FACTORS).init(&device);

    let mut target: Actor<B> =
        ActorConfig::new(TARGET_FACTORS, N_DIRECTIONS as usize).init(&device);
    let mut t_critic: Critic<B> = CriticConfig::new(TARGET_FACTORS).init(&device);

    // Actor optimizers
    let mut p_optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let mut t_optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    // Critic optimizers
    let mut pc_optimizer = AdamWConfig::new()
        .with_cautious_weight_decay(true)
        .with_weight_decay(1e-3)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let mut tc_optimizer = AdamWConfig::new()
        .with_cautious_weight_decay(true)
        .with_weight_decay(1e-3)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let p_lr_config = ExponentialLrSchedulerConfig::new(1e-4, LR_GAMMA);
    let pc_lr_config = ExponentialLrSchedulerConfig::new(3e-4, LR_GAMMA);
    let t_lr_config = ExponentialLrSchedulerConfig::new(1e-4, LR_GAMMA);
    let tc_lr_config = ExponentialLrSchedulerConfig::new(3e-4, LR_GAMMA);

    let mut p_scheduler = p_lr_config.init().unwrap();
    let mut pc_scheduler = pc_lr_config.init().unwrap();
    let mut t_scheduler = t_lr_config.init().unwrap();
    let mut tc_scheduler = tc_lr_config.init().unwrap();

    let mut checkpoint_timer = Instant::now();

    let mut p_rollout = BatchCollector::new();
    let mut t_rollout = BatchCollector::new();

    loop {
        env.reset();

        let (mut p_state, _) = env.state_tensor(Perspective::Pursuer, &device);
        let (mut t_state, _) = env.state_tensor(Perspective::Target, &device);

        loop {
            let _ = data_tx.send(Data {
                env: env.clone(),
                curiosity: Some(*curiosity),
                _phantom: PhantomData,
            });

            let p_action = gumbel_sample(pursuer.forward(p_state.clone()));
            let t_action = gumbel_sample(target.forward(t_state.clone()));

            let (p_step, t_step) =
                env.step_simultaneous(p_action.clone(), t_action.clone(), &device);

            p_rollout.push(p_state, p_action, p_step.clone());
            t_rollout.push(t_state, t_action, t_step.clone());

            p_state = p_step.next_state.detach();
            t_state = t_step.next_state.detach();

            if p_rollout.len() == BATCH_SIZE as usize {
                let p_context = UpdateContext {
                    actor_learner: Learner {
                        model: &mut pursuer,
                        optimizer: &mut p_optimizer,
                        scheduler: &mut p_scheduler,
                    },
                    critic_learner: Learner {
                        model: &mut p_critic,
                        optimizer: &mut pc_optimizer,
                        scheduler: &mut pc_scheduler,
                    },
                    rollout: p_rollout,
                    next_state: p_state.clone(),
                    curiosity: *curiosity,
                    device: &device,
                };

                update_agent(p_context);

                let t_context = UpdateContext {
                    actor_learner: Learner {
                        model: &mut target,
                        optimizer: &mut t_optimizer,
                        scheduler: &mut t_scheduler,
                    },
                    critic_learner: Learner {
                        model: &mut t_critic,
                        optimizer: &mut tc_optimizer,
                        scheduler: &mut tc_scheduler,
                    },
                    rollout: t_rollout,
                    next_state: t_state.clone(),
                    curiosity: *curiosity,
                    device: &device,
                };

                update_agent(t_context);

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
