use crate::consts::*;
use crate::env::{
    batch::BatchCollector,
    context::{Env, Perspective},
};
use crate::render::utils::Data;
use crate::rl::{
    actor::{Actor, ActorConfig},
    critic::{Critic, CriticConfig},
    stochastic::gumbel_sample,
};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::Module,
    nn::loss::{MseLoss, Reduction},
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::ToElement,
    record::CompactRecorder,
    tensor::{
        Tensor,
        activation::{log_softmax, softmax},
        backend::AutodiffBackend,
    },
};
use macroquad::prelude::*;
use std::{
    marker::PhantomData,
    path::Path,
    sync::mpsc::Sender,
    time::{Duration, Instant},
};

const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(5);

pub fn training<B: AutodiffBackend, E: Env<B> + Clone>(
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
        .with_cautious_weight_decay(true)
        .with_weight_decay(1e-2)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let mut tc_optimizer = AdamWConfig::new()
        .with_cautious_weight_decay(true)
        .with_weight_decay(1e-2)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let mut checkpoint_timer = Instant::now();
    let update_agent = |actor: &mut Actor<B>,
                        critic: &mut Critic<B>,
                        actor_optimizer: &mut OptimizerAdaptor<AdamW, Actor<B>, B>,
                        critic_optimizer: &mut OptimizerAdaptor<AdamW, Critic<B>, B>,
                        rollout: BatchCollector<B>,
                        curiosity: f32| {
        let states = Tensor::cat(rollout.states, 0);
        let actions = Tensor::cat(rollout.actions, 0);
        let rewards = Tensor::cat(rollout.rewards, 0);
        let next_states = Tensor::cat(rollout.next_states, 0);
        let dones = Tensor::cat(rollout.dones, 0);

        let values = critic.forward(states.clone());
        let next_values = critic.forward(next_states).detach();
        let td_target = rewards + (next_values * (dones.neg() + 1.0) * GAMMA);
        let advantage = td_target.clone() - values.clone();

        let critic_loss = MseLoss::new().forward(values, td_target, Reduction::Mean);
        *critic = critic_optimizer.step(
            3e-4,
            critic.clone(),
            GradientsParams::from_grads(critic_loss.backward(), critic),
        );

        let logits = actor.forward(states);
        let log_probs = log_softmax(logits.clone(), 1);
        let probs = softmax(logits, 1);

        let entropy_loss = -(probs * log_probs.clone()).sum_dim(1).mean();
        let pick = log_probs.gather(1, actions);
        let actor_loss = -(pick * advantage.detach()).mean() - curiosity * entropy_loss;

        *actor = actor_optimizer.step(
            1e-4,
            actor.clone(),
            GradientsParams::from_grads(actor_loss.backward(), actor),
        );
    };

    loop {
        env.reset();
        let mut p_rollout = BatchCollector::new(BATCH_SIZE);
        let mut t_rollout = BatchCollector::new(BATCH_SIZE);
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

            p_rollout.push(p_state.clone(), p_action, p_step.clone());
            t_rollout.push(t_state.clone(), t_action, t_step.clone());

            p_state = p_step.next_states.clone();
            t_state = t_step.next_states.clone();

            if p_rollout.states.len() >= BATCH_SIZE {
                update_agent(
                    &mut pursuer,
                    &mut p_critic,
                    &mut p_optimizer,
                    &mut pc_optimizer,
                    p_rollout,
                    *curiosity,
                );
                update_agent(
                    &mut target,
                    &mut t_critic,
                    &mut t_optimizer,
                    &mut tc_optimizer,
                    t_rollout,
                    *curiosity,
                );

                p_rollout = BatchCollector::new(BATCH_SIZE);
                t_rollout = BatchCollector::new(BATCH_SIZE);
            }

            *curiosity = (*curiosity * CURIOSITY_DECAY).max(CURIOSITY_MIN);

            if p_step.dones.clone().any().into_scalar().to_bool()
                || t_step.dones.clone().any().into_scalar().to_bool()
            {
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
