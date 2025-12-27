use crate::consts::*;
use crate::env::{
    context::{Env, Perspective},
    step::Step,
};
use crate::render::utils::Data;
use crate::rl::{
    actor::{Actor, ActorConfig},
    critic::{Critic, CriticConfig},
};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::Module,
    nn::loss::{MseLoss, Reduction},
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::ToElement,
    record::CompactRecorder,
    tensor::{
        Distribution, ElementConversion, Tensor,
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

    loop {
        env.reset();

        let (mut p_state, _) = env.state_tensor(Perspective::Pursuer, device);
        let (mut t_state, _) = env.state_tensor(Perspective::Target, device);

        let probs_sample = |logits: Tensor<B, 2>| {
            let gumbel_noise = logits
                .random_like(Distribution::Uniform(1e-5, 1.0))
                .log()
                .neg()
                .log()
                .neg();

            let sampled_indices = (logits + gumbel_noise.clone()).argmax(1);

            if gumbel_noise.contains_nan().into_scalar().to_bool() {
                panic!();
            }

            sampled_indices.into_scalar()
        };

        let update_agent = |actor: &mut Actor<B>,
                            critic: &mut Critic<B>,
                            actor_optimizer: &mut OptimizerAdaptor<AdamW, Actor<B>, B>,
                            critic_optimizer: &mut OptimizerAdaptor<AdamW, Critic<B>, B>,
                            state: &mut Tensor<B, 2>,
                            step: &Step<B>,
                            action: B::IntElem,
                            curiosity: f32| {
            let reward_tensor = Tensor::from_floats([[step.reward]], device);
            let value = critic.forward(state.clone());
            let next_value = if step.done {
                Tensor::zeros([1, 1], device)
            } else {
                critic.forward(step.next_state.clone())
            };
            let td_target: Tensor<B, 2> = reward_tensor + GAMMA * next_value.detach();
            let advantage = td_target.clone() - value.clone();

            let critic_loss = MseLoss::new().forward(value, td_target, Reduction::Mean);
            let grads = critic_loss.backward();
            let grads_params = GradientsParams::from_grads(grads, critic);
            *critic = critic_optimizer.step(3e-4, critic.clone(), grads_params);

            let logits = actor.forward(state.clone());
            if logits.clone().contains_nan().into_scalar().to_bool() {
                panic!();
            }

            let log_probs = log_softmax(logits.clone(), 1);
            let probs = softmax(logits, 1);
            let entropy = -(probs.clone() * log_probs.clone()).sum_dim(1);
            let entropy_loss = entropy.mean();

            let pick = log_probs.gather(1, Tensor::from_data([[action]], device));
            let actor_loss = -(pick * advantage.detach()).mean() - curiosity * entropy_loss;

            let grads = actor_loss.backward();
            let grads_params = GradientsParams::from_grads(grads, actor);
            *actor = actor_optimizer.step(1e-4, actor.clone(), grads_params);

            *state = step.next_state.clone();
        };

        loop {
            let _ = data_tx.send(Data {
                env: env.clone(),
                curiosity: Some(*curiosity),
                _phantom: PhantomData,
            });

            let p_logits = pursuer.forward(p_state.clone());
            let p_action = B::IntElem::from_elem(probs_sample(p_logits));

            let t_logits = target.forward(t_state.clone());
            let t_action = B::IntElem::from_elem(probs_sample(t_logits));

            let (p_step, t_step) = env.step_simultaneous(p_action, t_action, device);

            update_agent(
                &mut pursuer,
                &mut p_critic,
                &mut p_optimizer,
                &mut pc_optimizer,
                &mut p_state,
                &p_step,
                p_action,
                *curiosity,
            );

            update_agent(
                &mut target,
                &mut t_critic,
                &mut t_optimizer,
                &mut tc_optimizer,
                &mut t_state,
                &t_step,
                t_action,
                *curiosity,
            );

            *curiosity = (*curiosity * CURIOSITY_DECAY).max(CURIOSITY_MIN);

            if p_step.done || t_step.done {
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
