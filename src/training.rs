use crate::consts::*;
use crate::env::Env;
use crate::rl::{
    actor::{Actor, ActorConfig},
    critic::{Critic, CriticConfig},
};
use ::rand::{Rng, rng};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::Module,
    nn::loss::{CrossEntropyLossConfig, MseLoss, Reduction},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::ToElement,
    record::CompactRecorder,
    tensor::{ElementConversion, Tensor, backend::AutodiffBackend},
};
use macroquad::prelude::*;
use std::{path::Path, sync::mpsc::Sender};

pub fn training<B: AutodiffBackend, E: Env<B> + Clone + std::fmt::Debug>(
    actor_config: &ActorConfig,
    mut env: E,
    env_tx: &Sender<E>,
    epsilon_tx: &Sender<f32>,
    epsilon: &mut f32,
    device: &B::Device,
) {
    let mut actor_optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let critic_config = CriticConfig::new(FACTORS, 512);
    let mut actor: Actor<B> = actor_config.init(device);
    let mut critic: Critic<B> = critic_config.init(device);

    let mut critic_optimizer = AdamWConfig::new()
        .with_cautious_weight_decay(true)
        .with_weight_decay(1e-3)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    loop {
        let mut state = env.reset(device);

        loop {
            let logits = actor.forward(state.clone());
            if logits.clone().contains_nan().into_scalar().to_bool() {
                panic!("NaN contained")
            }

            let action = if rng().random::<f32>() < *epsilon {
                (rng().random_range(0..N_DIRECTIONS)).elem::<B::IntElem>()
            } else {
                logits.clone().argmax(1).into_scalar()
            };

            let step = env.step(action, device);
            let _ = env_tx.send(env.clone());

            let reward_tensor = Tensor::from_floats([[step.reward]], device);
            let value = critic.forward(state.clone());

            let next_value = if step.done {
                Tensor::zeros([1, 1], device)
            } else {
                critic.forward(step.next_state.clone())
            };

            let td_target: Tensor<B, 2> = reward_tensor + GAMMA * next_value;

            let advantage = td_target.clone() - value.clone();

            let mse_loss = MseLoss::new();
            let critic_loss = mse_loss.forward(value, td_target.detach(), Reduction::Mean);

            let grads = critic_loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &critic);
            critic = critic_optimizer.step(1e-4, critic, grads_params);

            let loss_fn = CrossEntropyLossConfig::new().init(device);
            let log_prob = -loss_fn.forward(logits, Tensor::from_data([action], device));
            let actor_loss = -log_prob * advantage.squeeze_dim(1).detach();

            let grads = actor_loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &actor);
            actor = actor_optimizer.step(1e-4, actor, grads_params);

            *epsilon = (*epsilon * EPSILON_DECAY).max(EPSILON_MIN);
            let _ = epsilon_tx.send(*epsilon);

            if step.done {
                break;
            } else {
                state = step.next_state;
            }
        }

        actor
            .clone()
            .save_file(
                Path::new(ARTIFACT_DIR).join("actor"),
                &CompactRecorder::new(),
            )
            .unwrap();
        critic
            .clone()
            .save_file(
                Path::new(ARTIFACT_DIR).join("critic"),
                &CompactRecorder::new(),
            )
            .unwrap()
    }
}
