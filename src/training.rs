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
use ::rand::{Rng, rng};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::Module,
    nn::loss::{MseLoss, Reduction},
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    record::CompactRecorder,
    tensor::{ElementConversion, Tensor, activation::log_softmax, backend::AutodiffBackend},
};
use macroquad::prelude::*;
use std::{marker::PhantomData, path::Path, sync::mpsc::Sender};

pub fn training<B: AutodiffBackend, E: Env<B> + Clone>(
    mut env: E,
    data_tx: &Sender<Data<B, E>>,
    epsilon: &mut f64,
    device: &B::Device,
) {
    let mut pursuer: Actor<B> = ActorConfig::new(FACTORS, N_DIRECTIONS as usize, 1024).init(device);
    let mut p_critic: Critic<B> = CriticConfig::new(FACTORS, 1024).init(device);

    let mut target: Actor<B> = ActorConfig::new(FACTORS, N_DIRECTIONS as usize, 1024).init(device);
    let mut t_critic: Critic<B> = CriticConfig::new(FACTORS, 1024).init(device);

    let mut p_optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let mut t_optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

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

    loop {
        env.reset();

        let mut p_state = env.state_tensor(Perspective::Pursuer, device);
        let mut t_state = env.state_tensor(Perspective::Target, device);

        let update_agent = |actor: &mut Actor<B>,
                            critic: &mut Critic<B>,
                            actor_optimizer: &mut OptimizerAdaptor<AdamW, Actor<B>, B>,
                            critic_optimizer: &mut OptimizerAdaptor<AdamW, Critic<B>, B>,
                            state: &mut Tensor<B, 2>,
                            step: &Step<B>,
                            action: B::IntElem| {
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
            *critic = critic_optimizer.step(1e-4, critic.clone(), grads_params);

            let logits = actor.forward(state.clone());
            let log_probs = log_softmax(logits, 1);
            let pick = log_probs.gather(1, Tensor::from_data([[action]], device));

            let actor_loss = -(pick * advantage.detach()).mean();

            let grads = actor_loss.backward();
            let grads_params = GradientsParams::from_grads(grads, actor);
            *actor = actor_optimizer.step(1e-4, actor.clone(), grads_params);

            *state = step.next_state.clone();
        };

        loop {
            let _ = data_tx.send(Data {
                env: env.clone(),
                epsilon: Some(*epsilon),
                _phantom: PhantomData,
            });

            let p_action = if rng().random_bool(*epsilon) {
                rng().random_range(0..N_DIRECTIONS).elem::<B::IntElem>()
            } else {
                pursuer.forward(p_state.clone()).argmax(1).into_scalar()
            };

            let t_action = if rng().random_bool(*epsilon) {
                rng().random_range(0..N_DIRECTIONS).elem::<B::IntElem>()
            } else {
                target.forward(t_state.clone()).argmax(1).into_scalar()
            };

            let (p_step, t_step) = env.step_simultaneous(p_action, t_action, device);

            update_agent(
                &mut pursuer,
                &mut p_critic,
                &mut p_optimizer,
                &mut pc_optimizer,
                &mut p_state,
                &p_step,
                p_action,
            );

            update_agent(
                &mut target,
                &mut t_critic,
                &mut t_optimizer,
                &mut tc_optimizer,
                &mut t_state,
                &t_step,
                t_action,
            );

            *epsilon = (*epsilon * EPSILON_DECAY).max(EPSILON_MIN);

            if p_step.done || t_step.done {
                break;
            }
        }

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
