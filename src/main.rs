mod rl;

use crate::rl::{
    actor::{Actor, ActorConfig},
    critic::{Critic, CriticConfig},
};
use ::rand::{Rng, rng};
use burn::{
    backend::{Autodiff, Cuda, cuda::CudaDevice},
    grad_clipping::GradientClippingConfig,
    nn::loss::{MseLoss, Reduction},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::{Tensor, activation::log_softmax, bf16},
};
use macroquad::prelude::*;
use miniquad::conf::{LinuxBackend, Platform};
use nalgebra::{Point2, Vector2};
use num_complex::{Complex32, ComplexFloat, c32};
use parry2d::{
    math::Isometry,
    query::{Ray, RayCast, contact},
    shape::Ball,
};
use serde::Deserialize;
use std::f32::consts::PI;

pub type TrainingBackend = Autodiff<Cuda<bf16>>;

pub const OBSERVATION: usize = 4;
pub const SIZE: Complex32 = Complex32::new(1920.0, 1080.0);
pub const GAMMA: f32 = 0.9;
pub const CONSUMED_REWARD: f32 = 5.0;
pub const N_DIRECTIONS: usize = 80;

pub const AGENT_RADIUS: f32 = 5.0;
pub const TARGET_RADIUS: f32 = 5.0;
pub const SPEED: f32 = 0.01;

#[derive(Deserialize)]
enum Mode {
    Train,
    Test,
}

#[derive(Deserialize)]
pub struct TomlConfig {
    mode: Mode,
}

struct BallEnv {
    body_pos: Complex32,
    target_pos: Complex32,
}

impl BallEnv {
    fn new() -> Self {
        Self {
            body_pos: c32(rng().random_range(0.0..=1.0), rng().random_range(0.0..=1.0)),
            target_pos: c32(rng().random_range(0.0..=1.0), rng().random_range(0.0..=1.0)),
        }
    }

    fn reset(&mut self) -> Tensor<TrainingBackend, 2> {
        *self = Self::new();
        self.state_tensor()
    }

    fn state_tensor(&self) -> Tensor<TrainingBackend, 2> {
        Tensor::from_floats(
            [[
                self.body_pos.im(),
                self.body_pos.re(),
                self.target_pos.im(),
                self.target_pos.re(),
            ]],
            &CudaDevice::default(),
        )
    }

    fn step(&mut self, action: i32) -> (Tensor<TrainingBackend, 2>, f32, bool) {
        let prev_pos = self.body_pos;
        let prev_distance = (prev_pos - self.target_pos).abs();

        let action_idx = action as usize % N_DIRECTIONS;
        let angle = 2.0 * PI * (action_idx as f32) / N_DIRECTIONS as f32;

        self.body_pos.re += SPEED * angle.cos();
        self.body_pos.im += SPEED * angle.sin();

        self.body_pos.re = self.body_pos.re().clamp(0.0, 1.0);
        self.body_pos.im = self.body_pos.im().clamp(0.0, 1.0);

        let new_distance = (self.body_pos - self.target_pos).abs();
        let distance_improvement = (prev_distance - new_distance) * CONSUMED_REWARD;

        let hit = self.hits(prev_pos);
        let reward = if hit {
            CONSUMED_REWARD
        } else {
            distance_improvement
        };

        (self.state_tensor(), reward, hit)
    }

    fn hits(&self, prev_pos: Complex32) -> bool {
        let agent_ball = Ball::new(AGENT_RADIUS / SIZE.re);
        let target_ball = Ball::new(TARGET_RADIUS / SIZE.re);

        let agent_pos = Isometry::translation(prev_pos.re(), prev_pos.im());
        let target_pos = Isometry::translation(self.target_pos.re(), self.target_pos.im());

        let ray_dir = Vector2::new(
            self.body_pos.re() - prev_pos.re(),
            self.body_pos.im() - prev_pos.im(),
        );
        let ray = Ray::new(Point2::new(prev_pos.re(), prev_pos.im()), ray_dir);

        if target_ball.cast_ray(&target_pos, &ray, 1.0, true).is_some() {
            return true;
        }

        contact(&agent_pos, &agent_ball, &target_pos, &target_ball, 0.0)
            .map(|c| c.is_some())
            .unwrap_or(false)
    }
}

async fn render(env: &BallEnv) {
    clear_background(WHITE);
    let to_screen = |pos: Complex32| vec2(SIZE.re() * pos.re(), SIZE.im() * pos.im());

    let p = to_screen(env.body_pos);
    let t = to_screen(env.target_pos);

    draw_circle(p.x, p.y, AGENT_RADIUS, BLUE);
    draw_circle(t.x, t.y, TARGET_RADIUS, GREEN);

    next_frame().await;
}

fn window_conf() -> Conf {
    Conf {
        window_title: "training".to_owned(),
        fullscreen: true,
        platform: Platform {
            linux_backend: LinuxBackend::WaylandWithX11Fallback,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
pub async fn main() {
    let device = CudaDevice::default();

    let actor_config = ActorConfig::new(OBSERVATION, N_DIRECTIONS, 512);
    let critic_config = CriticConfig::new(OBSERVATION, 512);

    let mut actor: Actor<TrainingBackend> = actor_config.init(&device);
    let mut critic: Critic<TrainingBackend> = critic_config.init(&device);

    let mut actor_optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let mut critic_optimizer = AdamWConfig::new()
        .with_cautious_weight_decay(true)
        .with_weight_decay(1e-3)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let mut env = BallEnv::new();
    let mut state = env.reset();
    let mut epsilon = 0.1;

    for epoch in 0..100_000 {
        let mut total_reward = 0.0;
        let mut done = false;

        while !done {
            let logits = actor.forward(state.clone());

            let action = if rng().random::<f32>() < epsilon {
                rng().random_range(0..N_DIRECTIONS) as i32
            } else {
                logits.clone().argmax(1).into_scalar()
            };

            let (next_state, reward, is_done) = env.step(action);
            total_reward += reward;
            render(&env).await;

            let reward_tensor = Tensor::from_floats([[reward]], &device);

            let value = critic.forward(state.clone());
            let next_value = if is_done {
                Tensor::zeros([1, 1], &device)
            } else {
                critic.forward(next_state.clone())
            };

            let td_target: Tensor<TrainingBackend, 2> = reward_tensor + GAMMA * next_value;
            let advantage = td_target.clone() - value.clone();

            let mse_loss = MseLoss::new();
            let critic_loss = mse_loss.forward(value, td_target.detach(), Reduction::Mean);

            let grads = critic_loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &critic);
            critic = critic_optimizer.step(1e-4, critic, grads_params);

            let log_probs = log_softmax(logits, 1);
            let log_prob = log_probs.gather(1, Tensor::from_data([[action]], &device));

            let actor_loss = -log_prob * advantage.detach();

            let grads = actor_loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &actor);
            actor = actor_optimizer.step(1e-4, actor, grads_params);

            state = if is_done { env.reset() } else { next_state };
            done = is_done;
        }

        epsilon = (epsilon * 0.995).max(0.01);
        println!("Epoch: {}, Total Reward: {:.3}", epoch, total_reward);
    }
}
