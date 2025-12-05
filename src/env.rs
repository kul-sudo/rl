use crate::consts::*;
use ::rand::{Rng, rng};
use burn::{
    nn::GaussianNoiseConfig,
    prelude::ToElement,
    tensor::{Tensor, backend::Backend},
};
use macroquad::prelude::*;
use num_complex::{Complex32, ComplexFloat, c32};
use parry2d::{math::Isometry, query::contact, shape::Ball};
use std::f32::consts::PI;

#[derive(Clone, Debug)]
pub struct Step<B: Backend> {
    pub next_state: Tensor<B, 2>,
    pub reward: f32,
    pub done: bool,
}

impl<B: Backend> Step<B> {
    fn new(next_state: Tensor<B, 2>, reward: f32, done: bool) -> Self {
        Self {
            next_state,
            reward,
            done,
        }
    }
}

pub trait Env<B: Backend> {
    fn reset(&mut self, device: &B::Device) -> Tensor<B, 2>;

    fn step(&mut self, action: B::IntElem, device: &B::Device) -> Step<B>;

    fn state_tensor(&self, device: &B::Device) -> Tensor<B, 2>;
}

const REWARD_FACTOR: f32 = 1000.0;

#[derive(Clone, Debug)]
pub struct BallEnv {
    pub body_pos: Complex32,
    pub target_pos: Complex32,
    pub time: u32,
}

impl<B: Backend> Env<B> for BallEnv {
    fn reset(&mut self, device: &B::Device) -> Tensor<B, 2> {
        let init = Self::new();
        *self = Self {
            body_pos: self.body_pos,
            ..init
        };
        self.state_tensor(device)
    }

    fn state_tensor(&self, device: &B::Device) -> Tensor<B, 2> {
        let tensor = Tensor::from_floats(
            [[
                self.body_pos.re(),
                self.body_pos.im(),
                self.target_pos.re() - self.body_pos.re(),
                self.target_pos.im() - self.body_pos.im(),
            ]],
            device,
        );

        let dist = (self.target_pos - self.body_pos).abs();
        let sigma = 0.4 * dist.powi(2);

        if B::ad_enabled() {
            println!("Noise: {}", sigma);
        }

        let noise = GaussianNoiseConfig::new(sigma as f64).init();
        noise.forward(tensor)
    }

    fn step(&mut self, action: B::IntElem, device: &B::Device) -> Step<B> {
        let reward;
        let done;

        if self.time == TIME_CAP {
            (reward, done) = (-20.0, true);
        } else {
            let prev_distance = (self.body_pos - self.target_pos).abs();
            if prev_distance.is_nan() {
                panic!("Distance is NaN")
            }

            let action_idx = action.to_u32() % N_DIRECTIONS;
            let angle = 2.0 * PI * (action_idx as f32) / N_DIRECTIONS as f32;

            self.body_pos.re += SPEED * angle.cos();
            self.body_pos.im += SPEED * angle.sin();

            self.body_pos.re = self.body_pos.re().clamp(0.0, 1.0);
            self.body_pos.im = self.body_pos.im().clamp(0.0, 1.0);

            let collision = self.hits();

            (reward, done, self.time) = if collision {
                (20.0, true, 0)
            } else {
                let new_distance = (self.body_pos - self.target_pos).abs();
                let distance_improvement = prev_distance - new_distance;

                let mut raw = distance_improvement * REWARD_FACTOR;
                if B::ad_enabled() {
                    println!("Pure: {}", raw);
                }

                let time_penalty = raw.abs() * (self.time as f32 / TIME_CAP as f32).powf(1.0 / 3.0);
                raw -= time_penalty;

                if B::ad_enabled() {
                    println!("Time penalty: {}", raw);
                }

                (raw, false, self.time + 1)
            };

            // dbg!(reward);

            // let new_distance = (self.body_pos - self.target_pos).abs();
            // let distance_improvement = prev_distance - new_distance;
            // dbg!(
            //     distance_improvement * REWARD_CONSUMED,
            //     (self.time as f32 / TIME_CAP as f32) * 1.5
            // );
        }

        Step::new(self.state_tensor(device), reward, done)
    }
}

impl BallEnv {
    pub fn new() -> Self {
        Self {
            body_pos: c32(rng().random_range(0.0..=1.0), rng().random_range(0.0..=1.0)),
            target_pos: c32(rng().random_range(0.0..=1.0), rng().random_range(0.0..=1.0)),
            time: 0,
        }
    }

    pub fn hits(&self) -> bool {
        let agent_pos = Isometry::translation(
            self.body_pos.re() * SIZE.re(),
            self.body_pos.im() * SIZE.im(),
        );
        let target_pos = Isometry::translation(
            self.target_pos.re() * SIZE.re(),
            self.target_pos.im() * SIZE.im(),
        );

        let agent_ball = Ball::new(AGENT_RADIUS);
        let target_ball = Ball::new(TARGET_RADIUS);

        contact(&agent_pos, &agent_ball, &target_pos, &target_ball, 0.0)
            .map(|c| c.is_some())
            .unwrap_or(false)
    }

    pub fn render(&self) {
        let to_screen = |pos: Complex32| vec2(SIZE.re() * pos.re(), SIZE.im() * pos.im());

        let p = to_screen(self.body_pos);
        let t = to_screen(self.target_pos);

        draw_circle(p.x, p.y, AGENT_RADIUS, BLUE);
        draw_circle(t.x, t.y, TARGET_RADIUS, GREEN);
    }
}
