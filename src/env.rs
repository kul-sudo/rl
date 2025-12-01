use crate::consts::*;
use ::rand::{Rng, rng};
use burn::{
    prelude::ToElement,
    tensor::{Tensor, backend::Backend},
};
use macroquad::prelude::*;
use num_complex::{Complex32, ComplexFloat, c32};
use parry2d::{math::Isometry, query::contact, shape::Ball};
use std::f32::consts::PI;

pub struct Step<B: Backend, const D: usize> {
    pub next_state: Tensor<B, D>,
    pub reward: f32,
    pub done: bool,
}

impl<B: Backend, const D: usize> Step<B, D> {
    fn new(next_state: Tensor<B, D>, reward: f32, done: bool) -> Self {
        Self {
            next_state,
            reward,
            done,
        }
    }
}

pub trait Env<B: Backend, const D: usize> {
    fn reset(&mut self, device: &B::Device) -> Tensor<B, D>;

    fn step(&mut self, action: B::IntElem, device: &B::Device) -> Step<B, D>;

    fn state_tensor(&self, device: &B::Device) -> Tensor<B, D>;
}

#[derive(Clone)]
pub struct BallEnv {
    pub body_pos: Complex32,
    pub target_pos: Complex32,
    pub life: u32,
}

impl<B: Backend, const D: usize> Env<B, D> for BallEnv {
    fn reset(&mut self, device: &B::Device) -> Tensor<B, D> {
        let init = Self::new();
        *self = init;
        self.state_tensor(device)
    }

    fn state_tensor(&self, device: &B::Device) -> Tensor<B, D> {
        Tensor::from_floats(
            [[
                self.body_pos.im(),
                self.body_pos.re(),
                self.target_pos.im(),
                self.target_pos.re(),
                self.life as f32 / LIFE_LENGTH as f32,
            ]],
            device,
        )
    }

    fn step(&mut self, action: B::IntElem, device: &B::Device) -> Step<B, D> {
        let reward;
        let done;

        if self.life == LIFE_LENGTH {
            (reward, done) = (f32::MIN, true);
        } else {
            let prev_distance = (self.body_pos - self.target_pos).abs();

            let action_idx = action.to_u32() % N_DIRECTIONS;
            let angle = 2.0 * PI * (action_idx as f32) / N_DIRECTIONS as f32;

            self.body_pos.re += SPEED * angle.cos();
            self.body_pos.im += SPEED * angle.sin();

            self.body_pos.re = self.body_pos.re().clamp(0.0, 1.0);
            self.body_pos.im = self.body_pos.im().clamp(0.0, 1.0);

            let collision = self.hits();

            (reward, done, self.life) = if collision {
                (REWARD_CONSUMED, true, 0)
            } else {
                let new_distance = (self.body_pos - self.target_pos).abs();
                let distance_improvement = prev_distance - new_distance;
                (
                    distance_improvement * REWARD_CONSUMED - self.life as f32 * 0.002,
                    false,
                    self.life + 1,
                )
            };
            let new_distance = (self.body_pos - self.target_pos).abs();
            let distance_improvement = prev_distance - new_distance;

            dbg!(
                distance_improvement * REWARD_CONSUMED,
                self.life as f32 * 0.002
            );
        }

        // self.energy -= ENERGY_LIFE + REWARD_CONSUMED * self.hits() as u8 as f32;
        //
        // let new_distance = (self.body_pos - self.target_pos).abs();
        // let distance_improvement = prev_distance - new_distance;
        // let reward = self.energy - prev_energy + distance_improvement * DISTANCE_IMPROVEMENT_FACTOR;

        Step::new(self.state_tensor(device), reward, done)
    }
}

impl BallEnv {
    pub fn new() -> Self {
        Self {
            body_pos: c32(rng().random_range(0.0..=1.0), rng().random_range(0.0..=1.0)),
            target_pos: c32(rng().random_range(0.0..=1.0), rng().random_range(0.0..=1.0)),
            life: 0,
        }
    }

    pub fn hits(&self) -> bool {
        let agent_ball = Ball::new(AGENT_RADIUS / SIZE.re());
        let target_ball = Ball::new(TARGET_RADIUS / SIZE.re());
        let agent_pos = Isometry::translation(self.body_pos.re(), self.body_pos.im());
        let target_pos = Isometry::translation(self.target_pos.re(), self.target_pos.im());
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
