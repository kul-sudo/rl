use super::step::*;
use super::walls::{WALLS, pos_invalid};
use crate::consts::*;
use ::rand::{Rng, rng};
use burn::{
    prelude::ToElement,
    tensor::{Tensor, backend::Backend},
};
use macroquad::prelude::*;
use num_complex::{Complex32, ComplexFloat, c32};
use parry2d::{
    math::Isometry,
    query::{contact, intersection_test},
    shape::{Ball, Compound, SharedShape},
};
use std::f32::consts::PI;

pub trait Env<B: Backend> {
    fn reset(&mut self);
    fn state_tensor(&self, perspective: Perspective, device: &B::Device) -> Tensor<B, 2>;
    fn step_simultaneous(
        &mut self,
        p_action: B::IntElem,
        t_action: B::IntElem,
        device: &B::Device,
    ) -> (Step<B>, Step<B>);
}

pub enum Perspective {
    Pursuer,
    Target,
}

fn advance(pos: &mut Complex32, angle: f32, speed: f32) {
    let prev = pos.clone();
    pos.re += speed * angle.cos();
    pos.im += speed * angle.sin();
    pos.re = pos.re().clamp(0.0, 1.0);
    pos.im = pos.im().clamp(0.0, 1.0);

    if pos_invalid(&pos, SharedShape::ball(RADIUS)) {
        *pos = prev;
    }
}

#[derive(Clone, Debug)]
pub struct Pursuer {
    pos: Complex32,
    time: u32,
}

#[derive(Clone, Debug)]
pub struct Target {
    pos: Complex32,
}

#[derive(Clone, Debug)]
pub struct BallEnv {
    pub pursuer: Pursuer,
    pub target: Target,
}

impl<B: Backend> Env<B> for BallEnv {
    fn reset(&mut self) {
        *self = Self::new();
    }

    fn state_tensor(&self, perspective: Perspective, device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_floats(
            [match perspective {
                Perspective::Pursuer => [
                    self.pursuer.pos.re(),
                    self.pursuer.pos.im(),
                    self.target.pos.re() - self.pursuer.pos.re(),
                    self.target.pos.im() - self.pursuer.pos.im(),
                ],
                Perspective::Target => [
                    self.target.pos.re(),
                    self.target.pos.im(),
                    self.pursuer.pos.re() - self.target.pos.re(),
                    self.pursuer.pos.im() - self.target.pos.im(),
                ],
            }],
            device,
        )
        // let dist = (self.pursuer.pos - self.target.pos).abs();
        // let sigma = 0.4 * dist.powi(2);
        //
        // if B::ad_enabled() {
        //     println!("Noise: {}", sigma);
        // }
        //
        // let noise = GaussianNoiseConfig::new(sigma as f64).init();
        // noise.forward(tensor)
    }

    fn step_simultaneous(
        &mut self,
        p_action: B::IntElem,
        t_action: B::IntElem,
        device: &B::Device,
    ) -> (Step<B>, Step<B>) {
        let initial = self.clone();
        let p_angle = 2.0 * PI * (p_action.to_u32() % N_DIRECTIONS) as f32 / N_DIRECTIONS as f32;
        let t_angle = 2.0 * PI * (t_action.to_u32() % N_DIRECTIONS) as f32 / N_DIRECTIONS as f32;

        advance(&mut self.pursuer.pos, p_angle, PURSUER_SPEED);
        advance(&mut self.target.pos, t_angle, TARGET_SPEED);

        let collision = self.hits();

        let prev_distance = (initial.pursuer.pos - initial.target.pos).abs();
        let new_distance = (self.pursuer.pos - self.target.pos).abs();

        let total_speed = PURSUER_SPEED + TARGET_SPEED;

        // Pursuer
        let distance_reward = (prev_distance - new_distance) / total_speed;
        let p_reward = if collision { 5.0 } else { distance_reward };
        self.pursuer.time += 1;
        let expired = self.pursuer.time == PURSUER_TIME_CAP;
        let p_done = collision || expired;

        let p_step = Step::new(
            self.state_tensor(Perspective::Pursuer, device),
            p_reward,
            p_done,
        );

        // Target
        let t_reward = if collision {
            -20.0
        } else if expired {
            5.0
        } else {
            0.001
        };
        let t_done = collision;

        let t_step = Step::new(
            self.state_tensor(Perspective::Target, device),
            t_reward,
            t_done,
        );

        (p_step, t_step)
    }
}

impl BallEnv {
    pub fn new() -> Self {
        let valid_spawn = || {
            loop {
                let pos = c32(rng().random_range(0.0..=1.0), rng().random_range(0.0..=1.0));
                if !pos_invalid(&pos, SharedShape::ball(RADIUS)) {
                    break pos;
                }
            }
        };

        Self {
            pursuer: Pursuer {
                pos: valid_spawn(),
                time: 0,
            },
            target: Target { pos: valid_spawn() },
        }
    }

    pub fn hits(&self) -> bool {
        let pursuer_pos = Isometry::translation(
            self.pursuer.pos.re() * SIZE.re(),
            self.pursuer.pos.im() * SIZE.im(),
        );
        let target_pos = Isometry::translation(
            self.target.pos.re() * SIZE.re(),
            self.target.pos.im() * SIZE.im(),
        );

        let ball = Ball::new(RADIUS);

        contact(&pursuer_pos, &ball, &target_pos, &ball, 0.0)
            .map(|c| c.is_some())
            .unwrap()
    }

    pub fn render(&self) {
        let to_screen = |pos: Complex32| vec2(SIZE.re() * pos.re(), SIZE.im() * pos.im());

        let p = to_screen(self.pursuer.pos);
        let t = to_screen(self.target.pos);

        draw_circle(p.x, p.y, RADIUS, BLUE);
        draw_circle(t.x, t.y, RADIUS, GREEN);
    }
}
