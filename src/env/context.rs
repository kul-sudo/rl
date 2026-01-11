use super::step::*;
use super::walls::{WALLS, pos_invalid};
use crate::consts::*;
use ::rand::{Rng, rng};
use burn::{
    prelude::ToElement,
    tensor::{Int, Tensor, backend::Backend},
};
use macroquad::prelude::*;
use num::{
    complex::{Complex32, ComplexFloat, c32},
    traits::identities::Zero,
};
use parry2d::{
    math::{Isometry, Point, Vector},
    query::{Ray, RayCast, contact},
    shape::Ball,
};
use std::f32::consts::{SQRT_2, TAU};

pub trait Env<B: Backend> {
    fn reset(&mut self);
    fn state_tensor(
        &mut self,
        perspective: Perspective,
        device: &B::Device,
    ) -> (Tensor<B, 2>, bool);
    fn step_simultaneous(
        &mut self,
        p_action: Tensor<B, 2, Int>,
        t_action: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> (Step<B>, Step<B>);
}

pub enum Perspective {
    Pursuer,
    Target,
}

fn advance(pos: &mut Complex32, angle: f32, speed: f32) {
    let movement = Complex32::from_polar(speed, angle);

    if !movement.is_zero() {
        let ray_dir = Vector::new(movement.re(), movement.im()).normalize();
        let ray = Ray::new(Point::new(pos.re(), pos.im()), ray_dir);

        match WALLS.cast_ray_and_get_normal(&Isometry::identity(), &ray, speed, true) {
            Some(hit) => {
                let normal = Vector::new(hit.normal.x, hit.normal.y);
                let tangent = Vector::new(-normal.y, normal.x);

                let remaining_movement =
                    movement - Complex32::from_polar(hit.time_of_impact, angle);
                let slide_dir = tangent.dot(&Vector::new(
                    remaining_movement.re(),
                    remaining_movement.im(),
                ));

                *pos += c32(tangent.x * slide_dir, tangent.y * slide_dir);
            }
            None => {
                *pos += movement;
            }
        }
    }

    pos.re = pos.re().clamp(1e-8, 1.0 - 1e-8);
    pos.im = pos.im().clamp(1e-8, 1.0 - 1e-8);
}

#[derive(Clone, Debug)]
pub struct Pursuer {
    pos: Complex32,
    pub time: u32,
}

impl Pursuer {
    pub fn age(&self) -> f32 {
        self.time as f32 / PURSUER_TIME_CAP as f32
    }
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

    fn state_tensor(
        &mut self,
        perspective: Perspective,
        device: &B::Device,
    ) -> (Tensor<B, 2>, bool) {
        let wall_blocks = self.wall_blocks();

        match perspective {
            Perspective::Pursuer => {
                let mut context = if wall_blocks {
                    vec![self.pursuer.pos.re(), self.pursuer.pos.im(), -1.0, -1.0]
                } else {
                    vec![
                        self.pursuer.pos.re(),
                        self.pursuer.pos.im(),
                        self.target.pos.re() - self.pursuer.pos.re(),
                        self.target.pos.im() - self.pursuer.pos.im(),
                    ]
                };

                let lasers = self.lasers(true, self.pursuer.pos);
                context.extend(lasers.iter().flat_map(|laser| [laser.re(), laser.im()]));

                let factors: [f32; PURSUER_FACTORS] = context.try_into().unwrap();
                (
                    Tensor::<B, 1>::from_data(factors, device).reshape([1, PURSUER_FACTORS]),
                    wall_blocks,
                )
            }
            Perspective::Target => {
                let wall_signal = (1.0 / 12.0).sqrt() * wall_blocks as u8 as f32;

                let mut context = vec![
                    wall_signal,
                    self.pursuer.age(),
                    self.target.pos.re(),
                    self.target.pos.im(),
                    self.pursuer.pos.re() - self.target.pos.re(),
                    self.pursuer.pos.im() - self.target.pos.im(),
                ];

                let lasers = self.lasers(true, self.target.pos);
                context.extend(lasers.iter().flat_map(|laser| [laser.re(), laser.im()]));

                let factors: [f32; TARGET_FACTORS] = context.try_into().unwrap();
                (
                    Tensor::<B, 1>::from_data(factors, device).reshape([1, TARGET_FACTORS]),
                    wall_blocks,
                )
            }
        }
    }

    fn step_simultaneous(
        &mut self,
        p_action: Tensor<B, 2, Int>,
        t_action: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> (Step<B>, Step<B>) {
        let initial = self.clone();

        let p_action_scalar = p_action.into_scalar();
        let t_action_scalar = t_action.into_scalar();

        if p_action_scalar.to_u32() != N_DIRECTIONS {
            let p_angle = TAU * p_action_scalar.to_f32() / N_DIRECTIONS as f32;
            advance(&mut self.pursuer.pos, p_angle, PURSUER_SPEED);
        }

        if t_action_scalar.to_u32() != N_DIRECTIONS {
            let t_angle = TAU * t_action_scalar.to_f32() / N_DIRECTIONS as f32;
            advance(&mut self.target.pos, t_angle, TARGET_SPEED);
        }

        let collision = self.hits();
        let expired = self.pursuer.time == PURSUER_TIME_CAP;
        let prev_distance = (initial.pursuer.pos - initial.target.pos).abs();
        let new_distance = (self.pursuer.pos - self.target.pos).abs();
        let distance_change = (prev_distance - new_distance) / (PURSUER_SPEED + TARGET_SPEED);

        let p_step = {
            let (state, _) = self.state_tensor(Perspective::Pursuer, device);

            self.pursuer.time += 1;

            let p_reward = if collision {
                Tensor::full([1, 1], 5.0, device)
            } else if expired {
                Tensor::full([1, 1], -5.0, device)
            } else {
                Tensor::full([1, 1], distance_change, device)
            };

            let p_done = Tensor::full([1, 1], collision || expired, device);
            Step::new(state, p_reward, p_done)
        };

        // Target
        let t_step = {
            let (state, wall_blocks) = self.state_tensor(Perspective::Target, device);

            let t_reward = if collision {
                Tensor::full([1, 1], -5.0, device)
            } else if expired {
                Tensor::full([1, 1], 5.0, device)
            } else if wall_blocks {
                let wall_already_blocks = initial.wall_blocks();
                Tensor::full([1, 1], if wall_already_blocks { 1.0 } else { 4.0 }, device)
            } else {
                let lasers = self.lasers(false, self.target.pos);
                let closest = 1.0
                    - lasers
                        .iter()
                        .map(|laser| laser.abs())
                        .min_by(|a, b| a.total_cmp(b))
                        .unwrap();

                let proximity = Tensor::<B, 2>::full([1, 1], closest, device);

                let base_reward = Tensor::full([1, 1], -distance_change, device);
                let penalty_reward = Tensor::full([1, 1], -1.0, device);

                let condition = proximity.clone().greater_elem(1.0 - (-3.0).exp());
                base_reward.mask_where(condition, penalty_reward)
            };

            let t_done = Tensor::full([1, 1], collision, device);
            Step::new(state, t_reward, t_done)
        };

        (p_step, t_step)
    }
}

impl BallEnv {
    pub fn new() -> Self {
        let valid_spawn = || {
            loop {
                let pos = c32(rng().random_range(0.0..=1.0), rng().random_range(0.0..=1.0));
                if !pos_invalid(&pos) {
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

    fn wall_blocks(&self) -> bool {
        let direction = self.target.pos - self.pursuer.pos;
        let distance = direction.abs();
        let ray_dir = Vector::new(direction.re(), direction.im()).normalize();
        let origin = Point::new(self.pursuer.pos.re(), self.pursuer.pos.im());
        let ray = Ray::new(origin, ray_dir);

        let wall_hit = WALLS.cast_ray(&Isometry::identity(), &ray, distance, true);
        let wall_blocks = wall_hit.is_some_and(|hit| hit < distance);
        wall_blocks
    }

    fn lasers(&self, walls: bool, origin: Complex32) -> Vec<Complex32> {
        let origin = Point::new(origin.re(), origin.im());

        let mut lasers = Vec::with_capacity(N_LASERS);

        for i in 0..N_LASERS {
            let angle = (i as f32) * TAU / N_LASERS as f32;
            let (sin, cos) = angle.sin_cos();
            let dir = Vector::new(cos, sin);
            let ray = Ray::new(origin, dir);

            let toi = if walls {
                WALLS.cast_ray(&Isometry::identity(), &ray, SQRT_2, true)
            } else {
                None
            }
            .unwrap_or_else(|| {
                let vertical = if cos > 0.0 {
                    (1.0 - origin.x) / cos
                } else if cos < 0.0 {
                    -origin.x / cos
                } else {
                    f32::INFINITY
                };

                let horizontal = if sin > 0.0 {
                    (1.0 - origin.y) / sin
                } else if sin < 0.0 {
                    -origin.y / sin
                } else {
                    f32::INFINITY
                };

                // Using the Pythagorean identity, we can derive that both sine and cosine can't be
                // zero.
                vertical.min(horizontal)
            });

            let normalized = toi / SQRT_2;
            let relative = c32(dir.x * normalized, dir.y * normalized);

            lasers.push(relative);
        }

        lasers
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
        let to_screen = |pos: Complex32| c32(SIZE.re() * pos.re(), SIZE.im() * pos.im());

        let p = to_screen(self.pursuer.pos);
        let t = to_screen(self.target.pos);

        draw_circle(p.re(), p.im(), RADIUS, BLUE);
        draw_circle(t.re(), t.im(), RADIUS, GREEN);
    }
}
