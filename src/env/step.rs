use burn::tensor::{Tensor, backend::Backend};
use macroquad::prelude::*;

#[derive(Clone, Debug)]
pub struct Step<B: Backend> {
    pub next_state: Tensor<B, 2>,
    pub reward: f32,
    pub done: bool,
}

impl<B: Backend> Step<B> {
    pub fn new(next_state: Tensor<B, 2>, reward: f32, done: bool) -> Self {
        Self {
            next_state,
            reward,
            done,
        }
    }
}
