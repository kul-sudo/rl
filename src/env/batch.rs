use crate::env::step::Step;
use burn::tensor::{Bool, Int, Tensor, backend::Backend};

pub struct BatchCollector<B: Backend> {
    pub states: Tensor<B, 2>,
    pub actions: Tensor<B, 2, Int>,
    pub rewards: Tensor<B, 2>,
    pub dones: Tensor<B, 2, Bool>,
    pub index: usize,
}

impl<B: Backend> BatchCollector<B> {
    pub fn new(capacity: usize, state_dim: usize, device: &B::Device) -> Self {
        Self {
            states: Tensor::zeros([capacity, state_dim], device),
            actions: Tensor::zeros([capacity, 1], device),
            rewards: Tensor::zeros([capacity, 1], device),
            dones: Tensor::zeros([capacity, 1], device),
            index: 0,
        }
    }

    pub fn push(&mut self, state: Tensor<B, 2>, action: Tensor<B, 2, Int>, step: Step<B>) {
        let i = self.index;

        self.states = self
            .states
            .clone()
            .slice_assign([i..i + 1, 0..state.dims()[1]], state);
        self.actions = self.actions.clone().slice_assign([i..i + 1, 0..1], action);
        self.rewards = self
            .rewards
            .clone()
            .slice_assign([i..i + 1, 0..1], step.reward);
        self.dones = self.dones.clone().slice_assign([i..i + 1, 0..1], step.done);
        self.index += 1;
    }

    pub fn consume(
        self,
    ) -> (
        Tensor<B, 2>,
        Tensor<B, 2, Int>,
        Tensor<B, 2>,
        Tensor<B, 2, Bool>,
    ) {
        (self.states, self.actions, self.rewards, self.dones)
    }
}
