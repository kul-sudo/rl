use crate::consts::BATCH_SIZE;
use crate::env::step::Step;
use burn::tensor::{Bool, Int, Tensor, backend::Backend, s};

pub struct BatchCollector<B: Backend> {
    pub states: Tensor<B, 2>,
    pub actions: Tensor<B, 2, Int>,
    pub rewards: Tensor<B, 2>,
    pub dones: Tensor<B, 2, Bool>,
    index: usize,
}

impl<B: Backend> BatchCollector<B> {
    pub fn new(device: &B::Device, state_dim: usize) -> Self {
        Self {
            states: Tensor::zeros([BATCH_SIZE, state_dim], device),
            actions: Tensor::zeros([BATCH_SIZE, 1], device),
            rewards: Tensor::zeros([BATCH_SIZE, 1], device),
            dones: Tensor::zeros([BATCH_SIZE, 1], device),
            index: 0,
        }
    }

    pub fn push(&mut self, state: Tensor<B, 2>, action: Tensor<B, 2, Int>, step: Step<B>) {
        let range = s![self.index..self.index + 1, ..];

        self.states = self.states.clone().slice_assign(range.clone(), state);
        self.actions = self.actions.clone().slice_assign(range.clone(), action);
        self.rewards = self
            .rewards
            .clone()
            .slice_assign(range.clone(), step.reward);
        self.dones = self.dones.clone().slice_assign(range, step.done);

        self.index += 1;
    }

    pub fn len(&self) -> usize {
        self.index
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
