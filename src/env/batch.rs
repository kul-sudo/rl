use crate::consts::BATCH_SIZE;
use crate::env::step::Step;
use burn::tensor::{Bool, Int, Tensor, backend::Backend};

pub struct Rollout<B: Backend> {
    pub states: Tensor<B, 2>,
    pub actions: Tensor<B, 2, Int>,
    pub rewards: Tensor<B, 2>,
    pub dones: Tensor<B, 2, Bool>,
    pub terminated: Tensor<B, 2, Bool>,
}

pub struct BatchCollector<B: Backend> {
    pub states: Vec<Tensor<B, 2>>,
    pub actions: Vec<Tensor<B, 2, Int>>,
    pub rewards: Vec<Tensor<B, 2>>,
    pub dones: Vec<Tensor<B, 2, Bool>>,
    pub terminated: Vec<Tensor<B, 2, Bool>>,
}

impl<B: Backend> BatchCollector<B> {
    pub fn new() -> Self {
        Self {
            states: Vec::with_capacity(BATCH_SIZE as usize),
            actions: Vec::with_capacity(BATCH_SIZE as usize),
            rewards: Vec::with_capacity(BATCH_SIZE as usize),
            dones: Vec::with_capacity(BATCH_SIZE as usize),
            terminated: Vec::with_capacity(BATCH_SIZE as usize),
        }
    }

    pub fn push(&mut self, state: Tensor<B, 2>, action: Tensor<B, 2, Int>, step: Step<B>) {
        self.states.push(state.detach());
        self.actions.push(action);
        self.rewards.push(step.reward.detach());
        self.dones.push(step.done);
        self.terminated.push(step.terminated);
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn consume(self) -> Rollout<B> {
        Rollout {
            states: Tensor::cat(self.states, 0),
            actions: Tensor::cat(self.actions, 0),
            rewards: Tensor::cat(self.rewards, 0),
            dones: Tensor::cat(self.dones, 0),
            terminated: Tensor::cat(self.terminated, 0),
        }
    }
}
