use crate::env::step::Step;
use burn::tensor::{Int, Tensor, backend::Backend};

pub struct BatchCollector<B: Backend> {
    pub states: Vec<Tensor<B, 2>>,
    pub actions: Vec<Tensor<B, 2, Int>>,
    pub rewards: Vec<Tensor<B, 2>>,
    pub next_states: Vec<Tensor<B, 2>>,
    pub dones: Vec<Tensor<B, 2>>,
}

impl<B: Backend> BatchCollector<B> {
    pub fn new(capacity: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            next_states: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, states: Tensor<B, 2>, actions: Tensor<B, 2, Int>, step: Step<B>) {
        self.states.push(states);
        self.actions.push(actions);
        self.rewards.push(step.rewards);
        self.next_states.push(step.next_states);
        self.dones.push(step.dones);
    }
}
