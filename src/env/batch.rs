use crate::env::step::Step;
use burn::tensor::{Bool, Int, Tensor, backend::Backend};

pub struct BatchCollector<B: Backend> {
    pub states: Vec<Tensor<B, 2>>,
    pub actions: Vec<Tensor<B, 2, Int>>,
    pub rewards: Vec<Tensor<B, 2>>,
    pub dones: Vec<Tensor<B, 2, Bool>>,
}

impl<B: Backend> BatchCollector<B> {
    pub fn new() -> Self {
        Self {
            states: vec![],
            actions: vec![],
            rewards: vec![],
            dones: vec![],
        }
    }

    pub fn push(&mut self, state: Tensor<B, 2>, action: Tensor<B, 2, Int>, step: Step<B>) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(step.reward);
        self.dones.push(step.done);
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn consume(
        self,
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 3, Int>,
        Tensor<B, 3>,
        Tensor<B, 3, Bool>,
    ) {
        (
            Tensor::stack(self.states, 0),
            Tensor::stack(self.actions, 0),
            Tensor::stack(self.rewards, 0),
            Tensor::stack(self.dones, 0),
        )
    }
}
