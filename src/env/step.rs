use burn::tensor::{Tensor, backend::Backend};

#[derive(Clone, Debug)]
pub struct Step<B: Backend> {
    pub next_states: Tensor<B, 2>,
    pub rewards: Tensor<B, 2>,
    pub dones: Tensor<B, 2>,
}

impl<B: Backend> Step<B> {
    pub fn new(next_states: Tensor<B, 2>, rewards: Tensor<B, 2>, dones: Tensor<B, 2>) -> Self {
        Self {
            next_states,
            rewards,
            dones,
        }
    }
}
