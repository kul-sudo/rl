use burn::tensor::{Bool, Tensor, backend::Backend};

#[derive(Clone, Debug)]
pub struct Step<B: Backend> {
    pub next_state: Tensor<B, 2>,
    pub reward: Tensor<B, 2>,
    pub done: Tensor<B, 2, Bool>,
    pub terminated: Tensor<B, 2, Bool>,
}

impl<B: Backend> Step<B> {
    pub fn new(
        next_state: Tensor<B, 2>,
        reward: Tensor<B, 2>,
        done: Tensor<B, 2, Bool>,
        terminated: Tensor<B, 2, Bool>,
    ) -> Self {
        Self {
            next_state,
            reward,
            done,
            terminated,
        }
    }
}
