use crate::Env;
use crate::env::Step;
use burn::tensor::backend::Backend;

pub struct Data<B: Backend, T: Env<B>> {
    pub env: T,
    pub step: Step<B>,
    pub epsilon: Option<f32>,
}
