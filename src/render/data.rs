use crate::Env;
use burn::tensor::backend::Backend;
use std::marker::PhantomData;

/// Description of the current situation.
pub struct Data<B: Backend, E: Env<B>> {
    pub env: E,
    pub curiosity: Option<f32>,
    pub _phantom: PhantomData<B>,
}
