use crate::Env;
use burn::tensor::backend::Backend;
use std::marker::PhantomData;

/// Description of the current situation.
pub struct Data<B: Backend, T: Env<B>> {
    pub env: T,
    pub epsilon: Option<f64>,
    pub _phantom: PhantomData<B>,
}
