use crate::Env;
use crate::env::vecenv::VecEnv;
use burn::tensor::backend::Backend;
use std::marker::PhantomData;

/// Description of the current situation.
pub struct Data<B: Backend, E: Env<B>, const N: usize> {
    pub env: VecEnv<B, E, N>,
    pub curiosity: Option<f32>,
    pub _phantom: PhantomData<B>,
}
