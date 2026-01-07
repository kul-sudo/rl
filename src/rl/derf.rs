use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{Tensor, backend::Backend},
};

#[derive(Config, Debug)]
pub struct DerfConfig {
    pub num_features: usize,
}

#[derive(Module, Debug)]
pub struct Derf<B: Backend> {
    alpha: Param<Tensor<B, 1>>,
    shift: Param<Tensor<B, 1>>,
}

impl DerfConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> Derf<B> {
        Derf {
            alpha: Param::from_tensor(Tensor::ones([self.num_features], device)),
            shift: Param::from_tensor(Tensor::zeros([self.num_features], device)),
        }
    }
}

impl<B: Backend> Derf<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let alpha = self.alpha.val().reshape([1, -1]);
        let shift = self.shift.val().reshape([1, -1]);
        (x * alpha + shift).erf()
    }
}
