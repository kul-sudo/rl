use burn::module::{Module, RunningState};
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct RunningStats<B: Backend, const D: usize> {
    pub mean: RunningState<Tensor<B, D>>,
    pub var: RunningState<Tensor<B, D>>,
    pub momentum: f64,
}

impl<B: Backend, const D: usize> RunningStats<B, D> {
    pub fn new(shape: [usize; D], device: &B::Device) -> Self {
        Self {
            mean: RunningState::new(Tensor::zeros(shape, device)),
            var: RunningState::new(Tensor::ones(shape, device)),
            momentum: 0.9,
        }
    }

    pub fn normalize(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let (var, mean) = input.clone().var_mean(0);

        let new_mean = self
            .mean
            .value()
            .mul_scalar(self.momentum)
            .add(mean.mul_scalar(1.0 - self.momentum));

        let new_var = self
            .var
            .value()
            .mul_scalar(self.momentum)
            .add(var.mul_scalar(1.0 - self.momentum));

        self.mean.update(new_mean);
        self.var.update(new_var);

        input
            .sub(self.mean.value())
            .div(self.var.value().sqrt().add_scalar(1e-8))
    }
}
