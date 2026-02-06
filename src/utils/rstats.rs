use burn::module::{Module, RunningState};
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug, Clone)]
pub enum Approach {
    Scale,
    ZScore,
}

#[derive(Module, Debug)]
pub struct RunningStats<B: Backend, const D: usize> {
    pub mean: RunningState<Tensor<B, D>>,
    pub var: RunningState<Tensor<B, D>>,
    pub momentum: f64,
    pub approach: Approach,
}

impl<B: Backend, const D: usize> RunningStats<B, D> {
    pub fn new(shape: [usize; D], approach: Approach, device: &B::Device) -> Self {
        Self {
            mean: RunningState::new(Tensor::zeros(shape, device)),
            var: RunningState::new(Tensor::ones(shape, device)),
            momentum: 0.999,
            approach,
        }
    }

    pub fn normalize(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let current_mean = self.mean.value().detach();
        let current_var = self.var.value().detach();

        let (batch_var, batch_mean) = input.clone().var_mean(0);

        let new_mean = current_mean
            .clone()
            .mul_scalar(self.momentum)
            .add(batch_mean.mul_scalar(1.0 - self.momentum));

        let new_var = current_var
            .clone()
            .mul_scalar(self.momentum)
            .add(batch_var.mul_scalar(1.0 - self.momentum));

        self.mean.update(new_mean);
        self.var.update(new_var);

        let centered = match self.approach {
            Approach::ZScore => input.sub(current_mean),
            Approach::Scale => input,
        };

        centered.div(current_var.sqrt().add_scalar(1e-8))
    }

    pub fn normalize_no_update(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let current_mean = self.mean.value().detach();
        let current_var = self.var.value().detach();

        let centered = match self.approach {
            Approach::ZScore => input.sub(current_mean),
            Approach::Scale => input,
        };

        centered.div(current_var.sqrt().add_scalar(1e-8))
    }
}
