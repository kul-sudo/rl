use super::{
    derf::{Derf, DerfConfig},
    serf::serf,
};
use burn::{
    config::Config,
    module::{Initializer, Module},
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::sigmoid, backend::Backend},
};
use std::f64::consts::SQRT_2;

#[derive(Config, Debug)]
pub struct CriticConfig {
    pub obs_dim: usize,
}

impl CriticConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> Critic<B> {
        Critic {
            gate: LinearConfig::new(self.obs_dim, self.obs_dim).init(device),
            fc1: LinearConfig::new(self.obs_dim, 1024)
                .with_initializer(Initializer::KaimingNormal {
                    gain: SQRT_2,
                    fan_out_only: false,
                })
                .init(device),
            derf1: DerfConfig::new(1024).init(device),
            fc2: LinearConfig::new(1024, 1024)
                .with_initializer(Initializer::KaimingNormal {
                    gain: SQRT_2,
                    fan_out_only: false,
                })
                .init(device),
            derf2: DerfConfig::new(1024).init(device),
            fc3: LinearConfig::new(1024, 512)
                .with_initializer(Initializer::KaimingNormal {
                    gain: SQRT_2,
                    fan_out_only: false,
                })
                .init(device),
            derf3: DerfConfig::new(512).init(device),
            fc4: LinearConfig::new(512, 1)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.01,
                })
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Critic<B: Backend> {
    gate: Linear<B>,
    pub fc1: Linear<B>,
    derf1: Derf<B>,
    fc2: Linear<B>,
    derf2: Derf<B>,
    fc3: Linear<B>,
    derf3: Derf<B>,
    fc4: Linear<B>,
}

impl<B: Backend> Critic<B> {
    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let mask = sigmoid(self.gate.forward(state.clone()));
        let x = state * mask;

        let x = self.fc1.forward(x);
        let x = self.derf1.forward(x);
        let x = serf(x);

        let x = self.fc2.forward(x);
        let x = self.derf2.forward(x);
        let x = serf(x);

        let x = self.fc3.forward(x);
        let x = self.derf3.forward(x);
        let x = serf(x);

        self.fc4.forward(x)
    }
}
