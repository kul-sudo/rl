use super::derf::{Derf, DerfConfig};
use burn::{
    config::Config,
    module::{Initializer, Module},
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::mish, backend::Backend},
};
use std::f64::consts::SQRT_2;

/// Configuration for the Actor network.
#[derive(Config, Debug)]
pub struct ActorConfig {
    pub obs_dim: usize,
    pub act_dim: usize,
}

impl ActorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Actor<B> {
        Actor {
            fc1: LinearConfig::new(self.obs_dim, 1024)
                .with_initializer(Initializer::KaimingNormal {
                    gain: SQRT_2,
                    fan_out_only: false,
                })
                .init(device),
            derf1: DerfConfig::new(1024).init(device),
            fc2: LinearConfig::new(1024, 512)
                .with_initializer(Initializer::KaimingNormal {
                    gain: SQRT_2,
                    fan_out_only: false,
                })
                .init(device),
            derf2: DerfConfig::new(512).init(device),
            fc3: LinearConfig::new(512, self.act_dim)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.0001,
                })
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Actor<B: Backend> {
    fc1: Linear<B>,
    derf1: Derf<B>,
    fc2: Linear<B>,
    derf2: Derf<B>,
    fc3: Linear<B>,
}

impl<B: Backend> Actor<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = self.derf1.forward(x);
        let x = mish(x);

        let x = self.fc2.forward(x);
        let x = self.derf2.forward(x);
        let x = mish(x);

        self.fc3.forward(x)
    }
}
