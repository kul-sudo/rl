use burn::{
    config::Config,
    module::{Initializer, Module},
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Tensor, activation::mish, backend::Backend},
};
use std::f64::consts::SQRT_2;

/// Configuration for the Critic network that estimates V(s).
#[derive(Config, Debug)]
pub struct CriticConfig {
    pub obs_dim: usize,
}

impl CriticConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Critic<B> {
        Critic {
            fc1: LinearConfig::new(self.obs_dim, 1024)
                .with_initializer(Initializer::KaimingNormal {
                    gain: SQRT_2,
                    fan_out_only: false,
                })
                .init(device),
            ln1: LayerNormConfig::new(1024).init(device),
            fc2: LinearConfig::new(1024, 1024)
                .with_initializer(Initializer::KaimingNormal {
                    gain: SQRT_2,
                    fan_out_only: false,
                })
                .init(device),
            ln2: LayerNormConfig::new(1024).init(device),
            fc3: LinearConfig::new(1024, 512)
                .with_initializer(Initializer::KaimingNormal {
                    gain: SQRT_2,
                    fan_out_only: false,
                })
                .init(device),
            ln3: LayerNormConfig::new(512).init(device),
            fc4: LinearConfig::new(512, 1)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.0003,
                })
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Critic<B: Backend> {
    fc1: Linear<B>,
    ln1: LayerNorm<B>,
    fc2: Linear<B>,
    ln2: LayerNorm<B>,
    fc3: Linear<B>,
    ln3: LayerNorm<B>,
    fc4: Linear<B>,
}

impl<B: Backend> Critic<B> {
    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(state);
        let x = self.ln1.forward(x);
        let x = mish(x);

        let x = self.fc2.forward(x);
        let x = self.ln2.forward(x);
        let x = mish(x);

        let x = self.fc3.forward(x);
        let x = self.ln3.forward(x);
        let x = mish(x);

        self.fc4.forward(x)
    }
}
