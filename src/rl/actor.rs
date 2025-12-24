use burn::{
    config::Config,
    module::{Initializer, Module},
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Tensor, activation::mish, backend::Backend},
};
use std::f64::consts::SQRT_2;

/// Configuration for the Actor network.
#[derive(Config, Debug)]
pub struct ActorConfig {
    /// Number of factors the actor observes from the environment.
    pub obs_dim: usize,
    /// Number of decisions the actor can make at once.
    pub act_dim: usize,
}

impl ActorConfig {
    /// Initialize the Actor network from the config.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Actor<B> {
        Actor {
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
            fc3: LinearConfig::new(1024, self.act_dim)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.001,
                })
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Actor<B: Backend> {
    fc1: Linear<B>,
    ln1: LayerNorm<B>,
    fc2: Linear<B>,
    ln2: LayerNorm<B>,
    fc3: Linear<B>,
}

impl<B: Backend> Actor<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        // let x = self.ln1.forward(x);
        let x = mish(x);

        let x = self.fc2.forward(x);
        // let x = self.ln2.forward(x);
        let x = mish(x);

        self.fc3.forward(x)
    }
}
