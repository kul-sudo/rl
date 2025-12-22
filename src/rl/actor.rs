use burn::{
    config::Config,
    module::{Initializer, Module},
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::mish, backend::Backend},
};

/// Configuration for the Actor network.
#[derive(Config, Debug)]
pub struct ActorConfig {
    /// Number of factors the actor observes from the environment.
    pub obs_dim: usize,
    /// Number of decisions the actor can make at once.
    pub act_dim: usize,
    pub hidden: usize,
}

impl ActorConfig {
    /// Initialize the Actor network from the config.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Actor<B> {
        Actor {
            fc1: LinearConfig::new(self.obs_dim, self.hidden)
                .with_initializer(Initializer::KaimingNormal {
                    gain: 1.0,
                    fan_out_only: false,
                })
                .init(device),
            fc2: LinearConfig::new(self.hidden, self.hidden)
                .with_initializer(Initializer::KaimingNormal {
                    gain: 1.0,
                    fan_out_only: false,
                })
                .init(device),
            fc3: LinearConfig::new(self.hidden, self.act_dim)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.01,
                })
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Actor<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl<B: Backend> Actor<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = mish(x);

        let x = self.fc2.forward(x);
        let x = mish(x);

        self.fc3.forward(x)
    }
}
