use burn::{
    config::Config,
    module::{Initializer, Module},
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::mish, backend::Backend},
};

/// Configuration for the Critic network that estimates V(s).
#[derive(Config, Debug)]
pub struct CriticConfig {
    pub obs_dim: usize,
    pub hidden: usize,
}

impl CriticConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Critic<B> {
        Critic {
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
            fc3: LinearConfig::new(self.hidden, 1)
                .with_initializer(Initializer::KaimingNormal {
                    gain: 1.0,
                    fan_out_only: false,
                })
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Critic<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl<B: Backend> Critic<B> {
    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(state);
        let x = mish(x);

        let x = self.fc2.forward(x);
        let x = mish(x);

        self.fc3.forward(x)
    }
}
