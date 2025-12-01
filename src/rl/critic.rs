use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::relu, backend::Backend},
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
            fc1: LinearConfig::new(self.obs_dim, self.hidden).init(device),
            fc2: LinearConfig::new(self.hidden, self.hidden).init(device),
            fc3: LinearConfig::new(self.hidden, 1).init(device),
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
    pub fn forward<const D: usize>(&self, state: Tensor<B, D>) -> Tensor<B, D> {
        let x = relu(self.fc1.forward(state));
        let x = relu(self.fc2.forward(x));
        self.fc3.forward(x)
    }
}
