use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::relu, backend::Backend},
};

/// Configuration for the Actor network.
#[derive(Config, Debug)]
pub struct ActorConfig {
    /// Number of factors the actor observes from the environment.
    pub observation: usize,
    /// Number of decisions the actor can make at once.
    pub actions: usize,
    pub hidden: usize,
}

impl ActorConfig {
    /// Initialize the Actor network from the config.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Actor<B> {
        Actor {
            fc1: LinearConfig::new(self.observation, self.hidden).init(device),
            fc2: LinearConfig::new(self.hidden, self.hidden).init(device),
            fc3: LinearConfig::new(self.hidden, self.actions).init(device),
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
        let x = relu(self.fc1.forward(x));
        let x = relu(self.fc2.forward(x));

        self.fc3.forward(x)
    }
}
