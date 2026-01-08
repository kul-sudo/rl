use burn::tensor::{Tensor, activation::softplus, backend::Backend};

pub fn serf<B: Backend>(x: Tensor<B, 1>) -> Tensor<B, 1> {
    x.clone() * softplus(x, 1.0).erf()
}
