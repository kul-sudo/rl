use burn::tensor::{Tensor, activation::softplus, backend::Backend};

pub fn serf<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    x.clone() * softplus(x, 1.0).erf()
}
