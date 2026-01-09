use burn::{
    prelude::ToElement,
    tensor::{Distribution, Int, Tensor, backend::Backend},
};

pub fn gumbel_sample<B: Backend>(logits: Tensor<B, 1>) -> Tensor<B, 1, Int> {
    let gumbel_noise = logits
        .random_like(Distribution::Uniform(1e-8, 1.0))
        .log()
        .neg()
        .log()
        .neg();

    let sampled = (logits + gumbel_noise.clone()).argmax(0);

    // if gumbel_noise.contains_nan().into_scalar().to_bool() {
    //     panic!("gumbel");
    // }

    sampled
}
