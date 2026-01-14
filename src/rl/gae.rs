use crate::consts::{GAMMA, STEPS_PER_ENV, THREADS_PER_BLOCK};
use crate::training::GaeBackend;
use burn::tensor::{Bool, Tensor as BurnTensor, TensorPrimitive};
use burn_cubecl::{
    CubeRuntime, FloatElement, cubecl,
    cubecl::{
        CubeCount, CubeDim, comptime, cube,
        frontend::{
            ABSOLUTE_POS, Cast, CompilationArg, Float, FloatExpand, Line,
            Tensor as CubeTensorFrontend,
        },
    },
    kernel::into_contiguous,
    tensor::CubeTensor,
};

const LAMBDA: f32 = 0.95;
const WARP_SIZE: u32 = 32;

#[cube(launch)]
pub fn fused_gae_kernel<F: Float>(
    rewards: &CubeTensorFrontend<F>,
    values: &CubeTensorFrontend<F>,
    dones: &CubeTensorFrontend<bool>,
    bootstrap_values: &CubeTensorFrontend<F>,
    advantages: &mut CubeTensorFrontend<F>,
) {
    let env_id = ABSOLUTE_POS;
    let steps = comptime!(STEPS_PER_ENV as u32);

    let offset = env_id * steps;
    let gamma = F::new(comptime!(GAMMA));
    let gl = gamma * F::new(comptime!(LAMBDA));

    let mut next_val = bootstrap_values[env_id];
    let mut last_adv = F::new(0.0);

    for i in 0..steps {
        let t = (steps - 1) - i;
        let idx = offset + t;

        let mask = F::cast_from(!dones[idx]);
        let delta = rewards[idx] + (gamma * next_val * mask) - values[idx];
        let adv = delta + (gl * mask * last_adv);

        advantages[idx] = adv;

        next_val = values[idx];
        last_adv = adv;
    }
}

pub fn compute_gae_fused<R: CubeRuntime, F: FloatElement>(
    rewards: CubeTensor<R>,
    values: CubeTensor<R>,
    dones: CubeTensor<R>,
    bootstrap_values: CubeTensor<R>,
) -> CubeTensor<R> {
    let rewards = into_contiguous(rewards);
    let values = into_contiguous(values);
    let dones = into_contiguous(dones);

    let shape = rewards.shape.clone();

    let buffer = rewards
        .client
        .empty(shape.num_elements() * core::mem::size_of::<F>());
    let output = CubeTensor::new_contiguous(
        rewards.client.clone(),
        rewards.device.clone(),
        shape,
        buffer,
        F::dtype(),
    );

    let cube_count = (THREADS_PER_BLOCK as u32).div_ceil(WARP_SIZE);

    fused_gae_kernel::launch::<F, R>(
        &rewards.client,
        CubeCount::Static(cube_count, 1, 1),
        CubeDim::new_1d(WARP_SIZE),
        rewards.as_handle_ref().as_tensor_arg(1),
        values.as_handle_ref().as_tensor_arg(1),
        dones.as_handle_ref().as_tensor_arg(1),
        bootstrap_values.as_handle_ref().as_tensor_arg(1),
        output.as_handle_ref().as_tensor_arg(1),
    )
    .unwrap();

    output
}

pub fn gae_custom<B: GaeBackend>(
    rewards: BurnTensor<B, 2>,
    values: BurnTensor<B, 2>,
    dones: BurnTensor<B, 2, Bool>,
    bootstrap_values: BurnTensor<B, 2>,
) -> BurnTensor<B, 2> {
    let shape = rewards.shape();

    let output = B::fused_gae(
        rewards.into_primitive().tensor(),
        values.detach().into_primitive().tensor(),
        dones.into_primitive(),
        bootstrap_values.into_primitive().tensor(),
    );

    BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(output)).reshape(shape)
}
