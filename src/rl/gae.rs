use crate::consts::{GAMMA, N_ENVS, STEPS_PER_ENV};
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
    rewards: &CubeTensorFrontend<Line<F>>,
    values: &CubeTensorFrontend<Line<F>>,
    dones: &CubeTensorFrontend<Line<bool>>,
    bootstrap_values: &CubeTensorFrontend<F>,
    advantages: &mut CubeTensorFrontend<Line<F>>,
) {
    let env_id = ABSOLUTE_POS;
    let steps = comptime!(STEPS_PER_ENV as u32);

    let r_line = rewards[env_id];
    let v_line = values[env_id];
    let d_line = dones[env_id];

    let gamma = F::new(comptime!(GAMMA));
    let gl = F::new(comptime!(GAMMA * LAMBDA));

    let mut next_val = bootstrap_values[env_id];
    let mut last_adv = F::new(0.0);
    let mut adv_out = Line::<F>::empty(comptime!(STEPS_PER_ENV as u32));

    #[unroll]
    for i in 0..steps {
        let t = steps - 1 - i;

        let mask = F::cast_from(!d_line[t]);
        let delta = r_line[t] + (gamma * next_val * mask) - v_line[t];
        let adv = delta + (gl * mask * last_adv);

        adv_out[t] = adv;
        next_val = v_line[t];
        last_adv = adv;
    }

    advantages[env_id] = adv_out;
}

pub fn compute_gae_fused<R: CubeRuntime, F: FloatElement>(
    rewards: CubeTensor<R>,
    values: CubeTensor<R>,
    dones: CubeTensor<R>,
    bootstrap_values: CubeTensor<R>,
    device: &R::Device,
) -> CubeTensor<R> {
    let client = R::client(device);

    let rewards = into_contiguous(rewards);
    let values = into_contiguous(values);
    let dones = into_contiguous(dones);

    let shape = rewards.shape.clone();
    let byte_size = shape.num_elements() * core::mem::size_of::<F>();

    let buffer = client.empty(byte_size);

    let output = CubeTensor::new_contiguous(
        client.clone(),
        rewards.device.clone(),
        shape,
        buffer,
        F::dtype(),
    );

    let cube_count = (N_ENVS as u32).div_ceil(WARP_SIZE);

    fused_gae_kernel::launch::<F, R>(
        &rewards.client,
        CubeCount::new_1d(cube_count),
        CubeDim::new_1d(WARP_SIZE),
        rewards.as_handle_ref().as_tensor_arg(STEPS_PER_ENV as u8),
        values.as_handle_ref().as_tensor_arg(STEPS_PER_ENV as u8),
        dones.as_handle_ref().as_tensor_arg(STEPS_PER_ENV as u8),
        bootstrap_values.as_handle_ref().as_tensor_arg(1),
        output.as_handle_ref().as_tensor_arg(STEPS_PER_ENV as u8),
    )
    .unwrap();

    output
}

pub fn gae_custom<B: GaeBackend>(
    rewards: BurnTensor<B, 2>,
    values: BurnTensor<B, 2>,
    dones: BurnTensor<B, 2, Bool>,
    bootstrap_values: BurnTensor<B, 2>,
    device: &B::Device,
) -> BurnTensor<B, 2> {
    let shape = rewards.shape();

    let output = B::fused_gae(
        rewards.into_primitive().tensor(),
        values.detach().into_primitive().tensor(),
        dones.into_primitive(),
        bootstrap_values.into_primitive().tensor(),
        device,
    );

    BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(output)).reshape(shape)
}
