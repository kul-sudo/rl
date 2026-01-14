use crate::consts::{GAMMA, STEPS_PER_ENV, THREADS_PER_BLOCK};
use crate::training::GaeBackend;
use burn::tensor::{Bool, Tensor as BurnTensor, TensorPrimitive};
use burn_cubecl::{
    CubeRuntime, FloatElement, cubecl,
    cubecl::{
        CubeCount, CubeDim, comptime, cube,
        frontend::{
            ABSOLUTE_POS, CompilationArg, Float, FloatExpand, Line, Tensor as CubeTensorFrontend,
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
    advantages: &mut CubeTensorFrontend<Line<F>>,
) {
    let env_id = ABSOLUTE_POS;

    let steps = comptime!(STEPS_PER_ENV as u32);
    let gamma = F::new(comptime!(GAMMA));
    let gl = gamma * F::new(comptime!(LAMBDA));

    let r_line = rewards[env_id];
    let v_line = values[env_id];
    let d_line = dones[env_id];

    let mut adv_out = Line::<F>::empty(steps);

    let mut next_val = F::new(0.0);
    let mut last_adv = F::new(0.0);

    for i in 0..steps {
        let t = steps - 1 - i;

        if d_line[t] {
            next_val = F::new(0.0);
            last_adv = F::new(0.0);
        }

        let delta = r_line[t] + (gamma * next_val) - v_line[t];
        let adv = delta + (gl * last_adv);

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
        rewards.as_handle_ref().as_tensor_arg(STEPS_PER_ENV as u8),
        values.as_handle_ref().as_tensor_arg(STEPS_PER_ENV as u8),
        dones.as_handle_ref().as_tensor_arg(STEPS_PER_ENV as u8),
        output.as_handle_ref().as_tensor_arg(STEPS_PER_ENV as u8),
    )
    .unwrap();

    output
}

pub fn gae_custom<B: GaeBackend>(
    rewards: BurnTensor<B, 2>,
    values: BurnTensor<B, 2>,
    dones: BurnTensor<B, 2, Bool>,
) -> BurnTensor<B, 2> {
    let shape = rewards.shape();

    let output = B::fused_gae(
        rewards.into_primitive().tensor(),
        values.detach().into_primitive().tensor(),
        dones.into_primitive(),
    );

    BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(output)).reshape(shape)
}
