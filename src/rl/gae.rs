use crate::consts::{BATCH_SIZE, GAMMA};
use crate::training::GaeBackend;
use burn::tensor::{Bool, Tensor as BurnTensor, TensorPrimitive};
use burn_cubecl::{
    CubeRuntime, FloatElement, cubecl,
    cubecl::{
        CubeCount, CubeDim, comptime, cube,
        frontend::{
            Cast, CompilationArg, CubeIndexExpand, CubeIndexMutExpand, Float, FloatExpand,
            SharedMemory, Tensor as CubeTensorFrontend, UNIT_POS, synchronization::sync_cube,
        },
    },
    kernel::into_contiguous,
    tensor::CubeTensor,
};

const LAMBDA: f32 = 0.95;

#[cube(launch)]
pub fn fused_gae_kernel<F: Float>(
    rewards: &CubeTensorFrontend<F>,
    values: &CubeTensorFrontend<F>,
    dones: &CubeTensorFrontend<bool>,
    terminated: &CubeTensorFrontend<bool>,
    bootstrap_values: &CubeTensorFrontend<F>,
    advantages: &mut CubeTensorFrontend<F>,
) {
    let t = UNIT_POS as usize;
    let steps = comptime!(BATCH_SIZE as usize);
    let gamma = F::new(comptime!(GAMMA));
    let gl = F::new(comptime!(GAMMA * LAMBDA));

    let mut shared_deltas = SharedMemory::<F>::new(comptime!(BATCH_SIZE as usize));
    let mut shared_dones = SharedMemory::<bool>::new(comptime!(BATCH_SIZE as usize));

    if t < steps {
        let v_next = if t == steps - 1 {
            bootstrap_values[0]
        } else {
            values[t + 1]
        };

        let boot_mask = F::cast_from(!terminated[t]);
        shared_deltas[t] = rewards[t] + (gamma * v_next * boot_mask) - values[t];
        shared_dones[t] = dones[t];
    }

    sync_cube();

    if t == 0 {
        let mut last_adv = F::new(0.0);

        for i in 0..steps {
            let idx = steps - 1 - i;

            let done_mask = F::cast_from(!shared_dones[idx]);
            let adv = shared_deltas[idx] + (gl * done_mask * last_adv);
            advantages[idx] = adv;
            last_adv = adv;
        }
    }
}

pub fn compute_gae_fused<R: CubeRuntime, F: FloatElement>(
    rewards: CubeTensor<R>,
    values: CubeTensor<R>,
    dones: CubeTensor<R>,
    terminated: CubeTensor<R>,
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

    fused_gae_kernel::launch::<F, R>(
        &rewards.client,
        CubeCount::new_1d(1),
        CubeDim::new_1d(BATCH_SIZE),
        rewards.as_handle_ref().as_tensor_arg(1),
        values.as_handle_ref().as_tensor_arg(1),
        dones.as_handle_ref().as_tensor_arg(1),
        terminated.as_handle_ref().as_tensor_arg(1),
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
    terminated: BurnTensor<B, 2, Bool>,
    bootstrap_values: BurnTensor<B, 2>,
    device: &B::Device,
) -> BurnTensor<B, 2> {
    let rewards = rewards.detach().into_primitive();
    let values = values.detach().into_primitive();
    let dones = dones.into_primitive();
    let terminated = terminated.into_primitive();
    let bootstrap = bootstrap_values.detach().into_primitive();

    let output = B::fused_gae(
        rewards.tensor(),
        values.tensor(),
        dones,
        terminated,
        bootstrap.tensor(),
        device,
    );

    BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(output))
}
