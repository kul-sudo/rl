use crate::consts::{BATCH_SIZE, GAMMA};
use crate::training::GaeBackend;
use burn::tensor::{Bool, Tensor as BurnTensor, TensorPrimitive};
use burn_cubecl::{
    CubeRuntime, FloatElement, cubecl,
    cubecl::{
        CubeCount, CubeDim, comptime, cube,
        frontend::{
            ABSOLUTE_POS, Cast, CompilationArg, Float, FloatExpand, Tensor as CubeTensorFrontend,
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
    bootstrap_values: &CubeTensorFrontend<F>,
    advantages: &mut CubeTensorFrontend<F>,
) {
    let env_id = ABSOLUTE_POS;

    if env_id == 0 {
        let steps = comptime!(BATCH_SIZE as u32);
        let gamma = F::new(comptime!(GAMMA));
        let gl = F::new(comptime!(GAMMA * LAMBDA));

        let mut next_val = bootstrap_values[env_id];
        let mut last_adv = F::new(0.0);

        #[unroll]
        for i in 0..steps {
            let t = steps - 1 - i;

            let r = rewards[t];
            let v = values[t];
            let d = dones[t];
            let mask = F::cast_from(!d);

            let delta = r + (gamma * next_val * mask) - v;
            let adv = delta + (gl * mask * last_adv);
            advantages[t] = adv;

            next_val = v;
            last_adv = adv;
        }
    }
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

    fused_gae_kernel::launch::<F, R>(
        &rewards.client,
        CubeCount::new_1d(1),
        CubeDim::new_1d(1),
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
