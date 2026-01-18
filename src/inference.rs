use crate::consts::*;
use crate::env::context::{Env, Perspective};
use crate::env::vecenv::VecEnv;
use crate::render::utils::Data;
use crate::rl::{
    actor::{Actor, ActorConfig},
    stochastic::gumbel_sample,
};
use burn::{module::Module, prelude::ToElement, record::CompactRecorder, tensor::backend::Backend};
use std::{marker::PhantomData, path::Path, sync::mpsc::SyncSender};

pub fn inference<B: Backend, E: Env<B> + Clone>(
    base_env: E,
    data_tx: &SyncSender<Data<B, E, N_ENVS>>,
    device: &B::Device,
) {
    let mut env = VecEnv::<B, E, N_ENVS>::new(base_env);
    let recorder = CompactRecorder::new();

    let pursuer: Actor<B> = ActorConfig::new(PURSUER_FACTORS, N_DIRECTIONS as usize)
        .init(device)
        .load_file(Path::new(ARTIFACT_DIR).join("pursuer"), &recorder, device)
        .expect("Failed to load pursuer.mpk");

    let target: Actor<B> = ActorConfig::new(TARGET_FACTORS, N_DIRECTIONS as usize)
        .init(device)
        .load_file(Path::new(ARTIFACT_DIR).join("target"), &recorder, device)
        .expect("Failed to load target.mpk");

    loop {
        env.reset();

        loop {
            let _ = data_tx.send(Data {
                env: env.clone(),
                curiosity: None,
                _phantom: PhantomData,
            });

            let p_state = env.state_tensor(Perspective::Pursuer, device);
            let t_state = env.state_tensor(Perspective::Target, device);

            let p_action = gumbel_sample(pursuer.forward(p_state));
            let t_action = gumbel_sample(target.forward(t_state));

            let (p_step, t_step) = env.step_simultaneous(p_action, t_action, device);

            if p_step.done.clone().any().into_scalar().to_bool()
                || t_step.done.clone().any().into_scalar().to_bool()
            {
                break;
            }
        }
    }
}
