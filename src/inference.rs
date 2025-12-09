use crate::consts::*;
use crate::env::Env;
use crate::rl::actor::{Actor, ActorConfig};
use crate::utils::Data;
use burn::{
    module::Module,
    record::CompactRecorder,
    tensor::{Distribution, Tensor, backend::Backend},
};
use std::{path::Path, sync::mpsc::SyncSender};

pub fn inference<B: Backend, E: Env<B> + Clone>(
    mut env: E,
    data_tx: &SyncSender<Data<B, E>>,
    device: &B::Device,
) {
    let actor_config = ActorConfig::new(FACTORS, N_DIRECTIONS as usize, 512);
    let mut actor: Actor<B> = actor_config.init(device);

    actor = actor
        .load_file(
            Path::new(ARTIFACT_DIR).join("actor"),
            &CompactRecorder::new(),
            device,
        )
        .unwrap();

    loop {
        let mut state = env.reset(device);

        loop {
            let logits = actor.forward(state.clone()) / 2.0;

            let eps = 1e-8;
            let uniform = Tensor::random_like(&logits, Distribution::Uniform(eps, 1.0 - eps));
            let gumbel = -(-uniform.log()).log();
            let action = (logits + gumbel).argmax(1).into_scalar();

            let step = env.step(action, device);
            let _ = data_tx.send(Data {
                env: env.clone(),
                step: step.clone(),
                epsilon: None,
            });

            if step.done {
                break;
            } else {
                state = step.next_state;
            }
        }
    }
}
