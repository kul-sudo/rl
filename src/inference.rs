use crate::consts::*;
use crate::env::Env;
use crate::rl::actor::{Actor, ActorConfig};
use burn::{module::Module, record::CompactRecorder, tensor::backend::Backend};
use std::{path::Path, sync::mpsc::Sender};

pub fn inference<B: Backend, E: Env<B, D> + Clone, const D: usize>(
    actor_config: &ActorConfig,
    mut env: E,
    env_tx: &Sender<E>,
    device: &B::Device,
) {
    let mut actor: Actor<B> = actor_config.init(device);

    actor = actor
        .load_file(
            Path::new(ARTIFACT_DIR).join("actor"),
            &CompactRecorder::new(),
            &device,
        )
        .unwrap();

    loop {
        let mut state = env.reset(&device);

        loop {
            let logits = actor.forward(state.clone());
            let action = logits.argmax(1).into_scalar();

            let step = env.step(action, &device);

            let _ = env_tx.send(env.clone());

            if step.done {
                break;
            } else {
                state = step.next_state;
            }
        }
    }
}
