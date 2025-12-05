use crate::consts::*;
use crate::env::Env;
use crate::rl::actor::{Actor, ActorConfig};
use burn::{
    module::Module,
    record::CompactRecorder,
    tensor::{ElementConversion, Tensor, activation::softmax, backend::Backend},
};
use rand::{Rng, random, rng};
use std::{path::Path, sync::mpsc::Sender};

pub fn inference<B: Backend, E: Env<B> + Clone>(
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
            device,
        )
        .unwrap();

    loop {
        let mut state = env.reset(device);

        loop {
            let logits = actor.forward(state.clone());
            let action_probs: Tensor<B, 2> = softmax(logits, 1);

            let cumulative_probs = action_probs.cumsum(1);
            let r: f32 = random();
            let selected = cumulative_probs
                .into_data()
                .as_slice()
                .unwrap()
                .iter()
                .position(|&p: &f32| p >= r)
                .unwrap_or(0) as u32;
            let action = selected.elem::<B::IntElem>();

            let step = env.step(action, device);
            let _ = env_tx.send(env.clone());

            if step.done {
                break;
            } else {
                state = step.next_state;
            }
        }
    }
}
