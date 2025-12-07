use crate::consts::*;
use crate::env::Env;
use crate::rl::actor::{Actor, ActorConfig};
use crate::utils::Data;
use burn::{
    module::Module,
    record::CompactRecorder,
    tensor::{ElementConversion, Tensor, activation::softmax, backend::Backend},
};
use rand::{Rng, rng};
use std::{path::Path, sync::mpsc::SyncSender};

pub fn inference<B: Backend, E: Env<B> + Clone>(
    actor_config: &ActorConfig,
    mut env: E,
    data_tx: &SyncSender<Data<B, E>>,
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
            let logits = actor.forward(state.clone()) / 0.5;
            let action_probs: Tensor<B, 2> = softmax(logits, 1);

            let cumulative_probs = action_probs.cumsum(1);
            let r: f32 = rng().random_range(0.0..=1.0);
            let selected = cumulative_probs
                .into_data()
                .as_slice()
                .unwrap()
                .iter()
                .position(|&p: &f32| p >= r)
                .unwrap_or(0) as u32;
            let action = selected.elem::<B::IntElem>();

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
