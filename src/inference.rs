#![allow(warnings)]

use crate::consts::*;
use crate::env::context::Env;
use crate::render::utils::Data;
use crate::rl::actor::{Actor, ActorConfig};
use burn::{module::Module, record::CompactRecorder, tensor::backend::Backend};
use std::{path::Path, sync::mpsc::SyncSender};

pub fn inference<B: Backend, E: Env<B> + Clone>(
    mut env: E,
    data_tx: &SyncSender<Data<B, E>>,
    device: &B::Device,
) {
    // let actor_config = ActorConfig::new(FACTORS, N_DIRECTIONS as usize, 512);
    // let mut actor: Actor<B> = actor_config.init(device);
    //
    // actor = actor
    //     .load_file(
    //         Path::new(ARTIFACT_DIR).join("actor"),
    //         &CompactRecorder::new(),
    //         device,
    //     )
    //     .unwrap();
    //
    // loop {
    //     let mut state = env.reset(device);
    //
    //     loop {
    //         let logits = actor.forward(state.clone());
    //         let action = logits.argmax(1).into_scalar();
    //
    //         let step = env.step(action, device);
    //         let _ = data_tx.send(Data {
    //             env: env.clone(),
    //             step: step.clone(),
    //             epsilon: None,
    //         });
    //
    //         if step.done {
    //             break;
    //         } else {
    //             state = step.next_state;
    //         }
    //     }
    // }
}
