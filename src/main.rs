mod consts;
mod env;
mod inference;
mod mode;
mod render;
mod rl;
mod training;

use burn::backend::{Autodiff, Cuda};
use consts::*;
use env::context::{BallEnv, Env};
use inference::*;
use macroquad::prelude::*;
use miniquad::conf::{LinuxBackend, Platform};
use mode::{MODE, Mode};
use render::{display::display, utils::Data};
use std::{
    sync::mpsc::{channel, sync_channel},
    thread::spawn,
};
use training::training;

pub type TrainingBackend = Autodiff<Cuda<f32>>;
pub type InferenceBackend = Cuda<f32>;

fn window_conf() -> Conf {
    Conf {
        window_title: "training".to_owned(),
        fullscreen: true,
        platform: Platform {
            linux_backend: LinuxBackend::WaylandWithX11Fallback,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
pub async fn main() {
    let (data_tx, data_rx) = channel();
    let (data_tx_sync, data_rx_sync) = sync_channel::<Data<InferenceBackend, BallEnv>>(300);

    spawn(move || {
        let device = Default::default();
        let env = BallEnv::new();

        match *MODE {
            Mode::Inference => {
                inference::<InferenceBackend, BallEnv>(env, &data_tx_sync, &device);
            }
            Mode::Training => {
                let mut curiosity = CURIOSITY_DEFAULT;

                training::<TrainingBackend, BallEnv>(env, &data_tx, &mut curiosity, &device);
            }
        }
    });

    display(&data_rx, &data_rx_sync).await;
}
