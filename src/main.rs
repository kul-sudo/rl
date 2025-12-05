mod consts;
mod env;
mod inference;
mod rl;
mod training;
mod utils;

use crate::rl::actor::ActorConfig;
use crate::utils::Data;
use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};
use consts::*;
use env::{BallEnv, Env};
use inference::*;
use macroquad::prelude::*;
use miniquad::conf::{LinuxBackend, Platform};
use serde::Deserialize;
use std::{
    env::var,
    sync::mpsc::{channel, sync_channel},
    thread::spawn,
};
use training::*;

pub type TrainingBackend = Autodiff<Cuda<f32>>;
pub type InferenceBackend = Cuda<f32>;

#[derive(Clone, Debug, Deserialize)]
enum Mode {
    Training,
    Inference,
}

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

    let mode = match var("MODE").unwrap().as_str() {
        "inference" => Mode::Inference,
        "training" => Mode::Training,
        _ => panic!("Unexpected mode"),
    };

    let mode_clone = mode.clone();

    spawn(move || {
        let device = CudaDevice::default();
        let actor_config = ActorConfig::new(FACTORS, N_DIRECTIONS as usize, 512);
        let env = BallEnv::new();

        match mode_clone {
            Mode::Inference => {
                inference::<InferenceBackend, BallEnv>(&actor_config, env, &data_tx_sync, &device);
            }
            Mode::Training => {
                let mut epsilon = EPSILON_DEFAULT;

                training::<TrainingBackend, BallEnv>(
                    &actor_config,
                    env,
                    &data_tx,
                    &mut epsilon,
                    &device,
                );
            }
        }
    });

    let mut latest_data = None;

    loop {
        clear_background(WHITE);

        match mode {
            Mode::Inference => {
                if let Ok(ref data) = data_rx_sync.try_recv() {
                    data.env.render();
                }
            }
            Mode::Training => {
                while let Ok(data) = data_rx.try_recv() {
                    latest_data = Some(data);
                }
                if let Some(ref data) = latest_data {
                    data.env.render();

                    let text = format!(
                        "Exploration = {:.2} Time = {:?}",
                        data.epsilon.unwrap(),
                        data.env.time
                    );
                    let size = measure_text(&text, None, FONT_SIZE, 1.0);
                    draw_text(&text, 0.0, size.height, FONT_SIZE as f32, BLACK);
                }
            }
        }

        next_frame().await;
    }
}
