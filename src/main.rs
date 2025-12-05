mod consts;
mod env;
mod inference;
mod rl;
mod training;

use crate::rl::actor::ActorConfig;
use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};
use consts::*;
use env::BallEnv;
use inference::*;
use macroquad::prelude::*;
use miniquad::conf::{LinuxBackend, Platform};
use serde::Deserialize;
use std::{env::var, sync::mpsc::channel, thread::spawn};
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
    let (env_tx, env_rx) = channel();
    let (epsilon_tx, epsilon_rx) = channel();

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
                inference::<InferenceBackend, BallEnv>(&actor_config, env, &env_tx, &device);
            }
            Mode::Training => {
                let mut epsilon = EPSILON_DEFAULT;

                training::<TrainingBackend, BallEnv>(
                    &actor_config,
                    env,
                    &env_tx,
                    &epsilon_tx,
                    &mut epsilon,
                    &device,
                );
            }
        }
    });

    loop {
        clear_background(WHITE);

        match mode {
            Mode::Inference => {
                if let Ok(ref env) = env_rx.try_recv() {
                    env.render();
                }
            }
            Mode::Training => {
                let mut latest_env = None;
                while let Ok(env) = env_rx.try_recv() {
                    latest_env = Some(env);
                }
                if let Some(ref env) = latest_env {
                    env.render();
                }

                let mut latest_epsilon = None;
                while let Ok(epsilon) = epsilon_rx.try_recv() {
                    latest_epsilon = Some(epsilon);
                }
                if let Some(ref epsilon) = latest_epsilon {
                    let text = format!(
                        "Exploration = {:.2} Time = {:?}",
                        epsilon,
                        latest_env.map(|x| x.time)
                    );
                    let size = measure_text(&text, None, FONT_SIZE, 1.0);
                    draw_text(&text, 0.0, size.height, FONT_SIZE as f32, BLACK);
                }

                next_frame().await;
            }
        }
    }
}
