mod consts;
mod env;
mod inference;
mod mode;
mod render;
mod rl;
mod training;
mod utils;

use burn::backend::Autodiff;
use consts::*;
use env::context::{BallEnv, Env};
use inference::*;
use macroquad::prelude::*;
use miniquad::conf::{LinuxBackend, Platform};
use mode::{MODE, Mode};
use render::{data::Data, display::display};
use std::{
    sync::mpsc::{channel, sync_channel},
    thread::spawn,
};
use training::training;

use burn_cubecl::{CubeBackend, cubecl::cuda::CudaRuntime};

pub type BaseBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
pub type TrainingBackend = Autodiff<BaseBackend>;
pub type InferenceBackend = BaseBackend;

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
        let env = BallEnv::new();

        match *MODE {
            Mode::Inference => {
                inference::<InferenceBackend, BallEnv>(env, &data_tx_sync);
            }
            Mode::Training => {
                let mut curiosity = CURIOSITY_DEFAULT;

                training::<TrainingBackend, BallEnv>(env, &data_tx, &mut curiosity);
            }
        }
    });

    display(&data_rx, &data_rx_sync).await;
}
