use super::utils::Data;
use crate::consts::FONT_SIZE;
use crate::env::context::BallEnv;
use crate::env::walls::WALLS;
use crate::mode::{MODE, Mode};
use burn::tensor::backend::Backend;
use macroquad::prelude::*;
use std::sync::mpsc::Receiver;

/// Render the situation based on the data received from either training or inference.
pub async fn display<B: Backend, Q: Backend>(
    data_rx: &Receiver<Data<B, BallEnv>>,
    data_rx_sync: &Receiver<Data<Q, BallEnv>>,
) {
    let mut latest_data = None;

    loop {
        clear_background(WHITE);

        match *MODE {
            Mode::Training => {
                while let Ok(data) = data_rx.try_recv() {
                    latest_data = Some(data);
                }
                if let Some(ref data) = latest_data {
                    data.env.render();

                    let text = format!("Exploration = {:.2}", data.epsilon.unwrap(),);
                    let size = measure_text(&text, None, FONT_SIZE, 1.0);
                    draw_text(&text, 0.0, size.height, FONT_SIZE as f32, BLACK);
                }
            }
            Mode::Inference => {
                if let Ok(ref data) = data_rx_sync.try_recv() {
                    data.env.render();
                }
            }
        }

        for (isometry, shape) in WALLS.shapes() {
            let cuboid = shape.as_cuboid().unwrap();
            let center = isometry.translation.vector;
            let w = cuboid.half_extents.x * 2.0;
            let h = cuboid.half_extents.y * 2.0;
            draw_rectangle(center.x - w / 2.0, center.y - h / 2.0, w, h, BLACK);
        }

        next_frame().await;
    }
}
