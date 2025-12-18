use super::utils::Data;
use crate::consts::{FONT_SIZE, SIZE};
use crate::env::{context::BallEnv, walls::WALLS};
use crate::mode::{MODE, Mode};
use burn::tensor::backend::Backend;
use macroquad::prelude::*;
use num_complex::{ComplexFloat, c32};
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
            let center = c32(
                isometry.translation.vector.x * SIZE.re(),
                isometry.translation.vector.y * SIZE.im(),
            );
            let size = c32(
                cuboid.half_extents.x * SIZE.re(),
                cuboid.half_extents.y * SIZE.im(),
            );

            let pad = (c32(screen_width(), screen_height()) - SIZE) / 2.0;

            draw_rectangle(
                pad.re() + center.re() - size.re(),
                pad.re() + center.im() - size.im(),
                size.re() * 2.0,
                size.im() * 2.0,
                BLACK,
            );
        }

        next_frame().await;
    }
}
