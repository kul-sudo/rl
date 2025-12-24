use super::utils::Data;
use crate::consts::{FONT_SIZE, SIZE};
use crate::env::{context::BallEnv, walls::WALLS};
use crate::mode::{MODE, Mode};
use burn::tensor::backend::Backend;
use macroquad::prelude::*;
use num::complex::{ComplexFloat, c32};
use std::sync::mpsc::Receiver;

/// Render the situation based on the data received from either training or inference.
pub async fn display<B: Backend, Q: Backend>(
    data_rx: &Receiver<Data<B, BallEnv>>,
    data_rx_sync: &Receiver<Data<Q, BallEnv>>,
) {
    let mut latest_data = None;

    loop {
        clear_background(BLACK);

        let pad = (c32(screen_width(), screen_height()) - SIZE) / 2.0;

        let padding_camera = Camera2D::from_display_rect(Rect::new(
            -pad.re(),
            -pad.im(),
            screen_width(),
            screen_height(),
        ));

        set_camera(&padding_camera);

        match *MODE {
            Mode::Training => {
                while let Ok(data) = data_rx.try_recv() {
                    latest_data = Some(data);
                }
                if let Some(ref data) = latest_data {
                    data.env.render();

                    let text = format!(
                        "Curiosity = {:.2} Pursuer time = {:.2}",
                        data.curiosity.unwrap(),
                        data.env.pursuer.age()
                    );
                    let size = measure_text(&text, None, FONT_SIZE, 1.0);
                    set_default_camera();
                    draw_text(&text, 0.0, size.height, FONT_SIZE as f32, WHITE);
                    set_camera(&padding_camera);
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

            draw_rectangle(
                center.re() - size.re(),
                center.im() - size.im(),
                size.re() * 2.0,
                size.im() * 2.0,
                DARKGRAY,
            );
        }

        draw_rectangle_lines(0.0, 0.0, SIZE.re(), SIZE.im(), 4.0, DARKGRAY);

        next_frame().await;
    }
}
