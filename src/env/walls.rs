use num::complex::{Complex32, ComplexFloat};
use parry2d::{
    math::{Isometry, Point},
    query::PointQuery,
    shape::{Compound, SharedShape},
};
use std::{f32::consts::TAU, sync::LazyLock};

pub const CORNERS: usize = 8;
pub const WALL_SIZE: f32 = 0.05;
pub const CENTER: Complex32 = Complex32::new(0.5, 0.5);
pub const RADIUS: f32 = 0.3;

pub static WALLS: LazyLock<Compound> = LazyLock::new(|| {
    let shapes = (0..CORNERS)
        .map(|i| {
            let angle = TAU * (i as f32) / CORNERS as f32;

            let offset = Complex32::from_polar(RADIUS, angle);
            let pos = CENTER + offset;

            (
                Isometry::translation(pos.re(), pos.im()),
                SharedShape::cuboid(WALL_SIZE, WALL_SIZE),
            )
        })
        .collect::<Vec<_>>();

    Compound::new(shapes)
});

pub fn pos_invalid(pos: &Complex32) -> bool {
    WALLS.contains_local_point(&Point::new(pos.re(), pos.im()))
}
