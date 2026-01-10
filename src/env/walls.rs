use num::complex::{Complex32, ComplexFloat, c32};
use parry2d::{
    math::{Isometry, Point},
    query::PointQuery,
    shape::{Compound, SharedShape},
};
use std::sync::LazyLock;

pub const WALL_SIZE: f32 = 0.05;
pub static WALLS_POS: [Complex32; 6] = [
    Complex32::new(0.2, 0.5),
    Complex32::new(0.2, 0.2),
    Complex32::new(0.6, 0.1),
    Complex32::new(0.7, 0.9),
    Complex32::new(0.8, 0.5),
    Complex32::new(0.1, 0.9),
];
pub static WALLS: LazyLock<Compound> = LazyLock::new(|| {
    Compound::new(
        WALLS_POS
            .into_iter()
            .map(|pos| {
                let scaled = c32(pos.re(), pos.im());
                (
                    Isometry::translation(scaled.re(), scaled.im()),
                    SharedShape::cuboid(WALL_SIZE, WALL_SIZE),
                )
            })
            .collect::<Vec<_>>(),
    )
});

pub fn pos_invalid(pos: &Complex32) -> bool {
    WALLS.contains_local_point(&Point::new(pos.re(), pos.im()))
}
