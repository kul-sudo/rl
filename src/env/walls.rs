use num::complex::{Complex32, ComplexFloat, c32};
use parry2d::{
    math::{Isometry, Point},
    query::PointQuery,
    shape::{Compound, SharedShape},
};
use std::sync::LazyLock;

pub const WALL_SIZE: f32 = 0.05;
pub static WALLS: LazyLock<Compound> = LazyLock::new(|| {
    let positions = [
        c32(0.2, 0.5),
        c32(0.2, 0.2),
        c32(0.6, 0.1),
        c32(0.7, 0.9),
        c32(0.2, 0.2),
    ];

    Compound::new(
        positions
            .into_iter()
            .map(|pos| {
                let scaled = c32(pos.re(), pos.im());
                (
                    Isometry::translation(scaled.re(), scaled.im()),
                    SharedShape::cuboid(WALL_SIZE * 2.0, WALL_SIZE),
                )
            })
            .collect::<Vec<_>>(),
    )
});

pub fn pos_invalid(pos: &Complex32) -> bool {
    WALLS.contains_local_point(&Point::new(pos.re(), pos.im()))
}
