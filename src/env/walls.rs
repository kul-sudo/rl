use num::complex::{Complex32, ComplexFloat, c32};
use parry2d::{
    math::{Isometry, Point},
    query::PointQuery,
    shape::{Compound, SharedShape},
};
use std::sync::LazyLock;

pub const WALL_SIZE: f32 = 0.1;
pub static WALLS: LazyLock<Compound> = LazyLock::new(|| {
    let positions = [c32(0.5, 0.5), c32(0.0, 0.0), c32(0.0, 0.8)];

    Compound::new(
        positions
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
