use crate::consts::SIZE;
use num_complex::{Complex32, ComplexFloat, c32};
use parry2d::{
    math::Isometry,
    query::intersection_test,
    shape::{Compound, SharedShape},
};
use std::sync::LazyLock;

pub const WALL_SIZE: f32 = 0.1;
pub static WALLS: LazyLock<Compound> = LazyLock::new(|| {
    let positions = [c32(0.1, 0.1), c32(0.5, 0.9), c32(0.2, 0.8)];

    Compound::new(
        positions
            .into_iter()
            .map(|pos| {
                let scaled = c32(pos.re() * SIZE.re(), pos.im() * SIZE.im());
                (
                    Isometry::translation(scaled.re(), scaled.im()),
                    SharedShape::cuboid(WALL_SIZE * SIZE.re(), WALL_SIZE * SIZE.im()),
                )
            })
            .collect::<Vec<_>>(),
    )
});

pub fn pos_invalid(pos: &Complex32, shape: SharedShape) -> bool {
    intersection_test(
        &Isometry::identity(),
        &*WALLS,
        &Isometry::translation(pos.re() * SIZE.re(), pos.im() * SIZE.im()),
        &*shape,
    )
    .unwrap()
}
