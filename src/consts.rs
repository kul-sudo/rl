use num::complex::Complex32;
use std::f32::consts::SQRT_2;

// Parameters
pub const GAMMA: f32 = 0.99;
pub const N_DIRECTIONS: u32 = 23;
pub const CURIOSITY_DECAY: f32 = 0.99995;
pub const CURIOSITY_DEFAULT: f32 = 0.1;
pub const CURIOSITY_MIN: f32 = 0.005;

// Pursuer
pub const PURSUER_SPEED: f32 = 0.01;
pub const PURSUER_TIME_CAP: u32 = 3000;
pub const PURSUER_FACTORS: usize = 5;

// Target
pub const TARGET_SPEED: f32 = SQRT_2 * 1.1 * PURSUER_SPEED;
pub const N_LASERS: usize = 8;
pub const TARGET_FACTORS: usize = 5 + N_LASERS;

// pub const N_TARGETS: usize = 5;

// Visuals
pub const SIZE: Complex32 = Complex32::new(1080.0, 1080.0);
pub const RADIUS: f32 = 5.0;
pub const FONT_SIZE: u16 = 17;

// Persistance
pub const ARTIFACT_DIR: &str = "artifact";
