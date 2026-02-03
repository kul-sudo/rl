use num::complex::Complex32;
use std::f32::consts::SQRT_2;

// Parameters
pub const BATCH_SIZE: u32 = 256;
pub const GAMMA: f32 = 0.995;
pub const LR_GAMMA: f64 = 0.999;
pub const N_DIRECTIONS: u32 = 23 + 1; // Last means not moving at all
pub const CURIOSITY_DECAY: f32 = 0.99999;
pub const CURIOSITY_DEFAULT: f32 = 1.0;
pub const CURIOSITY_MIN: f32 = 0.1;

// Pursuer
pub const PURSUER_SPEED: f32 = 0.01;
pub const PURSUER_TIME_CAP: u32 = 4000;
pub const PURSUER_FACTORS: usize = 6;

// Target
pub const TARGET_SPEED: f32 = SQRT_2 * PURSUER_SPEED;
pub const N_LASERS: usize = 32;
pub const TARGET_FACTORS: usize = 7 + N_LASERS * 2;

// Visuals
pub const SIZE: Complex32 = Complex32::new(1080.0, 1080.0);
pub const RADIUS: f32 = 5.0;
pub const FONT_SIZE: u16 = 17;

// Persistance
pub const ARTIFACT_DIR: &str = "artifact";
