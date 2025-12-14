use num_complex::Complex32;

// Parameters
pub const FACTORS: usize = 4;
pub const GAMMA: f32 = 0.99;
pub const N_DIRECTIONS: u32 = 79;
pub const EPSILON_DECAY: f64 = 0.99995;
pub const EPSILON_DEFAULT: f64 = 1.0;
pub const EPSILON_MIN: f64 = 0.0;

// Pursuer
pub const PURSUER_SPEED: f32 = 0.01;
pub const PURSUER_TIME_CAP: u32 = 3000;

// Target
pub const TARGET_SPEED: f32 = 3.0 * PURSUER_SPEED;

// pub const N_TARGETS: usize = 5;

// Visuals
pub const SIZE: Complex32 = Complex32::new(1920.0, 1080.0);
pub const RADIUS: f32 = 5.0;
pub const FONT_SIZE: u16 = 17;

// Persistance
pub const ARTIFACT_DIR: &str = "artifact";
