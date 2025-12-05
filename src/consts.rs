use num_complex::Complex32;

// Parameters
pub const FACTORS: usize = 4;
pub const GAMMA: f32 = 0.999;
pub const N_DIRECTIONS: u32 = 79;
pub const EPSILON_DECAY: f32 = 0.9999;
pub const EPSILON_DEFAULT: f32 = 0.5;
pub const EPSILON_MIN: f32 = 0.0;
pub const SPEED: f32 = 0.01;
pub const TIME_CAP: u32 = 5000;

// pub const N_TARGETS: usize = 5;

// Visuals
pub const SIZE: Complex32 = Complex32::new(1920.0, 1080.0);
pub const AGENT_RADIUS: f32 = 5.0;
pub const TARGET_RADIUS: f32 = 5.0;
pub const FONT_SIZE: u16 = 17;

// Persistance
pub const ARTIFACT_DIR: &str = "artifact";
