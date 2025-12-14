use std::{env::var, sync::LazyLock};

#[derive(Clone, Debug)]
pub enum Mode {
    Training,
    Inference,
}

pub static MODE: LazyLock<Mode> = LazyLock::new(|| match var("MODE").unwrap().as_str() {
    "inference" => Mode::Inference,
    "training" => Mode::Training,
    _ => panic!("Unexpected mode"),
});
