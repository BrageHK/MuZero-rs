pub mod search_parallel;
pub mod search_serial;

pub use search_parallel::{
    InferenceHandles, inference_channels, inference_master, search,
};