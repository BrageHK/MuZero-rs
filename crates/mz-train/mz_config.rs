use std::fs;

use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use crate::{networks::MuZeroNets, utils::BackendChoice};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum NetworkType {
    Linear,
    ResNet,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum Environment {
    CartPole,
    TicTacToe,
    Othello,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TemperatureSchedule {
    pub step: Option<usize>,
    pub tau: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NetworkSubConfig {
    pub latent_space_dims: usize,
    pub fc_hidden_size: usize,
    pub n_layers: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResNetSubConfig {
    pub obs_channels: usize,
    pub channels: usize,
    pub n_blocks: usize,
    pub board_height: usize,
    pub board_width: usize,
    pub fc_hidden_size: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MuZeroConfig {
    pub network_type: NetworkType,
    pub environment: Environment,

    pub representation: NetworkSubConfig,
    pub dynamic: NetworkSubConfig,
    pub prediction: NetworkSubConfig,
    #[serde(default)]
    pub resnet: Option<ResNetSubConfig>,

    pub action_space: usize,
    pub obs_dim: usize,

    pub n_steps: usize,
    pub unroll_steps: usize,
    pub batch_size: usize,
    pub discount: f32,
    pub learning_rate: f64,
    pub num_simulations: usize,
    pub dirichlet_noise: f32,
    pub root_exploration_fraction: f32,
    pub total_steps: usize,
    pub train_steps_per_game: usize,
    pub inference_update_interval: usize,
    pub checkpoint_interval: usize,

    pub max_thread_wait: f32,
    // None = set automatically
    pub num_search_threads: Option<usize>,
    pub init_batch_size: usize,
    pub rec_batch_size: usize,

    // Original muzero paper uses t = 1 first 500k steps, t = 0.5 for next 250k and 0.25 for remaining
    pub temperature_schedule: Vec<TemperatureSchedule>,

    // None => random init
    #[serde(default)]
    pub init_checkpoint: Option<String>,

    // Compute backends; a choice must be compiled in via cargo features. See utils::BackendChoice.
    #[serde(default)]
    pub training_backend: BackendChoice,
    #[serde(default)]
    pub inference_backend: BackendChoice,
}

impl Default for MuZeroConfig {
    fn default() -> Self {
        let file_content = fs::read_to_string("configs/config.yaml").or_else(|_| {
            fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../configs/config.yaml"
            ))
        })
        .expect("Failed to read configs/config.yaml");
        serde_yaml::from_str(&file_content).unwrap()
    }
}

impl MuZeroConfig {
    pub fn new<B: Backend>(path: &str) -> Self {
        let file_content = fs::read_to_string(path).expect("Failed to read file");
        serde_yaml::from_str(&file_content).unwrap()
    }
}

impl MuZeroConfig {
    /// `num_search_threads`, defaulting to available cores when unset.
    pub fn search_threads(&self) -> usize {
        self.num_search_threads.unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        })
    }

    /// Fresh random init of a network family, e.g. `mz_conf.init::<B, MlpNets<B>>(&device)`.
    pub fn init<B: Backend, N: MuZeroNets<B>>(&self, device: &B::Device) -> N {
        N::init(self, device)
    }

    /// Same as `init`, but loads weights from `init_checkpoint` if set in config.
    pub fn init_agent<B: Backend, N: MuZeroNets<B>>(&self, device: &B::Device) -> N {
        let agent: N = self.init(device);
        match &self.init_checkpoint {
            Some(path) => agent.load_file(path, &CompactRecorder::new(), device).unwrap_or_else(|e| {
                panic!("Failed to load init_checkpoint '{path}': {e}")
            }),
            None => agent,
        }
    }
}
