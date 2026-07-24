use std::fs;

use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use strum::AsRefStr;

use crate::{
    env::Environment,
    env::atari::env::{AtariGame, set_atari_game},
    mz_config::NetworkType::{Linear, ResNet},
    networks::MuZeroNets,
    utils::BackendChoice,
    with_env,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum OptimChoice {
    Adam,
    AdamW,
    Sgd,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum NetworkType {
    Linear,
    ResNet,
}

#[derive(Debug, Clone, Deserialize, Serialize, AsRefStr)]
pub enum EnvironmentName {
    CartPole,
    TicTacToe,
    Othello,
    Atari,
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
pub struct LinearSubConfig {
    pub representation: NetworkSubConfig,
    pub dynamic: NetworkSubConfig,
    pub prediction: NetworkSubConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResNetSubConfig {
    pub obs_channels: usize,
    pub channels: usize,
    pub n_blocks: usize,
    pub fc_hidden_size: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct MuZeroConfig {
    pub network_type: NetworkType,
    pub environment: EnvironmentName,

    #[serde(default)]
    pub atari_game: Option<AtariGame>,

    #[serde(default)]
    pub linear: Option<LinearSubConfig>,
    #[serde(default)]
    pub resnet: Option<ResNetSubConfig>,

    pub optimizer: OptimChoice,
    pub n_steps: usize,
    pub unroll_steps: usize,
    pub training_batch_size: usize,
    pub game_batch_size: usize,
    pub discount: f32,
    pub learning_rate: f64,
    pub grad_clip: f32,
    pub weight_decay: f32,
    pub momentum: f32,
    pub num_simulations: usize,
    #[serde(default = "default_support_size")]
    pub support_size: usize,
    pub dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub training_steps: usize,
    pub train_ratio: f32,
    pub buffer_size: usize,
    // Avg-reward metric averages over the last N finished games.
    #[serde(default = "default_avg_window")]
    pub avg_window: usize,
    // Env-steps/sec metric averages over the last N seconds.
    #[serde(default = "default_rate_window_secs")]
    pub rate_window_secs: f32,
    pub inference_update_interval: usize,
    pub checkpoint_interval: usize,

    // Per-step probability of running a reanalyze pass. 0.0 disables.
    #[serde(default)]
    pub reanalyze_fraction: f32,
    #[serde(default = "default_reanalyze_pool")]
    pub reanalyze_batch_size: usize,

    // rayon with_min_len chunk size: batches smaller than this run serially.
    pub rayon_min_chunk_len: usize,

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

    // !!!! Never set these from the config! It will be overwritten by the chosen env.
    #[serde(default)]
    pub action_space: usize,
    #[serde(default)]
    pub obs_dim: usize,
    #[serde(default)]
    pub is_twoplayer: bool,
    #[serde(default)]
    pub board_height: usize,
    #[serde(default)]
    pub board_width: usize,
}

fn default_avg_window() -> usize {
    100
}

fn default_rate_window_secs() -> f32 {
    10.0
}

fn default_reanalyze_pool() -> usize {
    4096
}

fn default_support_size() -> usize {
    50
}

impl Default for MuZeroConfig {
    fn default() -> Self {
        let file_content = fs::read_to_string("configs/config.yaml")
            .or_else(|_| {
                fs::read_to_string(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../configs/config.yaml"
                ))
            })
            .expect("Failed to read configs/config.yaml");
        get_conf(file_content)
    }
}

impl MuZeroConfig {
    pub fn new<B: Backend>(path: &str) -> Self {
        let file_content = fs::read_to_string(path).expect("Failed to read file");
        get_conf(file_content)
    }
}

fn validate(conf: &MuZeroConfig) {
    assert!(
        conf.discount > 0.0 && conf.discount <= 1.0,
        "discount must be in (0, 1], got {}",
        conf.discount
    );
    assert!(
        conf.learning_rate > 0.0,
        "learning_rate must be > 0, got {}",
        conf.learning_rate
    );
    assert!(
        conf.weight_decay >= 0.0 && conf.weight_decay < 1.0,
        "weight_decay must be in [0, 1), got {}",
        conf.weight_decay
    );
    assert!(
        (0.0..=1.0).contains(&conf.root_exploration_fraction),
        "root_exploration_fraction must be in [0, 1], got {}",
        conf.root_exploration_fraction
    );
    assert!(conf.training_batch_size >= 1, "training_batch_size must be >= 1");
    assert!(conf.game_batch_size >= 1, "game_batch_size must be >= 1");
    assert!(conf.num_simulations >= 1, "num_simulations must be >= 1");
    assert!(conf.unroll_steps >= 1, "unroll_steps must be >= 1");
    assert!(conf.n_steps >= 1, "n_steps must be >= 1");
    assert!(conf.buffer_size >= 1, "buffer_size must be >= 1");
    assert!(conf.support_size >= 1, "support_size must be >= 1");
}

fn get_conf(file_content: String) -> MuZeroConfig {
    let mut conf: MuZeroConfig =
        serde_yaml::from_str(&file_content).expect("Failed to parse configs/config.yaml");
    validate(&conf);
    if let EnvironmentName::Atari = conf.environment {
        let game = conf
            .atari_game
            .expect("environment: Atari requires `atari_game` in the config");
        set_atari_game(game);
    }
    with_env!(conf, E => {
        let env = E::default();

        let info = env.get_info();
        match conf.network_type {
            Linear => (),
            ResNet => {
                let shape = info.obs_shape;
                if shape.len() < 2 {
                    panic!("Cannot use ResNet with a 1D environment");
                }
                conf.board_height = shape[shape.len() - 2];
                conf.board_width = shape[shape.len() - 1];
            },
        };
        conf.action_space = info.action_size;
        conf.obs_dim = info.obs_dim();
        conf.is_twoplayer = info.num_players > 1;
    });
    conf
}

impl MuZeroConfig {
    pub fn linear(&self) -> &LinearSubConfig {
        self.linear
            .as_ref()
            .expect("network_type: Linear requires a `linear:` section in the config")
    }

    pub fn resnet(&self) -> &ResNetSubConfig {
        self.resnet
            .as_ref()
            .expect("network_type: ResNet requires a `resnet:` section in the config")
    }

    pub fn support_len(&self) -> usize {
        crate::support::support_len(self.support_size)
    }

    /// Fresh random init of a network family, e.g. `mz_conf.init::<B, MlpNets<B>>(&device)`.
    pub fn init<B: Backend, N: MuZeroNets<B>>(&self, device: &B::Device) -> N {
        N::init(self, device)
    }

    /// Same as `init`, but loads weights from `init_checkpoint` if set in config.
    pub fn init_agent<B: Backend, N: MuZeroNets<B>>(&self, device: &B::Device) -> N {
        let agent: N = self.init(device);
        match &self.init_checkpoint {
            Some(path) => agent
                .load_file(path, &CompactRecorder::new(), device)
                .unwrap_or_else(|e| panic!("Failed to load init_checkpoint '{path}': {e}")),
            None => agent,
        }
    }
}
