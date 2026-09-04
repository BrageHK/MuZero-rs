use std::fs;

use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use strum::AsRefStr;

pub use mz_core::config::{
    LinearSubConfig, NetConfig, NetworkSubConfig, NetworkType, ProjectionSubConfig,
    ResNetBlockConfig, ResNetRepresentationConfig, ResNetSubConfig,
};

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

/// Puct: MuZero PUCT with Dirichlet root noise and visit-count policy targets.
/// Gumbel: Gumbel MuZero (Danihelka et al. 2022) — Sequential Halving over
/// Gumbel-perturbed logits at the root, improved-policy targets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
pub enum SearchAlgorithm {
    #[default]
    Puct,
    Gumbel,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PuctSubConfig {
    pub dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    // Original muzero paper uses t = 1 first 500k steps, t = 0.5 for next 250k
    // and 0.25 for remaining.
    pub temperature_schedule: Vec<TemperatureSchedule>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GumbelSubConfig {
    pub max_num_considered_actions: usize,
    pub c_visit: f32,
    pub c_scale: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize, AsRefStr)]
pub enum EnvironmentName {
    CartPole,
    TicTacToe,
    Othello,
    Chess,
    Atari,
}

/// One step of the benchmark-opponent ladder. `elo` is a hand-picked anchor: the
/// absolute numbers are arbitrary, only movement between evals is meaningful.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "opponent", deny_unknown_fields)]
pub enum RungConfig {
    Random {
        elo: f32,
    },
    AlphaBeta {
        depth: usize,
        #[serde(default)]
        epsilon: f32,
        elo: f32,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalConfig {
    /// Training steps between evals. 0 disables evaluation.
    #[serde(default = "default_eval_interval")]
    pub interval: usize,
    #[serde(default = "default_eval_games")]
    pub games: usize,
    /// None = reuse the training `num_simulations`.
    #[serde(default)]
    pub num_simulations: Option<usize>,
    #[serde(default = "default_random_opening_plies")]
    pub random_opening_plies: usize,
    #[serde(default = "default_promote_score")]
    pub promote_score: f32,
    #[serde(default = "default_demote_score")]
    pub demote_score: f32,
    /// Fixed so every eval replays the same opening book, which makes successive
    /// readings a paired comparison instead of independent samples.
    #[serde(default = "default_eval_seed")]
    pub seed: u64,
    /// None = the built-in ladder for the configured environment.
    #[serde(default)]
    pub ladder: Option<Vec<RungConfig>>,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            interval: default_eval_interval(),
            games: default_eval_games(),
            num_simulations: None,
            random_opening_plies: default_random_opening_plies(),
            promote_score: default_promote_score(),
            demote_score: default_demote_score(),
            seed: default_eval_seed(),
            ladder: None,
        }
    }
}

fn default_eval_interval() -> usize {
    500
}

fn default_eval_games() -> usize {
    100
}

fn default_random_opening_plies() -> usize {
    4
}

fn default_promote_score() -> f32 {
    0.75
}

fn default_demote_score() -> f32 {
    0.25
}

fn default_eval_seed() -> u64 {
    0x5EED_0E10
}

/// Ladders for the environments that support evaluation. Other environments get
/// an empty ladder, which disables it.
pub fn default_ladder(environment: &EnvironmentName) -> Vec<RungConfig> {
    match environment {
        EnvironmentName::Othello => vec![
            RungConfig::Random { elo: 0.0 },
            RungConfig::AlphaBeta {
                depth: 1,
                epsilon: 0.05,
                elo: 500.0,
            },
            RungConfig::AlphaBeta {
                depth: 3,
                epsilon: 0.02,
                elo: 1000.0,
            },
            RungConfig::AlphaBeta {
                depth: 5,
                epsilon: 0.0,
                elo: 1400.0,
            },
        ],
        EnvironmentName::TicTacToe => vec![
            RungConfig::Random { elo: 0.0 },
            RungConfig::AlphaBeta {
                depth: 1,
                epsilon: 0.2,
                elo: 300.0,
            },
            RungConfig::AlphaBeta {
                depth: 3,
                epsilon: 0.1,
                elo: 600.0,
            },
            RungConfig::AlphaBeta {
                depth: 9,
                epsilon: 0.0,
                elo: 900.0,
            },
        ],
        EnvironmentName::Chess => vec![
            RungConfig::Random { elo: 0.0 },
            RungConfig::AlphaBeta {
                depth: 1,
                epsilon: 0.1,
                elo: 400.0,
            },
            RungConfig::AlphaBeta {
                depth: 2,
                epsilon: 0.05,
                elo: 800.0,
            },
            RungConfig::AlphaBeta {
                depth: 3,
                epsilon: 0.0,
                elo: 1200.0,
            },
        ],
        EnvironmentName::CartPole | EnvironmentName::Atari => Vec::new(),
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TemperatureSchedule {
    pub step: Option<usize>,
    pub tau: f32,
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
    #[serde(default = "default_eps")]
    pub eps: f32,
    pub num_simulations: usize,
    #[serde(default = "default_support_size")]
    pub support_size: usize,

    #[serde(default = "default_value_coef")]
    pub value_coef: f32,
    #[serde(default = "default_unit_coef")]
    pub policy_coef: f32,
    #[serde(default = "default_unit_coef")]
    pub reward_coef: f32,
    #[serde(default = "default_consistency_coef")]
    pub consistency_coef: f32,
    #[serde(default)]
    pub projection: ProjectionSubConfig,

    #[serde(default)]
    pub augmentation: bool,

    #[serde(default)]
    pub lr_warmup_steps: usize,
    #[serde(default = "default_lr_decay_rate")]
    pub lr_decay_rate: f64,
    #[serde(default = "default_lr_decay_steps")]
    pub lr_decay_steps: usize,

    #[serde(default)]
    pub search_algorithm: SearchAlgorithm,
    #[serde(default)]
    pub puct: Option<PuctSubConfig>,
    #[serde(default)]
    pub gumbel: Option<GumbelSubConfig>,

    #[serde(default)]
    pub eval: Option<EvalConfig>,

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

    // Run self-play and network training on two threads instead of one loop.
    #[serde(default)]
    pub async_training: bool,

    // Resumes networks, optimizer state, replay buffer, and training step from
    // model/<environment>/<checkpoint_name>/. false => random init, fresh buffer, step 0.
    #[serde(default)]
    pub load_from_checkpoint: bool,

    // Compute backends; a choice must be compiled in via cargo features. See utils::BackendChoice.
    #[serde(default)]
    pub training_backend: BackendChoice,
    #[serde(default)]
    pub inference_backend: BackendChoice,
    #[serde(default)]
    pub eval_backend: BackendChoice,

    pub checkpoint_name: String,

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
    #[serde(default)]
    pub obs_channels: usize,
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

fn default_eps() -> f32 {
    1e-5
}

fn default_value_coef() -> f32 {
    0.25
}

fn default_lr_decay_rate() -> f64 {
    1.0
}

fn default_lr_decay_steps() -> usize {
    350_000
}

fn default_unit_coef() -> f32 {
    1.0
}

fn default_consistency_coef() -> f32 {
    2.0
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
        conf.lr_decay_rate > 0.0,
        "lr_decay_rate must be > 0, got {}",
        conf.lr_decay_rate
    );
    assert!(
        conf.weight_decay >= 0.0 && conf.weight_decay < 1.0,
        "weight_decay must be in [0, 1), got {}",
        conf.weight_decay
    );
    assert!(conf.eps > 0.0, "eps must be > 0, got {}", conf.eps);
    assert!(
        conf.training_batch_size >= 1,
        "training_batch_size must be >= 1"
    );
    assert!(conf.game_batch_size >= 1, "game_batch_size must be >= 1");
    assert!(conf.num_simulations >= 1, "num_simulations must be >= 1");
    assert!(conf.unroll_steps >= 1, "unroll_steps must be >= 1");
    assert!(conf.n_steps >= 1, "n_steps must be >= 1");
    assert!(conf.buffer_size >= 1, "buffer_size must be >= 1");
    assert!(conf.support_size >= 1, "support_size must be >= 1");
    for (name, coef) in [
        ("value_coef", conf.value_coef),
        ("policy_coef", conf.policy_coef),
        ("reward_coef", conf.reward_coef),
        ("consistency_coef", conf.consistency_coef),
    ] {
        assert!(coef >= 0.0, "{name} must be >= 0, got {coef}");
    }
    if conf.consistency_coef > 0.0 {
        assert!(
            conf.projection.proj_hidden >= 1
                && conf.projection.proj_out >= 1
                && conf.projection.pred_hidden >= 1,
            "projection sizes must be >= 1"
        );
    }
    if let SearchAlgorithm::Puct = conf.search_algorithm {
        let puct = conf
            .puct
            .as_ref()
            .expect("search_algorithm: Puct requires a `puct:` section in the config");
        assert!(
            (0.0..=1.0).contains(&puct.root_exploration_fraction),
            "root_exploration_fraction must be in [0, 1], got {}",
            puct.root_exploration_fraction
        );
        assert!(
            puct.dirichlet_alpha > 0.0,
            "dirichlet_alpha must be > 0, got {}",
            puct.dirichlet_alpha
        );
        assert!(
            !puct.temperature_schedule.is_empty(),
            "temperature_schedule must have at least one entry"
        );
    }
    if let SearchAlgorithm::Gumbel = conf.search_algorithm {
        let gumbel = conf
            .gumbel
            .as_ref()
            .expect("search_algorithm: Gumbel requires a `gumbel:` section in the config");
        assert!(
            gumbel.max_num_considered_actions >= 1,
            "max_num_considered_actions must be >= 1, got {}",
            gumbel.max_num_considered_actions
        );
        assert!(
            gumbel.c_scale > 0.0,
            "c_scale must be > 0, got {}",
            gumbel.c_scale
        );
        assert!(
            gumbel.c_visit >= 0.0,
            "c_visit must be >= 0, got {}",
            gumbel.c_visit
        );
    }
    if let Some(eval) = conf.eval.as_ref() {
        assert!(
            eval.games >= 2,
            "eval.games must be >= 2, got {}",
            eval.games
        );
        assert!(
            (0.0..=1.0).contains(&eval.promote_score) && (0.0..=1.0).contains(&eval.demote_score),
            "eval promote/demote scores must be in [0, 1]"
        );
        assert!(
            eval.promote_score > eval.demote_score,
            "eval.promote_score ({}) must exceed eval.demote_score ({})",
            eval.promote_score,
            eval.demote_score
        );
        if let Some(ladder) = eval.ladder.as_ref() {
            assert!(
                !default_ladder(&conf.environment).is_empty(),
                "eval.ladder is only supported for board games (TicTacToe, Othello, Chess)"
            );
            assert!(
                !ladder.is_empty(),
                "eval.ladder must have at least one rung"
            );
            for rung in ladder {
                if let RungConfig::AlphaBeta { depth, epsilon, .. } = rung {
                    assert!(*depth >= 1, "eval ladder depth must be >= 1, got {depth}");
                    assert!(
                        (0.0..=1.0).contains(epsilon),
                        "eval ladder epsilon must be in [0, 1], got {epsilon}"
                    );
                }
            }
        }
    }
}

fn strip_numeric_underscores(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    for i in 0..chars.len() {
        let c = chars[i];
        if c == '_' {
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
            if prev_digit && next_digit {
                continue;
            }
        }
        out.push(c);
    }
    out
}

fn get_conf(file_content: String) -> MuZeroConfig {
    let file_content = strip_numeric_underscores(&file_content);
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
                conf.obs_channels = shape[..shape.len() - 2].iter().product::<usize>().max(1);
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

    pub fn puct(&self) -> &PuctSubConfig {
        self.puct
            .as_ref()
            .expect("search_algorithm: Puct requires a `puct:` section in the config")
    }

    pub fn gumbel(&self) -> &GumbelSubConfig {
        self.gumbel
            .as_ref()
            .expect("search_algorithm: Gumbel requires a `gumbel:` section in the config")
    }

    pub fn eval(&self) -> EvalConfig {
        self.eval.clone().unwrap_or_default()
    }

    pub fn support_len(&self) -> usize {
        crate::support::support_len(self.support_size)
    }

    /// Board games (2-player) use a plain scalar value head and no reward loss
    /// (paper App. F/G: l^v=(z-q)^2, l^r=0). Single-player envs (CartPole, Atari)
    /// use the categorical support for both.
    pub fn categorical(&self) -> bool {
        !self.is_twoplayer
    }

    /// Board games on a square board (Othello, TicTacToe) are invariant under
    /// the 8-cell dihedral group; the replay buffer samples a random rotation/
    /// reflection per game to multiply effective self-play data. Chess is also
    /// played on a square board but is not dihedrally symmetric (pawns only
    /// move one way, castling is side-specific, and the move encoding's 73
    /// direction/knight/underpromotion planes don't rotate with the board), so
    /// it is excluded even though it satisfies the square-board check.
    pub fn board_symmetric(&self) -> bool {
        self.is_twoplayer
            && self.board_height > 0
            && self.board_height == self.board_width
            && !matches!(self.environment, EnvironmentName::Chess)
    }

    /// The subset of the config that shapes the networks; the only part `mz-web`
    /// needs to rebuild the same modules for the trained weights.
    pub fn net_config(&self) -> NetConfig {
        NetConfig {
            network_type: self.network_type,
            obs_dim: self.obs_dim,
            action_space: self.action_space,
            support_size: self.support_size,
            categorical: self.categorical(),
            board_height: self.board_height,
            board_width: self.board_width,
            obs_channels: self.obs_channels,
            linear: self.linear.clone(),
            resnet: self.resnet.clone(),
            projection: self.projection.clone(),
        }
    }

    /// Fresh random init of a network family, e.g. `mz_conf.init::<B, MlpNets<B>>(&device)`.
    pub fn init<B: Backend, N: MuZeroNets<B>>(&self, device: &B::Device) -> N {
        N::init(&self.net_config(), device)
    }

    /// Directory holding this environment's checkpoint files: `latest`, `optimizer`,
    /// `buffer.mpk`, `training_step`, `env_steps`, `games_played`, `best_elo`,
    /// `model_best_{elo}`, `eval_state`, `config.yaml`. Namespaced under
    /// `model/{environment}/{checkpoint_name}/`.
    pub fn checkpoint_dir(&self) -> String {
        format!(
            "model/{}/{}",
            self.environment.as_ref(),
            self.checkpoint_name
        )
    }

    /// Same as `init`, but loads weights from the checkpoint dir if `load_from_checkpoint`.
    pub fn init_agent<B: Backend, N: MuZeroNets<B>>(&self, device: &B::Device) -> N {
        let agent: N = self.init(device);
        if self.load_from_checkpoint {
            let path = format!("{}/latest", self.checkpoint_dir());
            agent
                .load_file(&path, &CompactRecorder::new(), device)
                .unwrap_or_else(|e| panic!("Failed to load checkpoint '{path}': {e}"))
        } else {
            agent
        }
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::backend::NdArray;
    use mz_core::networks::resnet::ResNets;

    use super::*;

    /// Regression check: the shipped `configs/config.yaml` (including its
    /// `gpool`/`head_gpool` global-pooling keys) parses and builds a working
    /// ResNet network family end to end.
    #[test]
    fn shipped_config_parses_and_builds_resnet() {
        let conf = MuZeroConfig::default();
        assert_eq!(conf.network_type, NetworkType::ResNet);

        let device = Default::default();
        let _nets: ResNets<NdArray<f32>> = conf.init(&device);
    }

    /// Doesn't touch `configs/config.yaml`'s `environment:` (that stays whatever
    /// the shipped default is): reuses its ResNet hyperparameters but swaps in
    /// Chess's `EnvInfo` by hand, the way `get_conf` would for `environment: Chess`.
    #[test]
    fn chess_env_info_builds_a_resnet() {
        let mut conf = MuZeroConfig::default();
        let info = <crate::env::chess::env::Chess as Environment>::INFO;

        conf.environment = EnvironmentName::Chess;
        conf.obs_dim = info.obs_dim();
        conf.action_space = info.action_size;
        conf.is_twoplayer = info.num_players > 1;
        let shape = info.obs_shape;
        conf.board_height = shape[shape.len() - 2];
        conf.board_width = shape[shape.len() - 1];
        conf.obs_channels = shape[..shape.len() - 2].iter().product::<usize>().max(1);

        assert!(
            !conf.board_symmetric(),
            "chess must opt out of dihedral augmentation"
        );

        let device = Default::default();
        let _nets: ResNets<NdArray<f32>> = conf.init(&device);
    }
}
