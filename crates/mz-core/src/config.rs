use serde::{Deserialize, Serialize};

use crate::support::support_len;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum NetworkType {
    Linear,
    ResNet,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct NetworkSubConfig {
    pub latent_space_dims: usize,
    pub fc_hidden_size: usize,
    pub n_layers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct LinearSubConfig {
    pub representation: NetworkSubConfig,
    pub dynamic: NetworkSubConfig,
    pub prediction: NetworkSubConfig,
}

/// KataGo-style global pooling (arXiv:1902.10565 §3.3/A.2-A.5), scoped to a
/// single tower or head. `every == 0` (or `channels == 0` for heads) disables
/// it, reproducing the exact pre-global-pooling architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
pub struct GPoolConfig {
    /// Insert a global-pooling residual block every Nth block (1-indexed
    /// position). 0 disables global pooling in this tower entirely.
    #[serde(default)]
    pub every: usize,
    /// Channels carved out of the block's own width for the pooling branch.
    /// 0 = auto-pick channels/4 (minimum 4). Ignored when `every == 0`.
    #[serde(default)]
    pub pool_channels: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ResNetRepresentationConfig {
    pub channels: usize,
    pub n_blocks: usize,
    #[serde(default)]
    pub gpool: GPoolConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ResNetBlockConfig {
    pub channels: usize,
    pub n_blocks: usize,
    pub fc_hidden_size: usize,
    /// Inert for `prediction` (no residual tower there); only `dynamic` acts on this.
    #[serde(default)]
    pub gpool: GPoolConfig,
}

/// Global pooling injected directly into the policy/value heads (KataGo A.4/A.5).
/// 0 = disabled, reproducing the exact original head shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
pub struct HeadPoolConfig {
    /// Channels for the policy head's parallel pooling conv. 0 = disabled.
    #[serde(default)]
    pub policy_channels: usize,
    /// Channels for the value head's conv; when > 0 the value head pools
    /// (mean+max) instead of flattening the full h*w grid, making it
    /// resolution-agnostic. 0 = disabled (today's flatten(h*w) behavior).
    #[serde(default)]
    pub value_channels: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ResNetSubConfig {
    pub representation: ResNetRepresentationConfig,
    pub dynamic: ResNetBlockConfig,
    pub prediction: ResNetBlockConfig,
    #[serde(default)]
    pub head_gpool: HeadPoolConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProjectionSubConfig {
    pub proj_hidden: usize,
    pub proj_out: usize,
    pub pred_hidden: usize,
}

impl Default for ProjectionSubConfig {
    fn default() -> Self {
        ProjectionSubConfig {
            proj_hidden: 256,
            proj_out: 64,
            pred_hidden: 128,
        }
    }
}

/// Everything a network family needs to be shaped. Deliberately free of the
/// training-only knobs in `mz-train`'s `MuZeroConfig` so that the same weights
/// can be rebuilt without a filesystem or a yaml parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetConfig {
    pub network_type: NetworkType,
    pub obs_dim: usize,
    pub action_space: usize,
    pub support_size: usize,
    /// Categorical (two-hot) value/reward heads for single-player envs; a plain
    /// scalar head for board games (paper App. F/G: l^v=(z-q)^2, l^r=0).
    pub categorical: bool,
    pub board_height: usize,
    pub board_width: usize,
    pub obs_channels: usize,
    pub linear: Option<LinearSubConfig>,
    pub resnet: Option<ResNetSubConfig>,
    pub projection: ProjectionSubConfig,
}

impl NetConfig {
    pub fn support_len(&self) -> usize {
        support_len(self.support_size)
    }

    pub fn value_support_len(&self) -> usize {
        if self.categorical {
            self.support_len()
        } else {
            1
        }
    }

    pub fn reward_support_len(&self) -> usize {
        if self.categorical {
            self.support_len()
        } else {
            1
        }
    }

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
}
