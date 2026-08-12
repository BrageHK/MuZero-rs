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

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ResNetRepresentationConfig {
    pub channels: usize,
    pub n_blocks: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ResNetBlockConfig {
    pub channels: usize,
    pub n_blocks: usize,
    pub fc_hidden_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ResNetSubConfig {
    pub representation: ResNetRepresentationConfig,
    pub dynamic: ResNetBlockConfig,
    pub prediction: ResNetBlockConfig,
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
