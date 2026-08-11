use burn::{
    Tensor,
    module::Module,
    tensor::{Int, backend::Backend},
};

use crate::config::{NetConfig, NetworkType};
use crate::networks::MuZeroNets;
use crate::networks::{
    dynamic::{DynamicModelConfig, DynamicModelMLP},
    prediction::{PredictionModel, PredictionModelConfig},
    projector::{MlpProjection, MlpProjectionConfig},
    representation::{RepresentationModel, RepresentationModelConfig},
    resnet::ResNets,
};

/// The MLP (Linear) MuZero network family. Obs and hidden are flat vectors.
#[derive(Module, Debug)]
pub struct MlpNets<B: Backend> {
    pub representation: RepresentationModel<B>,
    pub dynamic: DynamicModelMLP<B>,
    pub prediction: PredictionModel<B>,
    pub projection: MlpProjection<B>,
}

impl<B: Backend> MuZeroNets<B> for MlpNets<B> {
    fn init(net_conf: &NetConfig, device: &B::Device) -> Self {
        let linear = net_conf.linear();
        MlpNets {
            representation: RepresentationModelConfig {
                hidden_size: linear.representation.latent_space_dims,
                fc_hidden_size: linear.representation.fc_hidden_size,
                input_size: net_conf.obs_dim,
                n_layers: linear.representation.n_layers,
            }
            .init::<B>(device),
            dynamic: DynamicModelConfig {
                hidden_input: linear.dynamic.latent_space_dims + net_conf.action_space,
                fc_hidden_size: linear.dynamic.fc_hidden_size,
                hidden_output: linear.dynamic.latent_space_dims,
                n_layers: linear.dynamic.n_layers,
                reward_support: net_conf.reward_support_len(),
            }
            .init::<B>(device),
            prediction: PredictionModelConfig {
                fc_hidden_size: linear.prediction.fc_hidden_size,
                hidden_size: linear.prediction.latent_space_dims,
                action_space: net_conf.action_space,
                n_layers: linear.prediction.n_layers,
                value_support: net_conf.value_support_len(),
            }
            .init::<B>(device),
            projection: MlpProjectionConfig {
                latent_size: linear.dynamic.latent_space_dims,
                proj_hidden: net_conf.projection.proj_hidden,
                proj_out: net_conf.projection.proj_out,
                pred_hidden: net_conf.projection.pred_hidden,
            }
            .init::<B>(device),
        }
    }

    fn represent(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.representation.forward(obs)
    }

    fn dynamics(
        &self,
        hidden: Tensor<B, 2>,
        action: Tensor<B, 1, Int>,
        action_size: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        self.dynamic.forward(hidden, action, action_size)
    }

    fn predict(&self, hidden: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        self.prediction.forward(hidden)
    }

    fn project(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        self.projection.project(hidden)
    }

    fn predict_projection(&self, projection: Tensor<B, 2>) -> Tensor<B, 2> {
        self.projection.predict(projection)
    }
}

/// The network family picked at runtime by `network_type` in the config.
/// Lets a single binary build either an `MlpNets` or a `ResNets` without
/// choosing the type at compile time.
#[derive(Module, Debug)]
pub enum AnyNets<B: Backend> {
    Mlp(MlpNets<B>),
    ResNet(ResNets<B>),
}

impl<B: Backend> MuZeroNets<B> for AnyNets<B> {
    fn init(net_conf: &NetConfig, device: &B::Device) -> Self {
        match net_conf.network_type {
            NetworkType::Linear => AnyNets::Mlp(MlpNets::init(net_conf, device)),
            NetworkType::ResNet => AnyNets::ResNet(ResNets::init(net_conf, device)),
        }
    }

    fn represent(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            AnyNets::Mlp(n) => n.represent(obs),
            AnyNets::ResNet(n) => n.represent(obs),
        }
    }

    fn dynamics(
        &self,
        hidden: Tensor<B, 2>,
        action: Tensor<B, 1, Int>,
        action_size: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        match self {
            AnyNets::Mlp(n) => n.dynamics(hidden, action, action_size),
            AnyNets::ResNet(n) => n.dynamics(hidden, action, action_size),
        }
    }

    fn predict(&self, hidden: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        match self {
            AnyNets::Mlp(n) => n.predict(hidden),
            AnyNets::ResNet(n) => n.predict(hidden),
        }
    }

    fn project(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            AnyNets::Mlp(n) => n.project(hidden),
            AnyNets::ResNet(n) => n.project(hidden),
        }
    }

    fn predict_projection(&self, projection: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            AnyNets::Mlp(n) => n.predict_projection(projection),
            AnyNets::ResNet(n) => n.predict_projection(projection),
        }
    }
}
