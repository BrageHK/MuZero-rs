use burn::{
    Tensor,
    module::Module,
    tensor::{Int, backend::Backend},
};

use crate::config::NetConfig;
use crate::networks::MuZeroNets;
use crate::networks::{
    dynamic::{DynamicModelConfig, DynamicModelMLP},
    prediction::{PredictionModel, PredictionModelConfig},
    projector::{MlpProjection, MlpProjectionConfig},
    representation::{RepresentationModel, RepresentationModelConfig},
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
                reward_support: net_conf.support_len(),
            }
            .init::<B>(device),
            prediction: PredictionModelConfig {
                fc_hidden_size: linear.prediction.fc_hidden_size,
                hidden_size: linear.prediction.latent_space_dims,
                action_space: net_conf.action_space,
                n_layers: linear.prediction.n_layers,
                value_support: net_conf.support_len(),
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
