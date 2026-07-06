use burn::{
    Tensor,
    module::Module,
    tensor::{Int, backend::Backend},
};

use crate::mz_config::MuZeroConfig;
use crate::networks::MuZeroNets;
use crate::networks::{
    dynamic::{DynamicModelConfig, DynamicModelMLP},
    prediction::{PredictionModel, PredictionModelConfig},
    representation::{RepresentationModel, RepresentationModelConfig},
};

/// The MLP (Linear) MuZero network family. Obs and hidden are flat vectors.
#[derive(Module, Debug)]
pub struct MlpNets<B: Backend> {
    pub representation: RepresentationModel<B>,
    pub dynamic: DynamicModelMLP<B>,
    pub prediction: PredictionModel<B>,
}

impl<B: Backend> MuZeroNets<B> for MlpNets<B> {
    fn init(mz_conf: &MuZeroConfig, device: &B::Device) -> Self {
        MlpNets {
            representation: RepresentationModelConfig {
                hidden_size: mz_conf.representation.latent_space_dims,
                fc_hidden_size: mz_conf.representation.fc_hidden_size,
                input_size: mz_conf.obs_dim,
                n_layers: mz_conf.representation.n_layers,
            }
            .init::<B>(device),
            dynamic: DynamicModelConfig {
                hidden_input: mz_conf.dynamic.latent_space_dims + mz_conf.action_space,
                fc_hidden_size: mz_conf.dynamic.fc_hidden_size,
                hidden_output: mz_conf.dynamic.latent_space_dims,
                n_layers: mz_conf.dynamic.n_layers,
            }
            .init::<B>(device),
            prediction: PredictionModelConfig {
                fc_hidden_size: mz_conf.prediction.fc_hidden_size,
                hidden_size: mz_conf.prediction.latent_space_dims,
                action_space: mz_conf.action_space,
                n_layers: mz_conf.prediction.n_layers,
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
}
