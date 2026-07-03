use std::fs;

use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use crate::{
    agent::MuZeroAgent,
    networks::{
        dynamic::DynamicModelConfig, prediction::PredictionModelConfig,
        representation::RepresentationModelConfig,
    },
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum NetworkType {
    Linear,
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

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct MuZeroConfig {
    pub network_type: NetworkType,

    pub representation: NetworkSubConfig,
    pub dynamic: NetworkSubConfig,
    pub prediction: NetworkSubConfig,

    pub action_space: usize,
    pub obs_dim: usize,

    pub n_steps: usize,
    pub unroll_steps: usize,
    pub batch_size: usize,
    pub discount: f32,
    pub num_simulations: usize,
    pub dirichlet_noise: f32,
    pub total_steps: usize,

    // Original muzero paper uses t = 1 first 500k steps, t = 0.5 for next 250k and 0.25 for remaining
    pub temperature_schedule: Vec<TemperatureSchedule>,
}

impl Default for MuZeroConfig {
    fn default() -> Self {
        let file_content = fs::read_to_string("configs/config.yaml").expect("Failed to read file");
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> MuZeroAgent<B> {
        MuZeroAgent {
            representation: RepresentationModelConfig {
                hidden_size: self.representation.latent_space_dims,
                fc_hidden_size: self.representation.fc_hidden_size,
                input_size: self.action_space,
            }
            .init::<B>(device),
            dynamic: DynamicModelConfig {
                hidden_input: self.dynamic.latent_space_dims + self.action_space,
                fc_hidden_size: self.dynamic.fc_hidden_size,
                hidden_output: self.dynamic.latent_space_dims,
            }
            .init::<B>(device),
            prediction: PredictionModelConfig {
                hidden_size: self.prediction.latent_space_dims,
                fc_hidden_size: self.prediction.fc_hidden_size,
                action_space: self.action_space,
            }
            .init::<B>(device),
        }
    }
}
