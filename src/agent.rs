use burn::{
    Tensor,
    module::Module,
    tensor::{Float, Int, backend::Backend},
};

use crate::networks::{
    dynamic::{DynamicModel, DynamicModelConfig},
    prediction::{PredictionModel, PredictionModelConfig},
    representation::{RepresentationModel, RepresentationModelConfig},
};

#[derive(Module, Debug)]
pub struct MuZeroAgent<B: Backend> {
    pub representation: RepresentationModel<B>,
    pub dynamic: DynamicModel<B>,
    pub prediction: PredictionModel<B>,
}

type MuzeroOutput<B> = (
    Tensor<B, 2, Float>,
    Tensor<B, 2, Float>,
    Tensor<B, 2, Float>,
    Tensor<B, 2, Float>,
);

impl<B: Backend> MuZeroAgent<B> {
    /// returns (hidden_state, reward, value, policy)
    pub fn initial_forward(&self, obs: Tensor<B, 2, Float>) -> MuzeroOutput<B> {
        let hidden_state = self.representation.forward(obs);
        let (value, policy) = self.prediction.forward(hidden_state.clone());

        (
            hidden_state.clone(),
            Tensor::zeros([1, 1], &hidden_state.device()),
            value,
            policy,
        )
    }

    /// returns (hidden_state, reward, value, policy)
    pub fn recurrent_forward(
        &self,
        obs: Tensor<B, 2, Float>,
        action: Tensor<B, 1, Int>,
        action_size: usize,
    ) -> (
        Tensor<B, 2, Float>,
        Tensor<B, 2, Float>,
        Tensor<B, 2, Float>,
        Tensor<B, 2, Float>,
    ) {
        let (hidden_state, reward) = self.dynamic.forward(obs, action, action_size);
        let (value, policy) = self.prediction.forward(hidden_state.clone());

        (hidden_state, reward, value, policy)
    }
}
