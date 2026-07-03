use burn::{
    Tensor,
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::{Float, Int, backend::Backend},
};

use crate::mz_config::MuZeroConfig;
use crate::networks::{
    dynamic::DynamicModel,
    prediction::PredictionModel,
    representation::RepresentationModel,
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

        let batch_size = hidden_state.dims()[0];

        (
            hidden_state.clone(),
            Tensor::zeros([batch_size, 1], &hidden_state.device()),
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

pub fn agent_to_backend<B1: Backend, B2: Backend>(
    agent: &MuZeroAgent<B1>,
    mz_conf: &MuZeroConfig,
    device: &B2::Device,
) -> MuZeroAgent<B2> {
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = recorder.record(agent.clone().into_record(), ()).unwrap();
    let record = recorder.load(bytes, device).unwrap();
    mz_conf.init::<B2>(device).load_record(record)
}
