pub mod dynamic;
pub mod prediction;
pub mod representation;
pub mod resblock;
pub mod resnet;

use burn::{
    Tensor,
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::{Int, backend::Backend},
};

use crate::mz_config::MuZeroConfig;

/// (hidden_state, reward_logits, value_logits, policy). reward_logits and
/// value_logits are categorical distributions over the support (see `support`).
pub type MuZeroOutput<B> = (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>);

/// Appendix G: scale the hidden state to the same range as the action input
/// ([0, 1]) per sample: s_scaled = (s - min(s)) / (max(s) - min(s)).
pub fn scale_hidden_state<B: Backend>(hidden: Tensor<B, 2>) -> Tensor<B, 2> {
    let min = hidden.clone().min_dim(1);
    let max = hidden.clone().max_dim(1);
    let range = (max - min.clone()).clamp_min(1e-5);
    (hidden - min) / range
}

pub trait MuZeroNets<B: Backend>: Module<B> + Sized {
    fn init(mz_conf: &MuZeroConfig, device: &B::Device) -> Self;

    fn represent(&self, obs: Tensor<B, 2>) -> Tensor<B, 2>;

    fn dynamics(
        &self,
        hidden: Tensor<B, 2>,
        action: Tensor<B, 1, Int>,
        action_size: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>);

    /// Returns (value_logits, policy)
    fn predict(&self, hidden: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>);

    /// returns (hidden_state, reward_logits, value_logits, policy). reward is a
    /// zero distribution at the root (softmax of zeros decodes to scalar 0).
    fn initial_inference(&self, obs: Tensor<B, 2>) -> MuZeroOutput<B> {
        let hidden_state = scale_hidden_state(self.represent(obs));
        let (value, policy) = self.predict(hidden_state.clone());
        let batch_size = hidden_state.dims()[0];
        let support_len = value.dims()[1];
        let reward = Tensor::zeros([batch_size, support_len], &hidden_state.device());
        (hidden_state, reward, value, policy)
    }

    /// returns (hidden_state, reward, value, policy)
    fn recurrent_inference(
        &self,
        hidden: Tensor<B, 2>,
        action: Tensor<B, 1, Int>,
        action_size: usize,
    ) -> MuZeroOutput<B> {
        let (hidden_state, reward) = self.dynamics(hidden, action, action_size);
        let hidden_state = scale_hidden_state(hidden_state);
        let (value, policy) = self.predict(hidden_state.clone());
        (hidden_state, reward, value, policy)
    }
}

pub fn nets_to_backend<B1: Backend, B2: Backend, N1: MuZeroNets<B1>, N2: MuZeroNets<B2>>(
    nets: &N1,
    mz_conf: &MuZeroConfig,
    device: &B2::Device,
) -> N2 {
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = recorder.record(nets.clone().into_record(), ()).unwrap();
    let record = recorder.load(bytes, device).unwrap();
    N2::init(mz_conf, device).load_record(record)
}
