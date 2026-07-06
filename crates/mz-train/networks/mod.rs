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

/// (hidden_state, reward, value, policy)
pub type MuZeroOutput<B> = (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>);

pub trait MuZeroNets<B: Backend>: Module<B> + Sized {
    fn init(mz_conf: &MuZeroConfig, device: &B::Device) -> Self;

    fn represent(&self, obs: Tensor<B, 2>) -> Tensor<B, 2>;

    fn dynamics(
        &self,
        hidden: Tensor<B, 2>,
        action: Tensor<B, 1, Int>,
        action_size: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>);

    /// Returns (value, policy)
    fn predict(&self, hidden: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>);

    /// returns (hidden_state, reward, value, policy). reward is zero at the root
    fn initial_inference(&self, obs: Tensor<B, 2>) -> MuZeroOutput<B> {
        let hidden_state = self.represent(obs);
        let (value, policy) = self.predict(hidden_state.clone());
        let batch_size = hidden_state.dims()[0];
        let reward = Tensor::zeros([batch_size, 1], &hidden_state.device());
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
