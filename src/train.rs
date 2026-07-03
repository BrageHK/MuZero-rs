use burn::{Tensor, backend::{Autodiff, wgpu::graphics::WebGpu}, tensor::backend::Backend};

use crate::{agent::MuZeroAgent, mz_config::MuZeroConfig, replay_buffer::ReplayBuffer};

pub fn train<B: Backend>(mz_agent: &MuZeroAgent<B>, mz_conf: &MuZeroConfig, buffer: &mut ReplayBuffer<B>, device: &B::Device) {
    if buffer.total_positions <= mz_conf.batch_size {
        return;
    }


    let sequence = buffer.sample_games(&mz_conf);



    // Unroll muzero
    for (i, buffer_batch) in sequence.iter().enumerate() {


        if i == 1 {
            let obs: Vec<Tensor<B, 2>> = buffer_batch.iter().map(|s| s.state.clone()).collect();

            let obs: Tensor<B, 3> = Tensor::stack(obs, 0);

            let (hidden_state, reward, value, policy)
                = mz_agent.initial_forward(obs.squeeze_dim(1));
        } else {

        }
    }
}