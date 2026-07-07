use std::hint::black_box;
use std::time::Instant;

use burn::{
    Tensor,
    backend::{NdArray, ndarray::NdArrayDevice},
    tensor::{Distribution, Shape, Transaction},
};
use mz_rs::agent::MlpNets;
use mz_rs::mz_config::MuZeroConfig;
use mz_rs::networks::MuZeroNets;

fn main() {
    // type B = Wgpu::<f32, i32>;
    type B = NdArray<f32>;
    let device = NdArrayDevice::default();
    // let device = WgpuDevice::default();
    let mut mz_conf = MuZeroConfig::default();

    // Batch size speed test from 1 -> 1024

    for i in 0..17 {
        let batch_size: usize = 2_usize.pow(i);
        mz_conf.batch_size = batch_size;
        let mz_agent: MlpNets<B> = mz_conf.init(&device);

        let distribution = Distribution::Uniform(0.0, 1.0); // Any random value between 0.0 and 1.0
        let dummy_tensor = Tensor::<B, 2>::random(
            Shape::new([batch_size, mz_conf.obs_dim]),
            distribution,
            &device,
        );
        println!("Dummy tensor shape: {:?}", dummy_tensor.shape());

        let num_iterations = 100;
        // warumup
        for _ in 0..3 {
            let (hidden_state, reward, value, policy) =
                black_box(mz_agent.initial_inference(black_box(dummy_tensor.clone())));
            // single sync point: batch all readbacks in one transaction instead of 4 blocking calls
            let [hidden_state, reward, value, policy] = black_box(
                Transaction::default()
                    .register(hidden_state)
                    .register(reward)
                    .register(value)
                    .register(policy)
                    .execute()
                    .try_into()
                    .expect("correct amount of tensor data"),
            );
            black_box((hidden_state, reward, value, policy));
        }

        let start_time = Instant::now();
        for _ in 0..num_iterations {
            let (hidden_state, reward, value, policy) =
                black_box(mz_agent.initial_inference(black_box(dummy_tensor.clone())));
            // single sync point: batch all readbacks in one transaction instead of 4 blocking calls
            let [hidden_state, reward, value, policy] = black_box(
                Transaction::default()
                    .register(hidden_state)
                    .register(reward)
                    .register(value)
                    .register(policy)
                    .execute()
                    .try_into()
                    .expect("correct amount of tensor data"),
            );
            black_box((hidden_state, reward, value, policy));
        }

        println!(
            "Time: {:?}, Time per data_point: {}s, Batch size: {}",
            start_time.elapsed(),
            start_time.elapsed().as_millis() as f32 / batch_size as f32,
            batch_size
        );
    }
}
