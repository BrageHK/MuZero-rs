use std::time::Instant;

use burn::backend::{
    Wgpu,
    ndarray::{NdArray, NdArrayDevice},
    wgpu::WgpuDevice,
};
use burn::tensor::backend::Backend;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use mz_rs::agent::MlpNets;
use mz_rs::env::Environment;
use mz_rs::env::cartpole::env::CartPoleWrapper;
use mz_rs::mz_config::MuZeroConfig;
use mz_rs::search::batched_search;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

fn play_games_lockstep<B: Backend>(
    mz_conf: &MuZeroConfig,
    agent: &MlpNets<B>,
    device: &B::Device,
    target_games: u64,
) {
    let mut envs = vec![CartPoleWrapper::default(); mz_conf.game_batch_size];
    for env in envs.iter_mut() {
        env.reset();
    }

    let mut rng = rand::rng();
    let mut completed = 0u64;
    while completed < target_games {
        let obs = CartPoleWrapper::batch_state_tensor::<B>(&envs, device);
        let results = batched_search(obs, None, mz_conf, agent, 1.0, false);
        for (env, res) in envs.iter_mut().zip(&results) {
            let dist = WeightedIndex::new(&res.distribution).unwrap();
            let action = dist.sample(&mut rng);
            let result = env.step(action);
            if result.done || result.truncated {
                completed += 1;
                env.reset();
            }
        }
    }
}

fn bench_throughput_ndarray(c: &mut Criterion) {
    type B = NdArray<f32>;
    let device = NdArrayDevice::default();
    let mz_conf = MuZeroConfig::default();
    let agent = mz_conf.init::<B, MlpNets<B>>(&device);

    let mut group = c.benchmark_group("games_per_sec");
    group.sample_size(10);
    group.throughput(Throughput::Elements(1));
    group.bench_function("batched_ndarray", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            play_games_lockstep(&mz_conf, &agent, &device, iters);
            start.elapsed()
        });
    });
    group.finish();
}

fn bench_throughput_wgpu(c: &mut Criterion) {
    type B = Wgpu<f32, i32>;
    let device = WgpuDevice::default();
    let mz_conf = MuZeroConfig::default();
    let agent = mz_conf.init::<B, MlpNets<B>>(&device);

    play_games_lockstep(&mz_conf, &agent, &device, 1);

    let mut group = c.benchmark_group("games_per_sec");
    group.sample_size(10);
    group.throughput(Throughput::Elements(1));
    group.bench_function("batched_wgpu", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            play_games_lockstep(&mz_conf, &agent, &device, iters);
            start.elapsed()
        });
    });
    group.finish();
}

criterion_group!(benches, bench_throughput_ndarray, bench_throughput_wgpu);
criterion_main!(benches);
