//! Batched MCTS throughput across batch sizes, with dummy Othello-shaped
//! inputs (3 channels, 8x8 board, 65 actions).
//!
//! The network family is picked from `network_type` in configs/config.yaml:
//! `ResNet` benches `ResNets`, `Linear` benches `MlpNets` (MLP layer sizes
//! come from the config's representation/dynamic/prediction sections).
//!
//! Each case runs one lockstep batched search over `n_games` trees with
//! NUM_SIMULATIONS expansions each. Criterion reports throughput in
//! elements/sec, where one element = one game's search — i.e. games/sec.

use burn::Dispatch;
use burn::tensor::{Distribution, Tensor};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mz_rs::agent::MlpNets;
use mz_rs::mz_config::{MuZeroConfig, NetworkType};
use mz_rs::networks::MuZeroNets;
use mz_rs::networks::resnet::ResNets;
use mz_rs::search::batched_search;
use mz_rs::utils::select_device;
use std::hint::black_box;

const BATCH_SIZES: [usize; 8] = [1, 32, 64, 128, 256, 512, 1024, 2048];
const NUM_SIMULATIONS: usize = 200;
const BOARD: usize = 8;
const OBS_CHANNELS: usize = 3;
const ACTION_SPACE: usize = BOARD * BOARD + 1; // 64 moves + pass

// Backend picked at runtime from `inference_backend` in configs/config.yaml.
type B = Dispatch;

fn othello_conf() -> MuZeroConfig {
    let mut conf = MuZeroConfig { 
        action_space: ACTION_SPACE, 
        obs_dim: OBS_CHANNELS * BOARD * BOARD, 
        num_simulations: NUM_SIMULATIONS, 
        ..Default::default() 
    };
    conf.action_space = ACTION_SPACE;
    conf.obs_dim = OBS_CHANNELS * BOARD * BOARD;
    conf.num_simulations = NUM_SIMULATIONS;
    if let Some(resnet) = conf.resnet.as_mut() {
        resnet.obs_channels = OBS_CHANNELS;
    }
    conf
}

fn run_group<N: MuZeroNets<B>>(c: &mut Criterion, group_name: &str) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);

    // min_rayon_threads is rayon's with_min_len chunk size: usize::MAX keeps the
    // whole batch in one chunk (serial), the config value lets rayon split
    // (parallel). Both run per batch size to compare which is faster.
    let config_min_len = MuZeroConfig::default().min_rayon_threads;

    for n_games in BATCH_SIZES {
        let mut mz_conf = othello_conf();
        let device = select_device(mz_conf.inference_backend);
        println!(
            "inference backend from config: {:?}",
            mz_conf.inference_backend
        );
        let agent: N = mz_conf.init(&device);
        println!("network parameters: {}", agent.num_params());
        let obs = Tensor::<B, 2>::random(
            [n_games, mz_conf.obs_dim],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );

        // One element = one game's full search, so throughput reads as games/sec.
        group.throughput(Throughput::Elements(n_games as u64));

        for (mode, min_len) in [("serial", usize::MAX), ("parallel", config_min_len)] {
            if mode == "parallel" && min_len >= n_games {
                println!(
                    "skipping parallel/{n_games}: min_rayon_threads ({min_len}) >= batch size, \
                     identical to serial — lower it in config.yaml to compare"
                );
                continue;
            }
            mz_conf.min_rayon_threads = min_len;
            group.bench_with_input(BenchmarkId::new(mode, n_games), &n_games, |b, _| {
                b.iter(|| black_box(batched_search(obs.clone(), None, &mz_conf, &agent, 1.0)))
            });
        }
    }

    group.finish();
}

fn bench_batched_throughput(c: &mut Criterion) {
    match MuZeroConfig::default().network_type {
        NetworkType::ResNet => run_group::<ResNets<B>>(c, "batched_mcts_resnet_othello"),
        NetworkType::Linear => run_group::<MlpNets<B>>(c, "batched_mcts_mlp_othello"),
    }
}

criterion_group!(benches, bench_batched_throughput);
criterion_main!(benches);
