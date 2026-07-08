//! Batched MCTS throughput on the ResNet nets across batch sizes, with dummy
//! Othello-shaped inputs (3 channels, 8x8 board, 65 actions).
//!
//! Each case runs one lockstep batched search over `n_games` trees with
//! NUM_SIMULATIONS expansions each. Criterion reports throughput in
//! elements/sec, where one element = one game's search — i.e. games/sec.

use burn::Dispatch;
use burn::tensor::{Distribution, Tensor};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mz_rs::mz_config::{MuZeroConfig, NetworkType};
use mz_rs::networks::resnet::ResNets;
use mz_rs::search::batched_search;
use mz_rs::utils::select_device;
use std::hint::black_box;

const BATCH_SIZES: [usize; 6] = [32, 64, 128, 256, 512, 1024];
const NUM_SIMULATIONS: usize = 200;
const BOARD: usize = 8;
const OBS_CHANNELS: usize = 3;
const ACTION_SPACE: usize = BOARD * BOARD + 1; // 64 moves + pass

fn othello_conf(n_games: usize) -> MuZeroConfig {
    let mut conf = MuZeroConfig::default();
    conf.network_type = NetworkType::ResNet;
    conf.action_space = ACTION_SPACE;
    conf.obs_dim = OBS_CHANNELS * BOARD * BOARD;
    conf.num_simulations = NUM_SIMULATIONS;
    conf.init_batch_size = n_games;
    conf.rec_batch_size = n_games;
    let resnet = conf.resnet.as_mut().expect("config.yaml has resnet section");
    resnet.obs_channels = OBS_CHANNELS;
    resnet.board_height = BOARD;
    resnet.board_width = BOARD;
    conf
}

fn bench_batched_throughput(c: &mut Criterion) {
    // Backend picked at runtime from `inference_backend` in configs/config.yaml.
    type B = Dispatch;

    let mut group = c.benchmark_group("batched_mcts_resnet_othello");
    group.sample_size(10);

    for n_games in BATCH_SIZES {
        let mz_conf = othello_conf(n_games);
        let device = select_device(mz_conf.inference_backend);
        println!("inference backend from config: {:?}", mz_conf.inference_backend);
        let agent: ResNets<B> = mz_conf.init(&device);
        let obs = Tensor::<B, 2>::random(
            [n_games, mz_conf.obs_dim],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );

        // One element = one game's full search, so throughput reads as games/sec.
        group.throughput(Throughput::Elements(n_games as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_games), &n_games, |b, _| {
            b.iter(|| black_box(batched_search(obs.clone(), &mz_conf, &agent, 1.0)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_batched_throughput);
criterion_main!(benches);
