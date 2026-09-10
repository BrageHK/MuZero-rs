//! Sweeps `batch_size` for `chess_mamba_mcts_batched::search_with_stats` on
//! the ROCm GPU (`tch::Device::Cuda(0)`) -- the wave-batched search that
//! bypasses burn's batch=1-baked-in onnx model (see that module's
//! docstring). Counterpart to `bench_mcts_tch_gpu`, which measures the old
//! one-leaf-per-NN-call path for comparison.
//!
//! Run with (same libtorch/ROCm setup as bench_mcts_tch_gpu):
//!   cargo run --release -p mz-web --example bench_mcts_batched_gpu \
//!       --features tch --no-default-features

use std::time::Instant;

use mz_web::chess_mamba_mcts_batched::{Model, SearchConfig, search_with_stats};
use tch::Device;

const POSITIONS: [&str; 3] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r2q1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9",
];
const SIMULATIONS: usize = 800;
const REPEATS: usize = 2;

struct BenchResult {
    batch_size: usize,
    total_time_s: f64,
    nodes_per_sec: f64,
    collisions: u64,
    collision_rate: f64,
    cache_hit_rate: f64,
}

fn bench(model: &Model, batch_size: usize) -> BenchResult {
    let mut total_time_s = 0.0;
    let mut total_sims = 0u64;
    let mut total_collisions = 0u64;
    let mut total_cache_hits = 0u64;
    let mut total_cache_misses = 0u64;

    for fen in POSITIONS {
        for _ in 0..REPEATS {
            let cfg = SearchConfig { simulations: SIMULATIONS, batch_size, ..SearchConfig::default() };
            let t0 = Instant::now();
            let outcome = search_with_stats(model, fen, &cfg);
            let dt = t0.elapsed().as_secs_f64();
            assert!(outcome.best_move.is_some());
            total_time_s += dt;
            total_sims += SIMULATIONS as u64;
            total_collisions += outcome.collisions;
            total_cache_hits += outcome.cache_hits;
            total_cache_misses += outcome.cache_misses;
        }
    }

    BenchResult {
        batch_size,
        total_time_s,
        nodes_per_sec: total_sims as f64 / total_time_s,
        collisions: total_collisions,
        collision_rate: total_collisions as f64 / total_sims as f64,
        cache_hit_rate: total_cache_hits as f64 / (total_cache_hits + total_cache_misses).max(1) as f64,
    }
}

fn main() {
    tch::set_num_threads(1);
    let model = Model::load_embedded(Device::Cuda(0));

    let batch_sizes: &[usize] = &[1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256];

    println!("{:>10} {:>8} {:>9} {:>10} {:>10} {:>10}", "batch_size", "time_s", "nodes/s", "collisions", "coll_rate", "cache_hit%");
    let mut results = Vec::new();
    for &batch_size in batch_sizes {
        let r = bench(&model, batch_size);
        println!(
            "{:>10} {:>8.2} {:>9.1} {:>10} {:>9.2}% {:>9.1}%",
            r.batch_size, r.total_time_s, r.nodes_per_sec, r.collisions, r.collision_rate * 100.0, r.cache_hit_rate * 100.0
        );
        results.push(r);
    }

    let best = results.iter().max_by(|a, b| a.nodes_per_sec.total_cmp(&b.nodes_per_sec)).unwrap();
    println!("\nbest: batch_size={} at {:.1} nodes/s", best.batch_size, best.nodes_per_sec);
}
