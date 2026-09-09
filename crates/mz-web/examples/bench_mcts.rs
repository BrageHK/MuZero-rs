//! Benchmarks `chess_mamba_mcts::search_with_stats` across thread counts and
//! virtual-loss on/off -- the Rust counterpart to bee-chess's
//! `bench_mcts.py` (same 400-simulation budget, same metric: nodes/sec and
//! the leaf-collision rate, i.e. how often two threads redundantly
//! NN-evaluate the identical position).
//!
//! Run with:
//!   cargo run --release -p mz-web --example bench_mcts \
//!       --features ndarray --no-default-features
//! (the `ndarray` backend, not the default `flex`/wgpu one, since this is a
//! plain CLI binary with no async GPU-init dance to do -- see
//! `chess_mamba_bot.rs`'s tests for the same choice.)

use std::time::Instant;

use mz_web::chess_mamba_mcts::{SearchConfig, search_with_stats};
use mz_web::model::{self, Be};

const POSITIONS: [&str; 3] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r2q1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9",
];
const SIMULATIONS: usize = 400;
const REPEATS: usize = 2;

struct Result {
    threads: usize,
    virtual_loss: f32,
    total_time_s: f64,
    nodes_per_sec: f64,
    collisions: u64,
    collision_rate: f64,
}

fn bench(model: &model::chess_mamba::Model<Be>, device: &model::Device, threads: usize, virtual_loss: f32) -> Result {
    let mut total_time_s = 0.0;
    let mut total_sims = 0u64;
    let mut total_collisions = 0u64;

    for fen in POSITIONS {
        for _ in 0..REPEATS {
            let cfg = SearchConfig { simulations: SIMULATIONS, threads, virtual_loss, ..SearchConfig::default() };
            let t0 = Instant::now();
            let outcome = search_with_stats(model, device, fen, &cfg);
            let dt = t0.elapsed().as_secs_f64();
            assert!(outcome.best_move.is_some());
            total_time_s += dt;
            total_sims += SIMULATIONS as u64;
            total_collisions += outcome.collisions;
        }
    }

    Result {
        threads,
        virtual_loss,
        total_time_s,
        nodes_per_sec: total_sims as f64 / total_time_s,
        collisions: total_collisions,
        collision_rate: total_collisions as f64 / total_sims as f64,
    }
}

fn main() {
    // libtorch defaults to its own intra-op thread pool spanning every CPU
    // core for *each* forward call. That's fine with 1 outer thread, but
    // once our own tree-parallel search also runs N worker threads, every
    // one of them fights over the same cores via libtorch's internal pool
    // too -- N outer threads x libtorch's own ~N-wide inner pool massively
    // oversubscribes the machine, which is exactly why the earlier tch
    // benchmark capped out around 4 threads and got *worse* past 8. Pinning
    // libtorch to 1 thread per call moves all the parallelism to our outer
    // threads instead, where it belongs.
    #[cfg(feature = "tch")]
    tch::set_num_threads(1);

    let device = model::Device::default();
    let model = model::chess_mamba::Model::from_embedded(&device);

    let configs: &[(usize, f32)] =
        &[(1, 1.0), (2, 0.0), (2, 1.0), (4, 0.0), (4, 1.0), (8, 0.0), (8, 1.0), (12, 0.0), (12, 1.0), (18, 0.0), (18, 1.0)];

    println!("{:>7} {:>12} {:>8} {:>9} {:>10} {:>10}", "threads", "virtual_loss", "time_s", "nodes/s", "collisions", "coll_rate");
    let mut results = Vec::new();
    for &(threads, vl) in configs {
        let r = bench(&model, &device, threads, vl);
        println!(
            "{:>7} {:>12.1} {:>8.2} {:>9.1} {:>10} {:>9.2}%",
            r.threads,
            r.virtual_loss,
            r.total_time_s,
            r.nodes_per_sec,
            r.collisions,
            r.collision_rate * 100.0
        );
        results.push(r);
    }

    let baseline = results[0].nodes_per_sec;
    println!("\nspeedup vs single-thread baseline:");
    for r in &results[1..] {
        println!(
            "  threads={} vl={}: {:.2}x  (collision rate {:.1}%)",
            r.threads,
            if r.virtual_loss > 0.0 { "on" } else { "off" },
            r.nodes_per_sec / baseline,
            r.collision_rate * 100.0
        );
    }
}
