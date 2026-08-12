// Full grid search over backends x batch sizes for ResNet inference timing.
// `B = Dispatch` routes to whichever `DispatchDevice` variant is requested at
// runtime, so this single binary can sweep every backend that was compiled
// in via cargo features (see `[features]` in Cargo.toml), e.g.:
//
//   cargo run --release --bin resnet_infer_bench --features "ndarray wgpu vulkan tch"
//
// Backends to sweep are the hardcoded `BACKENDS` list below. Any backend not
// compiled in is skipped with a note instead of panicking the whole run.
//
// Sweeps batch sizes 1..128 (powers of two) for each backend, printing each
// row as it completes and a combined summary grid at the end.
use std::hint::black_box;
use std::time::Instant;

use burn::Dispatch;
use burn::tensor::{Distribution, Shape, Tensor, Transaction};
use mz_rs::mz_config::MuZeroConfig;
use mz_rs::networks::MuZeroNets;
use mz_rs::networks::resnet::ResNets;
use mz_rs::utils::{BackendChoice, select_device};

const WARMUP_ITERS: usize = 10;
const TIMED_ITERS: usize = 100;
const BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];

const BACKENDS: &[BackendChoice] = &[
    //BackendChoice::LibTorchGpu,
    BackendChoice::Wgpu,
    BackendChoice::Vulkan,
    BackendChoice::Rocm,
    BackendChoice::LibTorch,
];

struct Row {
    backend: BackendChoice,
    batch: usize,
    us_per_sample: f64,
}

fn bench_backend<B: burn::prelude::Backend>(
    backend: BackendChoice,
    mz_conf: &MuZeroConfig,
    device: &B::Device,
    out: &mut Vec<Row>,
) {
    let agent: ResNets<B> = mz_conf.init(device);

    println!("\n== backend={backend:?} ==");
    println!(
        "{:>10} {:>12} {:>14} {:>14}",
        "batch", "total_ms", "ms/iter", "us/sample"
    );

    for &batch in BATCH_SIZES {
        let obs = Tensor::<B, 2>::random(
            Shape::new([batch, mz_conf.obs_dim]),
            Distribution::Uniform(0.0, 1.0),
            device,
        );

        let run = |obs: Tensor<B, 2>| {
            let (hidden, reward, value, policy) = agent.initial_inference(obs);
            Transaction::default()
                .register(hidden)
                .register(reward)
                .register(value)
                .register(policy)
                .execute()
        };

        for _ in 0..WARMUP_ITERS {
            black_box(run(obs.clone()));
        }

        let start = Instant::now();
        for _ in 0..TIMED_ITERS {
            black_box(run(black_box(obs.clone())));
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / TIMED_ITERS as u32;
        let ms_per_iter = per_iter.as_secs_f64() * 1000.0;
        let us_per_sample = per_iter.as_secs_f64() * 1_000_000.0 / batch as f64;

        println!(
            "{batch:>10} {:>12.3} {:>14.4} {:>14.4}",
            elapsed.as_secs_f64() * 1000.0,
            ms_per_iter,
            us_per_sample,
        );
        out.push(Row {
            backend,
            batch,
            us_per_sample,
        });
    }
}

fn main() {
    type B = Dispatch;

    let mz_conf = MuZeroConfig::default();
    let mut results = Vec::with_capacity(BACKENDS.len() * BATCH_SIZES.len());
    let mut skipped = Vec::new();

    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    for &backend in BACKENDS {
        let device = match std::panic::catch_unwind(|| select_device(backend)) {
            Ok(device) => device,
            Err(_) => {
                skipped.push(backend);
                continue;
            }
        };
        bench_backend::<B>(backend, &mz_conf, &device, &mut results);
    }

    std::panic::set_hook(prev_hook);

    if !skipped.is_empty() {
        println!("\n(skipped, not compiled in: {skipped:?})");
    }

    let ran_backends: Vec<BackendChoice> = BACKENDS
        .iter()
        .copied()
        .filter(|b| results.iter().any(|r| r.backend == *b))
        .collect();

    println!("\n== grid summary: us/sample (batch x, backend y) ==");
    print_grid_transposed(&ran_backends, &results);

    if let Some(best) = results.iter().min_by(|a, b| a.us_per_sample.total_cmp(&b.us_per_sample)) {
        println!(
            "\nbest: backend={:?} batch={} ({:.3} us/sample)",
            best.backend, best.batch, best.us_per_sample
        );
    }
}

const GREEN: &str = "\x1b[1;32m";
const RED: &str = "\x1b[1;31m";
const RESET: &str = "\x1b[0m";

fn print_grid_transposed(backends: &[BackendChoice], rows: &[Row]) {
    print!("{:>12}", "backend");
    for &batch in BATCH_SIZES {
        print!(" {batch:>10}");
    }
    println!();

    for backend in backends {
        let name = format!("{backend:?}");
        print!("{name:>12}");
        for &batch in BATCH_SIZES {
            let cell = rows
                .iter()
                .find(|r| r.backend == *backend && r.batch == batch)
                .map(|r| r.us_per_sample);
            let (min, max) = min_max_for_batch(backends, rows, batch);
            match cell {
                Some(v) if v == min && min < max => {
                    print!(" {GREEN}{v:>10.3}{RESET}")
                }
                Some(v) if v == max && min < max => print!(" {RED}{v:>10.3}{RESET}"),
                Some(v) => print!(" {v:>10.3}"),
                None => print!(" {:>10}", "-"),
            }
        }
        println!();
    }
}

fn min_max_for_batch(backends: &[BackendChoice], rows: &[Row], batch: usize) -> (f64, f64) {
    backends
        .iter()
        .filter_map(|b| {
            rows.iter()
                .find(|r| r.backend == *b && r.batch == batch)
                .map(|r| r.us_per_sample)
        })
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        })
}
