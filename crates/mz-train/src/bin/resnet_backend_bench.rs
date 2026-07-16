// Compares MuZero inference speed on ndarray (CPU) vs rocm (GPU) across many
// batch sizes, for both network families (ResNet and Linear/MLP), so you can
// see where GPU dispatch overhead stops dominating and the GPU starts winning.
//
// Build/run with rocm compiled in (needs an AMD GPU + ROCm drivers):
//   cargo run --release --bin resnet_backend_bench --features rocm
// Without --features rocm, only the ndarray side runs.

use std::hint::black_box;
use std::time::{Duration, Instant};

use burn::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Shape, Transaction};
use mz_rs::agent::MlpNets;
use mz_rs::mz_config::MuZeroConfig;
use mz_rs::networks::MuZeroNets;
use mz_rs::networks::resnet::{ResNetConfig, ResNets};

const WARMUP_ITERS: usize = 5;
const TIMED_ITERS: usize = 30;

fn resnet_config(mz_conf: &MuZeroConfig) -> ResNetConfig {
    let resnet = mz_conf
        .resnet
        .as_ref()
        .expect("configs/config.yaml needs a `resnet:` section to run this benchmark");
    ResNetConfig {
        obs_channels: resnet.obs_channels,
        channels: resnet.channels,
        n_blocks: resnet.n_blocks,
        board_height: resnet.board_height,
        board_width: resnet.board_width,
        action_space: mz_conf.action_space(),
        fc_hidden_size: resnet.fc_hidden_size,
    }
}

fn bench_backend<B: Backend, N: MuZeroNets<B>>(
    name: &str,
    device: &B::Device,
    nets: &N,
    input_dim: usize,
    batch_sizes: &[usize],
) -> Vec<(usize, Duration)> {
    println!("\n== {name} ==");
    println!(
        "{:>8} {:>14} {:>14} {:>14}",
        "batch", "total ms", "ms/iter", "us/sample"
    );

    let mut results = Vec::with_capacity(batch_sizes.len());
    for &batch in batch_sizes {
        let obs = Tensor::<B, 2>::random(
            Shape::new([batch, input_dim]),
            Distribution::Uniform(0.0, 1.0),
            device,
        );

        let run = |obs: Tensor<B, 2>| {
            let (hidden, reward, value, policy) = nets.initial_inference(obs);
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

        println!(
            "{batch:>8} {:>14.3} {:>14.4} {:>14.4}",
            elapsed.as_secs_f64() * 1000.0,
            per_iter.as_secs_f64() * 1000.0,
            per_iter.as_secs_f64() * 1_000_000.0 / batch as f64,
        );
        results.push((batch, per_iter));
    }
    results
}

// Only prints rows where both backends were benchmarked at the same batch
// size — cpu and gpu sweep different ranges (cpu gets slow fast).
#[cfg(feature = "rocm")]
fn print_comparison(label: &str, cpu: &[(usize, Duration)], gpu: &[(usize, Duration)]) {
    println!("\n== {label}: ndarray (cpu) vs rocm (gpu) ==");
    println!(
        "{:>8} {:>14} {:>14} {:>10}  {}",
        "batch", "cpu ms/iter", "gpu ms/iter", "speedup", "faster"
    );
    for &(batch, cpu_t) in cpu {
        let Some(&(_, gpu_t)) = gpu.iter().find(|(b, _)| *b == batch) else {
            continue;
        };
        let cpu_ms = cpu_t.as_secs_f64() * 1000.0;
        let gpu_ms = gpu_t.as_secs_f64() * 1000.0;
        let speedup = cpu_ms / gpu_ms;
        let faster = if speedup > 1.0 { "gpu" } else { "cpu" };
        println!("{batch:>8} {cpu_ms:>14.4} {gpu_ms:>14.4} {speedup:>9.2}x  {faster}");
    }
}

fn main() {
    let mz_conf = MuZeroConfig::default();
    let resnet_cfg = resnet_config(&mz_conf);

    // CPU gets brutally slow past a few dozen samples, so cap it low.
    let cpu_batch_sizes: Vec<usize> = (0..=5).map(|i| 1usize << i).collect(); // 1..=32
    // GPU stays cheap per-iter even at large batches — sweep much further.
    let gpu_batch_sizes: Vec<usize> = (0..=12).map(|i| 1usize << i).collect(); // 1..=4096

    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    let cpu_device = NdArrayDevice::default();

    println!("\n########## ResNet ##########");
    let resnet_input_dim =
        resnet_cfg.obs_channels * resnet_cfg.board_height * resnet_cfg.board_width;
    let resnet_cpu_nets: ResNets<NdArray<f32>> = resnet_cfg.init(&cpu_device);
    let resnet_cpu_results = bench_backend::<NdArray<f32>, ResNets<NdArray<f32>>>(
        "resnet ndarray (cpu)",
        &cpu_device,
        &resnet_cpu_nets,
        resnet_input_dim,
        &cpu_batch_sizes,
    );

    println!("\n########## Linear (MLP) ##########");
    let mlp_cpu_nets: MlpNets<NdArray<f32>> = mz_conf.init(&cpu_device);
    let mlp_cpu_results = bench_backend::<NdArray<f32>, MlpNets<NdArray<f32>>>(
        "linear ndarray (cpu)",
        &cpu_device,
        &mlp_cpu_nets,
        mz_conf.obs_dim(),
        &cpu_batch_sizes,
    );

    #[cfg(feature = "rocm")]
    {
        use burn::backend::Rocm;
        use burn::backend::rocm::RocmDevice;
        let gpu_device = RocmDevice::default();

        let resnet_gpu_nets: ResNets<Rocm> = resnet_cfg.init(&gpu_device);
        let resnet_gpu_results = bench_backend::<Rocm, ResNets<Rocm>>(
            "resnet rocm (gpu)",
            &gpu_device,
            &resnet_gpu_nets,
            resnet_input_dim,
            &gpu_batch_sizes,
        );
        print_comparison("resnet", &resnet_cpu_results, &resnet_gpu_results);

        let mlp_gpu_nets: MlpNets<Rocm> = mz_conf.init(&gpu_device);
        let mlp_gpu_results = bench_backend::<Rocm, MlpNets<Rocm>>(
            "linear rocm (gpu)",
            &gpu_device,
            &mlp_gpu_nets,
            mz_conf.obs_dim(),
            &gpu_batch_sizes,
        );
        print_comparison("linear", &mlp_cpu_results, &mlp_gpu_results);
    }

    #[cfg(not(feature = "rocm"))]
    {
        let _ = (resnet_cpu_results, mlp_cpu_results, gpu_batch_sizes);
        println!(
            "\nrocm backend not compiled in — rerun with \
             `cargo run --release --bin resnet_backend_bench --features rocm` \
             on a machine with an AMD GPU + ROCm drivers to compare."
        );
    }
}
