//! Raw single-leaf forward-pass throughput, CPU vs MPS, on the tch/libtorch
//! backend -- answers "is GPU (MPS) actually faster than CPU for this
//! workload" *before* wiring MPS into the full tree-parallel search, since
//! every other GPU path tried so far (wgpu, CoreML) lost to CPU on this
//! batch=1, single-leaf-at-a-time model.
//!
//! Run with:
//!   cargo run --release -p mz-web --example bench_device --features tch --no-default-features

use std::time::Instant;

use burn::backend::libtorch::LibTorchDevice;
use burn::prelude::*;
use mz_web::model::{self, Be};

const N_ITERS: usize = 2000;

fn bench(device: LibTorchDevice) -> f64 {
    let model = model::chess_mamba::Model::<Be>::from_embedded(&device);
    let planes = vec![0f32; 64 * 20];

    // Warmup -- first call on a fresh device pays one-time setup cost
    // (kernel compilation, context init) that shouldn't count against
    // steady-state throughput.
    for _ in 0..20 {
        let input: Tensor<Be, 3> = Tensor::<Be, 1>::from_floats(planes.as_slice(), &device).reshape([1, 64, 20]);
        let (p, v) = model.forward(input);
        let _: Vec<f32> = p.into_data().to_vec().unwrap();
        let _: Vec<f32> = v.into_data().to_vec().unwrap();
    }

    let t0 = Instant::now();
    for _ in 0..N_ITERS {
        let input: Tensor<Be, 3> = Tensor::<Be, 1>::from_floats(planes.as_slice(), &device).reshape([1, 64, 20]);
        let (p, v) = model.forward(input);
        let _: Vec<f32> = p.into_data().to_vec().unwrap();
        let _: Vec<f32> = v.into_data().to_vec().unwrap();
    }
    let dt = t0.elapsed().as_secs_f64();
    N_ITERS as f64 / dt
}

fn main() {
    tch::set_num_threads(1);

    println!("device        forward-passes/sec");
    let cpu_rate = bench(LibTorchDevice::Cpu);
    println!("{:12}  {:.1}", "cpu", cpu_rate);

    let mps_rate = std::panic::catch_unwind(|| bench(LibTorchDevice::Mps)).ok();
    match mps_rate {
        Some(rate) => {
            println!("{:12}  {:.1}", "mps", rate);
            println!("\nmps/cpu ratio: {:.2}x", rate / cpu_rate);
        }
        None => println!("{:12}  FAILED (see panic above)", "mps"),
    }
}
