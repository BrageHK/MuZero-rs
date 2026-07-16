use burn::backend::{
    Wgpu,
    ndarray::{NdArray, NdArrayDevice},
    wgpu::WgpuDevice,
};
use burn::tensor::Tensor;
use criterion::{Criterion, criterion_group, criterion_main};
use mz_rs::{agent::MlpNets, mz_config::MuZeroConfig, search::search_serial::search};
use std::hint::black_box;

// Compares MCTS search speed on the GPU (Wgpu) backend vs the CPU (NdArray)
// backend. Search does hundreds of batch-size-1 forward passes per move.
// Using NdArray is significantly faster
fn bench_search_wgpu(c: &mut Criterion) {
    type B = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    let mz_conf = MuZeroConfig::default();
    let agent = mz_conf.init::<B, MlpNets<B>>(&device);
    let obs = Tensor::<B, 2>::zeros([1, mz_conf.obs_dim()], &device);

    c.bench_function("mcts_search_wgpu", |b| {
        b.iter(|| {
            search(
                black_box(obs.clone()),
                black_box(&mz_conf),
                black_box(&agent),
                black_box(1.0),
            )
        })
    });
}

fn bench_search_ndarray(c: &mut Criterion) {
    type B = NdArray<f32>;
    let device = NdArrayDevice::default();

    let mz_conf = MuZeroConfig::default();
    let agent = mz_conf.init::<B, MlpNets<B>>(&device);
    let obs = Tensor::<B, 2>::zeros([1, mz_conf.obs_dim()], &device);

    c.bench_function("mcts_search_ndarray", |b| {
        b.iter(|| {
            search(
                black_box(obs.clone()),
                black_box(&mz_conf),
                black_box(&agent),
                black_box(1.0),
            )
        })
    });
}

criterion_group!(benches, bench_search_wgpu, bench_search_ndarray);
criterion_main!(benches);
