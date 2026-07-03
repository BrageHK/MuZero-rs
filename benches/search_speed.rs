use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn::tensor::Tensor;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use mz_rs::{agent::MuzeroConfig, search::search};

fn bench_search(c: &mut Criterion) {
    type B = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    let mz_conf = MuzeroConfig::default();
    let agent = mz_conf.init::<B>(&device);
    let obs = Tensor::<B, 2>::zeros([1, mz_conf.input_dim], &device);

    c.bench_function("mcts_search", |b| {
        b.iter(|| search(black_box(obs.clone()), black_box(&mz_conf), black_box(&agent)))
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);