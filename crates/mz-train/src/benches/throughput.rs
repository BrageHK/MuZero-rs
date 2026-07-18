use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use burn::backend::{
    Wgpu,
    ndarray::{NdArray, NdArrayDevice},
    wgpu::WgpuDevice,
};
use burn::tensor::Tensor;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use crossbeam::channel::{Sender, unbounded};
use mz_rs::agent::MlpNets;
use mz_rs::env::Environment;
use mz_rs::env::cartpole::env::CartPoleWrapper;
use mz_rs::mz_config::MuZeroConfig;
use mz_rs::search::search as search_parallel;
use mz_rs::search::search_serial::search as search_serial;
use mz_rs::search::{InferenceHandles, inference_channels, inference_master};
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

// Serial search is single-threaded and fastest on NdArray (see search_speed
// bench). Parallel search only pays off once there are enough concurrent
// leaf requests to fill a batch, which is the whole point of running it on
// Wgpu with many worker threads feeding one inference_master.
type SerialB = NdArray<f32>;
type ParallelB = Wgpu<f32, i32>;

fn play_one_game_serial(mz_conf: &MuZeroConfig, agent: &MlpNets<SerialB>, device: &NdArrayDevice) {
    let mut env = CartPoleWrapper::default();
    loop {
        let obs = env.state_tensor::<SerialB>(device);
        let (visit_distribution, _value, _action) = search_serial(obs, mz_conf, agent, 1.0);
        let dist = WeightedIndex::new(&visit_distribution).unwrap();
        let action = dist.sample(&mut rand::rng());
        let result = env.step(action);
        if result.done || result.truncated {
            return;
        }
    }
}

fn bench_throughput_serial_ndarray(c: &mut Criterion) {
    let device = NdArrayDevice::default();
    let mz_conf = MuZeroConfig::default();
    let agent = mz_conf.init::<SerialB, MlpNets<SerialB>>(&device);

    let mut group = c.benchmark_group("games_per_sec");
    group.throughput(Throughput::Elements(1));
    group.bench_function("serial_ndarray", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                play_one_game_serial(&mz_conf, &agent, &device);
            }
            start.elapsed()
        });
    });
    group.finish();
}

// One self-play worker: plays games back-to-back forever, reporting each
// completed game on `game_done_tx`. All NN calls go through `inference`,
// which forwards to the shared inference_master thread for batching.
fn worker_loop_bench(
    mz_conf: MuZeroConfig,
    inference: InferenceHandles<ParallelB>,
    infer_device: WgpuDevice,
    game_done_tx: Sender<()>,
    stop: Arc<AtomicBool>,
) {
    let mut env = CartPoleWrapper::default();
    while !stop.load(Ordering::Relaxed) {
        let obs = env.state_tensor::<ParallelB>(&infer_device);
        let (visit_distribution, _value, _action) = search_parallel(obs, &mz_conf, 1.0, &inference);
        let dist = WeightedIndex::new(&visit_distribution).unwrap();
        let action = dist.sample(&mut rand::rng());
        let result = env.step(action);
        if result.done || result.truncated {
            if game_done_tx.send(()).is_err() {
                return;
            }
            env.reset();
        }
    }
}

fn bench_throughput_parallel_wgpu(c: &mut Criterion) {
    let device = WgpuDevice::default();
    let mz_conf = MuZeroConfig::default();
    let agent = mz_conf.init::<ParallelB, MlpNets<ParallelB>>(&device);
    let agent_cell = Arc::new(Mutex::new(agent));

    let channels = inference_channels::<ParallelB>();
    let master_agent_cell = Arc::clone(&agent_cell);
    let action_space = mz_conf.action_space;
    let init_batch_size = mz_conf.init_batch_size;
    let rec_batch_size = mz_conf.rec_batch_size;
    let max_wait = Duration::from_secs_f32(mz_conf.max_thread_wait);

    let master_handle = thread::spawn(move || {
        inference_master(
            channels.init_rx,
            channels.rec_rx,
            master_agent_cell,
            action_space,
            init_batch_size,
            rec_batch_size,
            max_wait,
        );
    });

    let stop = Arc::new(AtomicBool::new(false));
    let (game_done_tx, game_done_rx) = unbounded::<()>();
    let mut worker_handles = Vec::with_capacity(mz_conf.search_threads());
    for _ in 0..mz_conf.search_threads() {
        let inference = channels.handles.clone();
        let worker_conf = mz_conf.clone();
        let worker_device = device.clone();
        let game_done_tx = game_done_tx.clone();
        let stop = Arc::clone(&stop);
        worker_handles.push(thread::spawn(move || {
            worker_loop_bench(worker_conf, inference, worker_device, game_done_tx, stop);
        }));
    }
    drop(game_done_tx);
    drop(channels.handles);

    // Let the pool past first-dispatch shader compilation before timing.
    for _ in 0..mz_conf.search_threads() {
        let _ = game_done_rx.recv();
    }

    let mut group = c.benchmark_group("games_per_sec");
    group.throughput(Throughput::Elements(1));
    group.bench_function("parallel_wgpu", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                game_done_rx.recv().expect("worker pool died");
            }
            start.elapsed()
        });
    });
    group.finish();

    stop.store(true, Ordering::Relaxed);
    for handle in worker_handles {
        let _ = handle.join();
    }
    let _ = master_handle.join();
}

criterion_group!(
    benches,
    bench_throughput_serial_ndarray,
    bench_throughput_parallel_wgpu
);
criterion_main!(benches);
