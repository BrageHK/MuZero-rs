use std::ops::{Deref, DerefMut};
use std::time::Duration;

use burn::Dispatch;
use burn::module::Module;
use burn::tensor::Tensor;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rayon::prelude::*;
use std::hint::black_box;

use mz_rs::env::Environment;
use mz_rs::env::atari::env::{AtariEnv, AtariGame, set_atari_game};
use mz_rs::mz_config::MuZeroConfig;
use mz_rs::networks::resnet::ResNets;
use mz_rs::search::batched_search;
use mz_rs::utils::select_device;

type B = Dispatch;

const N_ENVS: &[usize] = &[2, 4, 8, 16, 32, 48];
const MEASUREMENT_SECS: u64 = 10;

struct SendEnv(AtariEnv);
unsafe impl Send for SendEnv {}
impl Deref for SendEnv {
    type Target = AtariEnv;
    fn deref(&self) -> &AtariEnv {
        &self.0
    }
}
impl DerefMut for SendEnv {
    fn deref_mut(&mut self) -> &mut AtariEnv {
        &mut self.0
    }
}

#[derive(Clone, Copy)]
enum StepMode {
    Serial,
    Parallel,
}

fn parse_sizes() -> Vec<usize> {
    match std::env::var("ENV_STEP_SIZES") {
        Ok(s) => s.split(',').filter_map(|p| p.trim().parse().ok()).collect(),
        Err(_) => N_ENVS.to_vec(),
    }
}

fn build_envs(n: usize) -> Vec<SendEnv> {
    (0..n)
        .map(|_| {
            let mut e = AtariEnv::new(AtariGame::Breakout);
            e.reset();
            SendEnv(e)
        })
        .collect()
}

fn build_obs(envs: &[SendEnv], obs_dim: usize, device: &burn::DispatchDevice) -> Tensor<B, 2> {
    let mut data = Vec::with_capacity(envs.len() * obs_dim);
    for env in envs {
        data.extend(env.obs());
    }
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([envs.len(), obs_dim])
}

fn one_iter(
    envs: &mut [SendEnv],
    mode: StepMode,
    mz_conf: &MuZeroConfig,
    agent: &ResNets<B>,
    device: &burn::DispatchDevice,
    obs_dim: usize,
) -> usize {
    let obs = build_obs(envs, obs_dim, device);
    let legal_masks: Vec<Vec<bool>> = envs.iter().map(|e| e.legal_mask()).collect();
    let results = batched_search(obs, Some(&legal_masks), mz_conf, agent, 1.0, false);
    let actions: Vec<usize> = results.iter().map(|r| r.best_action).collect();

    match mode {
        StepMode::Serial => {
            for (env, &a) in envs.iter_mut().zip(&actions) {
                let res = env.step(a);
                if res.done || res.truncated {
                    env.reset();
                }
            }
        }
        StepMode::Parallel => {
            envs.par_iter_mut()
                .zip(actions.par_iter())
                .for_each(|(env, &a)| {
                    let res = env.step(a);
                    if res.done || res.truncated {
                        env.reset();
                    }
                });
        }
    }
    actions.len()
}

fn bench_env_step(c: &mut Criterion) {
    set_atari_game(AtariGame::Breakout);
    let mut mz_conf = MuZeroConfig::default();
    mz_conf.atari_game = Some(AtariGame::Breakout);

    let device = select_device(mz_conf.inference_backend);
    let agent: ResNets<B> = mz_conf.init(&device);
    let obs_dim = AtariEnv::INFO.obs_dim();

    println!("inference_backend: {:?}", mz_conf.inference_backend);
    println!("num_simulations: {}", mz_conf.num_simulations);
    println!("network params: {}", agent.num_params());
    println!("rayon threads: {}", rayon::current_num_threads());

    let mut group = c.benchmark_group("env_step_breakout");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let sizes = parse_sizes();
    let max_n = sizes.iter().copied().max().unwrap_or(0);
    let mut pool = build_envs(max_n);

    for n in sizes {
        group.throughput(Throughput::Elements(n as u64));
        for (name, mode) in [
            ("serial", StepMode::Serial),
            ("parallel", StepMode::Parallel),
        ] {
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, &n| {
                let envs = &mut pool[..n];
                for env in envs.iter_mut() {
                    env.reset();
                }
                one_iter(envs, mode, &mz_conf, &agent, &device, obs_dim);
                b.iter(|| black_box(one_iter(envs, mode, &mz_conf, &agent, &device, obs_dim)));
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_env_step);
criterion_main!(benches);
