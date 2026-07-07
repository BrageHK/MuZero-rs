use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use burn::module::{AutodiffModule, Module};
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::rl::Environment;
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::{Dispatch, DispatchDevice};
use crossbeam::channel::{Sender, unbounded};
use mz_rs::agent::MlpNets;
use mz_rs::env::cartpole::env::CartPoleWrapper;
use mz_rs::mz_config::{MuZeroConfig, NetworkType, TemperatureSchedule};
use mz_rs::networks::nets_to_backend;
use mz_rs::replay_buffer::{BufferData, ReplayBuffer};
use mz_rs::search::{InferenceHandles, inference_channels, inference_master, search};
use mz_rs::train::train;
use mz_rs::utils::select_device;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

fn tau_for_step(schedule: &[TemperatureSchedule], step: usize) -> f32 {
    for entry in schedule {
        match entry.step {
            Some(threshold) if step <= threshold => return entry.tau,
            None => return entry.tau,
            _ => {}
        }
    }
    schedule.last().map(|e| e.tau).unwrap_or(1.0)
}

/// One self-play worker: owns its own game and its own tree per move, talks
/// to the shared inference master thread for every NN call.
fn worker_loop<StoreB: Backend, InferB: Backend>(
    mz_conf: MuZeroConfig,
    inference: InferenceHandles<InferB>,
    game_tx: Sender<Vec<BufferData<StoreB>>>,
    store_device: StoreB::Device,
    infer_device: InferB::Device,
    global_step: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
) {
    let mut env = CartPoleWrapper::default();
    let mut game: Vec<BufferData<StoreB>> = Vec::new();

    while !stop.load(Ordering::Relaxed) {
        let s = env.state().state;
        let obs_floats = [[s[0] as f32, s[1] as f32, s[2] as f32, s[3] as f32]];
        let obs = Tensor::<InferB, 2>::from_floats(obs_floats, &infer_device);
        let obs_store = Tensor::<StoreB, 2>::from_floats(obs_floats, &store_device);

        let step = global_step.fetch_add(1, Ordering::Relaxed);
        let tau = tau_for_step(&mz_conf.temperature_schedule, step);

        let (visit_distribution, value, _action) = search(obs, &mz_conf, tau, &inference);
        let dist = WeightedIndex::new(&visit_distribution).unwrap();
        let action = dist.sample(&mut rand::rng());

        let result = env.step(action);

        let buffer_data = BufferData {
            state: obs_store,
            action,
            value,
            reward: result.reward as f32,
            policy: Tensor::<StoreB, 1>::from_floats(visit_distribution.as_slice(), &store_device)
                .unsqueeze_dim(0),
        };
        game.push(buffer_data);

        if result.truncated || result.done {
            let finished = std::mem::take(&mut game);
            if game_tx.send(finished).is_err() {
                break;
            }
            env.reset();
        }
    }
}

fn main() {
    // Dispatch picks the concrete backend at runtime from the config; autodiff
    // lives in the device, so TrainB/StoreB/InferB are all the same type.
    type TrainB = Dispatch;
    type StoreB = <TrainB as AutodiffBackend>::InnerBackend;
    type InferB = Dispatch;

    let mz_conf = MuZeroConfig::default();
    if let NetworkType::ResNet = mz_conf.network_type {
        panic!(
            "network_type: ResNet has no compatible environment yet \
             (cartpole obs is a flat vector) — use network_type: Linear"
        );
    }

    // Plain device for buffer/store tensors, autodiff-wrapped for the model.
    let device = select_device(mz_conf.training_backend);
    let train_device = DispatchDevice::autodiff(device.clone());
    let infer_device = select_device(mz_conf.inference_backend);

    let mut agent: MlpNets<TrainB> = mz_conf.init_agent(&train_device);
    let mut optimizer = AdamConfig::new().init::<TrainB, MlpNets<TrainB>>();

    let inference_agent: MlpNets<InferB> = nets_to_backend(&agent.valid(), &mz_conf, &infer_device);
    let agent_cell = Arc::new(Mutex::new(inference_agent));

    // Single master thread: one shared model, two independently-batched
    // request queues (initial_inference vs recurrent_inference).
    let channels = inference_channels::<InferB>();
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

    let (game_tx, game_rx) = unbounded::<Vec<BufferData<StoreB>>>();
    let global_step = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));

    let mut worker_handles = Vec::with_capacity(mz_conf.search_threads());
    for _ in 0..mz_conf.search_threads() {
        let inference = channels.handles.clone();
        let game_tx = game_tx.clone();
        let worker_conf = mz_conf.clone();
        let store_device = device.clone();
        let worker_infer_device = infer_device.clone();
        let global_step = Arc::clone(&global_step);
        let stop = Arc::clone(&stop);
        worker_handles.push(thread::spawn(move || {
            worker_loop::<StoreB, InferB>(
                worker_conf,
                inference,
                game_tx,
                store_device,
                worker_infer_device,
                global_step,
                stop,
            );
        }));
    }
    // Drop our copies so the channels close once every worker/master has
    // exited, instead of staying open forever.
    drop(game_tx);
    drop(channels.handles);

    let mut buffer = ReplayBuffer::<StoreB>::default();
    let mut games_played = 0usize;
    let mut total_train_steps = 0usize;
    let mut steps_since_inference_update = 0usize;
    let mut checkpoints_saved = 0usize;
    let mut best_reward = f32::NEG_INFINITY;
    std::fs::create_dir_all("model").expect("Failed to create model/ directory");

    while global_step.load(Ordering::Relaxed) < mz_conf.total_steps {
        let Ok(game) = game_rx.recv() else { break };
        games_played += 1;
        println!("Game {} finished, length {}", games_played, game.len());
        let game_reward: f32 = game.iter().map(|d| d.reward).sum();
        buffer.store_game(game);

        if game_reward > best_reward {
            best_reward = game_reward;
            agent
                .clone()
                .save_file("model/best", &CompactRecorder::new())
                .expect("Failed to save best model");
            println!("New best reward {best_reward}, saved model/best.mpk");
        }

        for _train_step in 0..mz_conf.train_steps_per_game {
            let _loss;
            (agent, _loss) = train(
                agent,
                &mut optimizer,
                &mz_conf,
                &mut buffer,
                mz_conf.learning_rate,
                &train_device,
            );
            total_train_steps += 1;
            steps_since_inference_update += 1;
        }

        if steps_since_inference_update >= mz_conf.inference_update_interval {
            let new_inference_agent: MlpNets<InferB> =
                nets_to_backend(&agent.valid(), &mz_conf, &infer_device);
            *agent_cell.lock().unwrap() = new_inference_agent;
            steps_since_inference_update = 0;
            println!(
                "Inference network updated at train step {}",
                total_train_steps
            );
        }

        let steps = global_step.load(Ordering::Relaxed);
        if steps / mz_conf.checkpoint_interval > checkpoints_saved {
            checkpoints_saved = steps / mz_conf.checkpoint_interval;
            agent
                .clone()
                .save_file("model/latest", &CompactRecorder::new())
                .expect("Failed to save checkpoint");
            println!("Checkpoint saved to model/latest.mpk at step {}", steps);
        }
    }

    stop.store(true, Ordering::Relaxed);
    for handle in worker_handles {
        let _ = handle.join();
    }
    let _ = master_handle.join();
}
