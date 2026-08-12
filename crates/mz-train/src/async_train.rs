//! Two-thread training: self-play on one thread, gradient steps on another.
//! Self-play streams finished games to the trainer over a non-blocking channel
//! and the trainer streams weight blobs back. Enabled by `async_training`.
//!
//! Both threads run free, so the realised train-steps-per-env-step is whatever
//! throughput allows rather than `train_ratio` as in the single-threaded loop.

use std::mem;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel};
use std::time::{Duration, Instant};

use burn::module::{AutodiffModule, Module};
use burn::optim::Optimizer;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::Interrupter;

use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

use crate::augment::Augmenter;
use crate::board_symmetry::BoardSymmetry;
use crate::env::Environment;
use crate::eval::{EloLadder, EvalReading};
use crate::mz_config::{MuZeroConfig, SearchAlgorithm};
use crate::networks::{MuZeroNets, nets_from_bytes, nets_to_bytes};
use crate::optim::AnyOptimizer;
use crate::replay_buffer::{BufferData, ReplayBuffer};
use crate::search::batched_search;
use crate::train::{reanalyze, reanalyze_due, train};
use crate::tui_metrics::TrainingTui;
use crate::utils::{lr_for_step, save_buffer, save_training_step, tau_for_step};

const RENDER_INTERVAL: Duration = Duration::from_millis(50);
const WARMUP_POLL: Duration = Duration::from_millis(100);

enum SelfPlayMsg {
    Game {
        data: Vec<BufferData>,
        reward: f32,
        length: usize,
    },
    EnvSteps(usize),
    Tau(f32),
    Eval(EvalReading),
}

struct WeightMsg {
    bytes: Vec<u8>,
    training_step: usize,
}

pub fn run<E, TrainB, InferB, NT, NI>(
    mz_conf: &MuZeroConfig,
    agent: NT,
    mut optimizer: AnyOptimizer<TrainB, NT>,
    mut buffer: ReplayBuffer,
    mut augmenter: Option<Augmenter>,
    mut board_sym: Option<BoardSymmetry>,
    mut tui: TrainingTui,
    train_device: TrainB::Device,
    inner_device: TrainB::Device,
    infer_device: InferB::Device,
    initial_training_step: usize,
) where
    E: Environment<Action = usize> + Default + Clone,
    TrainB: AutodiffBackend,
    InferB: Backend,
    NT: MuZeroNets<TrainB> + AutodiffModule<TrainB>,
    NT::InnerModule: MuZeroNets<TrainB::InnerBackend>,
    NI: MuZeroNets<InferB>,
{
    let (game_tx, game_rx) = channel::<SelfPlayMsg>();
    let (weight_tx, weight_rx) = channel::<WeightMsg>();

    let interrupter = tui.interrupter();
    let self_play_interrupter = interrupter.clone();
    let initial_weights = nets_to_bytes(&agent.valid());

    std::thread::scope(|scope| {
        scope.spawn(move || {
            self_play::<E, InferB, NI>(
                mz_conf,
                initial_weights,
                infer_device,
                game_tx,
                weight_rx,
                self_play_interrupter,
                initial_training_step,
            );
        });

        train_loop(
            mz_conf,
            agent,
            &mut optimizer,
            &mut buffer,
            &mut augmenter,
            &mut board_sym,
            &mut tui,
            &train_device,
            &inner_device,
            &game_rx,
            &weight_tx,
            initial_training_step,
        );

        interrupter.stop(None);
    });

    tui.close();
}

fn self_play<E, InferB, N>(
    mz_conf: &MuZeroConfig,
    initial_weights: Vec<u8>,
    infer_device: InferB::Device,
    tx: Sender<SelfPlayMsg>,
    weights: Receiver<WeightMsg>,
    interrupter: Interrupter,
    initial_training_step: usize,
) where
    E: Environment<Action = usize> + Default + Clone,
    InferB: Backend,
    N: MuZeroNets<InferB>,
{
    let net_conf = mz_conf.net_config();
    let mut agent: N = nets_from_bytes(initial_weights, &net_conf, &infer_device);
    let mut ladder = EloLadder::new(mz_conf);
    let mut training_step = initial_training_step;

    let mut game_batch: Vec<Vec<BufferData>> = vec![Vec::new(); mz_conf.game_batch_size];
    let mut game_reward_batch = vec![0.0f32; mz_conf.game_batch_size];
    let mut env_batch = vec![E::default(); mz_conf.game_batch_size];
    for env in env_batch.iter_mut() {
        env.reset();
    }

    while !interrupter.should_stop() {
        let mut latest = None;
        loop {
            match weights.try_recv() {
                Ok(msg) => latest = Some(msg),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        if let Some(msg) = latest {
            agent = nets_from_bytes(msg.bytes, &net_conf, &infer_device);
            training_step = msg.training_step;
        }

        let tau = match mz_conf.search_algorithm {
            SearchAlgorithm::Puct => {
                tau_for_step(&mz_conf.puct().temperature_schedule, training_step)
            }
            SearchAlgorithm::Gumbel => 0.0,
        };

        let obs = E::batch_state_tensor::<InferB>(&env_batch, &infer_device);
        let legal_masks: Vec<Vec<bool>> = env_batch.iter().map(|env| env.legal_mask()).collect();

        let results = batched_search(obs, Some(&legal_masks), mz_conf, &agent, tau, true);

        for (i, search_result) in results.iter().enumerate() {
            let action = match mz_conf.search_algorithm {
                SearchAlgorithm::Gumbel => search_result.best_action,
                SearchAlgorithm::Puct => match WeightedIndex::new(&search_result.distribution) {
                    Ok(dist) => dist.sample(&mut rand::rng()),
                    Err(_) => search_result.best_action,
                },
            };

            let state: Vec<f32> = env_batch[i].obs();
            let legal_mask: Vec<bool> = legal_masks[i].clone();
            let result = env_batch[i].step(action);

            game_batch[i].push(BufferData {
                state,
                action,
                value: search_result.value,
                reward: result.reward as f32,
                policy: search_result.policy_target.clone(),
                is_terminal: result.done || result.truncated,
                created_step: training_step,
                legal_mask,
                is_absorbing: false,
            });

            game_reward_batch[i] += result.reward as f32;

            if result.truncated || result.done {
                let msg = SelfPlayMsg::Game {
                    length: game_batch[i].len(),
                    data: mem::take(&mut game_batch[i]),
                    reward: game_reward_batch[i],
                };
                if tx.send(msg).is_err() {
                    return;
                }
                env_batch[i].reset();
                game_reward_batch[i] = 0.0;
            }
        }

        if tx
            .send(SelfPlayMsg::EnvSteps(mz_conf.game_batch_size))
            .is_err()
            || tx.send(SelfPlayMsg::Tau(tau)).is_err()
        {
            return;
        }

        if ladder.due(training_step) {
            let reading = ladder.run(mz_conf, &agent, &infer_device, training_step);
            if tx.send(SelfPlayMsg::Eval(reading)).is_err() {
                return;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn train_loop<TrainB, N>(
    mz_conf: &MuZeroConfig,
    mut agent: N,
    optimizer: &mut AnyOptimizer<TrainB, N>,
    buffer: &mut ReplayBuffer,
    augmenter: &mut Option<Augmenter>,
    board_sym: &mut Option<BoardSymmetry>,
    tui: &mut TrainingTui,
    train_device: &TrainB::Device,
    inner_device: &TrainB::Device,
    games: &Receiver<SelfPlayMsg>,
    weights: &Sender<WeightMsg>,
    initial_training_step: usize,
) where
    TrainB: AutodiffBackend,
    N: MuZeroNets<TrainB> + AutodiffModule<TrainB>,
    N::InnerModule: MuZeroNets<TrainB::InnerBackend>,
{
    let mut training_step = initial_training_step;
    let mut next_checkpoint = initial_training_step + mz_conf.checkpoint_interval;
    let mut last_render = Instant::now();

    while !tui.should_stop() && training_step < mz_conf.training_steps {
        if !drain(games, mz_conf, buffer, tui) {
            break;
        }

        if buffer.states.len() <= mz_conf.training_batch_size {
            match games.recv_timeout(WARMUP_POLL) {
                Ok(msg) => handle(msg, mz_conf, buffer, tui),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            render(tui, buffer, training_step, &mut last_render);
            continue;
        }

        if reanalyze_due(mz_conf) {
            reanalyze(mz_conf, buffer, training_step, inner_device, &agent.valid());
        }

        let metrics;
        (agent, metrics) = train(
            agent,
            optimizer,
            mz_conf,
            buffer,
            augmenter.as_mut(),
            board_sym.as_mut(),
            lr_for_step(
                mz_conf.learning_rate,
                mz_conf.lr_warmup_steps,
                mz_conf.lr_decay_rate,
                mz_conf.lr_decay_steps,
                training_step,
            ),
            train_device,
        );

        if let Some(metrics) = metrics {
            tui.set_loss(metrics.total);
            tui.set_consistency_loss(metrics.consistency);
            training_step += 1;
            tui.add_train_steps(1);

            if (training_step + 1) % mz_conf.inference_update_interval.max(1) == 0 {
                let msg = WeightMsg {
                    bytes: nets_to_bytes(&agent.valid()),
                    training_step,
                };
                if weights.send(msg).is_err() {
                    break;
                }
            }
        }

        if mz_conf.checkpoint_interval > 0 && training_step >= next_checkpoint {
            checkpoint(mz_conf, &agent, optimizer, buffer, training_step);
            next_checkpoint += mz_conf.checkpoint_interval;
        }

        render(tui, buffer, training_step, &mut last_render);
    }
}

fn handle(
    msg: SelfPlayMsg,
    mz_conf: &MuZeroConfig,
    buffer: &mut ReplayBuffer,
    tui: &mut TrainingTui,
) {
    match msg {
        SelfPlayMsg::Game {
            data,
            reward,
            length,
        } => {
            buffer.store_game(data, mz_conf);
            tui.game_finished(reward, length);
        }
        SelfPlayMsg::EnvSteps(n) => {
            let backprop_active = buffer.states.len() > mz_conf.training_batch_size;
            tui.add_env_steps(n, backprop_active);
        }
        SelfPlayMsg::Tau(tau) => tui.set_tau(tau),
        SelfPlayMsg::Eval(reading) => tui.set_eval(&reading),
    }
}

fn drain(
    games: &Receiver<SelfPlayMsg>,
    mz_conf: &MuZeroConfig,
    buffer: &mut ReplayBuffer,
    tui: &mut TrainingTui,
) -> bool {
    loop {
        match games.try_recv() {
            Ok(msg) => handle(msg, mz_conf, buffer, tui),
            Err(TryRecvError::Empty) => return true,
            Err(TryRecvError::Disconnected) => return false,
        }
    }
}

fn render(
    tui: &mut TrainingTui,
    buffer: &ReplayBuffer,
    training_step: usize,
    last_render: &mut Instant,
) {
    if last_render.elapsed() < RENDER_INTERVAL {
        return;
    }
    tui.set_buffer_states(buffer.states.len());
    tui.render(training_step + 1);
    *last_render = Instant::now();
}

fn checkpoint<TrainB, N>(
    mz_conf: &MuZeroConfig,
    agent: &N,
    optimizer: &AnyOptimizer<TrainB, N>,
    buffer: &ReplayBuffer,
    training_step: usize,
) where
    TrainB: AutodiffBackend,
    N: MuZeroNets<TrainB> + AutodiffModule<TrainB>,
{
    let path = mz_conf.checkpoint_dir();
    std::fs::create_dir_all(&path).expect("Failed to create directory");
    agent
        .valid()
        .save_file(format!("{path}/latest"), &CompactRecorder::new())
        .expect("Failed to save checkpoint");
    CompactRecorder::new()
        .record(optimizer.to_record(), format!("{path}/optimizer").into())
        .expect("Failed to save optimizer state");
    save_buffer(buffer, &format!("{path}/buffer.mpk"));
    save_training_step(training_step, &format!("{path}/training_step"));
}
