use std::mem;

use burn::module::{AutodiffModule, Module};
use burn::optim::Optimizer;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::{Dispatch, DispatchDevice};
use mz_rs::env::Environment;

use mz_rs::async_train;
use mz_rs::augment::Augmenter;
use mz_rs::board_symmetry::BoardSymmetry;
use mz_rs::eval::background::BackgroundLadder;
use mz_rs::mz_config::{MuZeroConfig, SearchAlgorithm};
use mz_rs::networks::{MuZeroNets, nets_to_backend};
use mz_rs::optim::AnyOptimizer;
use mz_rs::replay_buffer::{BufferData, ReplayBuffer};
use mz_rs::search::batched_search;
use mz_rs::train::{reanalyze, reanalyze_due, train};
use mz_rs::tui_metrics::TrainingTui;
use mz_rs::utils::{
    load_best_elo, load_buffer, load_env_steps, load_eval_state, load_games_played,
    load_training_step, lr_for_step, save_buffer, save_env_steps, save_games_played,
    save_training_step, select_device, tau_for_step,
};
use mz_rs::{with_env, with_net};

use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

fn main() {
    type TrainB = Dispatch;
    type InferB = Dispatch;

    let mz_conf = MuZeroConfig::default();

    // Plain device for buffer/store tensors, autodiff-wrapped for the model.
    let device = select_device(mz_conf.training_backend);
    let train_device = DispatchDevice::autodiff(device.clone());
    let infer_device = select_device(mz_conf.inference_backend);
    let eval_device = select_device(mz_conf.eval_backend);

    with_env!(mz_conf, E => {
        with_net!(mz_conf, Net => {
            run::<E, TrainB, InferB, Net<TrainB>, Net<InferB>>(
                &mz_conf,
                train_device.clone(),
                device.clone(),
                infer_device.clone(),
                eval_device.clone(),
            );
        });
    });
}

fn run<E, TrainB, InferB, NT, NI>(
    mz_conf: &MuZeroConfig,
    train_device: TrainB::Device,
    inner_device: TrainB::Device,
    infer_device: InferB::Device,
    eval_device: InferB::Device,
) where
    E: Environment<Action = usize> + Default + Clone,
    TrainB: AutodiffBackend,
    InferB: Backend,
    NT: MuZeroNets<TrainB> + AutodiffModule<TrainB>,
    NT::InnerModule: MuZeroNets<TrainB::InnerBackend>,
    NI: MuZeroNets<InferB> + 'static,
{
    let ckpt_dir = mz_conf.checkpoint_dir();
    std::fs::create_dir_all(&ckpt_dir).expect("Failed to create directory");
    std::fs::write(
        format!("{ckpt_dir}/config.yaml"),
        serde_yaml::to_string(mz_conf).expect("Failed to serialize config"),
    )
    .expect("Failed to write config snapshot");

    let mut agent: NT = mz_conf.init_agent(&train_device);
    let mut optimizer = AnyOptimizer::<TrainB, NT>::new(mz_conf);
    let mut buffer = ReplayBuffer::new(mz_conf);
    let mut training_step = 0usize;
    let mut best_elo = f32::NEG_INFINITY;
    let mut env_steps = 0usize;
    let mut games_played = 0usize;
    if mz_conf.load_from_checkpoint {
        let opt_path = format!("{ckpt_dir}/optimizer");
        match CompactRecorder::new().load(opt_path.clone().into(), &inner_device) {
            Ok(record) => optimizer = optimizer.load_record(record),
            Err(e) => panic!("Failed to load optimizer state from {opt_path}: {e}"),
        }
        buffer.states = load_buffer(&format!("{ckpt_dir}/buffer.mpk"));
        training_step = load_training_step(&format!("{ckpt_dir}/training_step"));
        best_elo = load_best_elo(&format!("{ckpt_dir}/best_elo")).unwrap_or(f32::NEG_INFINITY);
        env_steps = load_env_steps(&format!("{ckpt_dir}/env_steps")).unwrap_or(0);
        games_played = load_games_played(&format!("{ckpt_dir}/games_played")).unwrap_or(0);
    }

    let mut augmenter = Augmenter::from_config(mz_conf);
    let mut board_sym = BoardSymmetry::from_config(mz_conf);
    let mut tui = TrainingTui::new(mz_conf);
    tui.seed_counts(env_steps, games_played, training_step);

    let training_steps_per_iteration = ((mz_conf.game_batch_size as f32
        / mz_conf.training_batch_size as f32
        * mz_conf.train_ratio) as i32)
        .max(1);

    let total_steps = mz_conf.training_steps / training_steps_per_iteration as usize;
    let mut next_checkpoint = training_step + mz_conf.checkpoint_interval;

    if mz_conf.async_training {
        async_train::run::<E, TrainB, InferB, NT, NI>(
            mz_conf,
            agent,
            optimizer,
            buffer,
            augmenter,
            board_sym,
            tui,
            train_device,
            inner_device,
            infer_device,
            eval_device,
            training_step,
            best_elo,
        );
        return;
    }

    let mut inference_agent: NI = nets_to_backend(&agent.valid(), mz_conf, &infer_device);
    let initial_rung = mz_conf
        .load_from_checkpoint
        .then(|| load_eval_state(&format!("{ckpt_dir}/eval_state")))
        .flatten()
        .map(|(rung, _, _)| rung);
    let mut ladder = BackgroundLadder::<InferB>::new(mz_conf, eval_device, best_elo, initial_rung);

    let mut game_batch: Vec<Vec<BufferData>> = vec![Vec::new(); mz_conf.game_batch_size];
    let mut game_reward_batch = vec![0.0f32; mz_conf.game_batch_size];
    let mut env_batch = vec![E::default(); mz_conf.game_batch_size];
    for env in env_batch.iter_mut() {
        env.reset();
    }

    for _step in 0..total_steps {
        if tui.should_stop() {
            break;
        }
        let tau = match mz_conf.search_algorithm {
            SearchAlgorithm::Puct => {
                tau_for_step(&mz_conf.puct().temperature_schedule, training_step)
            }
            SearchAlgorithm::Gumbel => 0.0,
        };

        let obs = E::batch_state_tensor::<InferB>(&env_batch, &infer_device);
        let legal_masks: Vec<Vec<bool>> = env_batch.iter().map(|env| env.legal_mask()).collect();

        let results = batched_search(
            obs,
            Some(&legal_masks),
            mz_conf,
            &inference_agent,
            tau,
            true,
        );

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
                let length = game_batch[i].len();
                buffer.store_game(mem::take(&mut game_batch[i]), mz_conf);
                env_batch[i].reset();
                tui.game_finished(game_reward_batch[i], length);
                game_reward_batch[i] = 0.0;
            }
        }

        // Save model + buffer
        if mz_conf.checkpoint_interval > 0 && training_step >= next_checkpoint {
            std::fs::create_dir_all(&ckpt_dir).expect("Failed to create directory");
            agent
                .valid()
                .save_file(format!("{ckpt_dir}/latest"), &CompactRecorder::new())
                .expect("Failed to save checkpoint");
            CompactRecorder::new()
                .record(
                    optimizer.to_record(),
                    format!("{ckpt_dir}/optimizer").into(),
                )
                .expect("Failed to save optimizer state");
            next_checkpoint += mz_conf.checkpoint_interval;
            save_buffer(&buffer, &format!("{ckpt_dir}/buffer.mpk"));
            save_training_step(training_step, &format!("{ckpt_dir}/training_step"));
            save_env_steps(tui.env_steps(), &format!("{ckpt_dir}/env_steps"));
            save_games_played(tui.games_finished(), &format!("{ckpt_dir}/games_played"));
        }

        // Evaluate against the benchmark opponent ladder
        if ladder.due(training_step) {
            ladder.spawn(mz_conf, &inference_agent, training_step);
        }
        if let Some(reading) = ladder.poll() {
            tui.set_eval(&reading);
        }

        // Reanalyze
        if reanalyze_due(mz_conf) {
            reanalyze(
                mz_conf,
                &mut buffer,
                training_step,
                &infer_device,
                &inference_agent,
            );
        }

        // Train
        for _train_step in 0..training_steps_per_iteration {
            let metrics;
            (agent, metrics) = train(
                agent,
                &mut optimizer,
                mz_conf,
                &mut buffer,
                augmenter.as_mut(),
                board_sym.as_mut(),
                lr_for_step(
                    mz_conf.learning_rate,
                    mz_conf.lr_warmup_steps,
                    mz_conf.lr_decay_rate,
                    mz_conf.lr_decay_steps,
                    training_step,
                ),
                &train_device,
            );
            if let Some(metrics) = metrics {
                tui.set_loss(metrics.total);
                tui.set_consistency_loss(metrics.consistency);
            }

            if metrics.is_some() {
                training_step += 1;
                // Update inference agent every n training steps
                if (training_step + 1) % mz_conf.inference_update_interval.max(1) == 0 {
                    inference_agent = nets_to_backend(&agent.valid(), mz_conf, &infer_device);
                }
                tui.add_train_steps(1);
            }
        }

        // Tui stuff
        tui.add_env_steps(
            mz_conf.game_batch_size,
            buffer.states.len() > mz_conf.training_batch_size,
        );
        tui.set_tau(tau);
        tui.set_buffer_states(buffer.states.len());
        tui.render(training_step + 1);
    }

    tui.close();
}
