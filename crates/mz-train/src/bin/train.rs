use std::mem;

use burn::module::{AutodiffModule, Module};
use burn::optim::Optimizer;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::{Dispatch, DispatchDevice};
use mz_rs::env::Environment;

use mz_rs::async_train;
use mz_rs::mz_config::{MuZeroConfig, SearchAlgorithm};
use mz_rs::networks::{MuZeroNets, nets_to_backend};
use mz_rs::optim::AnyOptimizer;
use mz_rs::replay_buffer::{BufferData, ReplayBuffer};
use mz_rs::eval::EloLadder;
use mz_rs::search::batched_search;
use mz_rs::train::{reanalyze, reanalyze_due, train};
use mz_rs::tui_metrics::TrainingTui;
use mz_rs::augment::Augmenter;
use mz_rs::utils::{lr_for_step, save_buffer, select_device, tau_for_step};
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

    with_env!(mz_conf, E => {
        with_net!(mz_conf, Net => {
            run::<E, TrainB, InferB, Net<TrainB>, Net<InferB>>(
                &mz_conf,
                train_device.clone(),
                device.clone(),
                infer_device.clone(),
            );
        });
    });
}

fn run<E, TrainB, InferB, NT, NI>(
    mz_conf: &MuZeroConfig,
    train_device: TrainB::Device,
    inner_device: TrainB::Device,
    infer_device: InferB::Device,
) where
    E: Environment<Action = usize> + Default + Clone,
    TrainB: AutodiffBackend,
    InferB: Backend,
    NT: MuZeroNets<TrainB> + AutodiffModule<TrainB>,
    NT::InnerModule: MuZeroNets<TrainB::InnerBackend>,
    NI: MuZeroNets<InferB>,
{
    let mut agent: NT = mz_conf.init_agent(&train_device);
    let mut optimizer = AnyOptimizer::<TrainB, NT>::new(mz_conf);
    if let Some(ckpt) = &mz_conf.init_checkpoint {
        let opt_path = std::path::Path::new(ckpt).with_file_name("optimizer");
        match CompactRecorder::new().load(opt_path.clone(), &train_device) {
            Ok(record) => optimizer = optimizer.load_record(record),
            Err(e) => eprintln!("No optimizer state loaded from {opt_path:?}: {e}"),
        }
    }

    let mut buffer = ReplayBuffer::new(mz_conf);
    let mut augmenter = Augmenter::from_config(mz_conf);
    let mut tui = TrainingTui::new(mz_conf);

    let training_steps_per_iteration = ((mz_conf.game_batch_size as f32
        / mz_conf.training_batch_size as f32
        * mz_conf.train_ratio) as i32)
        .max(1);

    let total_steps = mz_conf.training_steps / training_steps_per_iteration as usize;
    let mut training_step = 0;
    let mut next_checkpoint = mz_conf.checkpoint_interval;

    if mz_conf.async_training {
        async_train::run::<E, TrainB, InferB, NT, NI>(
            mz_conf,
            agent,
            optimizer,
            buffer,
            augmenter,
            tui,
            train_device,
            inner_device,
            infer_device,
        );
        return;
    }

    let mut inference_agent: NI = nets_to_backend(&agent.valid(), mz_conf, &infer_device);
    let mut ladder = EloLadder::new(mz_conf);

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

        let results =
            batched_search(obs, Some(&legal_masks), mz_conf, &inference_agent, tau, true);

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
            let path = "model/".to_owned() + mz_conf.environment.as_ref();
            std::fs::create_dir_all(&path).expect("Failed to create directory");
            agent
                .valid()
                .save_file(format!("{path}/latest"), &CompactRecorder::new())
                .expect("Failed to save checkpoint");
            CompactRecorder::new()
                .record(optimizer.to_record(), format!("{path}/optimizer").into())
                .expect("Failed to save optimizer state");
            next_checkpoint += mz_conf.checkpoint_interval;
            save_buffer(&buffer, &format!("{path}/buffer.mpk"));
        }

        // Evaluate against the benchmark opponent ladder
        if ladder.due(training_step) {
            let reading = ladder.run(mz_conf, &inference_agent, &infer_device, training_step);
            tui.set_eval(&reading);
        }

        // Reanalyze
        if reanalyze_due(mz_conf) {
            reanalyze(mz_conf, &mut buffer, training_step, &infer_device, &inference_agent);
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
                lr_for_step(mz_conf.learning_rate, mz_conf.lr_warmup_steps, training_step),
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
