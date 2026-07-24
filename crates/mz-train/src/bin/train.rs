use std::mem;

use burn::module::{AutodiffModule, Module};
use burn::record::CompactRecorder;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::{Dispatch, DispatchDevice};
use mz_rs::env::Environment;

use mz_rs::agent::MlpNets;
use mz_rs::mz_config::MuZeroConfig;
use mz_rs::networks::nets_to_backend;
use mz_rs::optim::AnyOptimizer;
use mz_rs::replay_buffer::{BufferData, ReplayBuffer};
use mz_rs::search::batched_search;
use mz_rs::train::train;
use mz_rs::tui_metrics::TrainingTui;
use mz_rs::utils::{save_buffer, select_device, tau_for_step};
use mz_rs::with_env;

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

    let mut agent: MlpNets<TrainB> = mz_conf.init_agent(&train_device);
    let mut optimizer = AnyOptimizer::<TrainB, MlpNets<TrainB>>::new(&mz_conf);
    let mut inference_agent: MlpNets<InferB> =
        nets_to_backend(&agent.valid(), &mz_conf, &infer_device);

    let mut buffer = ReplayBuffer::new(&mz_conf);
    let mut tui = TrainingTui::new(&mz_conf);

    let training_steps_per_iteration = ((mz_conf.game_batch_size as f32
        / mz_conf.training_batch_size as f32
        * mz_conf.train_ratio) as i32)
        .max(1);

    let total_steps = mz_conf.training_steps / training_steps_per_iteration as usize;
    let mut training_step = 0;
    let mut env_steps = 0usize;
    let mut next_checkpoint = mz_conf.checkpoint_interval;

    with_env!(mz_conf, E => {
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
            let tau = tau_for_step(&mz_conf.temperature_schedule, training_step);
            tui.set_tau(tau);

            let obs = E::batch_state_tensor::<InferB>(&env_batch, &infer_device);
            let legal_masks: Vec<Vec<bool>> =
                env_batch.iter().map(|env| env.legal_mask()).collect();

            let results = batched_search(obs, Some(&legal_masks), &mz_conf, &inference_agent, tau);

            for (i, search_result) in results.iter().enumerate() {
                let action = match WeightedIndex::new(&search_result.distribution) {
                    Ok(dist) => dist.sample(&mut rand::rng()),
                    Err(_) => search_result.best_action,
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
                });

                game_reward_batch[i] += result.reward as f32;

                if result.truncated || result.done {
                    let length = game_batch[i].len();
                    buffer.store_game(mem::take(&mut game_batch[i]), &mz_conf);
                    env_batch[i].reset();
                    tui.game_finished(game_reward_batch[i], length);
                    game_reward_batch[i] = 0.0;
                }
            }

            // Save model + buffer
            env_steps += mz_conf.game_batch_size;
            if mz_conf.checkpoint_interval > 0 && env_steps >= next_checkpoint {
                let path = "model/".to_owned() + mz_conf.environment.as_ref();
                std::fs::create_dir_all(&path).expect("Failed to create directory");
                agent
                    .valid()
                    .save_file(format!("{path}/latest"), &CompactRecorder::new())
                    .expect("Failed to save checkpoint");
                next_checkpoint += mz_conf.checkpoint_interval;
                save_buffer(&buffer, &format!("{path}/buffer.mpk"));
            }

            // Reanalyze
            reanalyze(&mz_conf, &mut buffer, training_step, &infer_device, &inference_agent);

            // Train
            for _train_step in 0..training_steps_per_iteration {
                let loss;
                (agent, loss) = train(
                    agent,
                    &mut optimizer,
                    &mz_conf,
                    &mut buffer,
                    mz_conf.learning_rate,
                    &train_device,
                );
                if let Some(loss) = loss {
                    tui.set_loss(loss);
                }

                training_step += 1;
                // Update inference agent every n training steps
                if (training_step + 1) % mz_conf.inference_update_interval.max(1) == 0 {
                    inference_agent = nets_to_backend(&agent.valid(), &mz_conf, &infer_device);
                }
            }

            // Tui stuff
            tui.add_train_steps(training_steps_per_iteration as usize);
            tui.add_env_steps(mz_conf.game_batch_size, buffer.states.len() > mz_conf.training_batch_size);
            tui.set_buffer_states(buffer.states.len());
            tui.render(training_step + 1);
        }

        tui.close();
    });
}

fn reanalyze<InferB: Backend>(
    mz_conf: &MuZeroConfig,
    buffer: &mut ReplayBuffer,
    training_step: usize,
    infer_device: &InferB::Device,
    inference_agent: &MlpNets<InferB>,
) {
    if rand::random::<f32>() < mz_conf.reanalyze_fraction {
        let idxs = buffer.sample_reanalyze_indices(mz_conf.reanalyze_batch_size, training_step);
        let dim = mz_conf.obs_dim;
        let mut data = Vec::with_capacity(idxs.len() * dim);
        for &idx in &idxs {
            data.extend_from_slice(&buffer.states[idx].state);
        }
        let obs = Tensor::<InferB, 1>::from_floats(data.as_slice(), infer_device)
            .reshape([idxs.len(), dim]);
        let masks: Vec<Vec<bool>> = idxs
            .iter()
            .map(|&idx| buffer.states[idx].legal_mask.clone())
            .collect();
        let results = batched_search(obs, Some(&masks), mz_conf, inference_agent, 1.0);
        for (&idx, r) in idxs.iter().zip(results.iter()) {
            buffer.states[idx].policy = r.policy_target.clone();
            if !mz_conf.is_twoplayer {
                buffer.states[idx].value = r.value;
            }
            buffer.states[idx].created_step = training_step;
        }
    }
}
