use std::mem;

use burn::module::AutodiffModule;
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
use mz_rs::utils::{select_device, tau_for_step};
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

    let training_steps = (mz_conf.game_batch_size as f32 / mz_conf.training_batch_size as f32
        * mz_conf.train_ratio) as i32;

    with_env!(mz_conf, E => {
        let mut env_batch = vec![E::default(); mz_conf.game_batch_size];
        for env in env_batch.iter_mut() {
            env.reset();
        }

        let mut buffer = ReplayBuffer::new(&mz_conf);

        let mut game_batch: Vec<Vec<BufferData>> =
            vec![Vec::new(); mz_conf.game_batch_size];
        let mut game_len_batch = vec![0usize; mz_conf.game_batch_size];
        let mut game_reward_batch = vec![0.0f32; mz_conf.game_batch_size];

        let mut tui = TrainingTui::new(&mz_conf);

        for training_step in 0..mz_conf.total_steps {
            if tui.should_stop() {
                break;
            }
            let tau = tau_for_step(&mz_conf.temperature_schedule, training_step);

            let obs = E::batch_state_tensor::<InferB>(&env_batch, &infer_device);
            let legal_masks: Vec<Vec<bool>> =
                env_batch.iter().map(|env| env.legal_mask()).collect();

            let results = batched_search(obs, Some(&legal_masks), &mz_conf, &inference_agent, tau);

            for (i, search_result) in results.iter().enumerate() {
                game_len_batch[i] += 1;
                let dist = WeightedIndex::new(&search_result.distribution).unwrap();
                let action = dist.sample(&mut rand::rng());

                let state = env_batch[i].obs();
                let result = env_batch[i].step(action);

                game_batch[i].push(BufferData {
                    state,
                    action,
                    value: search_result.value,
                    reward: result.reward as f32,
                    policy: search_result.distribution.clone(),
                    is_terminal: result.done,
                });

                game_reward_batch[i] += result.reward as f32;

                if result.truncated || result.done {
                    buffer.store_game(mem::take(&mut game_batch[i]), &mz_conf);
                    env_batch[i].reset();
                    game_len_batch[i] = 0;
                    tui.game_finished(game_reward_batch[i]);
                    game_reward_batch[i] = 0.0;
                }
            }
            tui.add_env_steps(
                mz_conf.game_batch_size,
                buffer.states.len() > mz_conf.training_batch_size,
            );

            for _train_step in 0..training_steps {
                let _loss;
                (agent, _loss) = train(
                    agent,
                    &mut optimizer,
                    &mz_conf,
                    &mut buffer,
                    mz_conf.learning_rate,
                    &train_device,
                );
            }
            tui.add_train_steps(training_steps as usize);
            inference_agent = nets_to_backend(&agent.valid(), &mz_conf, &infer_device);

            tui.render(training_step + 1);
        }

        tui.close();
    });
}
