#![cfg(feature = "ndarray")]

use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::AutodiffModule;

use mz_rs::agent::MlpNets;
use mz_rs::env::Environment;
use mz_rs::env::cartpole::env::CartPoleWrapper;
use mz_rs::mz_config::{
    GumbelSubConfig, MuZeroConfig, NetworkType, PuctSubConfig, SearchAlgorithm, TemperatureSchedule,
};
use mz_rs::networks::nets_to_backend;
use mz_rs::optim::AnyOptimizer;
use mz_rs::replay_buffer::{BufferData, ReplayBuffer};
use mz_rs::search::batched_search;
use mz_rs::train::train;

type TrainB = Autodiff<NdArray>;

fn config(algorithm: SearchAlgorithm) -> MuZeroConfig {
    MuZeroConfig {
        search_algorithm: algorithm,
        puct: Some(PuctSubConfig {
            dirichlet_alpha: 0.25,
            root_exploration_fraction: 0.25,
            temperature_schedule: vec![TemperatureSchedule {
                step: None,
                tau: 1.0,
            }],
        }),
        gumbel: Some(GumbelSubConfig {
            max_num_considered_actions: 16,
            c_visit: 50.0,
            c_scale: 0.1,
        }),
        num_simulations: 16,
        support_size: 50,
        training_batch_size: 64,
        game_batch_size: 64,
        rayon_min_chunk_len: 8,
        network_type: NetworkType::Linear,
        obs_dim: CartPoleWrapper::INFO.obs_dim(),
        action_space: CartPoleWrapper::INFO.action_size,
        is_twoplayer: false,
        ..Default::default()
    }
}

fn run_loop(mz_conf: &MuZeroConfig, iterations: usize) -> (Vec<f32>, Vec<usize>) {
    let device = NdArrayDevice::default();
    let mut agent: MlpNets<TrainB> = mz_conf.init(&device);
    let mut optimizer = AnyOptimizer::<TrainB, MlpNets<TrainB>>::new(mz_conf);
    let mut inference_agent: MlpNets<NdArray> = nets_to_backend(&agent.valid(), mz_conf, &device);

    let mut buffer = ReplayBuffer::new(mz_conf);
    let mut env_batch = vec![CartPoleWrapper::default(); mz_conf.game_batch_size];
    for env in env_batch.iter_mut() {
        env.reset();
    }
    let mut game_batch: Vec<Vec<BufferData>> = vec![Vec::new(); mz_conf.game_batch_size];

    let mut losses = Vec::new();
    let mut episode_lengths = Vec::new();

    for _ in 0..iterations {
        let obs = CartPoleWrapper::batch_state_tensor::<NdArray>(&env_batch, &device);
        let legal_masks: Vec<Vec<bool>> = env_batch.iter().map(|env| env.legal_mask()).collect();
        let results = batched_search(
            obs,
            Some(&legal_masks),
            mz_conf,
            &inference_agent,
            0.0,
            true,
        );

        assert_eq!(results.len(), mz_conf.game_batch_size);
        for (i, result) in results.iter().enumerate() {
            assert!(
                legal_masks[i][result.best_action],
                "search picked illegal action {}",
                result.best_action
            );
            let target_sum: f32 = result.policy_target.iter().sum();
            assert!(
                (target_sum - 1.0).abs() < 1e-3,
                "policy target sums to {target_sum}"
            );
            assert!(
                result.value.is_finite(),
                "root value {} not finite",
                result.value
            );

            let state = env_batch[i].obs();
            let step = env_batch[i].step(result.best_action);
            game_batch[i].push(BufferData {
                state,
                action: result.best_action,
                value: result.value,
                reward: step.reward as f32,
                policy: result.policy_target.clone(),
                is_terminal: step.done || step.truncated,
                created_step: 0,
                legal_mask: legal_masks[i].clone(),
                is_absorbing: false,
            });

            if step.done || step.truncated {
                episode_lengths.push(game_batch[i].len());
                buffer.store_game(std::mem::take(&mut game_batch[i]), mz_conf);
                env_batch[i].reset();
            }
        }

        let metrics;
        (agent, metrics) = train(
            agent,
            &mut optimizer,
            mz_conf,
            &mut buffer,
            None,
            None,
            mz_conf.learning_rate,
            &device,
        );
        if let Some(metrics) = metrics {
            let loss = metrics.total;
            assert!(loss.is_finite(), "loss {loss} not finite");
            losses.push(loss);
            inference_agent = nets_to_backend(&agent.valid(), mz_conf, &device);
        }
    }

    (losses, episode_lengths)
}

#[test]
fn gumbel_selfplay_and_train_is_stable() {
    let mz_conf = config(SearchAlgorithm::Gumbel);
    let (losses, episode_lengths) = run_loop(&mz_conf, 60);

    assert!(!losses.is_empty(), "no training step ran");
    assert!(!episode_lengths.is_empty(), "no episode finished");

    let head: f32 = losses[..losses.len() / 4].iter().sum::<f32>() / (losses.len() / 4) as f32;
    let tail: f32 = losses[losses.len() * 3 / 4..].iter().sum::<f32>()
        / (losses.len() - losses.len() * 3 / 4) as f32;
    assert!(tail < head, "loss did not fall: {head} -> {tail}");
}

#[test]
fn puct_selfplay_and_train_is_stable() {
    let mz_conf = config(SearchAlgorithm::Puct);
    let (losses, episode_lengths) = run_loop(&mz_conf, 60);

    assert!(!losses.is_empty(), "no training step ran");
    assert!(!episode_lengths.is_empty(), "no episode finished");
}
