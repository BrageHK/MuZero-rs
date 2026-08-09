#![cfg(feature = "ndarray")]

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;

use mz_rs::agent::MlpNets;
use mz_rs::env::Environment;
use mz_rs::env::cartpole::env::CartPoleWrapper;
use mz_rs::mz_config::{GumbelSubConfig, MuZeroConfig, SearchAlgorithm};
use mz_rs::networks::nets_to_backend;
use mz_rs::optim::AnyOptimizer;
use mz_rs::replay_buffer::{BufferData, ReplayBuffer};
use mz_rs::search::batched_search;
use mz_rs::train::train;

type TrainB = Autodiff<NdArray>;

#[test]
fn consistency_improves_without_collapsing() {
    let mz_conf = MuZeroConfig {
        search_algorithm: SearchAlgorithm::Gumbel,
        gumbel: Some(GumbelSubConfig {
            max_num_considered_actions: 16,
            c_visit: 50.0,
            c_scale: 0.1,
        }),
        num_simulations: 16,
        training_batch_size: 64,
        game_batch_size: 64,
        rayon_min_chunk_len: 8,
        consistency_coef: 2.0,
        ..Default::default()
    };

    let device = NdArrayDevice::default();
    let mut agent: MlpNets<TrainB> = mz_conf.init(&device);
    let mut optimizer = AnyOptimizer::<TrainB, MlpNets<TrainB>>::new(&mz_conf);
    let mut inference_agent: MlpNets<NdArray> = nets_to_backend(&agent.valid(), &mz_conf, &device);

    let mut buffer = ReplayBuffer::new(&mz_conf);
    let mut env_batch = vec![CartPoleWrapper::default(); mz_conf.game_batch_size];
    for env in env_batch.iter_mut() {
        env.reset();
    }
    let mut game_batch: Vec<Vec<BufferData>> = vec![Vec::new(); mz_conf.game_batch_size];
    let mut consistency: Vec<f32> = Vec::new();

    for _ in 0..80 {
        let obs = CartPoleWrapper::batch_state_tensor::<NdArray>(&env_batch, &device);
        let results = batched_search(obs, None, &mz_conf, &inference_agent, 0.0, true);

        for (i, result) in results.iter().enumerate() {
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
                legal_mask: vec![true; mz_conf.action_space],
                is_absorbing: false,
            });
            if step.done || step.truncated {
                buffer.store_game(std::mem::take(&mut game_batch[i]), &mz_conf);
                env_batch[i].reset();
            }
        }

        let metrics;
        (agent, metrics) = train(
            agent,
            &mut optimizer,
            &mz_conf,
            &mut buffer,
            None,
            None,
            mz_conf.learning_rate,
            &device,
        );
        if let Some(metrics) = metrics {
            consistency.push(metrics.consistency);
            inference_agent = nets_to_backend(&agent.valid(), &mz_conf, &device);
        }
    }

    assert!(consistency.len() >= 20, "not enough training steps ran");
    let n = consistency.len();
    let head: f32 = consistency[..n / 4].iter().sum::<f32>() / (n / 4) as f32;
    let tail: f32 = consistency[n - n / 4..].iter().sum::<f32>() / (n / 4) as f32;
    println!("consistency head={head} tail={tail} n={n}");
    println!("last 5: {:?}", &consistency[n - 5..]);

    assert!(
        tail < head,
        "consistency term did not improve: head={head} tail={tail}"
    );
    assert!(
        tail > -2.0 + 1e-3,
        "consistency saturated at the coefficient, latents likely collapsed: tail={tail}"
    );
}
