use burn::{
    module::AutodiffModule,  optim::{GradientsParams, Optimizer}, tensor::{Int, Tensor, activation::log_softmax, backend::{AutodiffBackend, Backend}, cast::ToElement, linalg::cosine_similarity},
};

use crate::{
    augment::Augmenter, mz_config::MuZeroConfig, networks::{MuZeroNets, scale_hidden_state},
    replay_buffer::{BufferData, ReplayBuffer}, search::batched_search, support::two_hot_batch,
};

#[derive(Clone, Copy, Debug)]
pub struct TrainMetrics {
    pub total: f32,
    pub consistency: f32,
}

pub fn train<B: AutodiffBackend, N, O>(
    mut agent: N,
    optimizer: &mut O,
    mz_conf: &MuZeroConfig,
    buffer: &mut ReplayBuffer,
    mut augmenter: Option<&mut Augmenter>,
    lr: f64,
    device: &B::Device,
) -> (N, Option<TrainMetrics>)
where
    N: MuZeroNets<B> + AutodiffModule<B>,
    O: Optimizer<N, B>,
{
    if buffer.states.len() <= mz_conf.training_batch_size {
        return (agent, None);
    }

    let sequence = buffer.sample_games(mz_conf);
    let support_size = mz_conf.support_size;
    let support_len = mz_conf.support_len();
    let batch = sequence.len();
    let obs_dim = mz_conf.obs_dim;
    let unroll_steps = mz_conf.unroll_steps;
    let use_consistency = mz_conf.consistency_coef > 0.0 && unroll_steps > 1;

    let view = |data: &BufferData, aug: &mut Option<&mut Augmenter>| -> Vec<f32> {
        let mut state = data.state.clone();
        if let Some(aug) = aug.as_deref_mut() {
            aug.apply(&mut state);
        }
        state
    };

    let masks: Vec<Tensor<B, 2>> = (0..unroll_steps)
        .map(|step| {
            let mask: Vec<f32> = sequence
                .iter()
                .map(|game| if game[step].is_absorbing { 0.0 } else { 1.0 })
                .collect();
            Tensor::<B, 1>::from_floats(mask.as_slice(), device).reshape([batch, 1])
        })
        .collect();

    let target_projection = use_consistency.then(|| {
        let mut data = Vec::with_capacity(batch * (unroll_steps - 1) * obs_dim);
        for step in 1..unroll_steps {
            for game in &sequence {
                data.extend(view(&game[step], &mut augmenter));
            }
        }
        let obs = Tensor::<B, 1>::from_floats(data.as_slice(), device)
            .reshape([batch * (unroll_steps - 1), obs_dim]);
        agent
            .project(scale_hidden_state(agent.represent(obs)))
            .detach()
    });

    let mut loss = Tensor::<B, 1>::zeros([1], device);
    let mut consistency_total = Tensor::<B, 1>::zeros([1], device);
    let mut hidden_state: Option<Tensor<B, 2>> = None;

    for step in 0..unroll_steps {
        let mask = masks[step].clone();

        let target_value: Vec<f32> = sequence.iter().map(|game| game[step].value).collect();
        let target_value = Tensor::<B, 1>::from_floats(
            two_hot_batch(&target_value, support_size).as_slice(),
            device,
        )
        .reshape([batch, support_len]);

        let target_policy: Vec<Tensor<B, 2>> = sequence
            .iter()
            .map(|game| Tensor::<B, 1>::from_floats(game[step].policy.as_slice(), device).unsqueeze())
            .collect();
        let target_policy = Tensor::cat(target_policy, 0);

        let (new_hidden_state, reward, value, policy) = match &hidden_state {
            None => {
                let mut data = Vec::with_capacity(batch * obs_dim);
                for game in &sequence {
                    data.extend(view(&game[0], &mut augmenter));
                }
                let obs =
                    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([batch, obs_dim]);
                agent.initial_inference(obs)
            }
            Some(prev_hidden_state) => {
                let actions: Vec<i32> = sequence
                    .iter()
                    .map(|game| game[step - 1].action as i32)
                    .collect();
                let actions = Tensor::<B, 1, Int>::from_data(actions.as_slice(), device);
                // Appendix G: Training, trick to scale by 0.5
                let scaled_hidden_state =
                    prev_hidden_state.clone() * 0.5 + prev_hidden_state.clone().detach() * 0.5;
                agent.recurrent_inference(scaled_hidden_state, actions, mz_conf.action_space)
            }
        };

        // Appendix G: Training
        let step_scale = if step == 0 {
            1.0
        } else {
            1.0 / (unroll_steps as f32 - 1.0).max(1.0)
        };

        let value_loss = masked_mean(
            -(target_value * log_softmax(value, 1)).sum_dim(1),
            mask.clone(),
        ) * (step_scale * mz_conf.value_coef);
        let policy_loss = masked_mean(
            -(target_policy * log_softmax(policy, 1)).sum_dim(1),
            mask.clone(),
        ) * (step_scale * mz_conf.policy_coef);
        loss = loss + value_loss + policy_loss;

        if step > 0 {
            let reward_mask = masks[step - 1].clone();
            let target_reward: Vec<f32> =
                sequence.iter().map(|game| game[step - 1].reward).collect();
            let target_reward = Tensor::<B, 1>::from_floats(
                two_hot_batch(&target_reward, support_size).as_slice(),
                device,
            )
            .reshape([batch, support_len]);
            let reward_loss = masked_mean(
                -(target_reward * log_softmax(reward, 1)).sum_dim(1),
                reward_mask,
            ) * (step_scale * mz_conf.reward_coef);
            loss = loss + reward_loss;

            if let Some(targets) = &target_projection {
                let start = (step - 1) * batch;
                let target = targets.clone().narrow(0, start, batch);
                let online =
                    agent.predict_projection(agent.project(new_hidden_state.clone()));
                let similarity = cosine_similarity(online, target, 1, None);
                let consistency_loss = -masked_mean(similarity, mask)
                    * (step_scale * mz_conf.consistency_coef);
                consistency_total = consistency_total + consistency_loss.clone();
                loss = loss + consistency_loss;
            }
        }

        hidden_state = Some(new_hidden_state);
    }

    let metrics = TrainMetrics {
        total: loss.clone().into_scalar().to_f32(),
        consistency: consistency_total.into_scalar().to_f32(),
    };
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &agent);
    agent = optimizer.step(lr, agent, grads);

    (agent, Some(metrics))
}

fn masked_mean<B: AutodiffBackend>(values: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 1> {
    let valid = mask.clone().sum().clamp_min(1.0);
    (values * mask).sum() / valid
}

pub fn reanalyze_due(mz_conf: &MuZeroConfig) -> bool {
    rand::random::<f32>() < mz_conf.reanalyze_fraction
}

pub fn reanalyze<B: Backend, N: MuZeroNets<B>>(
    mz_conf: &MuZeroConfig,
    buffer: &mut ReplayBuffer,
    training_step: usize,
    device: &B::Device,
    agent: &N,
) {
    let idxs = buffer.sample_reanalyze_indices(mz_conf.reanalyze_batch_size, training_step);
    if idxs.is_empty() {
        return;
    }
    let dim = mz_conf.obs_dim;
    let mut data = Vec::with_capacity(idxs.len() * dim);
    for &idx in &idxs {
        data.extend_from_slice(&buffer.states[idx].state);
    }
    let obs = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([idxs.len(), dim]);
    let masks: Vec<Vec<bool>> = idxs
        .iter()
        .map(|&idx| buffer.states[idx].legal_mask.clone())
        .collect();
    let results = batched_search(obs, Some(&masks), mz_conf, agent, 1.0, false);
    for (&idx, r) in idxs.iter().zip(results.iter()) {
        buffer.states[idx].policy = r.policy_target.clone();
        if !mz_conf.is_twoplayer {
            buffer.states[idx].value = r.value;
        }
        buffer.states[idx].created_step = training_step;
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::backend::{Autodiff, NdArray, ndarray::NdArrayDevice};

    use super::*;
    use crate::agent::MlpNets;
    use crate::optim::AnyOptimizer;
    use crate::replay_buffer::BufferData;

    type TestB = Autodiff<NdArray>;

    fn config(consistency_coef: f32) -> MuZeroConfig {
        MuZeroConfig {
            training_batch_size: 8,
            consistency_coef,
            is_twoplayer: false,
            network_type: crate::mz_config::NetworkType::Linear,
            obs_dim: 4,
            action_space: 4,
            ..Default::default()
        }
    }

    fn filled_buffer(conf: &MuZeroConfig) -> ReplayBuffer {
        let mut buffer = ReplayBuffer::new(conf);
        for _ in 0..4 {
            let game: Vec<BufferData> = (0..10)
                .map(|i| BufferData {
                    state: vec![0.1 * i as f32, -0.2, 0.3, 0.05],
                    action: i % conf.action_space,
                    value: 1.0,
                    reward: 1.0,
                    policy: vec![1.0 / conf.action_space as f32; conf.action_space],
                    is_terminal: i == 9,
                    created_step: 0,
                    legal_mask: vec![true; conf.action_space],
                    is_absorbing: false,
                })
                .collect();
            buffer.store_game(game, conf);
        }
        buffer
    }

    fn probe<B: burn::tensor::backend::Backend>(
        agent: &MlpNets<B>,
        device: &B::Device,
    ) -> Vec<f32> {
        let obs = Tensor::<B, 1>::from_floats([0.4, -0.2, 0.3, 0.05], device).reshape([1, 4]);
        agent
            .represent(obs)
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap()
    }

    fn run_step(conf: &MuZeroConfig) -> (TrainMetrics, Vec<f32>, Vec<f32>) {
        let device = NdArrayDevice::default();
        let agent: MlpNets<TestB> = conf.init(&device);
        let mut optimizer = AnyOptimizer::<TestB, MlpNets<TestB>>::new(conf);
        let mut buffer = filled_buffer(conf);

        let before = probe(&agent, &device);
        let (agent, metrics) = train(
            agent,
            &mut optimizer,
            conf,
            &mut buffer,
            None,
            conf.learning_rate,
            &device,
        );
        let after = probe(&agent, &device);

        (metrics.expect("a training step should have run"), before, after)
    }

    #[test]
    fn consistency_loss_contributes_and_updates_params() {
        let conf = config(2.0);
        let (metrics, before, after) = run_step(&conf);

        assert!(metrics.total.is_finite(), "total loss {} not finite", metrics.total);
        assert!(
            metrics.consistency.is_finite() && metrics.consistency != 0.0,
            "consistency term should be active, got {}",
            metrics.consistency
        );
        assert!(
            metrics.consistency.abs() <= 2.0 + 1e-3,
            "cosine is bounded, so |consistency| <= coef, got {}",
            metrics.consistency
        );
        assert!(
            before.iter().zip(&after).any(|(a, b)| (a - b).abs() > 1e-9),
            "representation did not change after an optimizer step"
        );
    }

    #[test]
    fn zero_coefficient_disables_the_term() {
        let conf = config(0.0);
        let (metrics, _, _) = run_step(&conf);

        assert_eq!(metrics.consistency, 0.0);
        assert!(metrics.total.is_finite());
    }

    #[test]
    fn absorbing_steps_are_masked_out() {
        let conf = MuZeroConfig {
            training_batch_size: 4,
            consistency_coef: 2.0,
            is_twoplayer: false,
            network_type: crate::mz_config::NetworkType::Linear,
            obs_dim: 4,
            action_space: 2,
            ..Default::default()
        };
        let device = NdArrayDevice::default();
        let agent: MlpNets<TestB> = conf.init(&device);
        let mut optimizer = AnyOptimizer::<TestB, MlpNets<TestB>>::new(&conf);

        let mut buffer = ReplayBuffer::new(&conf);
        for _ in 0..3 {
            let game: Vec<BufferData> = (0..2)
                .map(|i| BufferData {
                    state: vec![0.0; 4],
                    action: 0,
                    value: 0.0,
                    reward: 0.0,
                    policy: vec![0.5; 2],
                    is_terminal: i == 1,
                    created_step: 0,
                    legal_mask: vec![true; 2],
                    is_absorbing: false,
                })
                .collect();
            buffer.store_game(game, &conf);
        }

        let (_, metrics) = train(
            agent,
            &mut optimizer,
            &conf,
            &mut buffer,
            None,
            conf.learning_rate,
            &device,
        );
        let metrics = metrics.expect("a training step should have run");
        assert!(
            metrics.total.is_finite() && metrics.consistency.is_finite(),
            "masked-out steps must not produce NaN: {metrics:?}"
        );
    }
}
