use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    tensor::{Int, Tensor, activation::log_softmax, backend::AutodiffBackend, cast::ToElement},
};

use crate::{
    mz_config::MuZeroConfig, networks::MuZeroNets, replay_buffer::ReplayBuffer,
    support::two_hot_batch,
};

const POLICY_LOSS_EPS: f32 = 1e-8;

pub fn train<B: AutodiffBackend, N, O>(
    mut agent: N,
    optimizer: &mut O,
    mz_conf: &MuZeroConfig,
    buffer: &mut ReplayBuffer,
    lr: f64,
    device: &B::Device,
) -> (N, Option<f32>)
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

    let mut loss = Tensor::<B, 1>::zeros([1], device);
    let mut hidden_state: Option<Tensor<B, 2>> = None;

    for step in 0..mz_conf.unroll_steps {
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
                let obs: Vec<Tensor<B, 2>> = sequence
                    .iter()
                    .map(|game| Tensor::<B, 1>::from_floats(game[0].state.as_slice(), device).unsqueeze())
                    .collect();
                let obs = Tensor::cat(obs, 0);
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
            1.0 / (mz_conf.unroll_steps as f32 - 1.0).max(1.0)
        };
        let value_loss =
            -(target_value * log_softmax(value, 1)).sum_dim(1).mean() * step_scale;
        let policy_loss = -(target_policy * (policy + POLICY_LOSS_EPS).log())
            .sum_dim(1)
            .mean()
            * step_scale;
        loss = loss + value_loss + policy_loss;

        if hidden_state.is_some() {
            let target_reward: Vec<f32> =
                sequence.iter().map(|game| game[step - 1].reward).collect();
            let target_reward = Tensor::<B, 1>::from_floats(
                two_hot_batch(&target_reward, support_size).as_slice(),
                device,
            )
            .reshape([batch, support_len]);
            let reward_loss =
                -(target_reward * log_softmax(reward, 1)).sum_dim(1).mean() * step_scale;
            loss = loss + reward_loss;
        }

        hidden_state = Some(new_hidden_state);
    }

    let loss_value = loss.clone().into_scalar().to_f32();
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &agent);
    agent = optimizer.step(lr, agent, grads);

    (agent, Some(loss_value))
}
