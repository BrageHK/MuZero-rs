use burn::{
    optim::{Adam, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Int, Tensor, backend::AutodiffBackend},
};

use crate::{agent::MuZeroAgent, mz_config::MuZeroConfig, replay_buffer::ReplayBuffer};

const POLICY_LOSS_EPS: f32 = 1e-8;

pub fn train<B: AutodiffBackend>(
    mut agent: MuZeroAgent<B>,
    optimizer: &mut OptimizerAdaptor<Adam, MuZeroAgent<B>, B>,
    mz_conf: &MuZeroConfig,
    buffer: &mut ReplayBuffer<B>,
    lr: f64,
    device: &B::Device,
) -> MuZeroAgent<B> {
    if buffer.total_positions <= mz_conf.batch_size {
        return agent;
    }

    let sequence = buffer.sample_games(mz_conf);

    let mut loss = Tensor::<B, 1>::zeros([1], device);
    let mut hidden_state: Option<Tensor<B, 2>> = None;

    for step in 0..mz_conf.unroll_steps {
        let target_value: Vec<f32> = sequence.iter().map(|game| game[step].value).collect();
        let target_value =
            Tensor::<B, 1>::from_floats(target_value.as_slice(), device).unsqueeze_dim(1);

        let target_policy: Vec<Tensor<B, 2>> =
            sequence.iter().map(|game| game[step].policy.clone()).collect();
        let target_policy = Tensor::cat(target_policy, 0);

        let (new_hidden_state, reward, value, policy) = match &hidden_state {
            None => {
                let obs: Vec<Tensor<B, 2>> =
                    sequence.iter().map(|game| game[0].state.clone()).collect();
                let obs = Tensor::cat(obs, 0);
                agent.initial_forward(obs)
            }
            Some(prev_hidden_state) => {
                let actions: Vec<i32> = sequence
                    .iter()
                    .map(|game| game[step - 1].action as i32)
                    .collect();
                let actions = Tensor::<B, 1, Int>::from_data(actions.as_slice(), device);
                agent.recurrent_forward(prev_hidden_state.clone(), actions, mz_conf.action_space)
            }
        };

        let value_loss = (value - target_value).powf_scalar(2.0).mean();
        let policy_loss = -(target_policy * (policy + POLICY_LOSS_EPS).log())
            .sum_dim(1)
            .mean();
        loss = loss + value_loss + policy_loss;

        if hidden_state.is_some() {
            let target_reward: Vec<f32> = sequence
                .iter()
                .map(|game| game[step - 1].reward)
                .collect();
            let target_reward =
                Tensor::<B, 1>::from_floats(target_reward.as_slice(), device).unsqueeze_dim(1);
            let reward_loss = (reward - target_reward).powf_scalar(2.0).mean();
            loss = loss + reward_loss;
        }

        hidden_state = Some(new_hidden_state);
    }

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &agent);
    agent = optimizer.step(lr, agent, grads);

    agent
}