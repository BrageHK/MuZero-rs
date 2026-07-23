pub mod cartpole;
pub mod othello;
pub mod tictactoe;

use burn::rl::StepResult;
use burn::tensor::{Tensor, backend::Backend};

#[derive(Debug, Clone, Copy)]
pub struct EnvInfo {
    pub obs_shape: &'static [usize],
    pub action_size: usize,
    pub num_players: usize,
    pub upper_reward_bound: Option<f32>,
    pub lower_reward_bound: Option<f32>
}

impl EnvInfo {
    pub const fn obs_dim(&self) -> usize {
        let mut dim = 1;
        let mut i = 0;
        while i < self.obs_shape.len() {
            dim *= self.obs_shape[i];
            i += 1;
        }
        dim
    }
}

/// Custom implementation of Burn RL's Env trait. This trait contains fields
/// that are needed for MuZero
pub trait Environment {
    /// The type of the state.
    type State;
    /// The type of actions.
    type Action;

    /// The maximum number of step for one episode.
    const MAX_STEPS: usize;
    /// Environment info
    const INFO: EnvInfo;

    /// Returns the current state.
    fn state(&self) -> Self::State;
    /// Flat observation of the current state, length `INFO.obs_dim()`.
    fn obs(&self) -> Vec<f32>;
    /// Current state as a `[1, obs_dim]` tensor, ready for the networks.
    fn state_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 1>::from_floats(self.obs().as_slice(), device)
            .reshape([1, Self::INFO.obs_dim()])
    }
    /// States of a batch of environments as a `[batch, obs_dim]` tensor.
    fn batch_state_tensor<B: Backend>(envs: &[Self], device: &B::Device) -> Tensor<B, 2>
    where
        Self: Sized,
    {
        let dim = Self::INFO.obs_dim();
        let mut data = Vec::with_capacity(envs.len() * dim);
        for env in envs {
            data.extend(env.obs());
        }
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([envs.len(), dim])
    }
    /// Take a step in the environment given an action.
    fn step(&mut self, action: Self::Action) -> StepResult<Self::State>;
    /// Reset the environment to an initial state.
    fn reset(&mut self);
    /// Legal moves in this position
    fn legal_mask(&self) -> Vec<bool>;
    /// Get static info about the environemnt
    fn get_info(&self) -> EnvInfo {
        Self::INFO
    }
}

/// ```ignore
/// use crate::env::Environment;
/// use mz_rs::with_env;
///
/// with_env!(mz_conf, E => {
///     let mut env = E::default();
///     env.reset();
///     // generic training/search code over E
/// });
/// ```
#[macro_export]
macro_rules! with_env {
    ($mz_conf:expr, $E:ident => $body:expr) => {
        match $mz_conf.environment {
            $crate::mz_config::EnvironmentName::CartPole => {
                type $E = $crate::env::cartpole::env::CartPoleWrapper;
                $body
            }
            $crate::mz_config::EnvironmentName::TicTacToe => {
                type $E = $crate::env::tictactoe::env::TicTacToe;
                $body
            }
            $crate::mz_config::EnvironmentName::Othello => {
                type $E = $crate::env::othello::env::Othello;
                $body
            }
        }
    };
}
