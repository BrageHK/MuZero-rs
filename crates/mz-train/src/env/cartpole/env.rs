use burn::rl::StepResult;
use gym_rs::{
    core::Env,
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
};

use crate::env::{EnvInfo, Environment};

#[derive(Clone)]
pub struct Action {
    pub action: usize,
}

#[derive(Clone)]
pub struct CartPoleState {
    pub state: [f64; 4],
}

impl From<CartPoleObservation> for CartPoleState {
    fn from(observation: CartPoleObservation) -> Self {
        let vec = Vec::<f64>::from(observation);
        Self {
            state: [vec[0], vec[1], vec[2], vec[3]],
        }
    }
}

#[derive(Clone)]
pub struct CartPoleWrapper {
    gym_env: CartPoleEnv,
    step_index: usize,
}

impl Default for CartPoleWrapper {
    fn default() -> Self {
        Self::new(gym_rs::utils::renderer::RenderMode::None)
    }
}

impl CartPoleWrapper {
    pub fn new(render_mode: gym_rs::utils::renderer::RenderMode) -> Self {
        Self {
            gym_env: CartPoleEnv::new(render_mode),
            step_index: 0,
        }
    }

    /// Frames collected since last reset requires RgbArray.
    pub fn frames(&mut self) -> Vec<gym_rs::utils::renderer::RenderFrame> {
        match self
            .gym_env
            .render(gym_rs::utils::renderer::RenderMode::RgbArray)
        {
            gym_rs::utils::renderer::Renders::RgbArray(frames) => frames,
            _ => Vec::new(),
        }
    }
}

impl Environment for CartPoleWrapper {
    type State = CartPoleState;
    type Action = usize;

    const MAX_STEPS: usize = 500;

    fn state(&self) -> Self::State {
        CartPoleState::from(self.gym_env.state)
    }

    fn obs(&self) -> Vec<f32> {
        self.state().state.iter().map(|&x| x as f32).collect()
    }

    fn step(&mut self, action: usize) -> StepResult<Self::State> {
        let action_reward = self.gym_env.step(action);
        self.step_index += 1;
        StepResult {
            next_state: CartPoleState::from(action_reward.observation),
            reward: action_reward.reward.into_inner(),
            done: action_reward.done,
            truncated: self.step_index >= Self::MAX_STEPS,
        }
    }

    fn reset(&mut self) {
        self.gym_env.reset(None, false, None);
        self.step_index = 0;
    }

    const INFO: EnvInfo = EnvInfo {
        obs_shape: &[4],
        action_size: 2,
        num_players: 1,
    };

    fn legal_mask(&self) -> Vec<bool> {
        vec![true; Self::INFO.action_size]
    }
}
