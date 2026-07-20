use std::collections::VecDeque;

use fastrand::Rng;

use crate::mz_config::MuZeroConfig;

#[derive(Clone)]
pub struct BufferData {
    pub state: Vec<f32>,
    pub action: usize,
    pub value: f32,
    pub reward: f32,
    pub policy: Vec<f32>,
    pub is_terminal: bool,
    pub is_boundary: bool,
}

pub struct ReplayBuffer {
    pub states: VecDeque<BufferData>,
    max_len: usize,
    rng: Rng,
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        ReplayBuffer {
            states: VecDeque::new(),
            max_len: 100_000,
            rng: Rng::new(),
        }
    }
}

impl ReplayBuffer {
    pub fn new(conf: &MuZeroConfig) -> Self {
        ReplayBuffer {
            states: VecDeque::with_capacity(conf.buffer_size),
            max_len: conf.buffer_size,
            rng: Rng::new(),
        }
    }
}

impl ReplayBuffer {
    pub fn store_game(&mut self, mut game: Vec<BufferData>, mz_config: &MuZeroConfig) {
        if mz_config.is_twoplayer {
            let mut last_reward = game.last().expect("Game should not be empty.").reward;
            for state in game.iter_mut().rev() {
                state.value = last_reward;
                state.reward = 0.0;
                last_reward = -last_reward;
            }
        }
        self.states.extend(game);
        while self.states.len() > self.max_len {
            self.states.pop_front();
        }
    }

    pub fn sample_games(&mut self, mz_config: &MuZeroConfig) -> Vec<Vec<BufferData>> {
        (0..mz_config.training_batch_size)
            .map(|_| self.sample_single(mz_config))
            .collect()
    }

    fn sample_single(&mut self, mz_config: &MuZeroConfig) -> Vec<BufferData> {
        if self.states.is_empty() {
            return Vec::new();
        }
        let idx = self.rng.usize(0..self.states.len());

        let mut sequence = Vec::with_capacity(mz_config.unroll_steps);
        let mut absorbing: Option<BufferData> = None;
        let uniform_policy = vec![1.0 / mz_config.action_space as f32; mz_config.action_space];

        for state_idx in idx..idx + mz_config.unroll_steps {
            if let Some(ref abs) = absorbing {
                sequence.push(abs.clone());
                continue;
            }
            if state_idx >= self.states.len() {
                let abs = BufferData {
                    value: 0.0,
                    reward: 0.0,
                    policy: uniform_policy.clone(),
                    ..sequence.last().expect("sequence has at least one state").clone()
                };
                sequence.push(abs.clone());
                absorbing = Some(abs);
                continue;
            }
            let state = &self.states[state_idx];
            let value = match mz_config.is_twoplayer {
                true => state.value,
                false => self.n_step_value(state_idx, mz_config),
            };
            sequence.push(BufferData {
                value,
                ..state.clone()
            });
            if state.is_boundary {
                absorbing = Some(BufferData {
                    value: 0.0,
                    reward: 0.0,
                    policy: uniform_policy.clone(),
                    ..state.clone()
                });
            }
        }

        sequence
    }

    fn n_step_value(
        &self,
        idx: usize,
        mz_config: &MuZeroConfig
    ) -> f32 {
        let mut value = 0.0;
        for k in 0..mz_config.n_steps {
            let curr_idx = idx + k;
            if curr_idx >= self.states.len() {
                return value;
            }
            let state = &self.states[curr_idx];
            if state.is_terminal {
                value += mz_config.discount.powi(k as i32) * state.reward;
                return value;
            }
            if state.is_boundary {
                value += mz_config.discount.powi(k as i32) * state.value;
                return value;
            }
            value += mz_config.discount.powi(k as i32) * state.reward;
        }
        let bootstrap_idx = idx + mz_config.n_steps;
        if bootstrap_idx < self.states.len() && !self.states[bootstrap_idx].is_terminal {
            value += mz_config.discount.powi(mz_config.n_steps as i32)
                * self.states[bootstrap_idx].value;
        }
        value
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn create_game(n: usize) -> Vec<BufferData> {
        (0..n)
            .map(|i| BufferData {
                state: vec![0.0; 4],
                action: 0,
                value: 0.0,
                reward: 0.0,
                policy: vec![0.25; 4],
                is_terminal: i == n - 1,
                is_boundary: i == n - 1,
            })
            .collect()
    }

    #[test]
    fn store_games() {
        let mz_config = MuZeroConfig { 
            training_batch_size: 1, 
            is_twoplayer: false, 
            ..Default::default() 
        };
        let mut buffer = ReplayBuffer::default();
        buffer.store_game(create_game(3), &mz_config);
        assert_eq!(
            buffer.sample_games(&mz_config)[0].len(),
            mz_config.unroll_steps
        );
        assert_eq!(
            buffer.sample_games(&mz_config)[0][mz_config.unroll_steps - 1].value,
            0.
        );
        assert_eq!(
            buffer.sample_games(&mz_config)[0][mz_config.unroll_steps - 1].reward,
            0.
        );
        buffer.store_game(create_game(100), &mz_config);
        for _ in 0..4 {
            assert_eq!(
                buffer.sample_games(&mz_config)[0].len(),
                mz_config.unroll_steps
            );
        }
    }

    #[test]
    fn store_1_game() {
        let mz_config = MuZeroConfig { 
            training_batch_size: 1, 
            is_twoplayer: false, 
            ..Default::default() 
        };
        let mut buffer = ReplayBuffer::default();
        buffer.store_game(create_game(1), &mz_config);
        for _ in 0..3 {
            assert_eq!(
                buffer.sample_games(&mz_config)[0].len(),
                mz_config.unroll_steps
            );
        }
        for i in 1..mz_config.unroll_steps {
            let sample = buffer.sample_games(&mz_config);
            assert_eq!(sample[0][i].value, 0.);
            assert_eq!(sample[0][i].reward, 0.);
        }
    }
}
