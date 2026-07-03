use burn::{Tensor, tensor::backend::Backend};
use fastrand::Rng;

use crate::mz_config::MuZeroConfig;

#[derive(Clone)]
pub struct BufferData<B: Backend> {
    pub state: Tensor<B, 2>,
    pub action: usize,
    pub value: f32,
    pub reward: f32,
    pub policy: Tensor<B, 2>,
}

pub struct ReplayBuffer<B: Backend> {
    games: Vec<Vec<BufferData<B>>>,
    cumulative_lengths: Vec<usize>,
    pub total_positions: usize,
    rng: Rng,
}

impl<B: Backend> Default for ReplayBuffer<B> {
    fn default() -> Self {
        ReplayBuffer {
            games: Vec::new(),
            cumulative_lengths: Vec::new(),
            total_positions: 0,
            rng: Rng::new(),
        }
    }
}

impl<B: Backend> ReplayBuffer<B> {
    pub fn store_game(&mut self, game: Vec<BufferData<B>>) {
        self.total_positions += game.len();
        self.cumulative_lengths.push(self.total_positions);
        self.games.push(game);
    }

    pub fn sample_games(
        &mut self,
        mz_config: &MuZeroConfig
    ) -> Vec<Vec<BufferData<B>>> {
        (0..mz_config.batch_size)
            .map(|_| self.sample_single(mz_config))
            .collect()
    }

    fn sample_single(&mut self, mz_config: &MuZeroConfig) -> Vec<BufferData<B>> {
        let flat_idx = self.rng.usize(0..self.total_positions);
        let game_idx = self.cumulative_lengths.partition_point(|&cum| cum <= flat_idx);
        let game_start = if game_idx == 0 { 0 } else { self.cumulative_lengths[game_idx - 1] };
        let pos = flat_idx - game_start;
        let game = &self.games[game_idx];

        let mut sequence = Vec::with_capacity(mz_config.unroll_steps);
        let mut absorbing: Option<BufferData<B>> = None;

        for i in 0..mz_config.unroll_steps {
            if let Some(ref abs) = absorbing {
                sequence.push(abs.clone());
                continue;
            }
            let idx = pos + i;
            if idx >= game.len() {
                let last = sequence.last().unwrap();
                let abs = BufferData {
                    state: last.state.clone(),
                    action: last.action,
                    value: 0.0,
                    reward: 0.0,
                    policy: last.policy.clone(),
                };
                absorbing = Some(abs.clone());
                sequence.push(abs);
            } else {
                let value = n_step_value(game, idx, mz_config.n_steps, mz_config.discount);
                sequence.push(BufferData {
                    state: game[idx].state.clone(),
                    action: game[idx].action,
                    value,
                    reward: game[idx].reward,
                    policy: game[idx].policy.clone(),
                });
            }
        }

        sequence
    }
}

fn n_step_value<B: Backend>(game: &[BufferData<B>], idx: usize, n_steps: usize, discount: f32) -> f32 {
    let mut value = 0.0;
    for k in 0..n_steps {
        let ridx = idx + k;
        if ridx >= game.len() {
            break;
        }
        value += discount.powi(k as i32) * game[ridx].reward;
    }
    let bootstrap_idx = idx + n_steps;
    if bootstrap_idx < game.len() {
        value += discount.powi(n_steps as i32) * game[bootstrap_idx].value;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::Wgpu, tensor::Shape};
    type B = Wgpu<f32, i32>;

    fn create_game<B: Backend>(n: usize, device: &B::Device) -> Vec<BufferData<B>> {
        (0..n)
            .map(|_| BufferData {
                state: Tensor::<B, 2>::random(
                    Shape::new([1, 4]),
                    burn::tensor::Distribution::Uniform(-1., 1.),
                    device,
                ),
                action: 0,
                value: 0.0,
                reward: 0.0,
                policy: Tensor::<B, 2>::random(
                    Shape::new([1, 4]),
                    burn::tensor::Distribution::Uniform(0., 1.),
                    device,
                ),
            })
            .collect()
    }

    #[test]
    fn store_games() {
        let mut mz_config = MuZeroConfig::default();
        mz_config.batch_size = 1;
        let device = Default::default();
        let mut buffer = ReplayBuffer::<B>::default();
        // Game shorter than n_steps: last step always absorbing
        buffer.store_game(create_game::<B>(9, &device));
        assert_eq!(buffer.sample_games(&mz_config)[0].len(), mz_config.n_steps);
        assert_eq!(buffer.sample_games(&mz_config)[0][mz_config.n_steps - 1].value, 0.);
        assert_eq!(buffer.sample_games(&mz_config)[0][mz_config.n_steps - 1].reward, 0.);
        buffer.store_game(create_game::<B>(100, &device));
        for _ in 0..4 {
            assert_eq!(buffer.sample_games(&mz_config)[0].len(), mz_config.n_steps);
        }
    }

    #[test]
    fn store_1_game() {
        let mut mz_config = MuZeroConfig::default();
        mz_config.batch_size = 1;
        let device = Default::default();
        let mut buffer = ReplayBuffer::<B>::default();
        buffer.store_game(create_game::<B>(1, &device));
        for _ in 0..3 {
            assert_eq!(buffer.sample_games(&mz_config)[0].len(), mz_config.n_steps);
        }
        for i in 1..mz_config.n_steps {
            let sample = buffer.sample_games(&mz_config);
            assert_eq!(sample[0][i].value, 0.);
            assert_eq!(sample[0][i].reward, 0.);
        }
    }
}
