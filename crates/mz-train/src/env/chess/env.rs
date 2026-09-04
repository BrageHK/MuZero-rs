use burn::rl::StepResult;

pub use mz_core::chess::{ACTION_SIZE, Chess, ChessState};

use crate::env::{EnvInfo, Environment};

impl Environment for Chess {
    type State = ChessState;
    type Action = usize;

    const MAX_STEPS: usize = mz_core::chess::MAX_STEPS;

    fn state(&self) -> Self::State {
        Chess::state(self)
    }

    fn obs(&self) -> Vec<f32> {
        Chess::obs(self)
    }

    /// Reward is from the perspective of the player taking the action, granted only
    /// on the terminal step: +1 win, -1 loss, 0 draw. Illegal moves end the game at -1.
    fn step(&mut self, action: usize) -> StepResult<Self::State> {
        let outcome = self.apply(action);
        StepResult {
            next_state: outcome.state,
            reward: outcome.reward,
            done: outcome.done,
            truncated: false,
        }
    }

    fn reset(&mut self) {
        Chess::reset(self);
    }

    const INFO: EnvInfo = EnvInfo {
        obs_shape: &[mz_core::chess::TOTAL_PLANES, 8, 8],
        action_size: ACTION_SIZE,
        num_players: 2,
        lower_reward_bound: Some(0.0),
        upper_reward_bound: Some(1.0),
    };

    fn legal_mask(&self) -> Vec<bool> {
        Chess::legal_mask(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_position_reports_20_legal_moves() {
        let env = Chess::new();
        assert_eq!(env.legal_mask().iter().filter(|&&l| l).count(), 20);
    }

    #[test]
    fn obs_matches_environment_obs_dim() {
        let env = Chess::default();
        assert_eq!(
            Environment::obs(&env).len(),
            <Chess as Environment>::INFO.obs_dim()
        );
    }

    #[test]
    fn random_games_stay_consistent_up_to_max_steps() {
        // No hard bound guarantees termination within any fixed ply count (the
        // 50-move rule alone permits sequences far longer than MAX_STEPS), so
        // this only checks the reward convention holds; running out of steps
        // without `done` is exactly what the self-play truncation is for.
        let mut rng = fastrand::Rng::with_seed(3);
        for _ in 0..20 {
            let mut env = Chess::new();
            for _ in 0..<Chess as Environment>::MAX_STEPS {
                let legal: Vec<usize> = env
                    .legal_mask()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &l)| l)
                    .map(|(a, _)| a)
                    .collect();
                assert!(!legal.is_empty(), "no legal move but game not over");
                let action = legal[rng.usize(..legal.len())];
                let r = env.step(action);
                if r.done {
                    assert!(
                        r.reward == 1.0 || r.reward == 0.0,
                        "unexpected reward {}",
                        r.reward
                    );
                    break;
                }
                assert_eq!(r.reward, 0.0);
            }
        }
    }
}
