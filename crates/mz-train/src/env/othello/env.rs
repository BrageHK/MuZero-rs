//  ░░░░░░░░░░░░░░░░░░░░░░░░░
//  ░  CLAUDE SEAL OF SLOP  ░
//  ░       ~ 卂丨 ~        ░
//  ░   grade: B+ (mild)    ░
//  ░  "shipped it anyway"  ░
//  ░░░░░░░░░░░░░░░░░░░░░░░░░

use burn::rl::StepResult;

pub use mz_core::othello::{Othello, OthelloState, PASS, moves};

use crate::env::{EnvInfo, Environment};

impl Environment for Othello {
    type State = OthelloState;
    type Action = usize;

    const MAX_STEPS: usize = mz_core::othello::MAX_STEPS;

    fn state(&self) -> Self::State {
        Othello::state(self)
    }

    fn obs(&self) -> Vec<f32> {
        Othello::obs(self)
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
        Othello::reset(self);
    }

    const INFO: EnvInfo = EnvInfo {
        obs_shape: &[1, 8, 8],
        action_size: mz_core::othello::ACTION_SIZE,
        num_players: 2,
        lower_reward_bound: Some(0.0),
        upper_reward_bound: Some(1.0),
    };

    fn legal_mask(&self) -> Vec<bool> {
        Othello::legal_mask(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference move generator to cross-check the bitboard fill.
    fn naive_moves(own: u64, opp: u64) -> u64 {
        let mut result = 0u64;
        for square in 0..64i32 {
            let (row, col) = (square / 8, square % 8);
            if (own | opp) & (1 << square) != 0 {
                continue;
            }
            'dirs: for (dr, dc) in [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ] {
                let (mut r, mut c) = (row + dr, col + dc);
                let mut seen_opp = false;
                while (0..8).contains(&r) && (0..8).contains(&c) {
                    let bit = 1u64 << (r * 8 + c);
                    if opp & bit != 0 {
                        seen_opp = true;
                    } else if own & bit != 0 {
                        if seen_opp {
                            result |= 1 << square;
                            break 'dirs;
                        }
                        break;
                    } else {
                        break;
                    }
                    r += dr;
                    c += dc;
                }
            }
        }
        result
    }

    #[test]
    fn initial_position() {
        let env = Othello::new();
        assert_eq!(env.counts(), (2, 2));
        assert!(env.black_to_move());
        // Black's four opening moves: d3, c4, f5, e6.
        let expected = (1u64 << 19) | (1 << 26) | (1 << 37) | (1 << 44);
        assert_eq!(env.legal_moves_mask(), expected);
        assert!(!env.is_legal(PASS));
    }

    #[test]
    fn opening_move_flips_one_stone() {
        let mut env = Othello::new();
        let r = env.step(19); // black d3 flips white d4 (bit 27)
        assert!(!r.done);
        assert_eq!(r.reward, 0.0);
        assert_eq!(env.counts(), (4, 1));
        assert!(!env.black_to_move());
    }

    #[test]
    fn bitboard_moves_match_naive_over_random_games() {
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..200 {
            let mut env = Othello::new();
            loop {
                let state = Environment::state(&env);
                assert_eq!(
                    env.legal_moves_mask(),
                    naive_moves(state.own, state.opp),
                    "mismatch at\n{env}"
                );
                let actions: Vec<usize> = env.legal_actions().collect();
                if actions.is_empty() {
                    break;
                }
                let action = actions[rng.usize(..actions.len())];
                if env.step(action).done {
                    break;
                }
            }
        }
    }

    #[test]
    fn random_games_terminate_with_consistent_outcome() {
        let mut rng = fastrand::Rng::with_seed(42);
        for _ in 0..100 {
            let mut env = Othello::new();
            let mut steps = 0;
            loop {
                let actions: Vec<usize> = env.legal_actions().collect();
                let mover_was_black = env.black_to_move();
                let r = env.step(actions[rng.usize(..actions.len())]);
                steps += 1;
                assert!(steps <= <Othello as Environment>::MAX_STEPS);
                if r.done {
                    let (black, white) = env.counts();
                    let expected = match black.cmp(&white) {
                        std::cmp::Ordering::Greater if mover_was_black => 1.0,
                        std::cmp::Ordering::Greater => -1.0,
                        std::cmp::Ordering::Less if mover_was_black => -1.0,
                        std::cmp::Ordering::Less => 1.0,
                        std::cmp::Ordering::Equal => 0.0,
                    };
                    assert_eq!(r.reward, expected);
                    assert!(env.is_over());
                    break;
                }
                assert_eq!(r.reward, 0.0);
            }
        }
    }

    #[test]
    fn pass_is_only_legal_when_stuck() {
        // Row 1 holds black e1/h1 around white f1/g1: the white run is capped by
        // black on both ends, so its entry squares are occupied and black has no
        // placement anywhere. White can still play d1 to flip black e1.
        let env = Othello::from_parts(
            (1 << 4) | (1 << 7), // black e1, h1
            (1 << 5) | (1 << 6), // white f1, g1
            true,
        );
        // Black has no legal placement (no own stone caps any run) => must pass.
        assert_eq!(env.legal_moves_mask(), 0);
        assert!(!env.is_over()); // white can still move
        assert!(env.is_legal(PASS));
        assert_eq!(env.legal_actions().collect::<Vec<_>>(), vec![PASS]);

        let mut env = env;
        let r = env.step(PASS);
        assert!(!r.done);
        assert!(!env.black_to_move());
        assert!(!env.is_legal(PASS)); // white has moves, may not pass
    }

    #[test]
    fn state_is_mover_relative() {
        let mut env = Othello::new();
        env.step(19);
        let s = Environment::state(&env); // white to move
        assert_eq!(s.own.count_ones(), 1);
        assert_eq!(s.opp.count_ones(), 4);
        let obs = s.to_obs();
        assert_eq!(obs[36], 1.0); // white e5
        assert_eq!(obs[19], -1.0); // black d3
        assert_eq!(obs[0], 0.0);
    }

    #[test]
    fn reset_restores_opening() {
        let mut env = Othello::new();
        env.step(19);
        env.step(18);
        Environment::reset(&mut env);
        assert_eq!(env.counts(), (2, 2));
        assert!(env.black_to_move());
    }
}
