//  ░░░░░░░░░░░░░░░░░░░░░░░░░
//  ░  CLAUDE SEAL OF SLOP  ░
//  ░       ~ 卂丨 ~        ░
//  ░   grade: B+ (mild)    ░
//  ░  "shipped it anyway"  ░
//  ░░░░░░░░░░░░░░░░░░░░░░░░░

use burn::rl::StepResult;

use crate::env::{EnvInfo, Environment};

/// Board cells are bits 0..64, row-major: bit = row * 8 + col, a1 = bit 0.
/// Action 64 is "pass", legal only when the mover has no placement.
pub const PASS: usize = 64;

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = 0x8080_8080_8080_8080;

/// The 8 ray directions as (shift, post-shift mask). Positive shifts go left,
/// negative go right; the mask kills wrap-around across the A/H files.
const DIRS: [(i8, u64); 8] = [
    (1, !FILE_A),  // east
    (-1, !FILE_H), // west
    (8, !0),       // south
    (-8, !0),      // north
    (9, !FILE_A),  // south-east
    (7, !FILE_H),  // south-west
    (-7, !FILE_A), // north-east
    (-9, !FILE_H), // north-west
];

#[inline(always)]
const fn shift(x: u64, dir: (i8, u64)) -> u64 {
    let s = dir.0;
    let moved = if s > 0 { x << s } else { x >> -s };
    moved & dir.1
}

/// Bitmask of legal placements for `own` against `opp` (Dumb7Fill ray walk).
#[inline]
pub fn moves(own: u64, opp: u64) -> u64 {
    let empty = !(own | opp);
    let mut result = 0;
    let mut d = 0;
    while d < 8 {
        let dir = DIRS[d];
        // Extend a run of opponent stones away from own stones; a legal move
        // is the empty square one step past the run.
        let mut run = shift(own, dir) & opp;
        run |= shift(run, dir) & opp;
        run |= shift(run, dir) & opp;
        run |= shift(run, dir) & opp;
        run |= shift(run, dir) & opp;
        run |= shift(run, dir) & opp;
        result |= shift(run, dir) & empty;
        d += 1;
    }
    result
}

/// Stones flipped by `own` placing on `square` (single set bit).
#[inline]
fn flips(own: u64, opp: u64, square: u64) -> u64 {
    let mut flipped = 0;
    let mut d = 0;
    while d < 8 {
        let dir = DIRS[d];
        let mut run = shift(square, dir) & opp;
        run |= shift(run, dir) & opp;
        run |= shift(run, dir) & opp;
        run |= shift(run, dir) & opp;
        run |= shift(run, dir) & opp;
        run |= shift(run, dir) & opp;
        // The run only flips if it is capped by an own stone.
        if shift(run, dir) & own != 0 {
            flipped |= run;
        }
        d += 1;
    }
    flipped
}

/// Board seen from the side to move: `own` is the mover's stones, `opp` the opponent's.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct OthelloState {
    pub own: u64,
    pub opp: u64,
}

impl OthelloState {
    /// Observation vector: 1.0 for the mover's stones, -1.0 for the opponent's, 0.0 empty.
    pub fn to_obs(self) -> [f64; 64] {
        let mut obs = [0.0; 64];
        let mut i = 0;
        while i < 64 {
            let bit = 1 << i;
            if self.own & bit != 0 {
                obs[i] = 1.0;
            } else if self.opp & bit != 0 {
                obs[i] = -1.0;
            }
            i += 1;
        }
        obs
    }
}

/// Bitboard Othello. Stones are stored relative to the side to move and the two
/// masks swap after every step, so move generation only ever runs one way.
#[derive(Clone, Copy)]
pub struct Othello {
    own: u64,
    opp: u64,
    black_is_mover: bool,
}

impl Default for Othello {
    fn default() -> Self {
        // Standard opening: black d5/e4 (bits 28, 35), white d4/e5 (bits 27, 36).
        Self {
            own: (1 << 28) | (1 << 35),
            opp: (1 << 27) | (1 << 36),
            black_is_mover: true,
        }
    }
}

impl Othello {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bitmask of legal placements for the side to move (excludes pass).
    #[inline]
    pub fn legal_moves_mask(&self) -> u64 {
        moves(self.own, self.opp)
    }

    /// The mover must pass: no placements, but the game is not over.
    #[inline]
    pub fn must_pass(&self) -> bool {
        self.legal_moves_mask() == 0 && !self.is_over()
    }

    #[inline]
    pub fn is_legal(&self, action: usize) -> bool {
        if action == PASS {
            self.must_pass()
        } else {
            action < 64 && self.legal_moves_mask() & (1 << action) != 0
        }
    }

    /// Legal actions ascending; yields only `PASS` when the mover must pass.
    pub fn legal_actions(&self) -> impl Iterator<Item = usize> {
        let mut mask = self.legal_moves_mask();
        let mut pass = self.must_pass();
        std::iter::from_fn(move || {
            if mask != 0 {
                let i = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                return Some(i);
            }
            if pass {
                pass = false;
                return Some(PASS);
            }
            None
        })
    }

    /// Game ends when neither side has a placement (covers the full board).
    #[inline]
    pub fn is_over(&self) -> bool {
        moves(self.own, self.opp) == 0 && moves(self.opp, self.own) == 0
    }

    #[inline]
    pub fn black_to_move(&self) -> bool {
        self.black_is_mover
    }

    /// (black stones, white stones)
    pub fn counts(&self) -> (u32, u32) {
        let (black, white) = if self.black_is_mover {
            (self.own, self.opp)
        } else {
            (self.opp, self.own)
        };
        (black.count_ones(), white.count_ones())
    }

    /// Terminal reward for the player whose stones are in `own`.
    #[inline]
    fn outcome(own: u64, opp: u64) -> f64 {
        match own.count_ones().cmp(&opp.count_ones()) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 0.0,
        }
    }
}

impl std::fmt::Display for Othello {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (black, white) = if self.black_is_mover {
            (self.own, self.opp)
        } else {
            (self.opp, self.own)
        };
        writeln!(f, "  a b c d e f g h")?;
        for row in 0..8 {
            write!(f, "{} ", row + 1)?;
            for col in 0..8 {
                let bit = 1u64 << (row * 8 + col);
                let cell = if black & bit != 0 {
                    'X'
                } else if white & bit != 0 {
                    'O'
                } else {
                    '.'
                };
                write!(f, "{cell} ")?;
            }
            if row < 7 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

impl Environment for Othello {
    type State = OthelloState;
    type Action = usize;

    // 60 placements plus at most one interleaved pass each.
    const MAX_STEPS: usize = 120;

    fn state(&self) -> Self::State {
        OthelloState {
            own: self.own,
            opp: self.opp,
        }
    }

    fn obs(&self) -> Vec<f32> {
        self.state().to_obs().iter().map(|&x| x as f32).collect()
    }

    /// Reward is from the perspective of the player taking the action, granted only
    /// on the terminal step: +1 win, -1 loss, 0 draw. Illegal moves end the game at -1.
    fn step(&mut self, action: usize) -> StepResult<Self::State> {
        debug_assert!(self.is_legal(action), "illegal action {action}\n{self}");
        if !self.is_legal(action) {
            return StepResult {
                next_state: self.state(),
                reward: -1.0,
                done: true,
                truncated: false,
            };
        }

        if action != PASS {
            let square = 1u64 << action;
            let flipped = flips(self.own, self.opp, square);
            self.own |= square | flipped;
            self.opp &= !flipped;
        }
        core::mem::swap(&mut self.own, &mut self.opp);
        self.black_is_mover = !self.black_is_mover;

        let done = self.is_over();
        StepResult {
            next_state: self.state(),
            // The mover's stones are in `opp` after the swap.
            reward: if done {
                Self::outcome(self.opp, self.own)
            } else {
                0.0
            },
            done,
            truncated: false,
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    const INFO: EnvInfo = EnvInfo {
        obs_shape: &[1, 8, 8],
        action_size: 65,
        num_players: 2,
        lower_reward_bound: Some(0.0),
        upper_reward_bound: Some(1.0)
    };

    fn legal_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; Self::INFO.action_size];
        let mut moves = self.legal_moves_mask();
        while moves != 0 {
            mask[moves.trailing_zeros() as usize] = true;
            moves &= moves - 1;
        }
        mask[PASS] = self.must_pass();
        mask
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
                assert_eq!(
                    env.legal_moves_mask(),
                    naive_moves(env.own, env.opp),
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
                assert!(steps <= Othello::MAX_STEPS);
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
        let env = Othello {
            own: (1 << 4) | (1 << 7), // black e1, h1
            opp: (1 << 5) | (1 << 6), // white f1, g1
            black_is_mover: true,
        };
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
        let s = env.state(); // white to move
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
        env.reset();
        assert_eq!(env.counts(), (2, 2));
        assert!(env.black_to_move());
    }
}
