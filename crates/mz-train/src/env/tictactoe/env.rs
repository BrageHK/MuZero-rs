//  ░░░░░░░░░░░░░░░░░░░░░░░░░
//  ░  CLAUDE SEAL OF SLOP  ░
//  ░       ~ 卂丨 ~        ░
//  ░   grade: B+ (mild)    ░
//  ░  "shipped it anyway"  ░
//  ░░░░░░░░░░░░░░░░░░░░░░░░░

use burn::rl::{Environment, StepResult};

use crate::env::{EnvInfo, MuZeroEnv};

/// Board cells are bits 0..9, row-major:
///
/// ```text
///  0 | 1 | 2
/// ---+---+---
///  3 | 4 | 5
/// ---+---+---
///  6 | 7 | 8
/// ```
const BOARD_MASK: u16 = 0x1FF;

/// The 8 winning lines as bitmasks (3 rows, 3 columns, 2 diagonals).
const LINES: [u16; 8] = [
    0b000_000_111,
    0b000_111_000,
    0b111_000_000,
    0b001_001_001,
    0b010_010_010,
    0b100_100_100,
    0b100_010_001,
    0b001_010_100,
];

const fn has_line(mask: u16) -> bool {
    let mut i = 0;
    while i < LINES.len() {
        if mask & LINES[i] == LINES[i] {
            return true;
        }
        i += 1;
    }
    false
}

/// Win detection for any 9-bit occupancy is a single table load.
static WIN: [bool; 512] = {
    let mut table = [false; 512];
    let mut mask = 0;
    while mask < 512 {
        table[mask] = has_line(mask as u16);
        mask += 1;
    }
    table
};

/// Board seen from the side to move: `own` is the mover's stones, `opp` the opponent's.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TicTacToeState {
    pub own: u16,
    pub opp: u16,
}

impl TicTacToeState {
    /// Observation vector: 1.0 for the mover's stones, -1.0 for the opponent's, 0.0 empty.
    pub fn to_obs(self) -> [f64; 9] {
        let mut obs = [0.0; 9];
        let mut i = 0;
        while i < 9 {
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

/// Bitboard tic-tac-toe. Stones are stored relative to the side to move and the two
/// masks swap after every step, so win checks only ever look at the mover's mask.
#[derive(Clone, Copy, Default)]
pub struct TicTacToe {
    own: u16,
    opp: u16,
}

impl TicTacToe {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bitmask of empty cells.
    #[inline]
    pub fn legal_moves_mask(&self) -> u16 {
        !(self.own | self.opp) & BOARD_MASK
    }

    #[inline]
    pub fn is_legal(&self, action: usize) -> bool {
        action < 9 && self.legal_moves_mask() & (1 << action) != 0
    }

    /// Indices of empty cells, ascending.
    pub fn legal_moves(&self) -> impl Iterator<Item = usize> {
        let mut mask = self.legal_moves_mask();
        std::iter::from_fn(move || {
            if mask == 0 {
                return None;
            }
            let i = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            Some(i)
        })
    }

    /// True if X is to move (X plays first, so equal stone counts mean X's turn).
    #[inline]
    pub fn x_to_move(&self) -> bool {
        self.own.count_ones() == self.opp.count_ones()
    }

    #[inline]
    fn done(&self) -> bool {
        // After the swap in `step`, the previous mover's stones sit in `opp`.
        WIN[self.opp as usize] || self.own | self.opp == BOARD_MASK
    }
}

impl std::fmt::Display for TicTacToe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (x, o) = if self.x_to_move() {
            (self.own, self.opp)
        } else {
            (self.opp, self.own)
        };
        for row in 0..3 {
            for col in 0..3 {
                let bit = 1 << (row * 3 + col);
                let cell = if x & bit != 0 {
                    'X'
                } else if o & bit != 0 {
                    'O'
                } else {
                    '.'
                };
                write!(f, "{cell}")?;
                if col < 2 {
                    write!(f, "|")?;
                }
            }
            if row < 2 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

impl Environment for TicTacToe {
    type State = TicTacToeState;
    type Action = usize;

    const MAX_STEPS: usize = 9;

    fn state(&self) -> Self::State {
        TicTacToeState {
            own: self.own,
            opp: self.opp,
        }
    }

    /// Reward is from the perspective of the player taking the action:
    /// +1 win, 0 draw or game still running, -1 illegal move (which also ends the game).
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

        self.own |= 1 << action;
        let won = WIN[self.own as usize];
        core::mem::swap(&mut self.own, &mut self.opp);

        StepResult {
            next_state: self.state(),
            reward: if won { 1.0 } else { 0.0 },
            done: won || self.done(),
            truncated: false,
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

impl MuZeroEnv for TicTacToe {
    const INFO: EnvInfo = EnvInfo {
        obs_shape: &[1, 3, 3],
        action_size: 9,
        num_players: 2,
    };

    fn legal_mask(&self) -> Vec<bool> {
        (0..Self::INFO.action_size)
            .map(|a| self.is_legal(a))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x_wins_top_row() {
        let mut env = TicTacToe::new();
        // X: 0, 1, 2  O: 3, 4
        for &(action, expect_done) in &[(0, false), (3, false), (1, false), (4, false)] {
            let r = env.step(action);
            assert_eq!(r.done, expect_done);
            assert_eq!(r.reward, 0.0);
        }
        let r = env.step(2);
        assert!(r.done);
        assert_eq!(r.reward, 1.0);
    }

    #[test]
    fn o_wins_column() {
        let mut env = TicTacToe::new();
        // X: 0, 1, 8  O: 2, 5, then 8 taken -> O plays 2,5 and X blocks nothing
        for action in [0, 2, 1, 5, 6] {
            assert!(!env.step(action).done);
        }
        let r = env.step(8); // O completes column 2|5|8
        assert!(r.done);
        assert_eq!(r.reward, 1.0);
    }

    #[test]
    fn diagonal_wins() {
        for diag in [[0usize, 4, 8], [2, 4, 6]] {
            let mut env = TicTacToe::new();
            let others: Vec<usize> = (0..9).filter(|c| !diag.contains(c)).collect();
            env.step(diag[0]);
            env.step(others[0]);
            env.step(diag[1]);
            env.step(others[1]);
            let r = env.step(diag[2]);
            assert!(r.done);
            assert_eq!(r.reward, 1.0);
        }
    }

    #[test]
    fn draw_game() {
        let mut env = TicTacToe::new();
        // X O X / X O O / O X X — no line for either side.
        let moves = [0, 1, 2, 4, 3, 5, 7, 6, 8];
        for (i, &action) in moves.iter().enumerate() {
            let r = env.step(action);
            assert_eq!(r.reward, 0.0, "move {i}");
            assert_eq!(r.done, i == 8, "move {i}");
        }
        assert_eq!(env.legal_moves_mask(), 0);
    }

    #[test]
    fn legal_moves_track_occupancy() {
        let mut env = TicTacToe::new();
        assert_eq!(env.legal_moves_mask(), BOARD_MASK);
        assert_eq!(env.legal_moves().count(), 9);
        env.step(4);
        assert!(!env.is_legal(4));
        assert!(!env.is_legal(9));
        assert_eq!(env.legal_moves().count(), 8);
        assert_eq!(env.legal_moves_mask() & (1 << 4), 0);
    }

    #[test]
    fn state_is_mover_relative() {
        let mut env = TicTacToe::new();
        env.step(0); // X plays 0
        let s = env.state(); // O to move: own = O stones (none), opp = X's bit 0
        assert_eq!(s.own, 0);
        assert_eq!(s.opp, 1);
        assert_eq!(s.to_obs()[0], -1.0);
        assert!(!env.x_to_move());
    }

    #[test]
    fn reset_clears_board() {
        let mut env = TicTacToe::new();
        env.step(0);
        env.step(1);
        env.reset();
        assert_eq!(env.legal_moves_mask(), BOARD_MASK);
        assert!(env.x_to_move());
    }

    #[test]
    fn win_table_matches_line_scan() {
        for mask in 0u16..512 {
            assert_eq!(WIN[mask as usize], has_line(mask));
        }
    }
}
