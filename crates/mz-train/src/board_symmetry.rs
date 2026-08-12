//! Random dihedral-group augmentation for square-board two-player games
//! (Othello, TicTacToe). One rotation/reflection is drawn per sampled game and
//! applied to every step's observation, policy target, and action so the
//! dynamics-conditioning chain stays consistent across the whole unroll.

use fastrand::Rng;

use crate::mz_config::MuZeroConfig;
use crate::replay_buffer::BufferData;

pub const NUM_SYMMETRIES: usize = 8;

/// Where cell `(r, c)` of an `h x w` board lands under symmetry `sym` (0..8):
/// the 4 rotations, then the 4 axis/diagonal reflections.
fn transform_cell(sym: usize, r: usize, c: usize, h: usize, w: usize) -> (usize, usize) {
    match sym {
        0 => (r, c),
        1 => (c, h - 1 - r),
        2 => (h - 1 - r, w - 1 - c),
        3 => (w - 1 - c, r),
        4 => (r, w - 1 - c),
        5 => (h - 1 - r, c),
        6 => (c, r),
        7 => (w - 1 - c, h - 1 - r),
        _ => unreachable!("symmetry index must be < NUM_SYMMETRIES"),
    }
}

/// `perm[i]` is where the cell at flat index `i` moves to under `sym`.
fn cell_permutation(sym: usize, h: usize, w: usize) -> Vec<usize> {
    let mut perm = vec![0usize; h * w];
    for r in 0..h {
        for c in 0..w {
            let (nr, nc) = transform_cell(sym, r, c, h, w);
            perm[r * w + c] = nr * w + nc;
        }
    }
    perm
}

pub struct BoardSymmetry {
    height: usize,
    width: usize,
    scratch: Vec<f32>,
    rng: Rng,
}

impl BoardSymmetry {
    pub fn new(height: usize, width: usize) -> Self {
        BoardSymmetry {
            height,
            width,
            scratch: vec![0.0; height * width],
            rng: Rng::new(),
        }
    }

    pub fn from_config(mz_conf: &MuZeroConfig) -> Option<Self> {
        if !mz_conf.board_symmetric() {
            return None;
        }
        Some(BoardSymmetry::new(
            mz_conf.board_height,
            mz_conf.board_width,
        ))
    }

    pub fn sample(&mut self) -> usize {
        self.rng.usize(0..NUM_SYMMETRIES)
    }

    /// Applies `sym` to every step of a sampled game in place: each channel
    /// plane of `state`, the board-cell prefix of `policy`, and `action` (left
    /// untouched if it addresses a non-board action, e.g. Othello's pass).
    pub fn apply_game(&mut self, game: &mut [BufferData], sym: usize) {
        if sym == 0 {
            return;
        }
        let cells = self.height * self.width;
        let perm = cell_permutation(sym, self.height, self.width);
        for data in game.iter_mut() {
            self.permute_plane(&mut data.state, &perm, cells);
            self.permute_plane(&mut data.policy, &perm, cells);
            if data.action < cells {
                data.action = perm[data.action];
            }
        }
    }

    /// `values` covers `channels` whole `h*w` planes, optionally followed by a
    /// non-board tail (e.g. Othello's pass logit) that is left untouched.
    fn permute_plane(&mut self, values: &mut [f32], perm: &[usize], cells: usize) {
        let channels = values.len() / cells;
        if self.scratch.len() < channels * cells {
            self.scratch.resize(channels * cells, 0.0);
        }
        for ch in 0..channels {
            let base = ch * cells;
            for i in 0..cells {
                self.scratch[base + perm[i]] = values[base + i];
            }
        }
        values[..channels * cells].copy_from_slice(&self.scratch[..channels * cells]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(state: Vec<f32>, action: usize, policy: Vec<f32>) -> BufferData {
        BufferData {
            state,
            action,
            value: 0.0,
            reward: 0.0,
            policy,
            is_terminal: false,
            created_step: 0,
            legal_mask: vec![],
            is_absorbing: false,
        }
    }

    #[test]
    fn identity_is_a_no_op() {
        let mut sym = BoardSymmetry::new(3, 3);
        let mut game = vec![data(
            (0..9).map(|i| i as f32).collect(),
            4,
            (0..9).map(|i| i as f32).collect(),
        )];
        let before = game[0].state.clone();
        sym.apply_game(&mut game, 0);
        assert_eq!(game[0].state, before);
    }

    #[test]
    fn rot90_moves_top_left_to_top_right() {
        // 3x3 board, single channel; cell 0 (top-left) rotates to the top-right corner.
        let mut sym = BoardSymmetry::new(3, 3);
        let mut game = vec![data(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            0,
            vec![0.0; 9],
        )];
        sym.apply_game(&mut game, 1);
        assert_eq!(game[0].state[2], 1.0);
        assert_eq!(game[0].action, 2);
    }

    #[test]
    fn pass_action_beyond_board_is_untouched() {
        // Othello-style: 8x8 board + a trailing pass action at index 64.
        let mut sym = BoardSymmetry::new(8, 8);
        let mut game = vec![data(vec![0.0; 3 * 64], 64, vec![0.0; 65])];
        sym.apply_game(&mut game, 5);
        assert_eq!(game[0].action, 64);
        assert_eq!(game[0].policy.len(), 65);
    }

    #[test]
    fn all_eight_symmetries_are_permutations() {
        for sym in 0..NUM_SYMMETRIES {
            let perm = cell_permutation(sym, 8, 8);
            let mut seen = vec![false; 64];
            for &p in &perm {
                assert!(!seen[p], "sym {sym} is not a bijection");
                seen[p] = true;
            }
        }
    }
}
