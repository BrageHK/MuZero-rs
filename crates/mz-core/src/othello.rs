/// Board cells are bits 0..64, row-major: bit = row * 8 + col, a1 = bit 0.
/// Action 64 is "pass", legal only when the mover has no placement.
pub const PASS: usize = 64;

/// 60 placements plus at most one interleaved pass each.
pub const MAX_STEPS: usize = 120;

pub const ACTION_SIZE: usize = 65;

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

/// Result of `Othello::apply`, mirroring burn-rl's `StepResult` without
/// depending on it.
pub struct StepOutcome {
    pub state: OthelloState,
    pub reward: f64,
    pub done: bool,
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

    /// Board from raw mover-relative masks. Only for tests and puzzle setups.
    pub fn from_parts(own: u64, opp: u64, black_is_mover: bool) -> Self {
        Self {
            own,
            opp,
            black_is_mover,
        }
    }

    pub fn state(&self) -> OthelloState {
        OthelloState {
            own: self.own,
            opp: self.opp,
        }
    }

    /// Flat observation, mover-relative, as the networks want it.
    pub fn obs(&self) -> Vec<f32> {
        self.state().to_obs().iter().map(|&x| x as f32).collect()
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
        core::iter::from_fn(move || {
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

    /// Length-65 legality mask, index 64 = pass.
    pub fn legal_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; ACTION_SIZE];
        let mut moves = self.legal_moves_mask();
        while moves != 0 {
            mask[moves.trailing_zeros() as usize] = true;
            moves &= moves - 1;
        }
        mask[PASS] = self.must_pass();
        mask
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
        let (black, white) = self.absolute();
        (black.count_ones(), white.count_ones())
    }

    /// (black stones, white stones) as bitboards, independent of who moves.
    #[inline]
    pub fn absolute(&self) -> (u64, u64) {
        if self.black_is_mover {
            (self.own, self.opp)
        } else {
            (self.opp, self.own)
        }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Play `action` for the side to move. Reward is from the perspective of the
    /// player taking the action, granted only on the terminal step: +1 win, -1
    /// loss, 0 draw. Illegal moves end the game at -1.
    pub fn apply(&mut self, action: usize) -> StepOutcome {
        debug_assert!(self.is_legal(action), "illegal action {action}\n{self}");
        if !self.is_legal(action) {
            return StepOutcome {
                state: self.state(),
                reward: -1.0,
                done: true,
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
        StepOutcome {
            state: self.state(),
            // The mover's stones are in `opp` after the swap.
            reward: if done {
                Self::outcome(self.opp, self.own)
            } else {
                0.0
            },
            done,
        }
    }

    /// Terminal reward for the player whose stones are in `own`.
    #[inline]
    fn outcome(own: u64, opp: u64) -> f64 {
        match own.count_ones().cmp(&opp.count_ones()) {
            core::cmp::Ordering::Greater => 1.0,
            core::cmp::Ordering::Less => -1.0,
            core::cmp::Ordering::Equal => 0.0,
        }
    }
}

impl core::fmt::Display for Othello {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (black, white) = self.absolute();
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
