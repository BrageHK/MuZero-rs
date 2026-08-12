use crate::env::othello::env::{Othello, moves};
use crate::eval::opponent::BoardGame;

const CORNERS: u64 = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63);

/// Classic positional table: corners are worth taking, the squares next to them
/// are traps that hand the corner away.
#[rustfmt::skip]
const SQUARE_WEIGHTS: [i32; 64] = [
    120, -20,  20,   5,   5,  20, -20, 120,
    -20, -40,  -5,  -5,  -5,  -5, -40, -20,
     20,  -5,  15,   3,   3,  15,  -5,  20,
      5,  -5,   3,   3,   3,   3,  -5,   5,
      5,  -5,   3,   3,   3,   3,  -5,   5,
     20,  -5,  15,   3,   3,  15,  -5,  20,
    -20, -40,  -5,  -5,  -5,  -5, -40, -20,
    120, -20,  20,   5,   5,  20, -20, 120,
];

/// Below this many empty squares the disc count is what actually decides the game.
const ENDGAME_EMPTIES: u32 = 12;

impl BoardGame for Othello {
    fn heuristic(&self) -> f32 {
        let state = self.state();
        let (own, opp) = (state.own, state.opp);

        let n_own = own.count_ones() as f32;
        let n_opp = opp.count_ones() as f32;
        let empties = 64 - own.count_ones() - opp.count_ones();

        let corners =
            ((own & CORNERS).count_ones() as f32 - (opp & CORNERS).count_ones() as f32) / 4.0;
        let discs = (n_own - n_opp) / (n_own + n_opp).max(1.0);

        if empties <= ENDGAME_EMPTIES {
            return (0.8 * discs + 0.2 * corners).clamp(-0.99, 0.99);
        }

        let mut weighted = 0i32;
        for (square, weight) in SQUARE_WEIGHTS.iter().enumerate() {
            let bit = 1u64 << square;
            if own & bit != 0 {
                weighted += weight;
            } else if opp & bit != 0 {
                weighted -= weight;
            }
        }
        let positional = weighted as f32 / 600.0;

        let m_own = moves(own, opp).count_ones() as f32;
        let m_opp = moves(opp, own).count_ones() as f32;
        let mobility = (m_own - m_opp) / (m_own + m_opp + 1.0);

        (0.25 * positional + 0.35 * corners + 0.30 * mobility + 0.10 * discs).clamp(-0.99, 0.99)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Environment;

    #[test]
    fn opening_position_is_symmetric() {
        assert_eq!(Othello::new().heuristic(), 0.0);
    }

    #[test]
    fn stays_in_range_over_random_games() {
        let mut rng = fastrand::Rng::with_seed(11);
        for _ in 0..50 {
            let mut env = Othello::new();
            loop {
                let value = env.heuristic();
                assert!(
                    value.is_finite() && value.abs() <= 0.99,
                    "out of range: {value}"
                );
                let actions: Vec<usize> = env.legal_actions().collect();
                if actions.is_empty() || env.step(actions[rng.usize(..actions.len())]).done {
                    break;
                }
            }
        }
    }
}
