use crate::env::Environment;
use crate::env::tictactoe::env::{LINES, TicTacToe};
use crate::eval::opponent::BoardGame;

impl BoardGame for TicTacToe {
    fn heuristic(&self) -> f32 {
        let state = self.state();
        let mut score = 0i32;
        for line in LINES {
            let own = (state.own & line).count_ones();
            let opp = (state.opp & line).count_ones();
            score += match (own, opp) {
                (n, 0) if n > 0 => line_value(n),
                (0, n) if n > 0 => -line_value(n),
                _ => 0,
            };
        }
        (score as f32 / 30.0).clamp(-0.99, 0.99)
    }
}

fn line_value(stones: u32) -> i32 {
    match stones {
        3 => 50,
        2 => 10,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_is_neutral() {
        assert_eq!(TicTacToe::new().heuristic(), 0.0);
    }

    #[test]
    fn two_in_a_row_favours_the_owner() {
        let mut env = TicTacToe::new();
        env.step(0); // X
        env.step(4); // O
        env.step(1); // X, now two on the top row
        // O to move, so the score is negative from the mover's view.
        assert!(env.heuristic() < 0.0);
    }
}
