use chess::{ALL_PIECES, Board, Color, Piece};

use crate::env::chess::env::Chess;
use crate::eval::opponent::BoardGame;

const PAWN: i32 = 100;
const KNIGHT: i32 = 320;
const BISHOP: i32 = 330;
const ROOK: i32 = 500;
const QUEEN: i32 = 900;

fn piece_value(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => PAWN,
        Piece::Knight => KNIGHT,
        Piece::Bishop => BISHOP,
        Piece::Rook => ROOK,
        Piece::Queen => QUEEN,
        Piece::King => 0,
    }
}

/// Classic "simplified evaluation function" piece-square tables (Tomasz
/// Michniewski), one row per rank starting from rank 8. Only pawn and knight
/// tables meaningfully change common alpha-beta lines; the rest are still
/// worth having since they're free once the table lookup exists.
#[rustfmt::skip]
const PAWN_PST: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const KNIGHT_PST: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP_PST: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const ROOK_PST: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
];

#[rustfmt::skip]
const QUEEN_PST: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

#[rustfmt::skip]
const KING_PST: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
];

fn pst(piece: Piece) -> &'static [i32; 64] {
    match piece {
        Piece::Pawn => &PAWN_PST,
        Piece::Knight => &KNIGHT_PST,
        Piece::Bishop => &BISHOP_PST,
        Piece::Rook => &ROOK_PST,
        Piece::Queen => &QUEEN_PST,
        Piece::King => &KING_PST,
    }
}

/// Tables above are written rank-8-first; white reads them top-to-bottom as
/// printed, black reads the same table upside down (rank-flipped), so the
/// bonuses for a piece on its own back rank match regardless of color.
fn pst_value(table: &[i32; 64], square: chess::Square, color: Color) -> i32 {
    let rank = square.get_rank().to_index();
    let file = square.get_file().to_index();
    let row = if color == Color::White {
        7 - rank
    } else {
        rank
    };
    table[row * 8 + file]
}

/// Material + piece-square-table diff, White minus Black, in centipawns.
fn material_and_positional(board: &Board) -> (i32, i32) {
    let mut material = 0;
    let mut positional = 0;
    for &piece in &ALL_PIECES {
        let value = piece_value(piece);
        let table = pst(piece);
        for square in *board.pieces(piece) & *board.color_combined(Color::White) {
            material += value;
            positional += pst_value(table, square, Color::White);
        }
        for square in *board.pieces(piece) & *board.color_combined(Color::Black) {
            material -= value;
            positional -= pst_value(table, square, Color::Black);
        }
    }
    (material, positional)
}

impl BoardGame for Chess {
    fn heuristic(&self) -> f32 {
        let board = self.board();
        let (material, positional) = material_and_positional(board);
        let sign = if board.side_to_move() == Color::White {
            1.0
        } else {
            -1.0
        };

        let my_moves = chess::MoveGen::new_legal(board).len() as f32;
        let opp_moves = board
            .null_move()
            .map(|b| chess::MoveGen::new_legal(&b).len() as f32)
            .unwrap_or(0.0);
        let mobility = (my_moves - opp_moves) / (my_moves + opp_moves + 1.0);

        let material = sign * material as f32 / 2000.0;
        let positional = sign * positional as f32 / 600.0;

        (0.7 * material + 0.2 * positional + 0.1 * mobility).clamp(-0.99, 0.99)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;
    use crate::env::Environment;

    #[test]
    fn opening_position_is_symmetric() {
        assert_eq!(Chess::new().heuristic(), 0.0);
    }

    #[test]
    fn stays_in_range_over_random_games() {
        let mut rng = fastrand::Rng::with_seed(19);
        for _ in 0..30 {
            let mut env = Chess::new();
            for _ in 0..200 {
                let value = env.heuristic();
                assert!(
                    value.is_finite() && value.abs() <= 0.99,
                    "out of range: {value}"
                );
                let legal: Vec<usize> = env
                    .legal_mask()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &l)| l)
                    .map(|(a, _)| a)
                    .collect();
                if legal.is_empty() {
                    break;
                }
                let action = legal[rng.usize(..legal.len())];
                if Environment::step(&mut env, action).done {
                    break;
                }
            }
        }
    }

    #[test]
    fn missing_queen_is_scored_worse_for_its_side() {
        let with_queen =
            Chess::from_board(Board::from_str("4k3/8/8/8/8/8/8/R2QK2R w KQ - 0 1").unwrap());
        let without_queen =
            Chess::from_board(Board::from_str("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap());
        assert!(with_queen.heuristic() > without_queen.heuristic());
    }
}
