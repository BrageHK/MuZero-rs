//! Benchmark opponent for the chess page, matching the alpha-beta bot in
//! `mz-train`'s `env/chess/heuristic.rs`. Ported rather than shared because
//! that crate pulls in rayon and the training env traits, neither
//! wasm-friendly (same reasoning as `opponent.rs`'s Othello port).
//!
//! Stands in for the trained MuZero chess agent, which doesn't exist yet:
//! chess self-play training is separate, compute-heavy work. The board's
//! authoritative state lives in the page's chess.js instance; this module is
//! stateless, taking a FEN in and handing a UCI move string back.

use core::str::FromStr;

use chess::{ALL_PIECES, Board, Color, Piece};
use mz_core::chess::Chess;

use crate::rng::Rng;

const WIN_SCORE: f32 = 100.0;

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

fn pst_value(table: &[i32; 64], square: chess::Square, color: Color) -> i32 {
    let rank = square.get_rank().to_index();
    let file = square.get_file().to_index();
    let row = if color == Color::White { 7 - rank } else { rank };
    table[row * 8 + file]
}

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

fn heuristic(game: &Chess) -> f32 {
    let board = game.board();
    let (material, positional) = material_and_positional(board);
    let sign = if board.side_to_move() == Color::White { 1.0 } else { -1.0 };

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

struct Child {
    action: usize,
    game: Chess,
    terminal: Option<f32>,
}

/// Value of this child for the parent's side to move, used only for ordering.
fn hint(child: &Child) -> f32 {
    match child.terminal {
        Some(value) => value,
        None => -heuristic(&child.game),
    }
}

fn expand(game: &Chess) -> Vec<Child> {
    game.legal_actions()
        .map(|action| {
            let mut child = game.clone();
            let outcome = child.apply(action);
            let terminal = outcome.done.then(|| outcome.reward as f32 * WIN_SCORE);
            Child { action, game: child, terminal }
        })
        .collect()
}

fn order(children: &mut [Child]) {
    children.sort_by(|a, b| hint(b).total_cmp(&hint(a)));
}

fn negamax(game: &Chess, depth: usize, mut alpha: f32, beta: f32) -> f32 {
    if depth == 0 {
        return heuristic(game);
    }

    let mut children = expand(game);
    if children.is_empty() {
        return heuristic(game);
    }
    order(&mut children);

    let mut best = f32::NEG_INFINITY;
    for child in &children {
        let value = match child.terminal {
            Some(value) => value,
            None => -negamax(&child.game, depth - 1, -beta, -alpha),
        };
        if value > best {
            best = value;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
    }
    best
}

/// Shuffling before the stable ordering sort breaks ties randomly, so repeated
/// games from the same position don't all follow one line.
fn alpha_beta_best(game: &Chess, depth: usize, rng: &mut Rng) -> usize {
    let mut children = expand(game);
    debug_assert!(!children.is_empty(), "alpha-beta root has no legal action");
    rng.shuffle(&mut children);
    order(&mut children);

    let mut best_action = children[0].action;
    let mut best = f32::NEG_INFINITY;
    for child in &children {
        let value = match child.terminal {
            Some(value) => value,
            None => -negamax(&child.game, depth - 1, f32::NEG_INFINITY, -best),
        };
        if value > best {
            best = value;
            best_action = child.action;
        }
    }
    best_action
}

/// The bot's move for the position given as FEN, in UCI notation (`e2e4`,
/// `e7e8q`). `None` if the FEN is malformed or the position has no legal move.
fn best_move_uci(fen: &str, depth: usize, seed: u64) -> Option<String> {
    let board = Board::from_str(fen).ok()?;
    let game = Chess::from_board(board);
    if game.is_over() {
        return None;
    }
    let mut rng = Rng::new(seed.max(1));
    let action = alpha_beta_best(&game, depth.max(1), &mut rng);
    game.action_to_move(action).map(|mv| mv.to_string())
}

/// `depth` is clamped to a range that stays responsive on the main thread
/// (this runs synchronously, same as Othello's `bot_move`).
#[cfg_attr(target_family = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub fn chess_bot_move(fen: &str, depth: u32, seed: f64) -> Option<String> {
    let depth = (depth as usize).clamp(1, 4);
    let seed = (seed.abs() as u64).max(1);
    best_move_uci(fen, depth, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plays_a_legal_move_from_the_start_position() {
        let mv = chess_bot_move(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            2,
            42.0,
        )
        .expect("start position has legal moves");
        assert_eq!(mv.len(), 4);
    }

    #[test]
    fn takes_a_free_queen() {
        // White to move, black queen hangs on h4 to the white queen on d8... use
        // a simpler hanging-piece position: black queen on a5 undefended, white
        // queen on a1 can just take it.
        let mv = chess_bot_move("4k3/8/8/q7/8/8/8/Q3K3 w - - 0 1", 1, 7.0).unwrap();
        assert_eq!(mv, "a1a5");
    }

    #[test]
    fn none_when_game_is_already_over() {
        // Fool's-mate final position: white to move, already checkmated.
        assert!(
            chess_bot_move(
                "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
                2,
                1.0
            )
            .is_none()
        );
    }
}
