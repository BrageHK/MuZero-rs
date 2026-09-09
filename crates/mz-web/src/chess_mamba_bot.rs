//! ChessMamba-powered chess opponent, a third alternative alongside
//! `chess_bot`'s alpha-beta heuristic and `chess_game`'s trained MuZero
//! agent. Stateless (unlike `ChessGame`) because ChessMamba was trained with
//! no game-history planes (`n_history=0` in bee-chess's `model.py`) -- unlike
//! the MuZero agent's, its observation is just the current position, so
//! there's nothing to keep in sync across moves via `play_uci`.
//!
//! ChessMamba is a plain policy+value net, not a MuZero net: it has no
//! dynamics/representation split and so cannot implement `MuZeroNets`
//! (`search.rs`'s `gumbel_search` requires `initial_inference` +
//! `recurrent_inference`, i.e. an expandable latent-space tree -- there's no
//! latent state to expand here). So instead of running a search, `best_move`
//! runs one forward pass and argmaxes the (from, to) policy logits over the
//! position's legal moves, exactly like bee-chess's own
//! `training/src/bee_training/chess_mamba/play.py`.

use core::str::FromStr;

use burn::prelude::*;
use chess::{ALL_SQUARES, Board, ChessMove, Color, MoveGen, Piece};
#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::wasm_bindgen;

use crate::model::{self, Be};

// Must match bee-chess's encode.py exactly: 12 one-hot piece planes (6 piece
// types x 2 colors) + 8 auxiliary scalars broadcast to every square.
const N_PIECE_TYPES: usize = 12;
const N_AUX: usize = 8;
const IN_DIM: usize = N_PIECE_TYPES + N_AUX;

/// `fen` -> flat (64 * IN_DIM) row-major (square, channel) plane data, the
/// same layout `encode_fen`'s (64, IN_DIM) tensor flattens to. `board` is
/// parsed from the same `fen` by the caller; passed in separately only so
/// the halfmove clock (below) can be read from the original string, since
/// `chess::Board` itself doesn't retain it.
fn encode_fen(fen: &str, board: &Board) -> [f32; 64 * IN_DIM] {
    let mut planes = [0f32; 64 * IN_DIM];

    for square in ALL_SQUARES {
        if let Some(piece) = board.piece_on(square) {
            let color_offset = if board.color_on(square) == Some(Color::White) { 0 } else { 6 };
            planes[square.to_index() * IN_DIM + piece.to_index() + color_offset] = 1.0;
        }
    }

    // `chess::Board` doesn't track the halfmove clock (see its FEN parsing),
    // so read it straight out of the FEN's 5th field; malformed/missing
    // falls back to 0, same as a fresh game.
    let halfmove_clock: f32 = fen.split_whitespace().nth(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let white_castle = board.castle_rights(Color::White);
    let black_castle = board.castle_rights(Color::Black);
    let aux = [
        f32::from(white_castle.has_kingside()),
        f32::from(white_castle.has_queenside()),
        f32::from(black_castle.has_kingside()),
        f32::from(black_castle.has_queenside()),
        f32::from(board.en_passant().is_some()),
        halfmove_clock / 100.0,
        f32::from(board.side_to_move() == Color::White),
        0.0, // reserved, unused -- see encode.py
    ];
    for square in ALL_SQUARES {
        let base = square.to_index() * IN_DIM + N_PIECE_TYPES;
        planes[base..base + N_AUX].copy_from_slice(&aux);
    }

    planes
}

/// Every legal move for the position given as FEN, ranked best-first by the
/// policy head, in UCI notation (`e2e4`, `e7e8q`). Empty if the FEN is
/// malformed or the position has no legal move.
///
/// Exposed as a ranked list (rather than just the top move) so callers can
/// fall back to the next-best candidate if the top one is ever rejected --
/// e.g. by chess.js's stricter FEN/legality checks disagreeing at the
/// margins with this crate's `chess` -- instead of the bot silently failing
/// to move.
fn ranked_moves_uci(model: &model::chess_mamba::Model<Be>, device: &model::Device, fen: &str) -> Vec<String> {
    let Some(board) = Board::from_str(fen).ok() else {
        return Vec::new();
    };
    let legal_moves: Vec<_> = MoveGen::new_legal(&board).collect();
    if legal_moves.is_empty() {
        return Vec::new();
    }

    let planes = encode_fen(fen, &board);
    let input: Tensor<Be, 3> = Tensor::<Be, 1>::from_floats(planes.as_slice(), device).reshape([1, 64, IN_DIM]);
    let (policy_logits, _value_logits) = model.forward(input);
    let policy: Vec<f32> = policy_logits.into_data().to_vec().expect("policy_logits is f32");

    let mut scored: Vec<(f32, ChessMove)> = legal_moves
        .into_iter()
        // `FromToPolicyHead` only scores (from, to) square pairs, so it
        // can't tell a queen promotion apart from an underpromotion to the
        // same square -- prune underpromotions, same as play.py's
        // `choose_move`. Queen is virtually always the right choice anyway.
        .filter(|mv| !mv.get_promotion().is_some_and(|p| p != Piece::Queen))
        .map(|mv| {
            let index = mv.get_source().to_index() * 64 + mv.get_dest().to_index();
            (policy[index], mv)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().map(|(_, mv)| mv.to_string()).collect()
}

/// The bot's top move for the position given as FEN, in UCI notation
/// (`e2e4`, `e7e8q`). `None` if the FEN is malformed or the position has no
/// legal move.
fn best_move_uci(model: &model::chess_mamba::Model<Be>, device: &model::Device, fen: &str) -> Option<String> {
    ranked_moves_uci(model, device, fen).into_iter().next()
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct ChessMambaBot {
    model: model::chess_mamba::Model<Be>,
    device: model::Device,
}

/// Builds the bot: sets up the backend and loads the embedded weights.
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub async fn create_chess_mamba() -> ChessMambaBot {
    let device = model::Device::default();
    model::init_backend(&device).await;
    ChessMambaBot { model: model::chess_mamba::Model::from_embedded(&device), device }
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl ChessMambaBot {
    /// Synchronous: a single forward pass, no search -- same reasoning as
    /// `chess_bot_move`'s (never blocks the main thread since this runs in
    /// `worker.js`, only the worker's own thread stalls while it computes).
    pub fn best_move(&self, fen: &str) -> Option<String> {
        best_move_uci(&self.model, &self.device, fen)
    }

    /// All legal moves ranked best-first by the policy head. `worker.js`
    /// uses this to fall back to the next-best candidate whenever
    /// `best_move`'s top pick turns out illegal by the page's own chess.js
    /// state, instead of leaving the game stuck.
    pub fn ranked_moves(&self, fen: &str) -> Vec<String> {
        ranked_moves_uci(&self.model, &self.device, fen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bot() -> ChessMambaBot {
        let device = model::Device::default();
        ChessMambaBot { model: model::chess_mamba::Model::from_embedded(&device), device }
    }

    #[test]
    fn plays_a_legal_move_from_the_start_position() {
        let mv = bot()
            .best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("start position has legal moves");
        assert!(mv.len() == 4 || mv.len() == 5);
    }

    #[test]
    fn none_when_game_is_already_over() {
        // Fool's-mate final position: white to move, already checkmated.
        assert!(
            bot()
                .best_move("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
                .is_none()
        );
    }

    #[test]
    fn ranked_moves_lists_every_legal_move_with_the_top_pick_first() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let ranked = bot().ranked_moves(fen);
        assert_eq!(ranked.first().cloned(), bot().best_move(fen));
        // Start position: 16 pawn/knight moves, no promotions to prune.
        assert_eq!(ranked.len(), 20);
    }

    #[test]
    fn ranked_moves_empty_when_game_is_already_over() {
        assert!(
            bot()
                .ranked_moves("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
                .is_empty()
        );
    }
}
