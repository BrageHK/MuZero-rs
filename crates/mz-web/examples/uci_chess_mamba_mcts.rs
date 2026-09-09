//! UCI-speaking wrapper around `ChessMambaBot::best_move_mcts` -- the Rust,
//! tch/libtorch-backed lc0-style MCTS from `chess_mamba_mcts.rs` -- so it can
//! play a 1v1 match against Bee (bee-chess's own UCI engine) through any
//! generic UCI harness, e.g. bee-chess's `tournament.py`.
//!
//! Mirrors bee-chess's `play_mcts_lc0.py` protocol exactly (same `uci`/
//! `isready`/`ucinewgame`/`position`/`go`/`quit` handling), just calling into
//! this crate's own search instead of the ONNX/Python one.
//!
//! `torch::set_num_threads(1)` is required here for the same reason as
//! `examples/bench_mcts.rs`: without it, libtorch's own intra-op thread
//! pool fights our outer tree-parallel MCTS threads for the same cores --
//! see that example's comment for the full explanation and the benchmark
//! that found it (unpinned: capped ~278 nodes/s and got worse past 8
//! threads; pinned: scaled to ~970 nodes/s at 8 threads).
//!
//! Run with:
//!   cargo run --release -p mz-web --example uci_chess_mamba_mcts \
//!       --features tch --no-default-features
//! Configure via env vars (both optional): SIMULATIONS (default 32),
//! THREADS (default 4).

use std::io::{self, BufRead, Write};
use std::str::FromStr;

use chess::{Board, ChessMove, Piece};
use mz_web::chess_mamba_bot;

const ENGINE_NAME: &str = "Bee-Mamba-lc0MCTS-tch";
const ENGINE_AUTHOR: &str = "bee-chess";

/// Tracks what `chess::Board` doesn't (see `chess_mamba_mcts.rs`'s
/// `advance_halfmove_clock`): the halfmove clock, needed to build a real FEN
/// for `best_move_mcts` since `Board::to_string()` always emits a placeholder
/// "0 1" for the last two FEN fields.
struct Game {
    board: Board,
    halfmove_clock: u32,
    fullmove_number: u32,
}

impl Game {
    fn new() -> Self {
        Self { board: Board::default(), halfmove_clock: 0, fullmove_number: 1 }
    }

    fn set_fen(&mut self, fen_fields: &[&str]) {
        let board_fen = fen_fields[..4].join(" ");
        self.board = Board::from_str(&board_fen).expect("valid FEN from a UCI position command");
        self.halfmove_clock = fen_fields.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        self.fullmove_number = fen_fields.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
    }

    fn push_uci(&mut self, uci: &str) {
        let mv = ChessMove::from_str(uci).expect("legal engine-supplied UCI move");
        let is_pawn_move = self.board.piece_on(mv.get_source()) == Some(Piece::Pawn);
        let is_capture = self.board.piece_on(mv.get_dest()).is_some();
        self.halfmove_clock = if is_pawn_move || is_capture { 0 } else { self.halfmove_clock + 1 };
        if self.board.side_to_move() == chess::Color::Black {
            self.fullmove_number += 1;
        }
        self.board = self.board.make_move_new(mv);
    }

    /// A full, correctly-fielded FEN -- unlike `self.board.to_string()`,
    /// which always has placeholder halfmove/fullmove fields.
    fn fen(&self) -> String {
        let board_fen = self.board.to_string();
        let mut fields: Vec<&str> = board_fen.split_whitespace().collect();
        fields.truncate(4);
        format!("{} {} {}", fields.join(" "), self.halfmove_clock, self.fullmove_number)
    }
}

fn apply_position_command(game: &mut Game, tokens: &[&str]) {
    let (fen_tokens, rest) = if tokens[0] == "startpos" {
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".split_whitespace().collect::<Vec<_>>(),
            &tokens[1..],
        )
    } else {
        assert_eq!(tokens[0], "fen");
        (tokens[1..7].to_vec(), &tokens[7..])
    };
    game.set_fen(&fen_tokens);
    if rest.first() == Some(&"moves") {
        for uci in &rest[1..] {
            game.push_uci(uci);
        }
    }
}

fn main() {
    // See the module docstring / examples/bench_mcts.rs for why this matters.
    #[cfg(feature = "tch")]
    tch::set_num_threads(1);

    let simulations: u32 = std::env::var("SIMULATIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let threads: u32 = std::env::var("THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);

    // `create_chess_mamba` is async only for wasm-bindgen's calling
    // convention; `init_backend` is a no-op on every backend this crate
    // supports (see model/mod.rs), so there's nothing to actually await --
    // a single manual poll is enough, no async runtime needed.
    let bot = pollster::block_on(chess_mamba_bot::create_chess_mamba());

    let mut game = Game::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line.expect("stdin readable");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();

        match tokens[0] {
            "uci" => {
                writeln!(stdout, "id name {ENGINE_NAME}").unwrap();
                writeln!(stdout, "id author {ENGINE_AUTHOR}").unwrap();
                writeln!(stdout, "uciok").unwrap();
            }
            "isready" => writeln!(stdout, "readyok").unwrap(),
            "ucinewgame" => game = Game::new(),
            "position" => apply_position_command(&mut game, &tokens[1..]),
            "go" => {
                let mv = bot.best_move_mcts(&game.fen(), simulations, threads);
                match mv {
                    Some(mv) => writeln!(stdout, "bestmove {mv}").unwrap(),
                    None => writeln!(stdout, "bestmove (none)").unwrap(),
                }
            }
            "quit" => return,
            _ => {}
        }
        stdout.flush().unwrap();
    }
}
