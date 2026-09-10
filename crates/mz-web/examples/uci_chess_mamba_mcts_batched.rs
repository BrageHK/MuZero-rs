//! UCI-speaking wrapper around `chess_mamba_mcts_batched::search_with_stats`
//! -- the wave-batched, dynamic-batch-TorchScript MCTS (~4800-5200
//! nodes/sec on an RX 7900 XTX over ROCm, vs. ~330 for the batch=1
//! `uci_chess_mamba_mcts` path; see that module's docstring for why) --
//! so it can be pointed at from a UCI GUI or a lichess-bot config, same
//! protocol as `uci_chess_mamba_mcts` and bee-chess's `play_mcts.py`.
//!
//! Falls back to CPU if no CUDA/ROCm device is available (e.g. a laptop
//! with no GPU); the dynamic-batch module works there too, just slower --
//! see `bench_mcts_batched_gpu`'s CPU comparison if you want numbers.
//!
//! Run with:
//!   cargo run --release -p mz-web --example uci_chess_mamba_mcts_batched \
//!       --features tch --no-default-features
//! Configure via UCI `setoption` (a GUI/lichess-bot does this for you) or
//! env vars for a quick manual run: SIMULATIONS (default 800), BATCH_SIZE
//! (default 64 -- see `bench_mcts_batched_gpu`'s README note: nodes/sec
//! peaks around 96-128, but the collision rate -- redundant same-leaf
//! re-evaluation within a wave -- climbs fast past ~64, wasting simulation
//! budget on search *quality*, not just throughput; 64 is a compromise, not
//! the raw-throughput-optimal value).

use std::io::{self, BufRead, Write};
use std::str::FromStr;

use chess::{Board, ChessMove, Piece};
use mz_web::chess_mamba_mcts_batched::{Model, SearchConfig, search_with_stats};
use tch::{Cuda, Device};

const ENGINE_NAME: &str = "Bee-Mamba-BatchedMCTS-tch";
const ENGINE_AUTHOR: &str = "bee-chess";

const DEFAULT_SIMULATIONS: u32 = 800;
const DEFAULT_BATCH_SIZE: u32 = 64;

/// Same FEN-tracking helper as `uci_chess_mamba_mcts::Game` -- `chess::Board`
/// doesn't track the halfmove clock itself, so it's carried alongside.
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

fn handle_setoption(rest: &[&str], simulations: &mut u32, batch_size: &mut u32) {
    if rest.len() < 4 || rest[0] != "name" || rest[2] != "value" {
        return;
    }
    let (name, value) = (rest[1], rest[3]);
    match name {
        "Simulations" => {
            if let Ok(v) = value.parse() {
                *simulations = v;
            }
        }
        "BatchSize" => {
            if let Ok(v) = value.parse() {
                *batch_size = v;
            }
        }
        _ => {}
    }
}

fn default_device() -> Device {
    if Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu }
}

fn main() {
    tch::set_num_threads(1);

    let device = default_device();
    eprintln!("[uci_chess_mamba_mcts_batched] loading model on {device:?}");
    let model = Model::load_embedded(device);

    let mut simulations: u32 =
        std::env::var("SIMULATIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_SIMULATIONS);
    let mut batch_size: u32 =
        std::env::var("BATCH_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_BATCH_SIZE);

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
                writeln!(stdout, "option name Simulations type spin default {DEFAULT_SIMULATIONS} min 1 max 100000")
                    .unwrap();
                writeln!(stdout, "option name BatchSize type spin default {DEFAULT_BATCH_SIZE} min 1 max 512").unwrap();
                writeln!(stdout, "uciok").unwrap();
            }
            "isready" => writeln!(stdout, "readyok").unwrap(),
            "ucinewgame" => game = Game::new(),
            "setoption" => handle_setoption(&tokens[1..], &mut simulations, &mut batch_size),
            "position" => apply_position_command(&mut game, &tokens[1..]),
            "go" => {
                let cfg = SearchConfig {
                    simulations: simulations as usize,
                    batch_size: batch_size as usize,
                    ..SearchConfig::default()
                };
                let outcome = search_with_stats(&model, &game.fen(), &cfg);
                match outcome.best_move {
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
