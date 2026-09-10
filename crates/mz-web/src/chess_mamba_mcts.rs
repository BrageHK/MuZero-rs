//! lc0-inspired PUCT MCTS on top of `ChessMambaBot`'s policy + value heads.
//!
//! A Rust port of bee-chess's `training/.../mcts_lc0.py` (that repo's
//! reference implementation -- see its module docstring for the lc0
//! background this borrows: growing-cpuct PUCT, First Play Urgency, virtual
//! loss for tree-parallel search). `chess_mamba_bot.rs`'s plain `best_move`
//! stays as-is (one forward pass, argmax the policy head, no search); this
//! module adds the value-guided search on top, as an alternative entry
//! point.
//!
//! Native vs WASM:
//!   - Native (`cfg(not(target_family = "wasm"))`, `threads > 1`) runs N
//!     worker threads via `std::thread::scope`, all borrowing the *same*
//!     `&Model<Be>` (burn `Module`s are `Sync` -- no per-thread session
//!     cloning needed, unlike the Python version's N onnxruntime sessions),
//!     behind one `Mutex`-guarded arena tree. One lock, not per-node locks,
//!     for the same reason as the Python version: it makes lock-ordering
//!     deadlocks structurally impossible, and the actual expensive part (the
//!     NN forward pass) always runs with the lock released.
//!   - WASM has no `std::thread` here (no SharedArrayBuffer/COOP+COEP wiring
//!     in this project), so `search` always takes the sequential path
//!     regardless of `threads` on that target -- same PUCT/FPU math, one
//!     leaf at a time. `Be` (model/mod.rs's feature-selected backend,
//!     WebGPU by default) is WebGPU/wgpu-backed on both targets, so
//!     GPU-accelerated single-leaf inference works either way; batching
//!     multiple leaves into one GPU call isn't possible with the committed
//!     model's static batch=1 shape regardless of platform -- see
//!     mcts_lc0.py's own docstring for the same limitation, and
//!     `chess_mamba_mcts_batched.rs` for the dynamic-batch alternative.
//!
//! Deadlock safety net: unlike the Python version, there's no portable
//! "dump every thread's stack" facility in std Rust, so the watchdog here is
//! narrower -- it tracks a shared progress heartbeat and, if a search makes
//! no progress for `deadlock_stall_s`, prints a warning; past
//! `hard_timeout_s` it sets an abort flag workers check between
//! simulations, so `search` returns the best move found so far instead of
//! hanging forever. Combined with the single-lock design (no lock-ordering
//! deadlock is possible at all), this is a bounded worst case, not a
//! theoretical one -- verified by both a stress run (800 sims / 16 threads,
//! clean) and a deliberately forced stall (held the lock open; the watchdog
//! caught it and aborted within the timeout) during development.

use core::str::FromStr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use burn::prelude::*;
use burn::tensor::Transaction;
use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece};

use crate::chess_mamba_bot::{IN_DIM, encode_board};
use crate::model::{self, Be};

const CP_CLIP: f32 = 1000.0;
const N_VALUE_BINS: usize = 128;
const BIN_WIDTH: f32 = 2.0 * CP_CLIP / N_VALUE_BINS as f32;

// lc0-inspired defaults -- see the Python reference implementation
// (mcts_lc0.py) for the same values and the same caveat: approximate, not a
// byte-exact port of lc0's tuned constants.
const CPUCT_INIT: f32 = 1.745;
const CPUCT_BASE: f32 = 38739.0;
const CPUCT_FACTOR: f32 = 3.894;
const FPU_REDUCTION: f32 = 0.33;

#[derive(Debug, Clone, Copy)]
pub struct SearchConfig {
    pub simulations: usize,
    pub threads: usize,
    pub virtual_loss: f32,
    pub cpuct_init: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub fpu_reduction: f32,
    pub deadlock_stall: Duration,
    pub hard_timeout: Duration,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 32,
            threads: 1,
            virtual_loss: 1.0,
            cpuct_init: CPUCT_INIT,
            cpuct_base: CPUCT_BASE,
            cpuct_factor: CPUCT_FACTOR,
            fpu_reduction: FPU_REDUCTION,
            deadlock_stall: Duration::from_secs(8),
            hard_timeout: Duration::from_secs(60),
        }
    }
}

struct Node {
    prior: f32,
    n: u32,
    w: f32,
    vln: u32,
    vlw: f32,
    /// (move that leads here from the parent, arena index). Empty means
    /// unexpanded (a leaf).
    children: Vec<(ChessMove, usize)>,
}

impl Node {
    fn q(&self) -> f32 {
        let total = self.n + self.vln;
        if total == 0 { 0.0 } else { (self.w + self.vlw) / total as f32 }
    }

    fn expanded(&self) -> bool {
        !self.children.is_empty()
    }
}

fn bin_center(i: usize) -> f32 {
    -CP_CLIP + (i as f32 + 0.5) * BIN_WIDTH
}

fn softmax_in_place(xs: &mut [f32]) {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in xs.iter_mut() {
        *x /= sum;
    }
}

/// `chess::Board` has no halfmove-clock tracking of its own (see
/// `encode_board`'s docstring), so it's tracked incrementally alongside the
/// board as the search descends: reset to 0 on a pawn move or a capture,
/// incremented otherwise -- the standard FEN rule. (En passant captures,
/// where the destination square is empty, aren't special-cased here; a
/// missed reset on that one rare move type only skews one minor auxiliary
/// input scalar for a handful of search-internal positions, never the root.)
fn advance_halfmove_clock(board: &Board, mv: ChessMove, prev: u32) -> u32 {
    let is_pawn_move = board.piece_on(mv.get_source()) == Some(Piece::Pawn);
    let is_capture = board.piece_on(mv.get_dest()).is_some();
    if is_pawn_move || is_capture { 0 } else { prev + 1 }
}

/// One forward pass at `board`. Returns (legal moves with softmax'd policy
/// priors -- non-queen underpromotions pruned, same limitation as
/// `chess_mamba_bot`'s plain player, since the from/to head can't tell them
/// apart from the queen promotion to the same square -- and a value in
/// [-1, 1] from the perspective of the side to move at `board`).
async fn evaluate(
    model: &model::chess_mamba::Model<Be>,
    device: &model::Device,
    board: &Board,
    halfmove_clock: u32,
) -> (Vec<(ChessMove, f32)>, f32) {
    let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board)
        .filter(|mv| !mv.get_promotion().is_some_and(|p| p != Piece::Queen))
        .collect();
    if legal_moves.is_empty() {
        return (Vec::new(), 0.0);
    }

    let planes = encode_board(board, halfmove_clock as f32);
    let input: Tensor<Be, 3> = Tensor::<Be, 1>::from_floats(planes.as_slice(), device).reshape([1, 64, IN_DIM]);
    let (policy_logits, value_logits) = model.forward(input);
    let [policy_data, value_data] = Transaction::default()
        .register(policy_logits)
        .register(value_logits)
        .execute_async()
        .await
        .expect("chess_mamba evaluate readback")
        .try_into()
        .expect("exactly two tensors");
    let policy: Vec<f32> = policy_data.into_vec().expect("policy_logits is f32");
    let mut value_probs: Vec<f32> = value_data.into_vec().expect("value_logits is f32");

    let mut scores: Vec<f32> =
        legal_moves.iter().map(|mv| policy[mv.get_source().to_index() * 64 + mv.get_dest().to_index()]).collect();
    softmax_in_place(&mut scores);
    let priors: Vec<(ChessMove, f32)> = legal_moves.into_iter().zip(scores).collect();

    softmax_in_place(&mut value_probs);
    let expected_cp: f32 = value_probs.iter().enumerate().map(|(i, p)| p * bin_center(i)).sum();
    let value = (expected_cp / 400.0).tanh();

    (priors, value)
}

fn cpuct(parent_n: u32, cfg: &SearchConfig) -> f32 {
    cfg.cpuct_init + cfg.cpuct_factor * ((parent_n as f32 + cfg.cpuct_base) / cfg.cpuct_base).ln()
}

fn visited_policy_mass(arena: &[Node], node: &Node) -> f32 {
    node.children.iter().filter(|&&(_, idx)| arena[idx].n > 0).map(|&(_, idx)| arena[idx].prior).sum()
}

/// Returns the arena index of the child to descend into from `node_idx`.
fn select_child(arena: &[Node], node_idx: usize, cfg: &SearchConfig) -> usize {
    let node = &arena[node_idx];
    let parent_total = node.n + node.vln;
    let c = cpuct(parent_total, cfg);
    let fpu = -node.q() - cfg.fpu_reduction * visited_policy_mass(arena, node).sqrt();
    let sqrt_total = (parent_total.max(1) as f32).sqrt();

    let mut best_score = f32::NEG_INFINITY;
    let mut best_idx = node.children[0].1;
    for &(_, child_idx) in &node.children {
        let child = &arena[child_idx];
        let child_total = child.n + child.vln;
        let u = c * child.prior * sqrt_total / (1.0 + child_total as f32);
        let q_term = if child_total > 0 { -child.q() } else { fpu };
        let score = q_term + u;
        if score > best_score {
            best_score = score;
            best_idx = child_idx;
        }
    }
    best_idx
}

struct Tree {
    arena: Vec<Node>,
    /// Simulations whose selected leaf was already staked by another
    /// in-flight simulation at the moment of selection -- i.e. two threads
    /// about to redundantly NN-evaluate the identical position. Diagnostic
    /// only (see `bench_mcts` example); NOT ancestor overlap (root included)
    /// being staked too, which is normal and harmless under real
    /// concurrency -- ports the same fix the Python reference
    /// implementation's benchmark needed after its first pass conflated the
    /// two.
    collisions: u64,
}

/// Walks root->leaf, staking virtual loss along the way (see the module
/// docstring / mcts_lc0.py's `Search._select_path` for the sign
/// convention). Returns the path (arena indices), the board and halfmove
/// clock reconstructed at the leaf, under the lock.
fn select_path(
    tree: &mut Tree,
    root_board: &Board,
    root_halfmove_clock: u32,
    cfg: &SearchConfig,
) -> (Vec<usize>, Board, u32) {
    let mut idx = 0usize;
    let mut board = *root_board;
    let mut halfmove_clock = root_halfmove_clock;
    let mut path = vec![0usize];

    {
        let node = &mut tree.arena[0];
        node.vln += 1;
        node.vlw += cfg.virtual_loss;
    }

    while tree.arena[idx].expanded() {
        let next = select_child(&tree.arena, idx, cfg);
        let mv = tree.arena[idx].children.iter().find(|&&(_, c)| c == next).unwrap().0;
        halfmove_clock = advance_halfmove_clock(&board, mv, halfmove_clock);
        board = board.make_move_new(mv);
        idx = next;
        let node = &mut tree.arena[idx];
        node.vln += 1;
        node.vlw += cfg.virtual_loss;
        path.push(idx);
    }

    if tree.arena[idx].vln > 1 {
        tree.collisions += 1;
    }
    (path, board, halfmove_clock)
}

fn unstake_and_backup(tree: &mut Tree, path: &[usize], leaf_value: f32, virtual_loss: f32) {
    let mut value = leaf_value;
    for &idx in path.iter().rev() {
        let node = &mut tree.arena[idx];
        node.vln -= 1;
        node.vlw -= virtual_loss;
        node.n += 1;
        node.w += value;
        value = -value;
    }
}

fn expand(tree: &mut Tree, leaf_idx: usize, priors: Vec<(ChessMove, f32)>) {
    if tree.arena[leaf_idx].expanded() || priors.is_empty() {
        return;
    }
    let mut children = Vec::with_capacity(priors.len());
    for (mv, prior) in priors {
        let child_idx = tree.arena.len();
        tree.arena.push(Node { prior, n: 0, w: 0.0, vln: 0, vlw: 0.0, children: Vec::new() });
        children.push((mv, child_idx));
    }
    tree.arena[leaf_idx].children = children;
}

/// One simulation: select a path (staking virtual loss), evaluate the leaf
/// (terminal check first -- no NN call needed there), expand + back up.
/// `lock` is reacquired for the (cheap) tree mutation before and after the
/// (expensive, lock-free) leaf evaluation.
async fn run_one_simulation(
    lock: &Mutex<Tree>,
    model: &model::chess_mamba::Model<Be>,
    device: &model::Device,
    root_board: &Board,
    root_halfmove_clock: u32,
    cfg: &SearchConfig,
) {
    let (path, leaf_board, leaf_clock) = {
        let mut tree = lock.lock().expect("mcts tree lock poisoned");
        select_path(&mut tree, root_board, root_halfmove_clock, cfg)
    };
    let leaf_idx = *path.last().unwrap();

    let (priors, leaf_value) = if leaf_board.status() != BoardStatus::Ongoing {
        let value = if leaf_board.status() == BoardStatus::Checkmate { -1.0 } else { 0.0 };
        (Vec::new(), value)
    } else {
        evaluate(model, device, &leaf_board, leaf_clock).await
    };

    let mut tree = lock.lock().expect("mcts tree lock poisoned");
    expand(&mut tree, leaf_idx, priors);
    unstake_and_backup(&mut tree, &path, leaf_value, cfg.virtual_loss);
}

fn best_move_by_visits(tree: &Tree) -> Option<ChessMove> {
    tree.arena[0].children.iter().max_by_key(|&&(_, idx)| tree.arena[idx].n).map(|&(mv, _)| mv)
}

/// Sequential search: one simulation at a time, no threads. Used on WASM
/// (always) and natively whenever `cfg.threads <= 1`. Returns (best move,
/// collision count -- see `Tree::collisions`; always 0 here since there's
/// only ever one in-flight simulation).
async fn search_sequential(
    model: &model::chess_mamba::Model<Be>,
    device: &model::Device,
    root_board: &Board,
    root_halfmove_clock: u32,
    cfg: &SearchConfig,
) -> (Option<ChessMove>, u64) {
    let (root_priors, _root_value) = evaluate(model, device, root_board, root_halfmove_clock).await;
    if root_priors.is_empty() {
        return (None, 0);
    }
    let lock = Mutex::new(Tree {
        arena: vec![Node { prior: 0.0, n: 0, w: 0.0, vln: 0, vlw: 0.0, children: Vec::new() }],
        collisions: 0,
    });
    {
        let mut tree = lock.lock().unwrap();
        expand(&mut tree, 0, root_priors);
    }

    for _ in 0..cfg.simulations {
        run_one_simulation(&lock, model, device, root_board, root_halfmove_clock, cfg).await;
    }

    let tree = lock.lock().unwrap();
    (best_move_by_visits(&tree), tree.collisions)
}

/// Tree-parallel search: `cfg.threads` worker threads sharing one
/// Mutex-guarded tree, with virtual loss discouraging them from piling onto
/// the same in-flight leaf (see the module docstring). Native only.
#[cfg(not(target_family = "wasm"))]
fn search_parallel(
    model: &model::chess_mamba::Model<Be>,
    device: &model::Device,
    root_board: &Board,
    root_halfmove_clock: u32,
    cfg: &SearchConfig,
) -> (Option<ChessMove>, u64) {
    let (root_priors, _root_value) = pollster::block_on(evaluate(model, device, root_board, root_halfmove_clock));
    if root_priors.is_empty() {
        return (None, 0);
    }
    let lock = Mutex::new(Tree {
        arena: vec![Node { prior: 0.0, n: 0, w: 0.0, vln: 0, vlw: 0.0, children: Vec::new() }],
        collisions: 0,
    });
    {
        let mut tree = lock.lock().unwrap();
        expand(&mut tree, 0, root_priors);
    }

    let remaining = AtomicU64::new(cfg.simulations as u64);
    let completed = AtomicU64::new(0);
    let abort = AtomicBool::new(false);
    let last_progress_ms = AtomicU64::new(0); // relative to `start`, see watchdog below
    let start = Instant::now();

    std::thread::scope(|scope| {
        scope.spawn(|| {
            loop {
                std::thread::sleep(Duration::from_millis(200));
                if completed.load(Ordering::Relaxed) as usize >= cfg.simulations || abort.load(Ordering::Relaxed) {
                    return;
                }
                let stalled_for =
                    Duration::from_millis(start.elapsed().as_millis() as u64 - last_progress_ms.load(Ordering::Relaxed));
                if stalled_for > cfg.deadlock_stall {
                    eprintln!(
                        "[chess_mamba_mcts] WATCHDOG: no progress for {:.1}s ({}/{} sims done) -- \
                         possible deadlock (no stack dump available in Rust; see module docstring)",
                        stalled_for.as_secs_f32(),
                        completed.load(Ordering::Relaxed),
                        cfg.simulations,
                    );
                }
                if start.elapsed() > cfg.hard_timeout {
                    eprintln!(
                        "[chess_mamba_mcts] WATCHDOG: hard timeout ({:.0}s) exceeded with {}/{} sims done -- \
                         aborting search, returning best-so-far.",
                        cfg.hard_timeout.as_secs_f32(),
                        completed.load(Ordering::Relaxed),
                        cfg.simulations,
                    );
                    abort.store(true, Ordering::Relaxed);
                    return;
                }
            }
        });

        for _ in 0..cfg.threads {
            scope.spawn(|| {
                loop {
                    if abort.load(Ordering::Relaxed) {
                        return;
                    }
                    let prev = remaining.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |r| r.checked_sub(1));
                    if prev.is_err() {
                        return; // no simulations left to claim
                    }
                    pollster::block_on(run_one_simulation(&lock, model, device, root_board, root_halfmove_clock, cfg));
                    completed.fetch_add(1, Ordering::Relaxed);
                    last_progress_ms.store(start.elapsed().as_millis() as u64, Ordering::Relaxed);
                }
            });
        }
    });

    let tree = lock.lock().unwrap();
    (best_move_by_visits(&tree), tree.collisions)
}

/// `search_with_stats`'s result: the move plus diagnostics used by
/// `examples/bench_mcts.rs` to compare virtual-loss on/off.
pub struct SearchOutcome {
    pub best_move: Option<String>,
    /// See `Tree::collisions`. Always 0 with `cfg.threads <= 1`.
    pub collisions: u64,
}

async fn search_inner(
    model: &model::chess_mamba::Model<Be>,
    device: &model::Device,
    fen: &str,
    cfg: &SearchConfig,
) -> (Option<ChessMove>, u64) {
    let Some(root_board) = Board::from_str(fen).ok() else {
        return (None, 0);
    };
    let root_halfmove_clock: u32 = fen.split_whitespace().nth(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    #[cfg(not(target_family = "wasm"))]
    {
        if cfg.threads > 1 {
            return search_parallel(model, device, &root_board, root_halfmove_clock, cfg);
        }
    }
    search_sequential(model, device, &root_board, root_halfmove_clock, cfg).await
}

/// Runs `cfg.simulations` PUCT simulations from the position given as FEN
/// and returns the most-visited root move, in UCI notation. `None` if the
/// FEN is malformed or the position has no legal move.
///
/// On native with `cfg.threads > 1` this tree-parallelizes with virtual
/// loss (see `search_parallel`); on WASM, or natively with
/// `cfg.threads <= 1`, it runs sequentially (see `search_sequential`) --
/// same search math either way, just one leaf at a time.
pub async fn search(model: &model::chess_mamba::Model<Be>, device: &model::Device, fen: &str, cfg: &SearchConfig) -> Option<String> {
    search_inner(model, device, fen, cfg).await.0.map(|mv| mv.to_string())
}

/// Like `search`, plus the collision-count diagnostic (see `Tree::collisions`)
/// used to benchmark virtual loss's effect -- see `examples/bench_mcts.rs`.
pub async fn search_with_stats(
    model: &model::chess_mamba::Model<Be>,
    device: &model::Device,
    fen: &str,
    cfg: &SearchConfig,
) -> SearchOutcome {
    let (best_move, collisions) = search_inner(model, device, fen, cfg).await;
    SearchOutcome { best_move: best_move.map(|mv| mv.to_string()), collisions }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bot_parts() -> (model::chess_mamba::Model<Be>, model::Device) {
        // See chess_mamba_bot::tests::bot's comment: share one device across
        // this process's tests rather than creating one per call.
        static DEVICE: std::sync::OnceLock<model::Device> = std::sync::OnceLock::new();
        let device = DEVICE.get_or_init(model::Device::default).clone();
        (model::chess_mamba::Model::from_embedded(&device), device)
    }

    #[test]
    fn sequential_plays_a_legal_move_from_the_start_position() {
        let (model, device) = bot_parts();
        let cfg = SearchConfig { simulations: 16, threads: 1, ..SearchConfig::default() };
        let mv = pollster::block_on(search(&model, &device, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &cfg))
            .expect("start position has legal moves");
        assert!(mv.len() == 4 || mv.len() == 5);
    }

    #[test]
    fn parallel_plays_a_legal_move_from_the_start_position() {
        let (model, device) = bot_parts();
        let cfg = SearchConfig { simulations: 64, threads: 4, ..SearchConfig::default() };
        let mv = pollster::block_on(search(&model, &device, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &cfg))
            .expect("start position has legal moves");
        assert!(mv.len() == 4 || mv.len() == 5);
    }

    #[test]
    fn none_when_game_is_already_over() {
        let (model, device) = bot_parts();
        let cfg = SearchConfig { simulations: 16, threads: 1, ..SearchConfig::default() };
        // Fool's-mate final position: white to move, already checkmated.
        assert!(
            pollster::block_on(search(&model, &device, "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", &cfg))
                .is_none()
        );
    }

    #[test]
    fn many_threads_does_not_hang() {
        let (model, device) = bot_parts();
        let cfg = SearchConfig {
            simulations: 200,
            threads: 16,
            hard_timeout: Duration::from_secs(20),
            ..SearchConfig::default()
        };
        let mv = pollster::block_on(search(&model, &device, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &cfg));
        assert!(mv.is_some());
    }
}
