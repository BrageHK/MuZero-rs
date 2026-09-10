//! Wave-batched PUCT MCTS through a *dynamic-batch* TorchScript module,
//! bypassing burn's onnx-generated `model::chess_mamba::Model<Be>` entirely.
//!
//! `chess_mamba_mcts`'s `evaluate` calls that burn model one leaf at a time
//! because it has to: `crates/mz-web/onnx/chess_mamba.onnx` was exported
//! with `--static-batch` (batch baked in as a literal `1` throughout the
//! traced graph -- see `build.rs`'s docstring and bee-chess's
//! `export_onnx.py`, which explains *why*: burn-onnx's ONNX->Rust codegen
//! can't compile the `Shape`/`Slice` ops a dynamic batch axis forces inside
//! `nn.MultiheadAttention`'s internal reshape). On a GPU backend, a batch=1
//! forward pass is dominated by kernel-launch/dispatch overhead rather than
//! actual compute, so more CPU threads hammering it with more batch=1 calls
//! doesn't move nodes/sec at all (measured: `bench_mcts_tch_gpu`, flat
//! ~330 nodes/sec from 1 to 18 threads on an RX 7900 XTX over ROCm) --
//! exactly the failure mode bee-chess's `mamba_mcts_native` README describes
//! ("batch=1 was consistently *slower* than CPU on every GPU backend
//! tried").
//!
//! That limitation is burn-onnx's codegen, not torch's: `nn.MultiheadAttention`
//! traces to a dynamic batch dim just fine via plain `torch.jit.trace` (see
//! this repo's export note below). So this module loads a *separately*
//! traced TorchScript module (`models/chess_mamba_dynamic.pt`, same
//! `checkpoints/ThisTimeForSure/best.pt` weights as the embedded burn model,
//! traced with `scan_backend="sequential"` -- matching `export_onnx.py`'s
//! own choice for the same tracing-safety reason -- via plain
//! `torch.jit.trace`, verified to match eager output exactly across batch
//! sizes 1/2/3/8/17/32/64 never seen while tracing) directly through raw
//! `tch::CModule`, and runs an lc0-style wave-batched search on top: each
//! wave selects `batch_size` leaves (staking virtual loss on each as it's
//! picked, so a wave doesn't just re-pick the same top leaf `batch_size`
//! times -- see `chess_mamba_mcts`'s module docstring for the same
//! technique under real thread concurrency), encodes all of them into ONE
//! tensor, and runs ONE forward pass for the whole wave -- ports
//! `mamba_mcts_native`'s design (see that crate's README/lib.rs docstring
//! for the original Python/PyO3 version and its measured numbers).
//!
//! Single-threaded by design, like `mamba_mcts_native`: batching alone is
//! what saturates the GPU, and a single wave loop keeps the tree mutation
//! lock-free (no `Mutex`, unlike `chess_mamba_mcts::search_parallel`).
//!
//! To regenerate `models/chess_mamba_dynamic.pt` after retraining:
//! ```python
//! # from bee-chess/training, with its own .venv active
//! import torch
//! from bee_training.chess_mamba.train import TrainConfig, build_model
//! ckpt = torch.load("checkpoints/ThisTimeForSure/best.pt", map_location="cpu", weights_only=False)
//! config = TrainConfig(**ckpt["config"]); config.scan_backend = "sequential"
//! model = build_model(config); model.load_state_dict(ckpt["model_state"]); model.eval()
//! dummy = torch.randn(5, 64, 20)  # batch size 5 is arbitrary -- see docstring above
//! traced = torch.jit.trace(model, (dummy,))
//! traced.save("chess_mamba_dynamic.pt")
//! ```

use core::str::FromStr;
use std::collections::HashMap;
use std::io::Cursor;

use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece};
use tch::{CModule, Device, IValue, Kind, Tensor};

use crate::chess_mamba_bot::{IN_DIM, encode_board};

const CP_CLIP: f32 = 1000.0;
const N_VALUE_BINS: usize = 128;
const BIN_WIDTH: f32 = 2.0 * CP_CLIP / N_VALUE_BINS as f32;

// Same lc0-inspired defaults as `chess_mamba_mcts` -- see that module for
// the caveat (approximate, not a byte-exact port of lc0's tuned constants).
const CPUCT_INIT: f32 = 1.745;
const CPUCT_BASE: f32 = 38739.0;
const CPUCT_FACTOR: f32 = 3.894;
const FPU_REDUCTION: f32 = 0.33;

#[derive(Debug, Clone, Copy)]
pub struct SearchConfig {
    pub simulations: usize,
    /// Pending leaves gathered per wave before the one NN forward call. See
    /// `mamba_mcts_native`'s README for measured sweet spots (32 on the
    /// hardware profiled there) -- this hardware's own sweet spot is
    /// whatever `bench_mcts_batched` finds, not assumed to transfer.
    pub batch_size: usize,
    pub virtual_loss: f32,
    pub cpuct_init: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub fpu_reduction: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 800,
            batch_size: 32,
            virtual_loss: 1.0,
            cpuct_init: CPUCT_INIT,
            cpuct_base: CPUCT_BASE,
            cpuct_factor: CPUCT_FACTOR,
            fpu_reduction: FPU_REDUCTION,
        }
    }
}

/// The traced TorchScript module plus the device it was loaded onto.
pub struct Model {
    module: CModule,
    device: Device,
}

impl Model {
    /// Loads the embedded dynamic-batch TorchScript module (see this
    /// module's docstring for how it's generated) onto `device` --
    /// `Device::Cuda(0)` is libtorch/tch's ROCm GPU handle on a ROCm build,
    /// same convention as `bench_mcts_tch_gpu`.
    pub fn load_embedded(device: Device) -> Self {
        let bytes: &[u8] = include_bytes!("../models/chess_mamba_dynamic.pt");
        let module = CModule::load_data_on_device(&mut Cursor::new(bytes), device)
            .expect("embedded chess_mamba_dynamic.pt failed to load");
        Self { module, device }
    }
}

struct Node {
    prior: f32,
    n: u32,
    w: f32,
    vln: u32,
    vlw: f32,
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

/// Same rule as `chess_mamba_mcts::advance_halfmove_clock` -- see there for
/// the en-passant caveat.
fn advance_halfmove_clock(board: &Board, mv: ChessMove, prev: u32) -> u32 {
    let is_pawn_move = board.piece_on(mv.get_source()) == Some(Piece::Pawn);
    let is_capture = board.piece_on(mv.get_dest()).is_some();
    if is_pawn_move || is_capture { 0 } else { prev + 1 }
}

fn cpuct(parent_n: u32, cfg: &SearchConfig) -> f32 {
    cfg.cpuct_init + cfg.cpuct_factor * ((parent_n as f32 + cfg.cpuct_base) / cfg.cpuct_base).ln()
}

fn visited_policy_mass(arena: &[Node], node: &Node) -> f32 {
    node.children.iter().filter(|&&(_, idx)| arena[idx].n > 0).map(|&(_, idx)| arena[idx].prior).sum()
}

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
    /// See `chess_mamba_mcts::Tree::collisions` -- same diagnostic, same
    /// caveat (ancestor overlap is normal, not counted; only a leaf staked
    /// twice within the same wave is).
    collisions: u64,
}

fn select_path(tree: &mut Tree, root_board: &Board, root_halfmove_clock: u32, cfg: &SearchConfig) -> (Vec<usize>, Board, u32) {
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

fn best_move_by_visits(tree: &Tree) -> Option<ChessMove> {
    tree.arena[0].children.iter().max_by_key(|&&(_, idx)| tree.arena[idx].n).map(|&(mv, _)| mv)
}

/// Runs one NN forward pass over every board in `boards`, returning
/// per-board (legal-move priors, value) -- the batched counterpart to
/// `chess_mamba_mcts::evaluate`. Boards with no legal moves must be filtered
/// out by the caller (terminal positions are handled without an NN call,
/// same as `chess_mamba_mcts::run_one_simulation`).
fn evaluate_batch(model: &Model, boards: &[(Board, u32)]) -> Vec<(Vec<(ChessMove, f32)>, f32)> {
    let n = boards.len();
    let mut planes_flat = vec![0f32; n * 64 * IN_DIM];
    let mut legal_moves_per: Vec<Vec<ChessMove>> = Vec::with_capacity(n);
    for (i, (board, halfmove_clock)) in boards.iter().enumerate() {
        let planes = encode_board(board, *halfmove_clock as f32);
        planes_flat[i * 64 * IN_DIM..(i + 1) * 64 * IN_DIM].copy_from_slice(&planes);
        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board)
            .filter(|mv| !mv.get_promotion().is_some_and(|p| p != Piece::Queen))
            .collect();
        legal_moves_per.push(legal_moves);
    }

    let input = Tensor::from_slice(&planes_flat)
        .to_device(model.device)
        .reshape([n as i64, 64, IN_DIM as i64]);
    let output = model.module.forward_is(&[IValue::Tensor(input)]).expect("chess_mamba_dynamic forward failed");
    let IValue::Tuple(mut outputs) = output else { panic!("expected a (policy, value) tuple output") };
    assert_eq!(outputs.len(), 2, "expected exactly 2 outputs");
    let IValue::Tensor(value_logits) = outputs.pop().unwrap() else { panic!("expected a value tensor") };
    let IValue::Tensor(policy_logits) = outputs.pop().unwrap() else { panic!("expected a policy tensor") };

    let policy_flat = policy_logits.reshape([n as i64, 64 * 64]).to_kind(Kind::Float).to_device(Device::Cpu);
    let value_flat = value_logits.to_kind(Kind::Float).to_device(Device::Cpu);
    let policy_rows: Vec<Vec<f32>> = (&policy_flat).try_into().expect("policy tensor -> Vec<Vec<f32>>");
    let value_rows: Vec<Vec<f32>> = (&value_flat).try_into().expect("value tensor -> Vec<Vec<f32>>");

    boards
        .iter()
        .zip(legal_moves_per)
        .zip(policy_rows)
        .zip(value_rows)
        .map(|(((_, legal_moves), policy), mut value_probs)| {
            if legal_moves.is_empty() {
                return (Vec::new(), 0.0);
            }
            let mut scores: Vec<f32> =
                legal_moves.iter().map(|mv| policy[mv.get_source().to_index() * 64 + mv.get_dest().to_index()]).collect();
            softmax_in_place(&mut scores);
            let priors: Vec<(ChessMove, f32)> = legal_moves.into_iter().zip(scores).collect();

            softmax_in_place(&mut value_probs);
            let expected_cp: f32 = value_probs.iter().enumerate().map(|(i, p)| p * bin_center(i)).sum();
            let value = (expected_cp / 400.0).tanh();
            (priors, value)
        })
        .collect()
}

/// `search_with_stats`'s result -- see `chess_mamba_mcts::SearchOutcome`.
/// `cache_hits`/`cache_misses` mirror `mamba_mcts_native::SearchStats` --
/// per lc0's own docs ("turning off lc0's cache makes NPS plummet"), and
/// measured here too: adding this cache closed most of the gap to
/// bee-chess's Python/PyO3 implementation (which had it from the start;
/// this Rust port didn't until benchmarked against it).
pub struct SearchOutcome {
    pub best_move: Option<String>,
    pub collisions: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

pub fn search_with_stats(model: &Model, fen: &str, cfg: &SearchConfig) -> SearchOutcome {
    let Some(root_board) = Board::from_str(fen).ok() else {
        return SearchOutcome { best_move: None, collisions: 0, cache_hits: 0, cache_misses: 0 };
    };
    let root_halfmove_clock: u32 = fen.split_whitespace().nth(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    let (root_priors, _root_value) = {
        let mut result = evaluate_batch(model, &[(root_board, root_halfmove_clock)]);
        result.pop().unwrap()
    };
    if root_priors.is_empty() {
        return SearchOutcome { best_move: None, collisions: 0, cache_hits: 0, cache_misses: 0 };
    }

    let mut tree =
        Tree { arena: vec![Node { prior: 0.0, n: 0, w: 0.0, vln: 0, vlw: 0.0, children: Vec::new() }], collisions: 0 };
    expand(&mut tree, 0, root_priors);

    // Keyed by `Board::get_hash()` (Zobrist -- doesn't fold in the halfmove
    // clock, same approximation `mamba_mcts_native::lib.rs` makes and
    // explains: two positions differing only in that one auxiliary scalar
    // are close enough to share a cache entry).
    let mut eval_cache: HashMap<u64, (Vec<(ChessMove, f32)>, f32)> = HashMap::new();
    let mut cache_hits = 0u64;
    let mut cache_misses = 0u64;

    let mut remaining = cfg.simulations;
    while remaining > 0 {
        let wave_size = remaining.min(cfg.batch_size);
        remaining -= wave_size;

        // Select `wave_size` leaves up front (staking virtual loss on each),
        // splitting them three ways: terminal (no NN call needed), a cache
        // hit (skip the NN call, expand+backup immediately), or a real miss
        // that needs a batched NN eval.
        let mut paths = Vec::with_capacity(wave_size);
        let mut nn_leaves: Vec<(Board, u32)> = Vec::new();
        let mut nn_path_indices = Vec::new();
        for _ in 0..wave_size {
            let (path, leaf_board, leaf_clock) = select_path(&mut tree, &root_board, root_halfmove_clock, cfg);
            let leaf_idx = *path.last().unwrap();
            if leaf_board.status() != BoardStatus::Ongoing {
                let value = if leaf_board.status() == BoardStatus::Checkmate { -1.0 } else { 0.0 };
                unstake_and_backup(&mut tree, &path, value, cfg.virtual_loss);
            } else if let Some((priors, value)) = eval_cache.get(&leaf_board.get_hash()) {
                cache_hits += 1;
                expand(&mut tree, leaf_idx, priors.clone());
                unstake_and_backup(&mut tree, &path, *value, cfg.virtual_loss);
            } else {
                cache_misses += 1;
                nn_path_indices.push((paths.len(), leaf_idx));
                paths.push((path, leaf_board.get_hash()));
                nn_leaves.push((leaf_board, leaf_clock));
            }
        }

        if !nn_leaves.is_empty() {
            let results = evaluate_batch(model, &nn_leaves);
            for ((path_idx, leaf_idx), (priors, value)) in nn_path_indices.into_iter().zip(results) {
                let (path, hash) = &paths[path_idx];
                eval_cache.insert(*hash, (priors.clone(), value));
                expand(&mut tree, leaf_idx, priors);
                unstake_and_backup(&mut tree, path, value, cfg.virtual_loss);
            }
        }
    }

    SearchOutcome {
        best_move: best_move_by_visits(&tree).map(|mv| mv.to_string()),
        collisions: tree.collisions,
        cache_hits,
        cache_misses,
    }
}

pub fn search(model: &Model, fen: &str, cfg: &SearchConfig) -> Option<String> {
    search_with_stats(model, fen, cfg).best_move
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plays_a_legal_move_from_the_start_position() {
        let model = Model::load_embedded(Device::Cpu);
        let cfg = SearchConfig { simulations: 16, batch_size: 8, ..SearchConfig::default() };
        let mv = search(&model, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &cfg)
            .expect("start position has legal moves");
        assert!(mv.len() == 4 || mv.len() == 5);
    }

    #[test]
    fn none_when_game_is_already_over() {
        let model = Model::load_embedded(Device::Cpu);
        let cfg = SearchConfig { simulations: 16, batch_size: 8, ..SearchConfig::default() };
        assert!(search(&model, "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", &cfg).is_none());
    }
}
