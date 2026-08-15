//! Benchmark opponents for the bot-battle mode, matching the eval ladder in
//! `mz-train`'s `crates/mz-train/src/eval/{mod,opponent}.rs` and
//! `env/othello/heuristic.rs`. Ported rather than shared because that crate
//! pulls in rayon and the training env traits, neither wasm-friendly.

use mz_core::othello::Othello;

use crate::rng::Rng;

/// Terminal outcomes are scaled past any heuristic score so a proven result
/// always dominates a static evaluation.
const WIN_SCORE: f32 = 100.0;

#[derive(Clone, Copy)]
pub enum Opponent {
    Random,
    AlphaBeta { depth: usize, epsilon: f32 },
}

impl Opponent {
    /// `depth` 0 selects the random opponent; any other depth selects
    /// alpha-beta search at that depth. Shallow depths keep a little epsilon
    /// noise so repeated games don't all follow one forced line.
    pub fn from_depth(depth: u32) -> Self {
        if depth == 0 {
            return Opponent::Random;
        }
        let epsilon = if depth <= 1 {
            0.05
        } else if depth <= 3 {
            0.02
        } else {
            0.0
        };
        Opponent::AlphaBeta {
            depth: depth as usize,
            epsilon,
        }
    }

    pub fn choose(&self, env: &Othello, rng: &mut Rng) -> usize {
        let legal: Vec<usize> = env.legal_actions().collect();
        debug_assert!(!legal.is_empty(), "opponent asked to move with no legal action");

        match self {
            Opponent::Random => legal[rng.below(legal.len())],
            Opponent::AlphaBeta { depth, epsilon } => {
                if *epsilon > 0.0 && rng.unit() < *epsilon {
                    return legal[rng.below(legal.len())];
                }
                alpha_beta_best(env, (*depth).max(1), rng)
            }
        }
    }
}

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

fn heuristic(env: &Othello) -> f32 {
    let state = env.state();
    let (own, opp) = (state.own, state.opp);

    let n_own = own.count_ones() as f32;
    let n_opp = opp.count_ones() as f32;
    let empties = 64 - own.count_ones() - opp.count_ones();

    let corners = ((own & CORNERS).count_ones() as f32 - (opp & CORNERS).count_ones() as f32) / 4.0;
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

    let m_own = mz_core::othello::moves(own, opp).count_ones() as f32;
    let m_opp = mz_core::othello::moves(opp, own).count_ones() as f32;
    let mobility = (m_own - m_opp) / (m_own + m_opp + 1.0);

    (0.25 * positional + 0.35 * corners + 0.30 * mobility + 0.10 * discs).clamp(-0.99, 0.99)
}

struct Child {
    action: usize,
    env: Othello,
    terminal: Option<f32>,
}

/// Value of this child for the parent's side to move, used only for ordering.
fn hint(child: &Child) -> f32 {
    match child.terminal {
        Some(value) => value,
        None => -heuristic(&child.env),
    }
}

fn expand(env: &Othello) -> Vec<Child> {
    env.legal_actions()
        .map(|action| {
            let mut child = *env;
            let outcome = child.apply(action);
            let terminal = outcome.done.then(|| outcome.reward as f32 * WIN_SCORE);
            Child {
                action,
                env: child,
                terminal,
            }
        })
        .collect()
}

fn order(children: &mut [Child]) {
    children.sort_by(|a, b| hint(b).total_cmp(&hint(a)));
}

fn negamax(env: &Othello, depth: usize, mut alpha: f32, beta: f32) -> f32 {
    if depth == 0 {
        return heuristic(env);
    }

    let mut children = expand(env);
    if children.is_empty() {
        return heuristic(env);
    }
    order(&mut children);

    let mut best = f32::NEG_INFINITY;
    for child in &children {
        let value = match child.terminal {
            Some(value) => value,
            None => -negamax(&child.env, depth - 1, -beta, -alpha),
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
/// games from the same position do not all follow one line.
fn alpha_beta_best(env: &Othello, depth: usize, rng: &mut Rng) -> usize {
    let mut children = expand(env);
    debug_assert!(!children.is_empty(), "alpha-beta root has no legal action");
    rng.shuffle(&mut children);
    order(&mut children);

    let mut best_action = children[0].action;
    let mut best = f32::NEG_INFINITY;
    for child in &children {
        let value = match child.terminal {
            Some(value) => value,
            None => -negamax(&child.env, depth - 1, f32::NEG_INFINITY, -best),
        };
        if value > best {
            best = value;
            best_action = child.action;
        }
    }
    best_action
}
