use crate::env::Environment;

/// Terminal outcomes are scaled past any heuristic score so a proven result
/// always dominates a static evaluation.
const WIN_SCORE: f32 = 100.0;

/// A two-player environment that can be searched by the benchmark opponents.
pub trait BoardGame: Environment<Action = usize> + Clone + Default + Send + Sync {
    /// Static evaluation from the side to move, roughly in `[-1, 1]`.
    fn heuristic(&self) -> f32;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Opponent {
    Random,
    AlphaBeta { depth: usize, epsilon: f32 },
}

impl Opponent {
    pub fn label(&self) -> String {
        match self {
            Opponent::Random => "Random".to_string(),
            Opponent::AlphaBeta { depth, epsilon } if *epsilon > 0.0 => {
                format!("AlphaBeta(d={depth}, e={epsilon})")
            }
            Opponent::AlphaBeta { depth, .. } => format!("AlphaBeta(d={depth})"),
        }
    }

    pub fn choose<E: BoardGame>(&self, env: &E, rng: &mut fastrand::Rng) -> usize {
        let legal = legal_actions(&env.legal_mask());
        assert!(
            !legal.is_empty(),
            "opponent asked to move with no legal action"
        );

        match self {
            Opponent::Random => legal[rng.usize(..legal.len())],
            Opponent::AlphaBeta { depth, epsilon } => {
                if *epsilon > 0.0 && rng.f32() < *epsilon {
                    return legal[rng.usize(..legal.len())];
                }
                alpha_beta_best(env, (*depth).max(1), rng)
            }
        }
    }
}

pub fn legal_actions(mask: &[bool]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter(|&(_, &l)| l)
        .map(|(a, _)| a)
        .collect()
}

struct Child<E> {
    action: usize,
    env: E,
    terminal: Option<f32>,
}

/// Value of this child for the parent's side to move, used only for ordering.
fn hint<E: BoardGame>(child: &Child<E>) -> f32 {
    match child.terminal {
        Some(value) => value,
        None => -child.env.heuristic(),
    }
}

fn expand<E: BoardGame>(env: &E) -> Vec<Child<E>> {
    legal_actions(&env.legal_mask())
        .into_iter()
        .map(|action| {
            let mut child = env.clone();
            let result = child.step(action);
            let terminal = if result.done || result.truncated {
                Some(result.reward as f32 * WIN_SCORE)
            } else {
                None
            };
            Child {
                action,
                env: child,
                terminal,
            }
        })
        .collect()
}

fn order<E: BoardGame>(children: &mut [Child<E>]) {
    children.sort_by(|a, b| hint(b).total_cmp(&hint(a)));
}

fn negamax<E: BoardGame>(env: &E, depth: usize, mut alpha: f32, beta: f32) -> f32 {
    if depth == 0 {
        return env.heuristic();
    }

    let mut children = expand(env);
    if children.is_empty() {
        return env.heuristic();
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
fn alpha_beta_best<E: BoardGame>(env: &E, depth: usize, rng: &mut fastrand::Rng) -> usize {
    let mut children = expand(env);
    assert!(!children.is_empty(), "alpha-beta root has no legal action");
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

/// Unpruned reference search, used by the tests to check the pruning is sound.
pub fn plain_negamax<E: BoardGame>(env: &E, depth: usize) -> f32 {
    if depth == 0 {
        return env.heuristic();
    }
    let children = expand(env);
    if children.is_empty() {
        return env.heuristic();
    }
    children
        .iter()
        .map(|child| match child.terminal {
            Some(value) => value,
            None => -plain_negamax(&child.env, depth - 1),
        })
        .fold(f32::NEG_INFINITY, f32::max)
}
