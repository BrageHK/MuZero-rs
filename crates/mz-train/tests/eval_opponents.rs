use mz_rs::env::Environment;
use mz_rs::env::othello::env::Othello;
use mz_rs::env::tictactoe::env::TicTacToe;
use mz_rs::eval::opponent::{BoardGame, Opponent, legal_actions, plain_negamax};

struct Tally {
    wins: usize,
    draws: usize,
    losses: usize,
}

impl Tally {
    fn games(&self) -> usize {
        self.wins + self.draws + self.losses
    }

    fn win_rate(&self) -> f32 {
        self.wins as f32 / self.games() as f32
    }

    fn score(&self) -> f32 {
        (self.wins as f32 + 0.5 * self.draws as f32) / self.games() as f32
    }
}

/// Plays `games` games, alternating who moves first. Result is from `first`'s view.
fn duel<E: BoardGame>(first: &Opponent, second: &Opponent, games: usize, seed: u64) -> Tally {
    let mut tally = Tally {
        wins: 0,
        draws: 0,
        losses: 0,
    };

    for game in 0..games {
        let mut rng = fastrand::Rng::with_seed(seed ^ game as u64);
        let mut env = E::default();
        // Half the games start with the other side, so neither gets a colour edge.
        let mut first_to_move = game % 2 == 0;
        let mut plies = 0;

        loop {
            let player = if first_to_move { first } else { second };
            let action = player.choose(&env, &mut rng);
            let result = env.step(action);
            plies += 1;

            if result.done || result.truncated {
                let value = if first_to_move {
                    result.reward as f32
                } else {
                    -(result.reward as f32)
                };
                if value > 0.0 {
                    tally.wins += 1;
                } else if value < 0.0 {
                    tally.losses += 1;
                } else {
                    tally.draws += 1;
                }
                break;
            }
            assert!(plies < E::MAX_STEPS, "game did not terminate");
            first_to_move = !first_to_move;
        }
    }

    tally
}

#[test]
fn othello_alpha_beta_crushes_random() {
    let ab = Opponent::AlphaBeta {
        depth: 3,
        epsilon: 0.0,
    };
    let tally = duel::<Othello>(&ab, &Opponent::Random, 200, 3);
    assert!(
        tally.win_rate() >= 0.99,
        "win rate {:.3} (W{} D{} L{})",
        tally.win_rate(),
        tally.wins,
        tally.draws,
        tally.losses
    );
}

#[test]
fn tictactoe_perfect_alpha_beta_never_loses_to_random() {
    let ab = Opponent::AlphaBeta {
        depth: 9,
        epsilon: 0.0,
    };
    let tally = duel::<TicTacToe>(&ab, &Opponent::Random, 200, 5);
    // Perfect play as the second player cannot force a win against every random
    // line, so a few draws are unavoidable. Never losing is the real property.
    assert_eq!(tally.losses, 0, "perfect play lost a game");
    assert!(
        tally.score() >= 0.95,
        "score {:.3} (W{} D{} L{})",
        tally.score(),
        tally.wins,
        tally.draws,
        tally.losses
    );
}

#[test]
fn deeper_othello_search_beats_shallower() {
    let deep = Opponent::AlphaBeta {
        depth: 3,
        epsilon: 0.0,
    };
    let shallow = Opponent::AlphaBeta {
        depth: 1,
        epsilon: 0.0,
    };
    let tally = duel::<Othello>(&deep, &shallow, 40, 7);
    assert!(
        tally.score() > 0.65,
        "score {:.3} (W{} D{} L{})",
        tally.score(),
        tally.wins,
        tally.draws,
        tally.losses
    );
}

#[test]
fn tictactoe_pruning_agrees_with_plain_negamax() {
    let ab = Opponent::AlphaBeta {
        depth: 9,
        epsilon: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(13);

    for _ in 0..200 {
        let mut env = TicTacToe::new();
        // Walk into a random position, checking the pruned root pick is optimal.
        loop {
            let legal = legal_actions(&env.legal_mask());
            if legal.is_empty() {
                break;
            }

            let reference = legal
                .iter()
                .map(|&action| {
                    let mut child = env;
                    let result = child.step(action);
                    if result.done || result.truncated {
                        result.reward as f32 * 100.0
                    } else {
                        -plain_negamax(&child, 8)
                    }
                })
                .fold(f32::NEG_INFINITY, f32::max);

            let chosen = ab.choose(&env, &mut rng);
            let mut child = env;
            let result = child.step(chosen);
            let chosen_value = if result.done || result.truncated {
                result.reward as f32 * 100.0
            } else {
                -plain_negamax(&child, 8)
            };
            assert!(
                (chosen_value - reference).abs() < 1e-4,
                "alpha-beta picked {chosen_value}, optimum is {reference}\n{env}"
            );

            if env.step(legal[rng.usize(..legal.len())]).done {
                break;
            }
        }
    }
}
