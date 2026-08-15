pub mod opponent;

use burn::tensor::{Tensor, backend::Backend};
use rayon::prelude::*;

use crate::env::othello::env::Othello;
use crate::env::tictactoe::env::TicTacToe;
use crate::eval::opponent::{BoardGame, Opponent, legal_actions};
use crate::mz_config::{EnvironmentName, EvalConfig, MuZeroConfig, RungConfig, default_ladder};
use crate::networks::MuZeroNets;
use crate::search::batched_search;

#[derive(Debug, Clone, Copy, Default)]
pub struct EvalResult {
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
}

impl EvalResult {
    pub fn games(&self) -> usize {
        self.wins + self.draws + self.losses
    }

    pub fn score(&self) -> f32 {
        let games = self.games();
        if games == 0 {
            return 0.5;
        }
        (self.wins as f32 + 0.5 * self.draws as f32) / games as f32
    }
}

#[derive(Debug, Clone)]
pub struct EvalReading {
    pub elo: f32,
    pub result: EvalResult,
    pub opponent: String,
}

pub struct Rung {
    pub opponent: Opponent,
    pub elo: f32,
}

impl Rung {
    fn from_config(config: &RungConfig) -> Self {
        match *config {
            RungConfig::Random { elo } => Rung {
                opponent: Opponent::Random,
                elo,
            },
            RungConfig::AlphaBeta {
                depth,
                epsilon,
                elo,
            } => Rung {
                opponent: Opponent::AlphaBeta { depth, epsilon },
                elo,
            },
        }
    }
}

/// A single Elo reading against a fixed anchor. The `+-1/(2n)` clamp keeps a
/// clean sweep finite instead of infinite.
pub fn elo_from_score(anchor: f32, score: f32, games: usize) -> f32 {
    let eps = 1.0 / (2.0 * games.max(1) as f32);
    let score = score.clamp(eps, 1.0 - eps);
    anchor + 400.0 * (score / (1.0 - score)).log10()
}

/// Benchmark opponents ordered by strength. The agent plays whichever rung it
/// currently sits on and moves up once it beats it convincingly, so the Elo
/// curve keeps resolving instead of saturating against a weak opponent.
pub struct EloLadder {
    rungs: Vec<Rung>,
    current: usize,
    next_eval: usize,
    conf: EvalConfig,
}

impl EloLadder {
    pub fn new(mz_conf: &MuZeroConfig) -> Self {
        let conf = mz_conf.eval();
        let rungs = conf
            .ladder
            .clone()
            .unwrap_or_else(|| default_ladder(&mz_conf.environment))
            .iter()
            .map(Rung::from_config)
            .collect();

        Self {
            rungs,
            current: 0,
            next_eval: 0,
            conf,
        }
    }

    pub fn enabled(&self) -> bool {
        self.conf.interval > 0 && !self.rungs.is_empty()
    }

    pub fn current(&self) -> usize {
        self.current
    }

    pub fn set_current(&mut self, rung: usize) {
        if rung < self.rungs.len() {
            self.current = rung;
        }
    }

    pub fn due(&self, training_step: usize) -> bool {
        self.enabled() && training_step >= self.next_eval
    }

    pub fn run<B: Backend, N: MuZeroNets<B>>(
        &mut self,
        mz_conf: &MuZeroConfig,
        agent: &N,
        device: &B::Device,
        training_step: usize,
    ) -> EvalReading {
        self.next_eval = training_step + self.conf.interval;

        let rung = &self.rungs[self.current];
        let eval_conf = eval_config(mz_conf, &self.conf);
        let result = match mz_conf.environment {
            EnvironmentName::TicTacToe => {
                eval_games::<B, N, TicTacToe>(&eval_conf, agent, device, &rung.opponent, &self.conf)
            }
            EnvironmentName::Othello => {
                eval_games::<B, N, Othello>(&eval_conf, agent, device, &rung.opponent, &self.conf)
            }
            _ => unreachable!("ladder is empty for single-player environments"),
        };

        let reading = EvalReading {
            elo: elo_from_score(rung.elo, result.score(), result.games()),
            result,
            opponent: rung.opponent.label(),
        };
        self.promote(result.score());
        reading
    }

    fn promote(&mut self, score: f32) {
        if score >= self.conf.promote_score && self.current + 1 < self.rungs.len() {
            self.current += 1;
        } else if score <= self.conf.demote_score && self.current > 0 {
            self.current -= 1;
        }
    }
}

/// Search config for evaluation: no root exploration, and optionally a deeper
/// search than training uses.
fn eval_config(mz_conf: &MuZeroConfig, conf: &EvalConfig) -> MuZeroConfig {
    let mut eval_conf = mz_conf.clone();
    if let Some(simulations) = conf.num_simulations {
        eval_conf.num_simulations = simulations;
    }
    if let Some(puct) = eval_conf.puct.as_mut() {
        puct.root_exploration_fraction = 0.0;
    }
    eval_conf
}

struct Game<E> {
    env: E,
    agent_to_move: bool,
    plies: usize,
    outcome: Option<f32>,
    rng: fastrand::Rng,
}

impl<E: BoardGame> Game<E> {
    fn live(&self) -> bool {
        self.outcome.is_none()
    }

    /// `reward` is relative to the side that just moved.
    fn record(&mut self, reward: f32, mover_was_agent: bool) {
        self.outcome = Some(if mover_was_agent { reward } else { -reward });
    }

    fn step(&mut self, action: usize) {
        let mover_was_agent = self.agent_to_move;
        let result = self.env.step(action);
        self.plies += 1;
        self.agent_to_move = !self.agent_to_move;

        if result.done || result.truncated {
            self.record(result.reward as f32, mover_was_agent);
        } else if self.plies >= E::MAX_STEPS {
            self.outcome = Some(0.0);
        }
    }
}

/// Plays `conf.games` games against `opponent`, all in lockstep so that every
/// agent move across the batch resolves in a single `batched_search`.
fn eval_games<B: Backend, N: MuZeroNets<B>, E: BoardGame>(
    mz_conf: &MuZeroConfig,
    agent: &N,
    device: &B::Device,
    opponent: &Opponent,
    conf: &EvalConfig,
) -> EvalResult {
    let mut games: Vec<Game<E>> = (0..conf.games)
        .map(|i| {
            // Games 2k and 2k+1 share an opening, played once from each side.
            let agent_first = i % 2 == 0;
            let mut rng = fastrand::Rng::with_seed(conf.seed ^ (i as u64 / 2));
            let mut env = E::default();
            let mut plies = 0;
            let opening = rng.usize(0..=conf.random_opening_plies);
            let mut finished = false;
            for _ in 0..opening {
                let legal = legal_actions(&env.legal_mask());
                if legal.is_empty() {
                    break;
                }
                let result = env.step(legal[rng.usize(..legal.len())]);
                plies += 1;
                if result.done || result.truncated {
                    finished = true;
                    break;
                }
            }
            Game {
                env,
                agent_to_move: agent_first == (plies % 2 == 0),
                plies,
                // A finished opening would bias the tally, so drop the game.
                outcome: if finished { Some(f32::NAN) } else { None },
                rng: fastrand::Rng::with_seed(conf.seed ^ 0x9E37_79B9 ^ i as u64),
            }
        })
        .collect();

    let dim = mz_conf.obs_dim;
    loop {
        games
            .par_iter_mut()
            .with_min_len(mz_conf.rayon_min_chunk_len)
            .filter(|game| game.live() && !game.agent_to_move)
            .for_each(|game| {
                let action = {
                    let Game { env, rng, .. } = game;
                    opponent.choose(env, rng)
                };
                game.step(action);
            });

        let active: Vec<usize> = games
            .iter()
            .enumerate()
            .filter(|(_, game)| game.live() && game.agent_to_move)
            .map(|(i, _)| i)
            .collect();
        if active.is_empty() {
            break;
        }

        let mut data = Vec::with_capacity(active.len() * dim);
        let mut masks = Vec::with_capacity(active.len());
        for &i in &active {
            data.extend(games[i].env.obs());
            masks.push(games[i].env.legal_mask());
        }
        let obs = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([active.len(), dim]);
        let results = batched_search(obs, Some(&masks), mz_conf, agent, 0.0, false);

        for (&i, result) in active.iter().zip(results.iter()) {
            games[i].step(result.best_action);
        }
    }

    let mut tally = EvalResult::default();
    for game in &games {
        match game.outcome {
            Some(outcome) if outcome > 0.0 => tally.wins += 1,
            Some(outcome) if outcome < 0.0 => tally.losses += 1,
            Some(0.0) => tally.draws += 1,
            _ => {}
        }
    }
    tally
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_score_returns_the_anchor() {
        assert_eq!(elo_from_score(1000.0, 0.5, 100), 1000.0);
    }

    #[test]
    fn a_76_percent_score_is_worth_about_200_elo() {
        let elo = elo_from_score(0.0, 0.76, 1000);
        assert!((elo - 200.0).abs() < 5.0, "got {elo}");
    }

    #[test]
    fn a_clean_sweep_is_finite_and_grows_with_sample_size() {
        let few = elo_from_score(0.0, 1.0, 20);
        let many = elo_from_score(0.0, 1.0, 400);
        assert!(few.is_finite() && many.is_finite());
        assert!(many > few);
    }

    fn test_ladder() -> EloLadder {
        EloLadder {
            rungs: (0..3)
                .map(|i| Rung {
                    opponent: Opponent::Random,
                    elo: i as f32 * 100.0,
                })
                .collect(),
            current: 0,
            next_eval: 0,
            conf: EvalConfig::default(),
        }
    }

    #[test]
    fn ladder_promotes_and_demotes_within_bounds() {
        let mut ladder = test_ladder();
        for _ in 0..5 {
            ladder.promote(1.0);
        }
        assert_eq!(ladder.current, 2);

        for _ in 0..5 {
            ladder.promote(0.0);
        }
        assert_eq!(ladder.current, 0);
    }

    #[test]
    fn ladder_holds_between_thresholds() {
        let mut ladder = test_ladder();
        ladder.promote(1.0);
        ladder.promote(0.5);
        assert_eq!(ladder.current, 1);
    }

    #[test]
    fn score_counts_draws_as_half() {
        let result = EvalResult {
            wins: 10,
            draws: 10,
            losses: 20,
        };
        assert_eq!(result.games(), 40);
        assert_eq!(result.score(), 0.375);
    }
}
