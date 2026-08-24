//! Standalone driver for burn's TUI metrics renderer, for training loops
//! that don't go through burn's Learner.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::data::dataloader::Progress;
use burn::train::Interrupter;
use burn::train::metric::{
    MetricAttributes, MetricDefinition, MetricEntry, MetricId, NumericAttributes, NumericEntry,
    SerializedEntry,
};
use burn::train::renderer::{
    MetricState, MetricsRenderer, MetricsRendererTraining, ProgressType, TrainingProgress,
    tui::TuiMetricsRendererWrapper,
};

use crate::eval::EvalReading;
use crate::mz_config::{MuZeroConfig, SearchAlgorithm};

pub struct TrainingTui {
    renderer: Option<TuiMetricsRendererWrapper>,
    interrupter: Interrupter,
    started: Instant,
    last_print: Instant,
    last_sps: f64,
    last_tps: f64,
    last_loss: f32,
    total_steps: usize,
    avg_window: usize,
    rate_window: Duration,
    best_id: Option<MetricId>,
    avg_id: Option<MetricId>,
    elo_id: Option<MetricId>,
    opponent_id: Option<MetricId>,
    tau_id: Option<MetricId>,
    loss_id: Option<MetricId>,
    consistency_id: Option<MetricId>,
    best_reward: f32,
    recent_rewards: VecDeque<f32>,
    recent_lengths: VecDeque<usize>,
    rate_samples: VecDeque<(Instant, usize)>,
    train_rate_samples: VecDeque<(Instant, usize)>,
    games_finished: usize,
    env_steps: usize,
    train_steps: usize,
    buffer_states: usize,
    avg_game_length: f64,
    win_pct: f64,
    draw_pct: f64,
    loss_pct: f64,
    board_game: bool,
}

impl TrainingTui {
    pub fn new(mz_conf: &MuZeroConfig) -> Self {
        let interrupter = Interrupter::new();
        // MZ_HEADLESS swaps the TUI for one metrics line per second on stdout.
        let mut renderer = std::env::var_os("MZ_HEADLESS")
            .is_none()
            .then(|| TuiMetricsRendererWrapper::new(interrupter.clone(), None));

        let mut register = |name: &str, attributes: MetricAttributes| {
            let id = MetricId::new(Arc::new(name.to_string()));
            if let Some(renderer) = renderer.as_mut() {
                renderer.register_metric(MetricDefinition {
                    metric_id: id.clone(),
                    name: name.to_string(),
                    description: None,
                    attributes,
                });
            }
            Some(id)
        };
        let mut numeric = |name: &str, higher_is_better: bool| {
            register(
                name,
                MetricAttributes::Numeric(NumericAttributes {
                    unit: None,
                    higher_is_better,
                }),
            )
        };

        // Board games are scored by Elo against benchmark opponents; episode
        // reward is ~0 on average no matter how strong the agent is.
        let board_game = mz_conf.is_twoplayer;
        let (best_id, avg_id) = match board_game {
            true => (None, None),
            false => (
                numeric("Best Game Reward", true),
                numeric("Avg Game Reward", true),
            ),
        };
        let elo_id = match board_game {
            true => numeric("Elo", true),
            false => None,
        };
        // Gumbel picks the root action deterministically, so tau is meaningless there.
        let tau_id = match mz_conf.search_algorithm {
            SearchAlgorithm::Puct => numeric("Tau", true),
            SearchAlgorithm::Gumbel => None,
        };
        let loss_id = numeric("Loss", false);
        let consistency_id = numeric("Consistency Loss", false);
        let opponent_id = match board_game {
            true => register("Eval Opponent", MetricAttributes::None),
            false => None,
        };

        Self {
            renderer,
            interrupter,
            started: Instant::now(),
            last_print: Instant::now(),
            last_sps: 0.0,
            last_tps: 0.0,
            last_loss: f32::NAN,
            total_steps: mz_conf.training_steps,
            avg_window: mz_conf.avg_window,
            rate_window: Duration::from_secs_f32(mz_conf.rate_window_secs),
            best_id,
            avg_id,
            elo_id,
            opponent_id,
            tau_id,
            loss_id,
            consistency_id,
            best_reward: f32::NEG_INFINITY,
            recent_rewards: VecDeque::with_capacity(mz_conf.avg_window),
            recent_lengths: VecDeque::with_capacity(mz_conf.avg_window),
            rate_samples: VecDeque::new(),
            train_rate_samples: VecDeque::new(),
            games_finished: 0,
            env_steps: 0,
            train_steps: 0,
            buffer_states: 0,
            avg_game_length: 0.0,
            win_pct: 0.0,
            draw_pct: 0.0,
            loss_pct: 0.0,
            board_game,
        }
    }

    fn set(renderer: &mut Option<TuiMetricsRendererWrapper>, id: &Option<MetricId>, value: f64) {
        if let (Some(renderer), Some(id)) = (renderer.as_mut(), id) {
            renderer.update_train(numeric_state(id, value));
        }
    }

    fn set_text(
        renderer: &mut Option<TuiMetricsRendererWrapper>,
        id: &Option<MetricId>,
        value: &str,
    ) {
        if let (Some(renderer), Some(id)) = (renderer.as_mut(), id) {
            renderer.update_train(MetricState::Generic(MetricEntry::new(
                id.clone(),
                SerializedEntry::new(value.to_string(), value.to_string()),
            )));
        }
    }

    pub fn game_finished(&mut self, total_reward: f32, length: usize) {
        self.games_finished += 1;
        if total_reward > self.best_reward {
            self.best_reward = total_reward;
        }

        if self.recent_rewards.len() == self.avg_window {
            self.recent_rewards.pop_front();
        }
        self.recent_rewards.push_back(total_reward);

        if self.recent_lengths.len() == self.avg_window {
            self.recent_lengths.pop_front();
        }
        self.recent_lengths.push_back(length);

        let avg = self.recent_rewards.iter().map(|&r| r as f64).sum::<f64>()
            / self.recent_rewards.len() as f64;
        let avg_len = self.recent_lengths.iter().map(|&l| l as f64).sum::<f64>()
            / self.recent_lengths.len() as f64;
        let best_reward = self.best_reward as f64;
        Self::set(&mut self.renderer, &self.best_id, best_reward);
        Self::set(&mut self.renderer, &self.avg_id, avg);
        self.avg_game_length = avg_len;
    }

    pub fn set_eval(&mut self, reading: &EvalReading) {
        let games = reading.result.games().max(1) as f64;
        Self::set(&mut self.renderer, &self.elo_id, reading.elo as f64);
        self.win_pct = 100.0 * reading.result.wins as f64 / games;
        self.draw_pct = 100.0 * reading.result.draws as f64 / games;
        self.loss_pct = 100.0 * reading.result.losses as f64 / games;
        Self::set_text(&mut self.renderer, &self.opponent_id, &reading.opponent);
    }

    pub fn set_tau(&mut self, tau: f32) {
        Self::set(&mut self.renderer, &self.tau_id, tau as f64);
    }

    pub fn set_loss(&mut self, loss: f32) {
        self.last_loss = loss;
        Self::set(&mut self.renderer, &self.loss_id, loss as f64);
    }

    pub fn set_consistency_loss(&mut self, loss: f32) {
        Self::set(&mut self.renderer, &self.consistency_id, loss as f64);
    }

    pub fn set_buffer_states(&mut self, n: usize) {
        self.buffer_states = n;
    }

    pub fn games_finished(&self) -> usize {
        self.games_finished
    }

    pub fn env_steps(&self) -> usize {
        self.env_steps
    }

    /// Restores counters from a checkpoint so a resumed run's display picks up
    /// where the previous one left off instead of restarting from zero.
    pub fn seed_counts(&mut self, env_steps: usize, games_finished: usize, train_steps: usize) {
        self.env_steps = env_steps;
        self.games_finished = games_finished;
        self.train_steps = train_steps;
    }

    pub fn add_env_steps(&mut self, n: usize, backprop_active: bool) {
        self.env_steps += n;

        if !backprop_active {
            self.rate_samples.clear();
            return;
        }

        let now = Instant::now();
        self.rate_samples.push_back((now, self.env_steps));
        while self.rate_samples.len() > 2
            && now.duration_since(self.rate_samples[0].0) > self.rate_window
        {
            self.rate_samples.pop_front();
        }

        let (first_t, first_steps) = self.rate_samples[0];
        let elapsed = now.duration_since(first_t).as_secs_f64();
        if elapsed > 0.0 {
            let rate = (self.env_steps - first_steps) as f64 / elapsed;
            self.last_sps = rate;
        }
    }

    pub fn add_train_steps(&mut self, n: usize) {
        self.train_steps += n;

        let now = Instant::now();
        self.train_rate_samples.push_back((now, self.train_steps));
        while self.train_rate_samples.len() > 2
            && now.duration_since(self.train_rate_samples[0].0) > self.rate_window
        {
            self.train_rate_samples.pop_front();
        }

        let (first_t, first_steps) = self.train_rate_samples[0];
        let elapsed = now.duration_since(first_t).as_secs_f64();
        if elapsed > 0.0 {
            self.last_tps = (self.train_steps - first_steps) as f64 / elapsed;
        }
    }

    pub fn render(&mut self, step: usize) {
        let progress = TrainingProgress {
            progress: None,
            global_progress: Progress::new(step, self.total_steps),
            iteration: Some(step),
        };
        let mut counters = if self.board_game {
            vec![
                ProgressType::Value {
                    tag: format!(
                        "Games (avg len {}, buffer {})",
                        self.avg_game_length.round() as usize,
                        self.buffer_states
                    ),
                    value: self.games_finished,
                },
                ProgressType::Value {
                    tag: format!(
                        "Steps/s env {} train {} (env {}, train {})",
                        self.last_sps.round() as usize,
                        self.last_tps.round() as usize,
                        self.env_steps,
                        self.train_steps
                    ),
                    value: self.train_steps,
                },
            ]
        } else {
            vec![
                ProgressType::Value {
                    tag: format!(
                        "Games (avg len {}, buffer {})",
                        self.avg_game_length.round() as usize,
                        self.buffer_states
                    ),
                    value: self.games_finished,
                },
                ProgressType::Value {
                    tag: format!("Env steps/s (train {})", self.last_tps.round() as usize),
                    value: self.last_sps.round() as usize,
                },
                ProgressType::Value {
                    tag: format!("Train steps (env {})", self.env_steps),
                    value: self.train_steps,
                },
            ]
        };
        if self.board_game {
            counters.push(ProgressType::Value {
                tag: format!(
                    "Win % (draw {}/loss {})",
                    self.draw_pct.round() as usize,
                    self.loss_pct.round() as usize
                ),
                value: self.win_pct.round() as usize,
            });
        }

        match self.renderer.as_mut() {
            Some(renderer) => renderer.render_train(progress, counters),
            None => {
                if self.last_print.elapsed() >= Duration::from_secs(1) {
                    self.last_print = Instant::now();
                    println!(
                        "t={:.1} step={step} env_steps={} train_steps={} games={} buffer={} sps={:.0} tps={:.0} loss={:.4}",
                        self.started.elapsed().as_secs_f64(),
                        self.env_steps,
                        self.train_steps,
                        self.games_finished,
                        self.buffer_states,
                        self.last_sps,
                        self.last_tps,
                        self.last_loss,
                    );
                }
            }
        }
    }

    pub fn should_stop(&self) -> bool {
        self.interrupter.should_stop()
    }

    pub fn interrupter(&self) -> Interrupter {
        self.interrupter.clone()
    }

    pub fn close(mut self) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.manual_close();
        }
    }
}

fn numeric_state(id: &MetricId, value: f64) -> MetricState {
    let entry = NumericEntry::Value(value);
    MetricState::Numeric(
        MetricEntry::new(
            id.clone(),
            SerializedEntry::new(format!("{value:.3}"), entry.serialize()),
        ),
        entry,
    )
}
