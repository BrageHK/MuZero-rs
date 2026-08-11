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
    last_loss: f32,
    total_steps: usize,
    avg_window: usize,
    rate_window: Duration,
    best_id: Option<MetricId>,
    avg_id: Option<MetricId>,
    elo_id: Option<MetricId>,
    eval_score_id: Option<MetricId>,
    win_id: Option<MetricId>,
    draw_id: Option<MetricId>,
    loss_pct_id: Option<MetricId>,
    opponent_id: Option<MetricId>,
    sps_id: Option<MetricId>,
    tau_id: Option<MetricId>,
    loss_id: Option<MetricId>,
    consistency_id: Option<MetricId>,
    len_id: Option<MetricId>,
    buf_id: Option<MetricId>,
    best_reward: f32,
    recent_rewards: VecDeque<f32>,
    recent_lengths: VecDeque<usize>,
    rate_samples: VecDeque<(Instant, usize)>,
    games_finished: usize,
    env_steps: usize,
    train_steps: usize,
    buffer_states: usize,
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
        let (elo_id, eval_score_id, win_id, draw_id, loss_pct_id) = match board_game {
            true => (
                numeric("Elo", true),
                numeric("Eval Score", true),
                numeric("Win %", true),
                numeric("Draw %", true),
                numeric("Loss %", false),
            ),
            false => (None, None, None, None, None),
        };
        let sps_id = numeric("Env Steps / sec", true);
        // Gumbel picks the root action deterministically, so tau is meaningless there.
        let tau_id = match mz_conf.search_algorithm {
            SearchAlgorithm::Puct => numeric("Tau", true),
            SearchAlgorithm::Gumbel => None,
        };
        let loss_id = numeric("Loss", false);
        let consistency_id = numeric("Consistency Loss", false);
        let len_id = numeric("Avg Game Length", true);
        let buf_id = numeric("Buffer States", true);
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
            last_loss: f32::NAN,
            total_steps: mz_conf.training_steps,
            avg_window: mz_conf.avg_window,
            rate_window: Duration::from_secs_f32(mz_conf.rate_window_secs),
            best_id,
            avg_id,
            elo_id,
            eval_score_id,
            win_id,
            draw_id,
            loss_pct_id,
            opponent_id,
            sps_id,
            tau_id,
            loss_id,
            consistency_id,
            len_id,
            buf_id,
            best_reward: f32::NEG_INFINITY,
            recent_rewards: VecDeque::with_capacity(mz_conf.avg_window),
            recent_lengths: VecDeque::with_capacity(mz_conf.avg_window),
            rate_samples: VecDeque::new(),
            games_finished: 0,
            env_steps: 0,
            train_steps: 0,
            buffer_states: 0,
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
        Self::set(&mut self.renderer, &self.len_id, avg_len);
    }

    pub fn set_eval(&mut self, reading: &EvalReading) {
        let games = reading.result.games().max(1) as f64;
        Self::set(&mut self.renderer, &self.elo_id, reading.elo as f64);
        Self::set(&mut self.renderer, &self.eval_score_id, reading.result.score() as f64);
        Self::set(&mut self.renderer, &self.win_id, 100.0 * reading.result.wins as f64 / games);
        Self::set(&mut self.renderer, &self.draw_id, 100.0 * reading.result.draws as f64 / games);
        Self::set(
            &mut self.renderer,
            &self.loss_pct_id,
            100.0 * reading.result.losses as f64 / games,
        );
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
        Self::set(&mut self.renderer, &self.buf_id, n as f64);
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
            Self::set(&mut self.renderer, &self.sps_id, rate);
        }
    }

    pub fn add_train_steps(&mut self, n: usize) {
        self.train_steps += n;
    }

    pub fn render(&mut self, step: usize) {
        let progress = TrainingProgress {
            progress: None,
            global_progress: Progress::new(step, self.total_steps),
            iteration: Some(step),
        };
        let counters = vec![
            ProgressType::Value {
                tag: "Env steps".to_string(),
                value: self.env_steps,
            },
            ProgressType::Value {
                tag: "Train steps".to_string(),
                value: self.train_steps,
            },
            ProgressType::Value {
                tag: "Games".to_string(),
                value: self.games_finished,
            },
            ProgressType::Value {
                tag: "Buffer states".to_string(),
                value: self.buffer_states,
            },
        ];

        match self.renderer.as_mut() {
            Some(renderer) => renderer.render_train(progress, counters),
            None => {
                if self.last_print.elapsed() >= Duration::from_secs(1) {
                    self.last_print = Instant::now();
                    println!(
                        "t={:.1} step={step} env_steps={} train_steps={} games={} buffer={} sps={:.0} loss={:.4}",
                        self.started.elapsed().as_secs_f64(),
                        self.env_steps,
                        self.train_steps,
                        self.games_finished,
                        self.buffer_states,
                        self.last_sps,
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
