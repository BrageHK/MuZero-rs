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

use crate::mz_config::MuZeroConfig;

pub struct TrainingTui {
    renderer: TuiMetricsRendererWrapper,
    interrupter: Interrupter,
    total_steps: usize,
    avg_window: usize,
    rate_window: Duration,
    best_id: MetricId,
    avg_id: MetricId,
    sps_id: MetricId,
    best_reward: f32,
    recent_rewards: VecDeque<f32>,
    rate_samples: VecDeque<(Instant, usize)>,
    games_finished: usize,
    env_steps: usize,
    train_steps: usize,
}

impl TrainingTui {
    pub fn new(mz_conf: &MuZeroConfig) -> Self {
        let interrupter = Interrupter::new();
        let mut renderer = TuiMetricsRendererWrapper::new(interrupter.clone(), None);

        let mut register = |name: &str| {
            let id = MetricId::new(Arc::new(name.to_string()));
            renderer.register_metric(MetricDefinition {
                metric_id: id.clone(),
                name: name.to_string(),
                description: None,
                attributes: MetricAttributes::Numeric(NumericAttributes {
                    unit: None,
                    higher_is_better: true,
                }),
            });
            id
        };
        let best_id = register("Best Game Reward");
        let avg_id = register("Avg Game Reward");
        let sps_id = register("Env Steps / sec");

        Self {
            renderer,
            interrupter,
            total_steps: mz_conf.total_steps,
            avg_window: mz_conf.avg_window,
            rate_window: Duration::from_secs_f32(mz_conf.rate_window_secs),
            best_id,
            avg_id,
            sps_id,
            best_reward: f32::NEG_INFINITY,
            recent_rewards: VecDeque::with_capacity(mz_conf.avg_window),
            rate_samples: VecDeque::new(),
            games_finished: 0,
            env_steps: 0,
            train_steps: 0,
        }
    }

    pub fn game_finished(&mut self, total_reward: f32) {
        self.games_finished += 1;
        if total_reward > self.best_reward {
            self.best_reward = total_reward;
        }

        if self.recent_rewards.len() == self.avg_window {
            self.recent_rewards.pop_front();
        }
        self.recent_rewards.push_back(total_reward);

        let avg = self.recent_rewards.iter().map(|&r| r as f64).sum::<f64>()
            / self.recent_rewards.len() as f64;
        let best = numeric_state(&self.best_id, self.best_reward as f64);
        let avg = numeric_state(&self.avg_id, avg);
        self.renderer.update_train(best);
        self.renderer.update_train(avg);
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
            let sps = numeric_state(&self.sps_id, rate);
            self.renderer.update_train(sps);
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
        self.renderer.render_train(
            progress,
            vec![
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
            ],
        );
    }

    pub fn should_stop(&self) -> bool {
        self.interrupter.should_stop()
    }

    pub fn close(mut self) {
        self.renderer.manual_close();
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
