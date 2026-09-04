//! Runs the Elo ladder on its own backend device in a background thread so
//! evaluation never blocks self-play or training. Only one eval runs at a
//! time; `due` stays false while one is in flight, and its result is picked
//! up later with `poll`.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

use burn::tensor::backend::Backend;

use crate::eval::{EloLadder, EvalReading};
use crate::mz_config::MuZeroConfig;
use crate::networks::{MuZeroNets, nets_to_backend};
use crate::utils::{save_best_elo, save_best_model, save_eval_state};

struct EvalDone {
    ladder: EloLadder,
    reading: EvalReading,
}

pub struct BackgroundLadder<B: Backend> {
    ladder: Option<EloLadder>,
    best_elo: f32,
    ckpt_dir: String,
    device: B::Device,
    tx: Sender<EvalDone>,
    rx: Receiver<EvalDone>,
    running: bool,
}

impl<B: Backend> BackgroundLadder<B> {
    pub fn new(
        mz_conf: &MuZeroConfig,
        device: B::Device,
        initial_best_elo: f32,
        initial_rung: Option<usize>,
    ) -> Self {
        let mut ladder = EloLadder::new(mz_conf);
        if let Some(rung) = initial_rung {
            ladder.set_current(rung);
        }
        let (tx, rx) = channel();
        Self {
            ladder: Some(ladder),
            best_elo: initial_best_elo,
            ckpt_dir: mz_conf.checkpoint_dir(),
            device,
            tx,
            rx,
            running: false,
        }
    }

    pub fn due(&self, training_step: usize) -> bool {
        !self.running
            && self
                .ladder
                .as_ref()
                .is_some_and(|ladder| ladder.due(training_step))
    }

    pub fn current(&self) -> usize {
        self.ladder.as_ref().map_or(0, EloLadder::current)
    }

    /// Snapshots `agent` onto the eval backend's device and evaluates it
    /// against the current rung on a background thread.
    pub fn spawn<N: MuZeroNets<B> + 'static>(
        &mut self,
        mz_conf: &MuZeroConfig,
        agent: &N,
        training_step: usize,
    ) {
        let Some(mut ladder) = self.ladder.take() else {
            return;
        };
        let eval_agent: N = nets_to_backend(agent, mz_conf, &self.device);
        let mz_conf = mz_conf.clone();
        let ckpt_dir = self.ckpt_dir.clone();
        let device = self.device.clone();
        let best_elo = self.best_elo;
        let tx = self.tx.clone();
        self.running = true;

        thread::spawn(move || {
            let reading = ladder.run(&mz_conf, &eval_agent, &device, training_step);
            if reading.elo > best_elo {
                let prev_best = best_elo.is_finite().then_some(best_elo);
                save_best_model(&ckpt_dir, eval_agent, reading.elo, prev_best);
                save_best_elo(reading.elo, &format!("{ckpt_dir}/best_elo"));
            }
            save_eval_state(
                ladder.current(),
                reading.elo,
                &reading.opponent,
                &format!("{ckpt_dir}/eval_state"),
            );
            let _ = tx.send(EvalDone { ladder, reading });
        });
    }

    /// Non-blocking: returns a finished reading once the background eval completes.
    pub fn poll(&mut self) -> Option<EvalReading> {
        let done = self.rx.try_recv().ok()?;
        if done.reading.elo > self.best_elo {
            self.best_elo = done.reading.elo;
        }
        self.ladder = Some(done.ladder);
        self.running = false;
        Some(done.reading)
    }
}
