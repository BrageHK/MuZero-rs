//! Main training node: gRPC server, canonical replay buffer + weights, and the
//! synchronous DDP round loop. Trainer workers pull a batch and the current
//! weights via `TrainerSync::GetBatch`, compute a local gradient, and submit it
//! via `TrainerSync::SyncGradients`; the coordinator averages every submission
//! that arrives within `sync_timeout_ms` together with its own gradient,
//! applies one optimizer step, and publishes the result. Self-play nodes push
//! finished games via `SelfPlayIngest::SubmitGame` and receive a fresh network
//! every `inference_update_interval` steps via `SelfPlayIngest::WatchWeights`.

use std::pin::Pin;
use std::sync::mpsc::{self as std_mpsc, RecvTimeoutError};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use burn::module::{AutodiffModule, Module};
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use tokio::sync::{oneshot, watch};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::WatchStream;
use tonic::{Request, Response, Status, transport::Server};

use mz_net::{
    Ack, BatchRequest, BatchResponse, Empty, GamePayload, GradientSubmission, SelfPlayIngest,
    SelfPlayIngestServer, SyncResult, TrainerSync, TrainerSyncServer, WeightUpdate,
};

use crate::distributed::grad_sync::{average_into, flatten_grads, unflatten_grads};
use crate::mz_config::MuZeroConfig;
use crate::networks::{MuZeroNets, nets_to_bytes};
use crate::optim::AnyOptimizer;
use crate::replay_buffer::{BufferData, ReplayBuffer};
use crate::train::{apply_grads, compute_grads_on_batch};
use crate::utils::{save_buffer, save_env_steps, save_games_played, save_training_step};

struct PendingGrad {
    worker_id: u32,
    step: u64,
    grad: Vec<f32>,
    reply: oneshot::Sender<SyncResult>,
}

struct Shared {
    buffer: Mutex<ReplayBuffer>,
    weights: RwLock<(u64, Vec<u8>)>,
    weight_broadcast: watch::Sender<WeightUpdate>,
    grad_tx: std_mpsc::Sender<PendingGrad>,
}

struct SelfPlayIngestSvc {
    shared: Arc<Shared>,
    mz_conf: MuZeroConfig,
}

#[tonic::async_trait]
impl SelfPlayIngest for SelfPlayIngestSvc {
    async fn submit_game(
        &self,
        request: Request<GamePayload>,
    ) -> Result<Response<Ack>, Status> {
        let payload = request.into_inner();
        let data: Vec<BufferData> = rmp_serde::from_slice(&payload.data)
            .map_err(|e| Status::invalid_argument(format!("bad game payload: {e}")))?;
        self.shared
            .buffer
            .lock()
            .expect("replay buffer mutex poisoned")
            .store_game(data, &self.mz_conf);
        Ok(Response::new(Ack {}))
    }

    type WatchWeightsStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<WeightUpdate, Status>> + Send + 'static>>;

    async fn watch_weights(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::WatchWeightsStream>, Status> {
        let rx = self.shared.weight_broadcast.subscribe();
        let current = rx.borrow().clone();
        let stream = tokio_stream::once(current).chain(WatchStream::new(rx)).map(Ok);
        Ok(Response::new(Box::pin(stream)))
    }
}

struct TrainerSyncSvc {
    shared: Arc<Shared>,
    mz_conf: MuZeroConfig,
}

#[tonic::async_trait]
impl TrainerSync for TrainerSyncSvc {
    async fn get_batch(
        &self,
        _request: Request<BatchRequest>,
    ) -> Result<Response<BatchResponse>, Status> {
        let batch = {
            let mut buffer = self.shared.buffer.lock().expect("replay buffer mutex poisoned");
            if buffer.states.len() > self.mz_conf.training_batch_size {
                buffer.sample_games(&self.mz_conf)
            } else {
                // Buffer isn't warmed up yet; tell the worker to back off instead
                // of handing out a batch full of empty (unpopulated) sequences.
                Vec::new()
            }
        };
        let batch = rmp_serde::to_vec(&batch)
            .map_err(|e| Status::internal(format!("failed to encode batch: {e}")))?;
        let (step, weights) = {
            let w = self.shared.weights.read().expect("weights lock poisoned");
            (w.0, w.1.clone())
        };
        Ok(Response::new(BatchResponse {
            batch,
            weights,
            step,
        }))
    }

    async fn sync_gradients(
        &self,
        request: Request<GradientSubmission>,
    ) -> Result<Response<SyncResult>, Status> {
        let req = request.into_inner();
        let (reply_tx, reply_rx) = oneshot::channel();
        self.shared
            .grad_tx
            .send(PendingGrad {
                worker_id: req.worker_id,
                step: req.step,
                grad: req.grad,
                reply: reply_tx,
            })
            .map_err(|_| Status::unavailable("coordinator training loop is not running"))?;
        reply_rx
            .await
            .map(Response::new)
            .map_err(|_| Status::internal("coordinator dropped the reply channel"))
    }
}

/// Runs the coordinator: binds the gRPC server on `mz_conf.distributed().listen_addr`
/// and drives the DDP round loop on a dedicated thread until `training_steps` is
/// reached. Never returns before then (or before an unrecoverable transport error).
pub async fn run<TrainB, NT>(
    mz_conf: MuZeroConfig,
    agent: NT,
    optimizer: AnyOptimizer<TrainB, NT>,
    buffer: ReplayBuffer,
    train_device: TrainB::Device,
    initial_training_step: usize,
) where
    TrainB: AutodiffBackend,
    NT: MuZeroNets<TrainB> + AutodiffModule<TrainB> + Send + 'static,
    NT::InnerModule: MuZeroNets<TrainB::InnerBackend>,
    TrainB::Device: Send + 'static,
{
    let listen_addr = mz_conf.distributed().listen_addr.clone();
    let initial_weights = nets_to_bytes(&agent.valid());

    let (weight_tx, _) = watch::channel(WeightUpdate {
        weights: initial_weights.clone(),
        training_step: initial_training_step as u64,
    });
    let (grad_tx, grad_rx) = std_mpsc::channel();

    let shared = Arc::new(Shared {
        buffer: Mutex::new(buffer),
        weights: RwLock::new((initial_training_step as u64, initial_weights)),
        weight_broadcast: weight_tx,
        grad_tx,
    });

    {
        let shared = shared.clone();
        let mz_conf = mz_conf.clone();
        std::thread::spawn(move || {
            training_loop(
                mz_conf,
                agent,
                optimizer,
                shared,
                grad_rx,
                train_device,
                initial_training_step as u64,
            );
        });
    }

    let self_play_svc = SelfPlayIngestServer::new(SelfPlayIngestSvc {
        shared: shared.clone(),
        mz_conf: mz_conf.clone(),
    });
    let trainer_svc = TrainerSyncServer::new(TrainerSyncSvc {
        shared: shared.clone(),
        mz_conf: mz_conf.clone(),
    });

    let addr = listen_addr
        .parse()
        .unwrap_or_else(|e| panic!("invalid distributed.listen_addr '{listen_addr}': {e}"));

    println!("coordinator listening on {addr}");
    Server::builder()
        .add_service(self_play_svc)
        .add_service(trainer_svc)
        .serve(addr)
        .await
        .expect("gRPC server failed");
}

#[allow(clippy::too_many_arguments)]
fn training_loop<TrainB, NT>(
    mz_conf: MuZeroConfig,
    mut agent: NT,
    mut optimizer: AnyOptimizer<TrainB, NT>,
    shared: Arc<Shared>,
    grad_rx: std_mpsc::Receiver<PendingGrad>,
    train_device: TrainB::Device,
    mut training_step: u64,
) where
    TrainB: AutodiffBackend,
    NT: MuZeroNets<TrainB> + AutodiffModule<TrainB>,
    NT::InnerModule: MuZeroNets<TrainB::InnerBackend>,
{
    use crate::utils::lr_for_step;

    let dist = mz_conf.distributed().clone();
    let sync_timeout = Duration::from_millis(dist.sync_timeout_ms);
    let ckpt_dir = mz_conf.checkpoint_dir();
    std::fs::create_dir_all(&ckpt_dir).expect("Failed to create checkpoint directory");
    let mut next_checkpoint = training_step + mz_conf.checkpoint_interval as u64;
    let started = Instant::now();
    let mut last_print = Instant::now();

    while (training_step as usize) < mz_conf.training_steps {
        let has_batch = shared
            .buffer
            .lock()
            .expect("replay buffer mutex poisoned")
            .states
            .len()
            > mz_conf.training_batch_size;
        if !has_batch {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }

        let local_batch = shared
            .buffer
            .lock()
            .expect("replay buffer mutex poisoned")
            .sample_games(&mz_conf);
        let (metrics, mut local_grad) = {
            let (metrics, grads) =
                compute_grads_on_batch(&agent, &mz_conf, local_batch, None, &train_device);
            (metrics, flatten_grads(&grads, &agent.valid()))
        };

        let mut others = Vec::with_capacity(dist.expected_trainer_workers);
        let mut replies = Vec::with_capacity(dist.expected_trainer_workers);
        let deadline = Instant::now() + sync_timeout;
        while others.len() < dist.expected_trainer_workers {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match grad_rx.recv_timeout(remaining) {
                Ok(sub) if sub.step == training_step => {
                    others.push(sub.grad);
                    replies.push(sub.reply);
                }
                Ok(sub) => {
                    // Stale submission from a worker still finishing a round we
                    // already closed (or a round we haven't started yet); ack it
                    // immediately with the current step so its RPC doesn't hang,
                    // but don't count it toward this round's average.
                    let _ = sub.reply.send(SyncResult {
                        weights: Vec::new(),
                        step: training_step,
                    });
                    println!(
                        "coordinator: dropped stale gradient from worker {} for step {} (currently at {})",
                        sub.worker_id, sub.step, training_step
                    );
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        average_into(&mut local_grad, &others, 1 + others.len());
        let grads = unflatten_grads(&local_grad, &agent.valid());
        let lr = lr_for_step(
            mz_conf.learning_rate,
            mz_conf.lr_warmup_steps,
            mz_conf.lr_decay_rate,
            mz_conf.lr_decay_steps,
            training_step as usize,
        );
        agent = apply_grads(agent, &mut optimizer, lr, grads);
        training_step += 1;

        let new_weights = nets_to_bytes(&agent.valid());
        {
            let mut w = shared.weights.write().expect("weights lock poisoned");
            *w = (training_step, new_weights.clone());
        }
        for reply in replies {
            let _ = reply.send(SyncResult {
                weights: Vec::new(),
                step: training_step,
            });
        }

        if training_step % mz_conf.inference_update_interval.max(1) as u64 == 0 {
            let _ = shared.weight_broadcast.send(WeightUpdate {
                weights: new_weights,
                training_step,
            });
        }

        if mz_conf.checkpoint_interval > 0 && training_step >= next_checkpoint {
            agent
                .valid()
                .save_file(format!("{ckpt_dir}/latest"), &CompactRecorder::new())
                .expect("Failed to save checkpoint");
            save_buffer(
                &shared.buffer.lock().expect("replay buffer mutex poisoned"),
                &format!("{ckpt_dir}/buffer.mpk"),
            );
            save_training_step(training_step as usize, &format!("{ckpt_dir}/training_step"));
            save_env_steps(0, &format!("{ckpt_dir}/env_steps"));
            save_games_played(0, &format!("{ckpt_dir}/games_played"));
            next_checkpoint += mz_conf.checkpoint_interval as u64;
        }

        if last_print.elapsed() >= Duration::from_secs(1) {
            last_print = Instant::now();
            println!(
                "t={:.1} step={training_step} loss={:.4} consistency={:.4} workers_in_round={}",
                started.elapsed().as_secs_f64(),
                metrics.total,
                metrics.consistency,
                others.len(),
            );
        }
    }
}
