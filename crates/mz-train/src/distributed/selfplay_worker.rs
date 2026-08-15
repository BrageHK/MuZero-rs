//! Self-play node: runs the same self-play loop as in-process `async_training`
//! (see `async_train::self_play`), but the finished-game and weight-update
//! channels are bridged over gRPC instead of local `mpsc` queues. Self-play
//! itself never touches the network directly — it only ever pushes onto an
//! unbounded channel — so a slow or backed-up coordinator can't stall
//! simulation; game submissions are fired off as independent, concurrent async
//! tasks instead of being awaited one at a time.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use burn::tensor::backend::Backend;
use burn::train::Interrupter;

use mz_net::{Empty, GamePayload, SelfPlayIngestClient};

use crate::async_train::{SelfPlayMsg, WeightMsg, self_play};
use crate::env::Environment;
use crate::mz_config::MuZeroConfig;
use crate::networks::MuZeroNets;

/// Connects to `mz_conf.distributed().coordinator_addr`, streams finished games
/// there, and applies weight updates received every `inference_update_interval`
/// training steps. Runs until the connection is lost or the process is killed.
pub async fn run<E, InferB, N>(mz_conf: MuZeroConfig, infer_device: InferB::Device)
where
    E: Environment<Action = usize> + Default + Clone,
    InferB: Backend,
    N: MuZeroNets<InferB>,
{
    let dist = mz_conf.distributed().clone();
    let addr = format!("http://{}", dist.coordinator_addr);
    println!(
        "self-play worker {} connecting to coordinator at {addr}",
        dist.worker_id
    );

    let mut client = SelfPlayIngestClient::connect(addr)
        .await
        .expect("failed to connect to coordinator");

    let mut weight_stream = client
        .watch_weights(Empty {})
        .await
        .expect("watch_weights RPC failed")
        .into_inner();
    let first = weight_stream
        .message()
        .await
        .expect("watch_weights stream error")
        .expect("coordinator closed the weight stream before sending anything");
    let initial_weights = first.weights;
    let initial_training_step = first.training_step as usize;

    let (game_tx, game_rx) = channel::<SelfPlayMsg>();
    let (weight_tx, weight_rx) = channel::<WeightMsg>();

    let interrupter = Interrupter::new();
    {
        let mz_conf = mz_conf.clone();
        let interrupter = interrupter.clone();
        std::thread::spawn(move || {
            self_play::<E, InferB, N>(
                &mz_conf,
                initial_weights,
                infer_device,
                game_tx,
                weight_rx,
                interrupter,
                initial_training_step,
                f32::NEG_INFINITY,
            );
        });
    }

    tokio::spawn(async move {
        while let Ok(Some(update)) = weight_stream.message().await {
            let msg = WeightMsg {
                bytes: update.weights,
                training_step: update.training_step as usize,
            };
            if weight_tx.send(msg).is_err() {
                break;
            }
        }
    });

    let games_sent = Arc::new(AtomicUsize::new(0));
    let handle = tokio::runtime::Handle::current();
    let worker_id = dist.worker_id;
    let drain = tokio::task::spawn_blocking(move || {
        let mut last_print = Instant::now();
        while let Ok(msg) = game_rx.recv() {
            match msg {
                SelfPlayMsg::Game { data, reward, .. } => {
                    let payload = GamePayload {
                        data: rmp_serde::to_vec(&data).expect("failed to encode game"),
                        total_reward: reward,
                    };
                    let mut client = client.clone();
                    let games_sent = games_sent.clone();
                    handle.spawn(async move {
                        if let Err(e) = client.submit_game(payload).await {
                            eprintln!("submit_game failed: {e}");
                        } else {
                            games_sent.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                }
                SelfPlayMsg::EnvSteps(_) | SelfPlayMsg::Tau(_) | SelfPlayMsg::Eval(_) => {}
            }

            if last_print.elapsed() >= Duration::from_secs(5) {
                last_print = Instant::now();
                println!(
                    "self-play worker {worker_id}: {} games submitted",
                    games_sent.load(Ordering::Relaxed)
                );
            }
        }
    });

    let _ = drain.await;
}
