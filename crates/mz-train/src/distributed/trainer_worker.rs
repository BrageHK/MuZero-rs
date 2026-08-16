//! Worker trainer: pulls a batch + the canonical weights from the coordinator
//! each round, computes a local gradient (its half of synchronous DDP), and
//! submits it back. It never runs an optimizer itself — the coordinator
//! averages every replica's gradient and is the single source of truth for
//! the weights, so workers can't drift from each other or from the coordinator
//! even when running on different hardware/backends.

use burn::module::AutodiffModule;
use burn::tensor::backend::AutodiffBackend;

use mz_net::{BatchRequest, GradientSubmission, MAX_GRPC_MESSAGE_SIZE, TrainerSyncClient};

use crate::distributed::grad_sync::flatten_grads;
use crate::mz_config::MuZeroConfig;
use crate::networks::{MuZeroNets, nets_from_bytes};
use crate::replay_buffer::BufferData;
use crate::train::compute_grads_on_batch;

/// Connects to `mz_conf.distributed().coordinator_addr` and runs rounds until
/// the connection is lost or the process is killed. Reconnects are the
/// operator's responsibility (e.g. a process supervisor restarting the binary).
pub async fn run<TrainB, NT>(mz_conf: MuZeroConfig, device: TrainB::Device)
where
    TrainB: AutodiffBackend,
    NT: MuZeroNets<TrainB> + AutodiffModule<TrainB>,
{
    let dist = mz_conf.distributed().clone();
    let worker_id = dist.worker_id;
    let addr = format!("http://{}", dist.coordinator_addr);
    println!("trainer worker {worker_id} connecting to coordinator at {addr}");

    let mut client = TrainerSyncClient::connect(addr)
        .await
        .expect("failed to connect to coordinator")
        .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE);

    let net_conf = mz_conf.net_config();
    let retry_backoff = std::time::Duration::from_millis(500);

    loop {
        let response = match client.get_batch(BatchRequest { worker_id }).await {
            Ok(r) => r.into_inner(),
            Err(e) => {
                eprintln!("trainer worker {worker_id}: get_batch failed ({e}), retrying");
                tokio::time::sleep(retry_backoff).await;
                continue;
            }
        };

        let agent: NT = nets_from_bytes(response.weights, &net_conf, &device);
        let batch: Vec<Vec<BufferData>> = rmp_serde::from_slice(&response.batch)
            .expect("failed to decode batch payload from coordinator");

        if batch.is_empty() {
            // Coordinator's buffer isn't warmed up yet; back off briefly.
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            continue;
        }

        let (_, grads) = compute_grads_on_batch(&agent, &mz_conf, batch, None, &device);
        let grad = flatten_grads(&grads, &agent.valid());

        if let Err(e) = client
            .sync_gradients(GradientSubmission {
                worker_id,
                step: response.step,
                grad,
            })
            .await
        {
            eprintln!(
                "trainer worker {worker_id}: sync_gradients failed ({e}), \
                 discarding this round's gradient and retrying"
            );
            tokio::time::sleep(retry_backoff).await;
        }
    }
}
