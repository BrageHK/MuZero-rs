pub mod pb {
    tonic::include_proto!("muzero");
}

pub const MAX_GRPC_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

pub use pb::self_play_ingest_client::SelfPlayIngestClient;
pub use pb::self_play_ingest_server::{SelfPlayIngest, SelfPlayIngestServer};
pub use pb::trainer_sync_client::TrainerSyncClient;
pub use pb::trainer_sync_server::{TrainerSync, TrainerSyncServer};
pub use pb::{
    Ack, BatchRequest, BatchResponse, Empty, GamePayload, GradientSubmission, SyncResult,
    WeightUpdate,
};
