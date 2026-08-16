use burn::{Dispatch, DispatchDevice};

use mz_rs::distributed::trainer_worker;
use mz_rs::mz_config::{MuZeroConfig, NodeRole};
use mz_rs::utils::select_device;
use mz_rs::with_net;

#[tokio::main]
async fn main() {
    type TrainB = Dispatch;

    let mz_conf = MuZeroConfig::new::<TrainB>("configs/distributed/trainer_worker.yaml");
    assert!(
        matches!(mz_conf.distributed_role(), NodeRole::TrainerWorker),
        "bin/trainer_worker requires `distributed: {{ role: TrainerWorker, ... }}` in the config"
    );

    let device = select_device(mz_conf.training_backend);
    let train_device = DispatchDevice::autodiff(device);

    with_net!(mz_conf, Net => {
        trainer_worker::run::<TrainB, Net<TrainB>>(mz_conf.clone(), train_device).await;
    });
}
