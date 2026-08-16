use burn::Dispatch;

use mz_rs::distributed::selfplay_worker;
use mz_rs::mz_config::{MuZeroConfig, NodeRole};
use mz_rs::utils::select_device;
use mz_rs::{with_env, with_net};

#[tokio::main]
async fn main() {
    type InferB = Dispatch;

    let mz_conf = MuZeroConfig::new::<InferB>("configs/distributed/selfplay_worker.yaml");
    assert!(
        matches!(mz_conf.distributed_role(), NodeRole::SelfPlayWorker),
        "bin/selfplay_worker requires `distributed: {{ role: SelfPlayWorker, ... }}` in the config"
    );

    let infer_device = select_device(mz_conf.inference_backend);

    with_env!(mz_conf, E => {
        with_net!(mz_conf, Net => {
            selfplay_worker::run::<E, InferB, Net<InferB>>(mz_conf.clone(), infer_device.clone()).await;
        });
    });
}
