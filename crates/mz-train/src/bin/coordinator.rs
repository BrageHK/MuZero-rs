use burn::module::AutodiffModule;
use burn::optim::Optimizer;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::{Dispatch, DispatchDevice};

use mz_rs::distributed::coordinator;
use mz_rs::mz_config::{MuZeroConfig, NodeRole};
use mz_rs::networks::MuZeroNets;
use mz_rs::optim::AnyOptimizer;
use mz_rs::replay_buffer::ReplayBuffer;
use mz_rs::utils::{load_buffer, load_training_step, select_device};
use mz_rs::with_net;

#[tokio::main]
async fn main() {
    type TrainB = Dispatch;

    let mz_conf = MuZeroConfig::new::<TrainB>("configs/distributed/coordinator.yaml");
    assert!(
        matches!(mz_conf.distributed_role(), NodeRole::Coordinator),
        "bin/coordinator requires `distributed: {{ role: Coordinator, ... }}` in the config"
    );

    let device = select_device(mz_conf.training_backend);
    let train_device = DispatchDevice::autodiff(device.clone());

    with_net!(mz_conf, Net => {
        run::<TrainB, Net<TrainB>>(mz_conf, device, train_device).await;
    });
}

async fn run<TrainB, NT>(mz_conf: MuZeroConfig, inner_device: TrainB::Device, train_device: TrainB::Device)
where
    TrainB: AutodiffBackend,
    NT: MuZeroNets<TrainB> + AutodiffModule<TrainB> + Send + 'static,
    NT::InnerModule: MuZeroNets<TrainB::InnerBackend>,
    TrainB::Device: Send + 'static,
{
    let ckpt_dir = mz_conf.checkpoint_dir();
    std::fs::create_dir_all(&ckpt_dir).expect("Failed to create directory");
    std::fs::write(
        format!("{ckpt_dir}/config.yaml"),
        serde_yaml::to_string(&mz_conf).expect("Failed to serialize config"),
    )
    .expect("Failed to write config snapshot");

    let agent: NT = mz_conf.init_agent(&train_device);
    let mut optimizer = AnyOptimizer::<TrainB, NT>::new(&mz_conf);
    let mut buffer = ReplayBuffer::new(&mz_conf);
    let mut training_step = 0usize;

    if mz_conf.load_from_checkpoint {
        let opt_path = format!("{ckpt_dir}/optimizer");
        match CompactRecorder::new().load(opt_path.clone().into(), &inner_device) {
            Ok(record) => optimizer = optimizer.load_record(record),
            Err(e) => panic!("Failed to load optimizer state from {opt_path}: {e}"),
        }
        buffer.states = load_buffer(&format!("{ckpt_dir}/buffer.mpk"));
        training_step = load_training_step(&format!("{ckpt_dir}/training_step"));
    }

    coordinator::run(mz_conf, agent, optimizer, buffer, train_device, training_step).await;
}
