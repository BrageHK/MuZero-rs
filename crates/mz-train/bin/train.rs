use burn::module::AutodiffModule;
use burn::{Dispatch, DispatchDevice};
use burn::optim::AdamConfig;
use burn::rl::Environment;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use mz_rs::agent::MlpNets;
use mz_rs::mz_config::{MuZeroConfig, NetworkType};
use mz_rs::networks::nets_to_backend;
use mz_rs::replay_buffer::BufferData;
use mz_rs::search::search_serial::search;
use mz_rs::train::train;
use mz_rs::utils::select_device;
use mz_rs::{env::cartpole::env::CartPoleWrapper, replay_buffer::ReplayBuffer};
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

fn main() {
    // TODO: make training parallel so GPU search is faster than CPU search
    // Dispatch picks the concrete backend at runtime from the config; autodiff
    // lives in the device, so TrainB/StoreB/InferB are all the same type.
    type TrainB = Dispatch;
    type StoreB = <TrainB as AutodiffBackend>::InnerBackend;
    type InferB = Dispatch;

    let mz_conf = MuZeroConfig::default();
    if let NetworkType::ResNet = mz_conf.network_type {
        panic!(
            "network_type: ResNet has no compatible environment yet \
             (cartpole obs is a flat vector) — use network_type: Linear"
        );
    }

    // Plain device for buffer/store tensors, autodiff-wrapped for the model.
    let device = select_device(mz_conf.training_backend);
    let train_device = DispatchDevice::autodiff(device.clone());
    let infer_device = select_device(mz_conf.inference_backend);

    let mut agent: MlpNets<TrainB> = mz_conf.init_agent(&train_device);
    let mut optimizer = AdamConfig::new().init::<TrainB, MlpNets<TrainB>>();
    let mut inference_agent: MlpNets<InferB> =
        nets_to_backend(&agent.valid(), &mz_conf, &infer_device);
    let mut env = CartPoleWrapper::default();

    let mut buffer = ReplayBuffer::<StoreB>::default();

    let mut game = Vec::new();

    let mut game_len = 0usize;
    let mut tau_idx = 0usize;
    let mut tau = mz_conf.temperature_schedule[tau_idx].tau;


    for training_step in 0..mz_conf.total_steps {
        game_len += 1;
        let s = env.state().state;
        let obs_floats = [[s[0] as f32, s[1] as f32, s[2] as f32, s[3] as f32]];
        let obs = Tensor::<InferB, 2>::from_floats(obs_floats, &infer_device);
        let obs_store = Tensor::<StoreB, 2>::from_floats(obs_floats, &device);

        match mz_conf.temperature_schedule[tau_idx].step {
            Some(n) => if training_step > n {
                tau_idx += 1;
                tau = mz_conf.temperature_schedule[tau_idx].tau;
            },
            None => (),
        }

        let (visit_distribution, value, _action) =
            search(obs, &mz_conf, &inference_agent, tau);
        let dist = WeightedIndex::new(&visit_distribution).unwrap();
        let action = dist.sample(&mut rand::rng());

        let result = env.step(action);

        let buffer_data = BufferData {
            state: obs_store,
            action,
            value,
            reward: result.reward as f32,
            policy: Tensor::<StoreB, 1>::from_floats(visit_distribution.as_slice(), &device)
                .unsqueeze_dim(0),
        };

        game.push(buffer_data);

        if result.truncated || result.done {
            println!("Died after {} steps", game_len);
            buffer.store_game(game.clone());
            env.reset();
            game.clear();
            game_len = 0;
            println!("N games: {}", buffer.games.len());

            for train_step in 0..mz_conf.train_steps_per_game {
                let loss;
                print!("Training!");
                (agent, loss) = train(
                    agent,
                    &mut optimizer,
                    &mz_conf,
                    &mut buffer,
                    mz_conf.learning_rate,
                    &train_device,
                );
                // if let Some(loss) = loss {
                //     println!(
                //         "Train step {}/{}: loss = {:.4}",
                //         train_step + 1,
                //         mz_conf.train_steps_per_game,
                //         loss
                //     );
                // }
            }
            inference_agent = nets_to_backend(&agent.valid(), &mz_conf, &infer_device);
        }
    }
}
