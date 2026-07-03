use burn::backend::{
    Autodiff, Wgpu,
    ndarray::{NdArray, NdArrayDevice},
    wgpu::WgpuDevice,
};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::rl::Environment;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use mz_rs::agent::{MuZeroAgent, agent_to_backend};
use mz_rs::env::CartPoleAction;
use mz_rs::mz_config::MuZeroConfig;
use mz_rs::replay_buffer::BufferData;
use mz_rs::search::search;
use mz_rs::train::train;
use mz_rs::{env::CartPoleWrapper, replay_buffer::ReplayBuffer};
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

fn main() {
    // TODO: make training parallel so GPU search is faster than CPU search
    type TrainB = Autodiff<Wgpu<f32, i32>>;
    type StoreB = <TrainB as AutodiffBackend>::InnerBackend;
    type InferB = NdArray<f32>;
    let device = WgpuDevice::default();
    let infer_device = NdArrayDevice::default();

    let mz_conf = MuZeroConfig::default();
    let mut agent = mz_conf.init::<TrainB>(&device);
    let mut optimizer = AdamConfig::new().init::<TrainB, MuZeroAgent<TrainB>>();
    let mut inference_agent =
        agent_to_backend::<_, InferB>(&agent.valid(), &mz_conf, &infer_device);
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

        let result = env.step(CartPoleAction { action });

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
                (agent, loss) = train(
                    agent,
                    &mut optimizer,
                    &mz_conf,
                    &mut buffer,
                    mz_conf.learning_rate,
                    &device,
                );
                if let Some(loss) = loss {
                    println!(
                        "Train step {}/{}: loss = {:.4}",
                        train_step + 1,
                        mz_conf.train_steps_per_game,
                        loss
                    );
                }
            }
            inference_agent =
                agent_to_backend::<_, InferB>(&agent.valid(), &mz_conf, &infer_device);
        }
    }
}
