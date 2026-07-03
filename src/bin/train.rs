use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn::rl::Environment;
use burn::tensor::Tensor;
use mz_rs::env::CartPoleAction;
use mz_rs::mz_config::MuZeroConfig;
use mz_rs::replay_buffer::BufferData;
use mz_rs::search::search;
use mz_rs::{env::CartPoleWrapper, replay_buffer::ReplayBuffer};
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

fn main() {
    println!("Hello");
    type B = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    let mz_conf = MuZeroConfig::default();
    let agent = mz_conf.init::<B>(&device);
    let mut env = CartPoleWrapper::default();

    let mut buffer = ReplayBuffer::<B>::default();

    let mut game = Vec::new();

    let mut game_len = 0usize;
    let mut tau_idx = 0usize;
    let mut tau = mz_conf.temperature_schedule[tau_idx].tau;


    for training_step in 0..mz_conf.total_steps {
        game_len += 1;
        let s = env.state().state;
        let obs = Tensor::<B, 2>::from_floats(
            [[s[0] as f32, s[1] as f32, s[2] as f32, s[3] as f32]],
            &device,
        );

        match mz_conf.temperature_schedule[tau_idx].step {
            Some(n) => if training_step > n {
                tau_idx += 1;
                tau = mz_conf.temperature_schedule[tau_idx].tau;
            },
            None => (),
        }

        let (visit_distribution, value, _action) = search(obs.clone(), &mz_conf, &agent, tau);
        let dist = WeightedIndex::new(&visit_distribution).unwrap();
        let action = dist.sample(&mut rand::rng());

        let result = env.step(CartPoleAction { action });

        let buffer_data = BufferData {
            state: obs,
            action,
            value,
            reward: result.reward as f32,
            policy: Tensor::<B, 1>::from_floats(visit_distribution.as_slice(), &device).unsqueeze_dim(0),
        };

        game.push(buffer_data);
        

        if result.truncated || result.done {
            println!("Died after {} steps", game_len);
            buffer.store_game(game.clone());
            env.reset();
            game.clear();
            game_len = 0;
        }
    }
}
