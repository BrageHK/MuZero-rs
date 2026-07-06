use burn::rl::Environment;
use mz_rs::env::cartpole::env::CartPoleWrapper;

fn main() {
    let mut env = CartPoleWrapper::new(gym_rs::utils::renderer::RenderMode::Human);
    let mut rng = fastrand::Rng::new();

    let num_episodes = 10;
    for episode in 0..num_episodes {
        env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;

        loop {
            let action = rng.usize(0..2);
            let result = env.step(action);
            total_reward += result.reward;
            steps += 1;
            if result.done || result.truncated {
                break;
            }
        }

        println!("Episode {episode}: steps={steps}, reward={total_reward}");
    }
}
