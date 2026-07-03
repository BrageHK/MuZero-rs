use burn::rl::Environment;
use burn::{
    backend::{Wgpu, wgpu::WgpuDevice},
    tensor::Tensor,
};
use mz_rs::{
    mz_config::MuZeroConfig,
    env::{CartPoleAction, CartPoleWrapper},
    search::search,
};

fn main() {
    type B = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    let mz_conf = MuZeroConfig::default();
    let agent = mz_conf.init::<B>(&device);
    let mut env = CartPoleWrapper::default();

    for episode in 0..10 {
        env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;

        loop {
            let s = env.state().state;
            let obs = Tensor::<B, 2>::from_floats(
                [[s[0] as f32, s[1] as f32, s[2] as f32, s[3] as f32]],
                &device,
            );

            let (_dist, _value, action) = search(obs, &mz_conf, &agent, 0.0);
            let result = env.step(CartPoleAction { action });
            total_reward += result.reward;
            steps += 1;

            if result.done || result.truncated {
                break;
            }
        }

        println!("Episode {episode}: steps={steps}, reward={total_reward}");
    }
}
