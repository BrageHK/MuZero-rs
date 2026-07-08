use burn::Dispatch;
use burn::rl::Environment;
use burn::tensor::Tensor;
use gif::{Encoder, Frame, Repeat};
use gym_rs::utils::renderer::{RenderColor, RenderFrame, RenderMode};
use mz_rs::utils::select_device;
use mz_rs::{
    agent::MlpNets, env::cartpole::env::CartPoleWrapper, mz_config::MuZeroConfig,
    search::search_serial::search,
};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use std::fs::File;

fn save_gif(frames: &[RenderFrame], path: &str) {
    let height = frames[0].0.len() as u16;
    let width = frames[0].0[0].len() as u16;

    std::fs::create_dir_all("media").unwrap();
    let mut file = File::create(path).unwrap();
    let mut encoder = Encoder::new(&mut file, width, height, &[]).unwrap();
    encoder.set_repeat(Repeat::Infinite).unwrap();

    // Every 2nd frame at half speed keeps playback rate but halves file size.
    for frame in frames.iter().step_by(2) {
        let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
        for row in &frame.0 {
            for RenderColor::RGB(r, g, b) in row {
                rgb.extend_from_slice(&[*r, *g, *b]);
            }
        }
        let mut gif_frame = Frame::from_rgb(width, height, &rgb);
        gif_frame.delay = 4; // 25 fps (env renders at 50)
        encoder.write_frame(&gif_frame).unwrap();
    }
}

fn main() {
    // Backend picked at runtime from config (see BackendChoice).
    type B = Dispatch;

    let mz_conf = MuZeroConfig::new::<B>("configs/config_inference.yaml");
    assert!(
        mz_conf.init_checkpoint.is_some(),
        "Set init_checkpoint in config.yaml (e.g. \"model/best\") to play from a trained model"
    );
    let device = select_device(mz_conf.inference_backend);
    let agent: MlpNets<B> = mz_conf.init_agent(&device);
    let mut env = CartPoleWrapper::new(RenderMode::RgbArray);
    let mut rng = rand::rng();

    let mut best_frames: Vec<RenderFrame> = Vec::new();
    let mut best_reward = f64::NEG_INFINITY;

    for episode in 0..1 {
        env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;

        loop {
            let s = env.state().state;
            let obs = Tensor::<B, 2>::from_floats(
                [[s[0] as f32, s[1] as f32, s[2] as f32, s[3] as f32]],
                &device,
            );

            let (dist, _value, _action) = search(obs, &mz_conf, &agent, 0.10);
            let action = WeightedIndex::new(&dist).unwrap().sample(&mut rng);
            let result = env.step(action);
            total_reward += result.reward;
            steps += 1;

            if result.done || result.truncated {
                break;
            }
        }

        if total_reward > best_reward {
            best_reward = total_reward;
            best_frames = env.frames();
        }

        println!("Episode {episode}: steps={steps}, reward={total_reward}");
    }

    if !best_frames.is_empty() {
        save_gif(&best_frames, "media/cartpole.gif");
        println!("Saved best episode (reward={best_reward}) to media/cartpole.gif");
    }
}
