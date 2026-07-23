use std::fs::File;
use std::io::{self, Write};

use burn::Dispatch;
use burn::tensor::backend::Backend;
use gif::{Encoder, Frame, Repeat};
use gym_rs::utils::renderer::{RenderColor, RenderFrame, RenderMode};
use mz_rs::env::cartpole::env::CartPoleWrapper;
use mz_rs::env::othello::env::{Othello, PASS};
use mz_rs::env::tictactoe::env::TicTacToe;
use mz_rs::env::Environment;
use mz_rs::mz_config::{EnvironmentName, MuZeroConfig};
use mz_rs::utils::select_device;
use mz_rs::{agent::MlpNets, search::batched_search};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;

trait Playable: Environment<Action = usize> + Default {
    fn render(&self) -> String;
    fn parse_action(input: &str) -> Option<usize>;
    fn action_label(action: usize) -> String;
}

impl Playable for TicTacToe {
    fn render(&self) -> String {
        format!("{self}")
    }

    fn parse_action(input: &str) -> Option<usize> {
        match input.trim().parse::<usize>() {
            Ok(cell) if cell < 9 => Some(cell),
            _ => None,
        }
    }

    fn action_label(action: usize) -> String {
        action.to_string()
    }
}

impl Playable for Othello {
    fn render(&self) -> String {
        format!("{self}")
    }

    fn parse_action(input: &str) -> Option<usize> {
        let s = input.trim().to_ascii_lowercase();
        if s == "pass" {
            return Some(PASS);
        }
        let mut chars = s.chars();
        let col = chars.next()?;
        let row = chars.next()?;
        if chars.next().is_some() {
            return None;
        }
        let col = (col as u8).checked_sub(b'a')?;
        let row = (row as u8).checked_sub(b'1')?;
        if col < 8 && row < 8 {
            Some(row as usize * 8 + col as usize)
        } else {
            None
        }
    }

    fn action_label(action: usize) -> String {
        if action == PASS {
            "pass".to_string()
        } else {
            let col = (b'a' + (action % 8) as u8) as char;
            let row = action / 8 + 1;
            format!("{col}{row}")
        }
    }
}

fn legal_labels<E: Playable>(mask: &[bool]) -> String {
    mask.iter()
        .enumerate()
        .filter(|&(_, &l)| l)
        .map(|(a, _)| E::action_label(a))
        .collect::<Vec<_>>()
        .join(" ")
}

fn read_human_action<E: Playable>(mask: &[bool]) -> usize {
    loop {
        print!("Your move (legal: {}): ", legal_labels::<E>(mask));
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if io::stdin().read_line(&mut line).unwrap() == 0 {
            std::process::exit(0);
        }

        match E::parse_action(&line) {
            Some(action) if mask.get(action).copied().unwrap_or(false) => return action,
            _ => println!("Illegal move, try again."),
        }
    }
}

fn play_two_player<B: Backend, E: Playable>(
    mz_conf: &MuZeroConfig,
    agent: &MlpNets<B>,
    device: &B::Device,
    human_first: bool,
) {
    let mut env = E::default();
    env.reset();
    let mut human_turn = human_first;

    println!("\n{}\n", env.render());

    let result = loop {
        let mask = env.legal_mask();

        let action = if human_turn {
            read_human_action::<E>(&mask)
        } else {
            let obs = env.state_tensor::<B>(device);
            let results =
                batched_search(obs, Some(std::slice::from_ref(&mask)), mz_conf, agent, 1.0);
            let result = &results[0];
            print!("Search distribution:");
            for (a, &p) in result.distribution.iter().enumerate() {
                if mask[a] {
                    print!(" {}={:.3}", E::action_label(a), p);
                }
            }
            println!("\nValue: {:.3}", result.value);
            let action = result.best_action;
            println!("Agent plays: {}", E::action_label(action));
            action
        };

        let result = env.step(action);
        println!("\n{}\n", env.render());
        human_turn = !human_turn;

        if result.done || result.truncated {
            break result;
        }
    };

    let mover_was_human = !human_turn;
    let msg = match result.reward {
        r if r > 0.0 && mover_was_human => "You win!",
        r if r > 0.0 => "Agent wins!",
        r if r < 0.0 && mover_was_human => "You lose!",
        r if r < 0.0 => "You win!",
        _ => "Draw.",
    };
    println!("{msg}");
}

fn play_single_player<B: Backend>(mz_conf: &MuZeroConfig, agent: &MlpNets<B>, device: &B::Device) {
    let mut env = CartPoleWrapper::new(RenderMode::RgbArray);
    let mut rng = rand::rng();

    env.reset();
    let mut total_reward = 0.0;
    let mut steps = 0;

    loop {
        let obs = env.state_tensor::<B>(device);
        let results = batched_search(obs, None, mz_conf, agent, 0.10);
        let action = WeightedIndex::new(&results[0].distribution)
            .unwrap()
            .sample(&mut rng);
        let result = env.step(action);
        total_reward += result.reward;
        steps += 1;

        if result.done || result.truncated {
            break;
        }
    }

    println!("steps={steps}, reward={total_reward}");
    let frames = env.frames();
    if !frames.is_empty() {
        save_gif(&frames, "media/cartpole.gif");
        println!("Saved episode to media/cartpole.gif");
    }
}

fn save_gif(frames: &[RenderFrame], path: &str) {
    let height = frames[0].0.len() as u16;
    let width = frames[0].0[0].len() as u16;

    std::fs::create_dir_all("media").unwrap();
    let mut file = File::create(path).unwrap();
    let mut encoder = Encoder::new(&mut file, width, height, &[]).unwrap();
    encoder.set_repeat(Repeat::Infinite).unwrap();

    for frame in frames.iter().step_by(2) {
        let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
        for row in &frame.0 {
            for RenderColor::RGB(r, g, b) in row {
                rgb.extend_from_slice(&[*r, *g, *b]);
            }
        }
        let mut gif_frame = Frame::from_rgb(width, height, &rgb);
        gif_frame.delay = 4;
        encoder.write_frame(&gif_frame).unwrap();
    }
}

fn prompt_human_first() -> bool {
    loop {
        print!("Play first? [y/n]: ");
        io::stdout().flush().unwrap();
        let mut line = String::new();
        if io::stdin().read_line(&mut line).unwrap() == 0 {
            std::process::exit(0);
        }
        match line.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" | "" => return true,
            "n" | "no" => return false,
            _ => {}
        }
    }
}

fn main() {
    type B = Dispatch;

    let mut mz_conf = MuZeroConfig::new::<B>("configs/config.yaml");
    mz_conf.root_exploration_fraction = 0.0;
    assert!(
        mz_conf.init_checkpoint.is_some(),
        "Set init_checkpoint in config.yaml (e.g. \"model/TicTacToe/latest\") to play a trained model"
    );

    let device = select_device(mz_conf.inference_backend);
    let agent: MlpNets<B> = mz_conf.init_agent(&device);

    match mz_conf.environment {
        EnvironmentName::TicTacToe => {
            let human_first = prompt_human_first();
            play_two_player::<B, TicTacToe>(&mz_conf, &agent, &device, human_first);
        }
        EnvironmentName::Othello => {
            let human_first = prompt_human_first();
            play_two_player::<B, Othello>(&mz_conf, &agent, &device, human_first);
        }
        EnvironmentName::CartPole => play_single_player::<B>(&mz_conf, &agent, &device),
    }
}
