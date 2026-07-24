use std::sync::RwLock;

use ale::{Ale, BundledRom};
use burn::rl::StepResult;
use serde::{Deserialize, Serialize};

use crate::env::{EnvInfo, Environment};

macro_rules! atari_games {
    ($($variant:ident),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
        pub enum AtariGame {
            $($variant),+
        }

        impl AtariGame {
            pub fn rom(self) -> BundledRom {
                match self {
                    $(AtariGame::$variant => BundledRom::$variant),+
                }
            }
        }
    };
}

atari_games! {
    Adventure, AirRaid, Alien, Amidar, Assault, Asterix, Asteroids, Atlantis, BankHeist,
    BattleZone, BeamRider, Berzerk, Bowling, Boxing, Breakout, Carnival, Centipede,
    ChopperCommand, CrazyClimber, Defender, DemonAttack, DoubleDunk, ElevatorAction, Enduro,
    FishingDerby, Freeway, Frostbite, Gopher, Gravitar, Hero, IceHockey, JamesBond,
    JourneyEscape, Kaboom, Kangaroo, Krull, KungFuMaster, MontezumaRevenge, MsPacman,
    NameThisGame, Phoenix, Pitfall, Pong, Pooyan, PrivateEye, QBert, RiverRaid, RoadRunner,
    RoboTank, Seaquest, Skiing, SpaceInvaders, StarGunner, Tennis, TimePilot, Tutankham,
    UpNDown, Venture, VideoPinball, WizardOfWor, YarsRevenge, Zaxxon,
}

static SELECTED_GAME: RwLock<Option<AtariGame>> = RwLock::new(None);

pub fn set_atari_game(game: AtariGame) {
    *SELECTED_GAME.write().unwrap() = Some(game);
}

fn selected_game() -> AtariGame {
    SELECTED_GAME
        .read()
        .unwrap()
        .expect("environment: Atari requires `atari_game` to be set in the config")
}

const FRAME_WIDTH: usize = 84;
const FRAME_HEIGHT: usize = 84;
const FULL_ACTION_SET: usize = 18;
const FRAME_SKIP: usize = 4;

#[derive(Clone)]
pub struct AtariState {
    pub obs: Vec<f32>,
}

pub struct AtariEnv {
    ale: Ale,
    game: AtariGame,
    step_index: usize,
    current_obs: Vec<f32>,
    legal_actions: Vec<i32>,
}

impl Default for AtariEnv {
    fn default() -> Self {
        Self::new(selected_game())
    }
}

impl Clone for AtariEnv {
    fn clone(&self) -> Self {
        Self::new(self.game)
    }
}

impl AtariEnv {
    pub fn new(game: AtariGame) -> Self {
        let mut ale = Ale::new();
        ale.load_rom(game.rom()).expect("failed to load Atari ROM");
        let mut env = Self {
            ale,
            game,
            step_index: 0,
            current_obs: vec![0.0; FRAME_WIDTH * FRAME_HEIGHT],
            legal_actions: Vec::new(),
        };
        env.legal_actions = env.ale.minimal_action_set().to_vec();
        env.capture_obs();
        env
    }

    fn capture_obs(&mut self) {
        let w = self.ale.screen_width();
        let h = self.ale.screen_height();
        let mut buf = vec![0u8; w * h];
        self.ale.get_screen_grayscale(&mut buf);
        let mut out = vec![0.0f32; FRAME_WIDTH * FRAME_HEIGHT];
        for y in 0..FRAME_HEIGHT {
            let sy = y * h / FRAME_HEIGHT;
            for x in 0..FRAME_WIDTH {
                let sx = x * w / FRAME_WIDTH;
                out[y * FRAME_WIDTH + x] = buf[sy * w + sx] as f32 / 255.0;
            }
        }
        self.current_obs = out;
    }

    pub fn rgb_frame(&mut self) -> (usize, usize, Vec<u8>) {
        let w = self.ale.screen_width();
        let h = self.ale.screen_height();
        let mut buf = vec![0u8; w * h * 3];
        self.ale.get_screen_rgb(&mut buf);
        (w, h, buf)
    }

    pub fn game(&self) -> AtariGame {
        self.game
    }
}

impl Environment for AtariEnv {
    type State = AtariState;
    type Action = usize;

    const MAX_STEPS: usize = 27_000;

    const INFO: EnvInfo = EnvInfo {
        obs_shape: &[1, FRAME_HEIGHT, FRAME_WIDTH],
        action_size: FULL_ACTION_SET,
        num_players: 1,
        lower_reward_bound: None,
        upper_reward_bound: None,
    };

    fn state(&self) -> Self::State {
        AtariState {
            obs: self.current_obs.clone(),
        }
    }

    fn obs(&self) -> Vec<f32> {
        self.current_obs.clone()
    }

    fn step(&mut self, action: usize) -> StepResult<Self::State> {
        let ale_action = action as i32;
        let mut reward = 0.0f64;
        let mut done = false;
        for _ in 0..FRAME_SKIP {
            reward += self.ale.act(ale_action) as f64;
            if self.ale.is_game_over() {
                done = true;
                break;
            }
        }
        self.step_index += 1;
        self.capture_obs();
        StepResult {
            next_state: self.state(),
            reward,
            done,
            truncated: self.step_index >= Self::MAX_STEPS,
        }
    }

    fn reset(&mut self) {
        self.ale.reset_game();
        self.step_index = 0;
        self.legal_actions = self.ale.minimal_action_set().to_vec();
        self.capture_obs();
    }

    fn legal_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; Self::INFO.action_size];
        for &a in &self.legal_actions {
            if (a as usize) < mask.len() {
                mask[a as usize] = true;
            }
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_rom_and_steps() {
        let mut env = AtariEnv::new(AtariGame::Breakout);
        assert_eq!(env.obs().len(), FRAME_WIDTH * FRAME_HEIGHT);
        assert_eq!(env.legal_mask().len(), FULL_ACTION_SET);
        assert!(env.legal_mask().iter().any(|&b| b));

        env.reset();
        let action = env.legal_mask().iter().position(|&b| b).unwrap();
        let result = env.step(action);
        assert_eq!(result.next_state.obs.len(), FRAME_WIDTH * FRAME_HEIGHT);
        assert!(!result.truncated);
    }
}
