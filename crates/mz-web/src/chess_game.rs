//! MuZero-powered chess opponent, an alternative to `chess_bot`'s alpha-beta
//! heuristic. Stateful (unlike `chess_bot_move`) because the trained net's
//! observation stacks up to 8 plies of history: the page must replay every
//! ply — human and bot — through `play_uci` so that history matches what the
//! net saw during self-play, not just hand it the current FEN each move.

use mz_core::chess::Chess;
#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::wasm_bindgen;

use crate::model::{self, Be, Device};
use crate::rng::Rng;
use crate::search::gumbel_search;

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct ChessAgentMove {
    #[cfg_attr(target_family = "wasm", wasm_bindgen(getter_with_clone))]
    pub uci: String,
    /// Root value for the side to move, in [-1, 1].
    pub value: f32,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct ChessGame {
    env: Chess,
    net: model::chess::Net<Be>,
    device: Device,
    simulations: usize,
}

// Same reasoning as `Game`'s impl: search never mutates the game.
impl core::panic::RefUnwindSafe for ChessGame {}

/// Builds the agent: sets up the backend and loads the embedded weights.
#[cfg_attr(target_family = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub async fn create_chess(simulations: u32) -> ChessGame {
    let device = Device::default();
    model::init_backend(&device).await;
    ChessGame {
        env: Chess::new(),
        net: model::chess::load(&device),
        device,
        simulations: (simulations as usize).clamp(1, 800),
    }
}

#[cfg_attr(target_family = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
impl ChessGame {
    /// Applies `uci` (e.g. `e2e4`, `e7e8q`) if it names a legal move in the
    /// current position. Returns whether it was applied.
    pub fn play_uci(&mut self, uci: &str) -> bool {
        let action = self
            .env
            .legal_actions()
            .find(|&action| self.env.action_to_move(action).is_some_and(|mv| mv.to_string() == uci));
        match action {
            Some(action) => {
                self.env.apply(action);
                true
            }
            None => false,
        }
    }

    pub fn reset(&mut self) {
        self.env.reset();
    }

    pub fn is_over(&self) -> bool {
        self.env.is_over()
    }

    pub fn simulations(&self) -> u32 {
        self.simulations as u32
    }

    pub fn set_simulations(&mut self, simulations: u32) {
        self.simulations = (simulations as usize).clamp(1, 800);
    }

    /// Runs the search for the side to move without touching the game state;
    /// commit the result with `play_uci`. `seed` of 0 makes the move deterministic.
    pub async fn think(&self, seed: f64) -> ChessAgentMove {
        let seed = seed.abs() as u64;
        let result = gumbel_search(
            &self.net,
            &self.device,
            &self.env.obs(),
            &self.env.legal_mask(),
            &model::chess::SEARCH,
            self.simulations,
            (seed != 0).then(|| Rng::new(seed)),
        )
        .await;
        let uci = self
            .env
            .action_to_move(result.best_action)
            .map(|mv| mv.to_string())
            .unwrap_or_default();
        ChessAgentMove { uci, value: result.value }
    }
}
