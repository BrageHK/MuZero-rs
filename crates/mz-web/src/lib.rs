pub mod model;
pub mod rng;
pub mod search;

use mz_core::othello::{Othello, PASS};

use crate::model::{Be, Device, Net, SEARCH};
use crate::rng::Rng;
use crate::search::gumbel_search;

#[cfg_attr(target_family = "wasm", wasm_bindgen::prelude::wasm_bindgen(start))]
pub fn start() {
    #[cfg(target_family = "wasm")]
    console_error_panic_hook::set_once();
}

#[cfg_attr(target_family = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
#[derive(Clone, Copy)]
pub struct AgentMove {
    pub action: u32,
    /// Root value for the side to move, in [-1, 1].
    pub value: f32,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct Game {
    env: Othello,
    net: Net<Be>,
    device: Device,
    simulations: usize,
    history: Vec<Othello>,
    last_move: Option<usize>,
}

// The wasm shims assert this for every exported `&self` method. Nothing here is
// interior-mutable from the JS side: search never mutates the game.
impl core::panic::RefUnwindSafe for Game {}

/// Builds the agent: sets up the backend and loads the embedded weights.
#[cfg_attr(target_family = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub async fn create(simulations: u32) -> Game {
    let device = Device::default();
    model::init_backend(&device).await;
    Game {
        env: Othello::new(),
        net: model::load(&device),
        device,
        simulations: (simulations as usize).clamp(1, 512),
        history: Vec::new(),
        last_move: None,
    }
}

#[cfg_attr(target_family = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
impl Game {
    /// 64 squares in absolute colours: 1 black, -1 white, 0 empty.
    pub fn board(&self) -> Vec<i8> {
        let (black, white) = self.env.absolute();
        (0..64)
            .map(|square| {
                let bit = 1u64 << square;
                if black & bit != 0 {
                    1
                } else if white & bit != 0 {
                    -1
                } else {
                    0
                }
            })
            .collect()
    }

    /// 65 flags; index 64 is pass.
    pub fn legal_mask(&self) -> Vec<u8> {
        self.env
            .legal_mask()
            .into_iter()
            .map(|legal| legal as u8)
            .collect()
    }

    pub fn black_to_move(&self) -> bool {
        self.env.black_to_move()
    }

    pub fn is_over(&self) -> bool {
        self.env.is_over()
    }

    pub fn must_pass(&self) -> bool {
        self.env.must_pass()
    }

    /// [black stones, white stones]
    pub fn counts(&self) -> Vec<u32> {
        let (black, white) = self.env.counts();
        vec![black, white]
    }

    /// Square of the most recent move: 64 for a pass, 255 for none.
    pub fn last_move(&self) -> u32 {
        self.last_move.map_or(255, |action| action as u32)
    }

    pub fn simulations(&self) -> u32 {
        self.simulations as u32
    }

    pub fn set_simulations(&mut self, simulations: u32) {
        self.simulations = (simulations as usize).clamp(1, 512);
    }

    pub fn play(&mut self, action: usize) -> bool {
        if !self.env.is_legal(action) {
            return false;
        }
        self.history.push(self.env);
        self.env.apply(action);
        self.last_move = Some(action);
        true
    }

    /// Runs the search for the side to move without touching the game state;
    /// commit the result with `play`. `seed` of 0 makes the move deterministic.
    pub async fn think(&self, seed: f64) -> AgentMove {
        let seed = seed.abs() as u64;
        let result = gumbel_search(
            &self.net,
            &self.device,
            &self.env.obs(),
            &self.env.legal_mask(),
            &SEARCH,
            self.simulations,
            (seed != 0).then(|| Rng::new(seed)),
        )
        .await;
        AgentMove {
            action: result.best_action as u32,
            value: result.value,
        }
    }

    /// Rewinds a full round so the human is on turn again.
    pub fn undo(&mut self) {
        for _ in 0..2 {
            if let Some(previous) = self.history.pop() {
                self.env = previous;
            }
        }
        self.last_move = None;
    }

    pub fn can_undo(&self) -> bool {
        !self.history.is_empty()
    }

    pub fn reset(&mut self) {
        self.env = Othello::new();
        self.history.clear();
        self.last_move = None;
    }

    pub fn pass_action(&self) -> u32 {
        PASS as u32
    }
}
