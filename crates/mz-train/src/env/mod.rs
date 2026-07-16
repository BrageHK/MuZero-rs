pub mod cartpole;
pub mod othello;
pub mod tictactoe;

use crate::mz_config::MuZeroConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvInfo {
    pub obs_shape: &'static [usize],
    pub action_size: usize,
    pub num_players: usize,
}

impl EnvInfo {
    pub const fn obs_dim(&self) -> usize {
        let mut dim = 1;
        let mut i = 0;
        while i < self.obs_shape.len() {
            dim *= self.obs_shape[i];
            i += 1;
        }
        dim
    }
}

pub trait MuZeroEnv: burn::rl::Environment {
    const INFO: EnvInfo;

    fn legal_mask(&self) -> Vec<bool>;
}

impl MuZeroConfig {
    pub fn env_info(&self) -> EnvInfo {
        crate::with_env!(self, E => <E as MuZeroEnv>::INFO)
    }

    pub fn action_space(&self) -> usize {
        self.env_info().action_size
    }

    pub fn obs_dim(&self) -> usize {
        self.env_info().obs_dim()
    }
}

/// Runs `$body` with `$E` bound to the concrete environment type selected by
/// `mz_conf.environment`.
///
/// `burn::rl::Environment` has associated types (`State`, `Action`) and a
/// `const MAX_STEPS`, so it is not dyn-compatible and a function cannot return
/// "some environment". Instead, dispatch into generic code:
///
/// ```ignore
/// use burn::rl::Environment;
/// use mz_rs::with_env;
///
/// with_env!(mz_conf, E => {
///     let mut env = E::default();
///     env.reset();
///     // generic training/search code over E
/// });
/// ```
#[macro_export]
macro_rules! with_env {
    ($mz_conf:expr, $E:ident => $body:expr) => {
        match $mz_conf.environment {
            $crate::mz_config::Environment::CartPole => {
                type $E = $crate::env::cartpole::env::CartPoleWrapper;
                $body
            }
            $crate::mz_config::Environment::TicTacToe => {
                type $E = $crate::env::tictactoe::env::TicTacToe;
                $body
            }
            $crate::mz_config::Environment::Othello => {
                type $E = $crate::env::othello::env::Othello;
                $body
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use burn::rl::Environment;

    use crate::mz_config::{self, MuZeroConfig};

    #[test]
    fn dispatches_to_env_selected_by_config() {
        let mut conf = MuZeroConfig::default();

        conf.environment = mz_config::Environment::TicTacToe;
        let max_steps = with_env!(conf, E => E::MAX_STEPS);
        assert_eq!(max_steps, crate::env::tictactoe::env::TicTacToe::MAX_STEPS);

        conf.environment = mz_config::Environment::CartPole;
        with_env!(conf, E => {
            let mut env = E::default();
            env.reset();
            let _state = env.state();
        });
    }
}
