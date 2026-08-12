#![cfg(feature = "ndarray")]

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;

use mz_rs::agent::MlpNets;
use mz_rs::env::Environment;
use mz_rs::env::othello::env::Othello;
use mz_rs::env::tictactoe::env::TicTacToe;
use mz_rs::eval::EloLadder;
use mz_rs::mz_config::{
    EnvironmentName, EvalConfig, GumbelSubConfig, MuZeroConfig, RungConfig, SearchAlgorithm,
};

fn config(environment: EnvironmentName, ladder: Vec<RungConfig>) -> MuZeroConfig {
    let (action_space, obs_dim) = match environment {
        EnvironmentName::TicTacToe => (TicTacToe::INFO.action_size, TicTacToe::INFO.obs_dim()),
        EnvironmentName::Othello => (Othello::INFO.action_size, Othello::INFO.obs_dim()),
        _ => panic!("not a board game"),
    };

    MuZeroConfig {
        environment,
        action_space,
        obs_dim,
        is_twoplayer: true,
        search_algorithm: SearchAlgorithm::Gumbel,
        gumbel: Some(GumbelSubConfig {
            max_num_considered_actions: 8,
            c_visit: 50.0,
            c_scale: 0.1,
        }),
        num_simulations: 4,
        rayon_min_chunk_len: 8,
        eval: Some(EvalConfig {
            interval: 100,
            games: 20,
            random_opening_plies: 2,
            ladder: Some(ladder),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn run(mz_conf: &MuZeroConfig) -> mz_rs::eval::EvalReading {
    let device = NdArrayDevice::default();
    let agent: MlpNets<NdArray> = mz_conf.init(&device);
    let mut ladder = EloLadder::new(mz_conf);
    assert!(ladder.due(0), "ladder should evaluate on the first step");
    ladder.run(mz_conf, &agent, &device, 0)
}

#[test]
fn tictactoe_eval_tallies_every_game() {
    let mz_conf = config(
        EnvironmentName::TicTacToe,
        vec![RungConfig::Random { elo: 0.0 }],
    );
    let reading = run(&mz_conf);

    assert_eq!(reading.result.games(), 20);
    assert_eq!(reading.opponent, "Random");
    assert!(reading.elo.is_finite());
}

#[test]
fn othello_eval_tallies_every_game() {
    let mz_conf = config(
        EnvironmentName::Othello,
        vec![RungConfig::AlphaBeta {
            depth: 1,
            epsilon: 0.0,
            elo: 500.0,
        }],
    );
    let reading = run(&mz_conf);

    assert_eq!(reading.result.games(), 20);
    assert_eq!(reading.opponent, "AlphaBeta(d=1)");
    // An untrained net should not be beating a heuristic opponent.
    assert!(
        reading.result.score() < 0.5,
        "score {}",
        reading.result.score()
    );
    assert!(reading.elo < 500.0);
}

#[test]
fn eval_only_runs_on_the_interval() {
    let mz_conf = config(
        EnvironmentName::TicTacToe,
        vec![RungConfig::Random { elo: 0.0 }],
    );
    let device = NdArrayDevice::default();
    let agent: MlpNets<NdArray> = mz_conf.init(&device);
    let mut ladder = EloLadder::new(&mz_conf);

    ladder.run(&mz_conf, &agent, &device, 0);
    assert!(!ladder.due(99));
    assert!(ladder.due(100));
}

#[test]
fn single_player_environments_have_no_ladder() {
    let mz_conf = MuZeroConfig {
        environment: EnvironmentName::CartPole,
        is_twoplayer: false,
        ..Default::default()
    };
    assert!(!EloLadder::new(&mz_conf).enabled());
}
