//! The wasm search is a trimmed re-implementation of mz-train's `batched_search`.
//! This pins the two together: same weights, same position, deterministic Gumbel
//! (all zeros) => same move and same root value.
//!
//! cargo test -p mz-web --no-default-features --features ndarray

#![cfg(all(feature = "ndarray", not(feature = "webgpu")))]

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use mz_core::networks::mlp::MlpNets;
use mz_core::othello::Othello;
use mz_rs::env::Environment;
use mz_rs::mz_config::{GumbelSubConfig, MuZeroConfig, SearchAlgorithm};
use mz_rs::search::batched_search;
use mz_web::model::SEARCH;
use mz_web::search::gumbel_search;

type B = NdArray;

/// The ndarray backend resolves its readbacks immediately, so a single poll is
/// enough to drive the async search on the host.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        match future.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(output) => return output,
            std::task::Poll::Pending => std::hint::spin_loop(),
        }
    }
}

fn config(num_simulations: usize) -> MuZeroConfig {
    MuZeroConfig {
        search_algorithm: SearchAlgorithm::Gumbel,
        gumbel: Some(GumbelSubConfig {
            max_num_considered_actions: SEARCH.max_num_considered_actions,
            c_visit: SEARCH.c_visit,
            c_scale: SEARCH.c_scale,
        }),
        num_simulations,
        discount: SEARCH.discount,
        support_size: SEARCH.support_size,
        action_space: SEARCH.action_space,
        obs_dim: 64,
        is_twoplayer: true,
        board_height: 8,
        board_width: 8,
        rayon_min_chunk_len: 1,
        ..Default::default()
    }
}

#[test]
fn minimal_search_matches_batched_search() {
    let device = NdArrayDevice::default();
    let num_simulations = 16;
    let mz_conf = config(num_simulations);
    // Same shapes as the exported net, freshly initialised: parity is about the
    // search, and both sides see identical weights.
    let net: MlpNets<B> = mz_conf.init(&device);
    let params = SEARCH;

    let mut rng = fastrand::Rng::with_seed(5);
    let mut env = Othello::new();
    let mut positions = 0;

    while positions < 20 {
        if env.is_over() {
            env = Othello::new();
            continue;
        }

        let obs = env.obs();
        let mask = env.legal_mask();

        let batched = batched_search(
            env.state_tensor::<B>(&device),
            Some(std::slice::from_ref(&mask)),
            &mz_conf,
            &net,
            0.0,
            false,
        );
        let minimal = block_on(gumbel_search(
            &net,
            &device,
            &obs,
            &mask,
            &params,
            num_simulations,
            None,
        ));

        assert_eq!(
            batched[0].best_action, minimal.best_action,
            "action mismatch at\n{env}"
        );
        assert!(
            (batched[0].value - minimal.value).abs() < 1e-4,
            "value mismatch: {} vs {}",
            batched[0].value,
            minimal.value
        );
        for (a, (want, got)) in batched[0]
            .policy_target
            .iter()
            .zip(&minimal.policy)
            .enumerate()
        {
            assert!(
                (want - got).abs() < 1e-4,
                "policy mismatch on action {a}: {want} vs {got}"
            );
        }

        let actions: Vec<usize> = env.legal_actions().collect();
        env.apply(actions[rng.usize(..actions.len())]);
        positions += 1;
    }
}
