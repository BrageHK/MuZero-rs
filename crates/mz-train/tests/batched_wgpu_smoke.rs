#![cfg(feature = "wgpu")]
use burn::Tensor;
use burn::backend::Wgpu;
use mz_rs::agent::MlpNets;
use mz_rs::mz_config::MuZeroConfig;
use mz_rs::search::batched_search;

#[test]
fn batched_search_wgpu() {
    let mz_conf = MuZeroConfig::default();
    let device = Default::default();
    let agent: MlpNets<Wgpu> = mz_conf.init(&device);
    let obs = Tensor::<Wgpu, 2>::random(
        [4, mz_conf.obs_dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let results = batched_search(obs, &mz_conf, &agent, 1.0);
    assert_eq!(results.len(), 4);
    for res in &results {
        let sum: f32 = res.distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "distribution sums to {sum}");
    }
}
