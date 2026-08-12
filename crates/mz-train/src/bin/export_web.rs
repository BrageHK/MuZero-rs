//! Converts a trained Othello checkpoint into the two artifacts `mz-web` embeds:
//! a flat `.bin` record (no filesystem needed to load it) and the network/search
//! constants as generated Rust.
//!
//! `cargo run -r --bin export_web [checkpoint-without-extension]`

use std::fs;
use std::path::Path;

use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{BinFileRecorder, CompactRecorder, HalfPrecisionSettings, Recorder};
use mz_rs::mz_config::{EnvironmentName, MuZeroConfig, SearchAlgorithm};
use mz_rs::networks::mlp::MlpNets;

type B = NdArray;

const ASSETS: &str = "crates/mz-web/assets";

fn main() {
    let mz_conf = MuZeroConfig::default();
    assert!(
        matches!(mz_conf.environment, EnvironmentName::Othello),
        "mz-web is Othello-only, but configs/config.yaml says {:?}",
        mz_conf.environment
    );
    assert!(
        matches!(mz_conf.search_algorithm, SearchAlgorithm::Gumbel),
        "mz-web only implements Gumbel search, but configs/config.yaml says {:?}",
        mz_conf.search_algorithm
    );

    let checkpoint = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("model/{}/latest", mz_conf.environment.as_ref()));
    let device = Default::default();

    // The training binaries always build MlpNets, whatever `network_type` says.
    let agent: MlpNets<B> = mz_conf.init(&device);
    let agent = agent
        .load_file(&checkpoint, &CompactRecorder::new(), &device)
        .unwrap_or_else(|e| panic!("Failed to load checkpoint '{checkpoint}': {e}"));

    fs::create_dir_all(ASSETS).expect("Failed to create the assets directory");
    let weights = Path::new(ASSETS).join("othello");
    BinFileRecorder::<HalfPrecisionSettings>::new()
        .record(agent.into_record(), weights.clone())
        .expect("Failed to write the web weights");

    let net = mz_conf.net_config();
    let linear = net.linear();
    let gumbel = mz_conf.gumbel();
    let layer = |name: &str, conf: &mz_rs::mz_config::NetworkSubConfig| {
        format!(
            "        {name}: NetworkSubConfig {{\n            latent_space_dims: {},\n            fc_hidden_size: {},\n            n_layers: {},\n        }},\n",
            conf.latent_space_dims, conf.fc_hidden_size, conf.n_layers
        )
    };

    let generated = format!(
        "pub const NET: NetConfig = NetConfig {{\n\
         \x20   network_type: NetworkType::Linear,\n\
         \x20   obs_dim: {obs_dim},\n\
         \x20   action_space: {action_space},\n\
         \x20   support_size: {support_size},\n\
         \x20   categorical: {categorical},\n\
         \x20   board_height: {board_height},\n\
         \x20   board_width: {board_width},\n\
         \x20   obs_channels: {obs_channels},\n\
         \x20   linear: Some(LinearSubConfig {{\n{repr}{dynamic}{prediction}\x20   }}),\n\
         \x20   resnet: None,\n\
         \x20   projection: ProjectionSubConfig {{\n\
         \x20       proj_hidden: {proj_hidden},\n\
         \x20       proj_out: {proj_out},\n\
         \x20       pred_hidden: {pred_hidden},\n\
         \x20   }},\n\
         }};\n\
         \n\
         pub const SEARCH: SearchParams = SearchParams {{\n\
         \x20   num_simulations: {num_simulations},\n\
         \x20   action_space: {action_space},\n\
         \x20   support_size: {support_size},\n\
         \x20   discount: {discount:?},\n\
         \x20   max_num_considered_actions: {max_considered},\n\
         \x20   c_visit: {c_visit:?},\n\
         \x20   c_scale: {c_scale:?},\n\
         \x20   two_player: {two_player},\n\
         }};\n",
        obs_dim = net.obs_dim,
        action_space = net.action_space,
        support_size = net.support_size,
        categorical = net.categorical,
        board_height = net.board_height,
        board_width = net.board_width,
        obs_channels = net.obs_channels,
        repr = layer("representation", &linear.representation),
        dynamic = layer("dynamic", &linear.dynamic),
        prediction = layer("prediction", &linear.prediction),
        proj_hidden = net.projection.proj_hidden,
        proj_out = net.projection.proj_out,
        pred_hidden = net.projection.pred_hidden,
        num_simulations = mz_conf.num_simulations,
        discount = mz_conf.discount,
        max_considered = gumbel.max_num_considered_actions,
        c_visit = gumbel.c_visit,
        c_scale = gumbel.c_scale,
        two_player = mz_conf.is_twoplayer,
    );

    let config_path = Path::new(ASSETS).join("net_config.rs");
    fs::write(&config_path, generated).expect("Failed to write the web net config");

    let size = fs::metadata(format!("{}.bin", weights.display()))
        .map(|m| m.len())
        .unwrap_or(0);
    println!("wrote {}.bin ({size} bytes)", weights.display());
    println!("wrote {}", config_path.display());
}
