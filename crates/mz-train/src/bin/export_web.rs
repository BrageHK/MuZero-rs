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
use mz_rs::mz_config::{EnvironmentName, MuZeroConfig, NetworkType, SearchAlgorithm};
use mz_rs::with_net;

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

    fs::create_dir_all(ASSETS).expect("Failed to create the assets directory");
    let weights = Path::new(ASSETS).join("othello");

    // The checkpoint was trained with whatever `network_type` says, so the
    // export must build the matching family or the record fields won't line up.
    with_net!(mz_conf, Net => {
        let agent: Net<B> = mz_conf.init(&device);
        let agent = agent
            .load_file(&checkpoint, &CompactRecorder::new(), &device)
            .unwrap_or_else(|e| panic!("Failed to load checkpoint '{checkpoint}': {e}"));
        BinFileRecorder::<HalfPrecisionSettings>::new()
            .record(agent.into_record(), weights.clone())
            .expect("Failed to write the web weights");
    });

    let net_family_path = match mz_conf.network_type {
        NetworkType::Linear => "mz_core::networks::mlp::MlpNets",
        NetworkType::ResNet => "mz_core::networks::resnet::ResNets",
    };

    let net = mz_conf.net_config();
    let gumbel = mz_conf.gumbel();

    let layer = |name: &str, conf: &mz_rs::mz_config::NetworkSubConfig| {
        format!(
            "        {name}: NetworkSubConfig {{\n            latent_space_dims: {},\n            fc_hidden_size: {},\n            n_layers: {},\n        }},\n",
            conf.latent_space_dims, conf.fc_hidden_size, conf.n_layers
        )
    };
    let gpool_repr = |every: usize, pool_channels: usize| {
        format!("GPoolConfig {{ every: {every}, pool_channels: {pool_channels} }}")
    };

    let (network_type, linear_field, resnet_field) = match mz_conf.network_type {
        NetworkType::Linear => {
            let linear = net.linear();
            let repr = layer("representation", &linear.representation);
            let dynamic = layer("dynamic", &linear.dynamic);
            let prediction = layer("prediction", &linear.prediction);
            (
                "NetworkType::Linear",
                format!("Some(LinearSubConfig {{\n{repr}{dynamic}{prediction}\x20   }})"),
                "None".to_string(),
            )
        }
        NetworkType::ResNet => {
            let resnet = net.resnet();
            let repr = format!(
                "        representation: ResNetRepresentationConfig {{\n            channels: {},\n            n_blocks: {},\n            gpool: {},\n        }},\n",
                resnet.representation.channels,
                resnet.representation.n_blocks,
                gpool_repr(
                    resnet.representation.gpool.every,
                    resnet.representation.gpool.pool_channels
                ),
            );
            let block = |name: &str, conf: &mz_rs::mz_config::ResNetBlockConfig| {
                format!(
                    "        {name}: ResNetBlockConfig {{\n            channels: {},\n            n_blocks: {},\n            fc_hidden_size: {},\n            gpool: {},\n        }},\n",
                    conf.channels,
                    conf.n_blocks,
                    conf.fc_hidden_size,
                    gpool_repr(conf.gpool.every, conf.gpool.pool_channels),
                )
            };
            let dynamic = block("dynamic", &resnet.dynamic);
            let prediction = block("prediction", &resnet.prediction);
            let head_gpool = format!(
                "HeadPoolConfig {{ policy_channels: {}, value_channels: {} }}",
                resnet.head_gpool.policy_channels, resnet.head_gpool.value_channels
            );
            (
                "NetworkType::ResNet",
                "None".to_string(),
                format!(
                    "Some(ResNetSubConfig {{\n{repr}{dynamic}{prediction}\x20       head_gpool: {head_gpool},\n\x20   }})"
                ),
            )
        }
    };

    let generated = format!(
        "pub type Net<B> = {net_family_path}<B>;\n\
         \n\
         pub const NET: NetConfig = NetConfig {{\n\
         \x20   network_type: {network_type},\n\
         \x20   obs_dim: {obs_dim},\n\
         \x20   action_space: {action_space},\n\
         \x20   support_size: {support_size},\n\
         \x20   categorical: {categorical},\n\
         \x20   board_height: {board_height},\n\
         \x20   board_width: {board_width},\n\
         \x20   obs_channels: {obs_channels},\n\
         \x20   linear: {linear_field},\n\
         \x20   resnet: {resnet_field},\n\
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
