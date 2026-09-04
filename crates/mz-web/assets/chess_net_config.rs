pub type Net<B> = mz_core::networks::resnet::ResNets<B>;

pub const NET: NetConfig = NetConfig {
    network_type: NetworkType::ResNet,
    obs_dim: 7616,
    action_space: 4672,
    support_size: 35,
    categorical: false,
    board_height: 8,
    board_width: 8,
    obs_channels: 119,
    linear: None,
    resnet: Some(ResNetSubConfig {
        representation: ResNetRepresentationConfig {
            channels: 64,
            n_blocks: 4,
            gpool: GPoolConfig { every: 0, pool_channels: 16 },
        },
        dynamic: ResNetBlockConfig {
            channels: 64,
            n_blocks: 2,
            fc_hidden_size: 64,
            gpool: GPoolConfig { every: 0, pool_channels: 0 },
        },
        prediction: ResNetBlockConfig {
            channels: 64,
            n_blocks: 4,
            fc_hidden_size: 64,
            gpool: GPoolConfig { every: 0, pool_channels: 0 },
        },
        head_gpool: HeadPoolConfig { policy_channels: 64, value_channels: 64 },
    }),
    projection: ProjectionSubConfig {
        proj_hidden: 64,
        proj_out: 64,
        pred_hidden: 32,
    },
};

pub const SEARCH: SearchParams = SearchParams {
    num_simulations: 32,
    action_space: 4672,
    support_size: 35,
    discount: 1.0,
    max_num_considered_actions: 8,
    c_visit: 50.0,
    c_scale: 0.1,
    two_player: true,
};
