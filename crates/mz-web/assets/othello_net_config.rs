pub type Net<B> = mz_core::networks::resnet::ResNets<B>;

pub const NET: NetConfig = NetConfig {
    network_type: NetworkType::ResNet,
    obs_dim: 192,
    action_space: 65,
    support_size: 35,
    categorical: false,
    board_height: 8,
    board_width: 8,
    obs_channels: 3,
    linear: None,
    resnet: Some(ResNetSubConfig {
        representation: ResNetRepresentationConfig {
            channels: 64,
            n_blocks: 15,
            gpool: GPoolConfig { every: 3, pool_channels: 16 },
        },
        dynamic: ResNetBlockConfig {
            channels: 64,
            n_blocks: 2,
            fc_hidden_size: 1024,
            gpool: GPoolConfig { every: 0, pool_channels: 0 },
        },
        prediction: ResNetBlockConfig {
            channels: 64,
            n_blocks: 10,
            fc_hidden_size: 1024,
            gpool: GPoolConfig { every: 0, pool_channels: 0 },
        },
        head_gpool: HeadPoolConfig { policy_channels: 16, value_channels: 16 },
    }),
    projection: ProjectionSubConfig {
        proj_hidden: 128,
        proj_out: 32,
        pred_hidden: 64,
    },
};

pub const SEARCH: SearchParams = SearchParams {
    num_simulations: 16,
    action_space: 65,
    support_size: 35,
    discount: 1.0,
    max_num_considered_actions: 8,
    c_visit: 50.0,
    c_scale: 0.1,
    two_player: true,
};
