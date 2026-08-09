pub const NET: NetConfig = NetConfig {
    network_type: NetworkType::Linear,
    obs_dim: 64,
    action_space: 65,
    support_size: 35,
    categorical: false,
    board_height: 8,
    board_width: 8,
    obs_channels: 1,
    linear: Some(LinearSubConfig {
        representation: NetworkSubConfig {
            latent_space_dims: 32,
            fc_hidden_size: 32,
            n_layers: 3,
        },
        dynamic: NetworkSubConfig {
            latent_space_dims: 32,
            fc_hidden_size: 32,
            n_layers: 3,
        },
        prediction: NetworkSubConfig {
            latent_space_dims: 32,
            fc_hidden_size: 64,
            n_layers: 3,
        },
    }),
    resnet: None,
    projection: ProjectionSubConfig {
        proj_hidden: 256,
        proj_out: 64,
        pred_hidden: 128,
    },
};

pub const SEARCH: SearchParams = SearchParams {
    num_simulations: 16,
    action_space: 65,
    support_size: 35,
    discount: 0.997,
    max_num_considered_actions: 8,
    c_visit: 50.0,
    c_scale: 0.1,
    two_player: true,
};
