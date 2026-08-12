use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{IndexingUpdateOp, Int, backend::Backend},
};

use crate::config::NetConfig;
use crate::networks::MuZeroNets;
use crate::networks::projector::{ConvProjection, ConvProjectionConfig};
use crate::networks::resblock::{ResBlock, ResBlockConfig};

#[derive(Config, Debug)]
pub struct ResNetConfig {
    pub obs_channels: usize,
    pub board_height: usize,
    pub board_width: usize,
    pub action_space: usize,
    pub value_support: usize,
    pub reward_support: usize,
    pub proj_hidden: usize,
    pub proj_out: usize,
    pub pred_hidden: usize,

    // Hidden-state channel count must match across all three blocks: the
    // representation output is fed straight into dynamics' fuse conv, and
    // dynamics/representation outputs both flow through prediction/projection.
    pub representation_channels: usize,
    pub representation_n_blocks: usize,

    pub dynamic_channels: usize,
    pub dynamic_n_blocks: usize,
    pub dynamic_fc_hidden_size: usize,

    pub prediction_channels: usize,
    pub prediction_n_blocks: usize,
    pub prediction_fc_hidden_size: usize,
}

fn conv3x3<B: Backend>(c_in: usize, c_out: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([c_in, c_out], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .init(device)
}

fn conv1x1<B: Backend>(c_in: usize, c_out: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([c_in, c_out], [1, 1]).init(device)
}

#[derive(Module, Debug)]
pub struct ResNetRepresentation<B: Backend> {
    stem: Conv2d<B>,
    stem_bn: BatchNorm<B>,
    blocks: Vec<ResBlock<B>>,
    relu: Relu,
}

impl<B: Backend> ResNetRepresentation<B> {
    pub fn forward(&self, obs: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self
            .relu
            .forward(self.stem_bn.forward(self.stem.forward(obs)));
        for block in &self.blocks {
            x = block.forward(x);
        }
        x
    }
}

#[derive(Module, Debug)]
pub struct ResNetDynamics<B: Backend> {
    fuse: Conv2d<B>,
    fuse_bn: BatchNorm<B>,
    blocks: Vec<ResBlock<B>>,
    reward_conv: Conv2d<B>,
    reward_bn: BatchNorm<B>,
    reward_fc1: Linear<B>,
    reward_fc2: Linear<B>,
    relu: Relu,
}

impl<B: Backend> ResNetDynamics<B> {
    /// Returns (hidden_state, reward).
    pub fn forward(
        &self,
        hidden: Tensor<B, 4>,
        action: Tensor<B, 1, Int>,
        action_size: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 2>) {
        let [n, _, h, w] = hidden.dims();
        let device = hidden.device();
        let action_planes = Tensor::<B, 2>::zeros([n, action_size], &device)
            .scatter(
                1,
                action.reshape([n, 1]),
                Tensor::<B, 2>::ones([n, 1], &device),
                IndexingUpdateOp::Add,
            )
            .reshape([n, action_size, 1, 1])
            .expand([n, action_size, h, w]);

        let x = Tensor::cat(vec![hidden, action_planes], 1);
        let mut x = self
            .relu
            .forward(self.fuse_bn.forward(self.fuse.forward(x)));
        for block in &self.blocks {
            x = block.forward(x);
        }

        let reward = self
            .relu
            .forward(self.reward_bn.forward(self.reward_conv.forward(x.clone())));
        let reward = reward.reshape([n as i32, -1]);
        let reward = self.relu.forward(self.reward_fc1.forward(reward));
        let reward = self.reward_fc2.forward(reward);

        (x, reward)
    }
}

#[derive(Module, Debug)]
pub struct ResNetPrediction<B: Backend> {
    policy_conv: Conv2d<B>,
    policy_bn: BatchNorm<B>,
    policy_fc: Linear<B>,
    value_conv: Conv2d<B>,
    value_bn: BatchNorm<B>,
    value_fc1: Linear<B>,
    value_fc2: Linear<B>,
    relu: Relu,
}

impl<B: Backend> ResNetPrediction<B> {
    /// Returns (value_logits, policy_logits)
    pub fn forward(&self, hidden: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let n = hidden.dims()[0] as i32;

        let policy = self.relu.forward(
            self.policy_bn
                .forward(self.policy_conv.forward(hidden.clone())),
        );
        let policy = self.policy_fc.forward(policy.reshape([n, -1]));

        let value = self
            .relu
            .forward(self.value_bn.forward(self.value_conv.forward(hidden)));
        let value = self
            .relu
            .forward(self.value_fc1.forward(value.reshape([n, -1])));
        let value = self.value_fc2.forward(value);

        (value, policy)
    }
}

/// The ResNet (conv) MuZero network family.
#[derive(Module, Debug)]
pub struct ResNets<B: Backend> {
    pub representation: ResNetRepresentation<B>,
    pub dynamics: ResNetDynamics<B>,
    pub prediction: ResNetPrediction<B>,
    pub projection: ConvProjection<B>,
    dynamic_channels: usize,
    prediction_channels: usize,
    obs_channels: usize,
    board_height: usize,
    board_width: usize,
}

impl ResNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResNets<B> {
        let (h, w) = (self.board_height, self.board_width);
        let blocks = |c: usize, n: usize| -> Vec<ResBlock<B>> {
            (0..n)
                .map(|_| ResBlockConfig::new(c, c, c).init(device))
                .collect()
        };

        let c_repr = self.representation_channels;
        let c_dyn = self.dynamic_channels;
        let c_pred = self.prediction_channels;

        ResNets {
            representation: ResNetRepresentation {
                stem: conv3x3(self.obs_channels, c_repr, device),
                stem_bn: BatchNormConfig::new(c_repr).init(device),
                blocks: blocks(c_repr, self.representation_n_blocks),
                relu: Relu,
            },
            dynamics: ResNetDynamics {
                fuse: conv3x3(c_dyn + self.action_space, c_dyn, device),
                fuse_bn: BatchNormConfig::new(c_dyn).init(device),
                blocks: blocks(c_dyn, self.dynamic_n_blocks),
                reward_conv: conv1x1(c_dyn, 1, device),
                reward_bn: BatchNormConfig::new(1).init(device),
                reward_fc1: LinearConfig::new(h * w, self.dynamic_fc_hidden_size).init(device),
                reward_fc2: LinearConfig::new(self.dynamic_fc_hidden_size, self.reward_support)
                    .init(device),
                relu: Relu,
            },
            prediction: ResNetPrediction {
                policy_conv: conv1x1(c_pred, 2, device),
                policy_bn: BatchNormConfig::new(2).init(device),
                policy_fc: LinearConfig::new(2 * h * w, self.action_space).init(device),
                value_conv: conv1x1(c_pred, 1, device),
                value_bn: BatchNormConfig::new(1).init(device),
                value_fc1: LinearConfig::new(h * w, self.prediction_fc_hidden_size).init(device),
                value_fc2: LinearConfig::new(self.prediction_fc_hidden_size, self.value_support)
                    .init(device),
                relu: Relu,
            },
            projection: ConvProjectionConfig {
                channels: c_dyn,
                board_height: h,
                board_width: w,
                proj_hidden: self.proj_hidden,
                proj_out: self.proj_out,
                pred_hidden: self.pred_hidden,
            }
            .init(device),
            dynamic_channels: c_dyn,
            prediction_channels: c_pred,
            obs_channels: self.obs_channels,
            board_height: h,
            board_width: w,
        }
    }
}

impl<B: Backend> ResNets<B> {
    fn unflatten(&self, flat: Tensor<B, 2>, channels: usize) -> Tensor<B, 4> {
        let n = flat.dims()[0];
        flat.reshape([n, channels, self.board_height, self.board_width])
    }

    fn flatten(&self, spatial: Tensor<B, 4>) -> Tensor<B, 2> {
        let n = spatial.dims()[0] as i32;
        spatial.reshape([n, -1])
    }
}

impl<B: Backend> MuZeroNets<B> for ResNets<B> {
    fn init(net_conf: &NetConfig, device: &B::Device) -> Self {
        let resnet = net_conf.resnet();
        ResNetConfig {
            obs_channels: net_conf.obs_channels,
            board_height: net_conf.board_height,
            board_width: net_conf.board_width,
            action_space: net_conf.action_space,
            value_support: net_conf.value_support_len(),
            reward_support: net_conf.reward_support_len(),
            proj_hidden: net_conf.projection.proj_hidden,
            proj_out: net_conf.projection.proj_out,
            pred_hidden: net_conf.projection.pred_hidden,

            representation_channels: resnet.representation.channels,
            representation_n_blocks: resnet.representation.n_blocks,

            dynamic_channels: resnet.dynamic.channels,
            dynamic_n_blocks: resnet.dynamic.n_blocks,
            dynamic_fc_hidden_size: resnet.dynamic.fc_hidden_size,

            prediction_channels: resnet.prediction.channels,
            prediction_n_blocks: resnet.prediction.n_blocks,
            prediction_fc_hidden_size: resnet.prediction.fc_hidden_size,
        }
        .init(device)
    }

    fn represent(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let obs = self.unflatten(obs, self.obs_channels);
        self.flatten(self.representation.forward(obs))
    }

    fn dynamics(
        &self,
        hidden: Tensor<B, 2>,
        action: Tensor<B, 1, Int>,
        action_size: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let hidden = self.unflatten(hidden, self.dynamic_channels);
        let (hidden_state, reward) = self.dynamics.forward(hidden, action, action_size);
        (self.flatten(hidden_state), reward)
    }

    fn predict(&self, hidden: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let hidden = self.unflatten(hidden, self.prediction_channels);
        self.prediction.forward(hidden)
    }

    fn project(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden = self.unflatten(hidden, self.dynamic_channels);
        self.projection.project(hidden)
    }

    fn predict_projection(&self, projection: Tensor<B, 2>) -> Tensor<B, 2> {
        self.projection.predict(projection)
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::activation::softmax;

    use super::*;
    use crate::networks::MuZeroNets;

    type MyBackend = NdArray<f32>;
    type MyDevice = burn::backend::ndarray::NdArrayDevice;

    fn test_nets(device: &MyDevice) -> ResNets<MyBackend> {
        ResNetConfig {
            obs_channels: 3,
            board_height: 4,
            board_width: 4,
            action_space: 5,
            value_support: 7,
            reward_support: 7,
            proj_hidden: 32,
            proj_out: 16,
            pred_hidden: 16,

            representation_channels: 8,
            representation_n_blocks: 2,

            dynamic_channels: 8,
            dynamic_n_blocks: 2,
            dynamic_fc_hidden_size: 16,

            prediction_channels: 8,
            prediction_n_blocks: 2,
            prediction_fc_hidden_size: 16,
        }
        .init(device)
    }

    #[test]
    fn forward_shapes() {
        let device = Default::default();
        let nets = test_nets(&device);

        // obs arrives flat: [N, obs_channels * h * w]
        let obs = Tensor::<MyBackend, 2>::zeros([2, 3 * 4 * 4], &device);
        let hidden = nets.represent(obs);
        assert_eq!(hidden.dims(), [2, 8 * 4 * 4]);

        let action = Tensor::<MyBackend, 1, Int>::from_data([1, 3], &device);
        let (next_hidden, reward) = nets.dynamics(hidden.clone(), action, 5);
        assert_eq!(next_hidden.dims(), [2, 8 * 4 * 4]);
        assert_eq!(reward.dims(), [2, 7]);

        let (value, policy) = nets.predict(hidden);
        assert_eq!(value.dims(), [2, 7]);
        assert_eq!(policy.dims(), [2, 5]);
    }

    #[test]
    fn projection_accepts_flat_hidden() {
        let device = Default::default();
        let nets = test_nets(&device);

        let obs = Tensor::<MyBackend, 2>::random(
            [4, 3 * 4 * 4],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let hidden = nets.represent(obs);

        let projection = nets.project(hidden);
        assert_eq!(projection.dims(), [4, 16]);
        assert_eq!(nets.predict_projection(projection).dims(), [4, 16]);
    }

    #[test]
    fn policy_softmaxes_and_root_reward_zero() {
        let device = Default::default();
        let nets = test_nets(&device);

        let obs = Tensor::<MyBackend, 2>::random(
            [2, 3 * 4 * 4],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let (_, reward, _, policy) = nets.initial_inference(obs);

        let rewards = reward.into_data().to_vec::<f32>().unwrap();
        assert!(rewards.iter().all(|r| *r == 0.0));

        let rows = softmax(policy, 1).into_data().to_vec::<f32>().unwrap();
        for row in rows.chunks(5) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "policy row sums to {sum}");
        }
    }
}
