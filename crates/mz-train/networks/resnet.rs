use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Int, activation::softmax, backend::Backend},
};

use crate::mz_config::MuZeroConfig;
use crate::networks::MuZeroNets;
use crate::networks::resblock::{ResBlock, ResBlockConfig};

#[derive(Config, Debug)]
pub struct ResNetConfig {
    pub obs_channels: usize,
    pub channels: usize,
    pub n_blocks: usize,
    pub board_height: usize,
    pub board_width: usize,
    pub action_space: usize,
    pub fc_hidden_size: usize,
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
        let action_planes = action
            .one_hot::<2>(action_size)
            .float()
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
    /// Returns (value, policy)
    pub fn forward(&self, hidden: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let n = hidden.dims()[0] as i32;

        let policy = self.relu.forward(
            self.policy_bn
                .forward(self.policy_conv.forward(hidden.clone())),
        );
        let policy = self.policy_fc.forward(policy.reshape([n, -1]));
        let policy = softmax(policy, 1);

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
    channels: usize,
    obs_channels: usize,
    board_height: usize,
    board_width: usize,
}

impl ResNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResNets<B> {
        let c = self.channels;
        let (h, w) = (self.board_height, self.board_width);
        let blocks = |n: usize| -> Vec<ResBlock<B>> {
            (0..n)
                .map(|_| ResBlockConfig::new(c, c, c).init(device))
                .collect()
        };

        ResNets {
            representation: ResNetRepresentation {
                stem: conv3x3(self.obs_channels, c, device),
                stem_bn: BatchNormConfig::new(c).init(device),
                blocks: blocks(self.n_blocks),
                relu: Relu,
            },
            dynamics: ResNetDynamics {
                fuse: conv3x3(c + self.action_space, c, device),
                fuse_bn: BatchNormConfig::new(c).init(device),
                blocks: blocks(self.n_blocks),
                reward_conv: conv1x1(c, 1, device),
                reward_bn: BatchNormConfig::new(1).init(device),
                reward_fc1: LinearConfig::new(h * w, self.fc_hidden_size).init(device),
                reward_fc2: LinearConfig::new(self.fc_hidden_size, 1).init(device),
                relu: Relu,
            },
            prediction: ResNetPrediction {
                policy_conv: conv1x1(c, 2, device),
                policy_bn: BatchNormConfig::new(2).init(device),
                policy_fc: LinearConfig::new(2 * h * w, self.action_space).init(device),
                value_conv: conv1x1(c, 1, device),
                value_bn: BatchNormConfig::new(1).init(device),
                value_fc1: LinearConfig::new(h * w, self.fc_hidden_size).init(device),
                value_fc2: LinearConfig::new(self.fc_hidden_size, 1).init(device),
                relu: Relu,
            },
            channels: c,
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
    fn init(mz_conf: &MuZeroConfig, device: &B::Device) -> Self {
        let resnet = mz_conf
            .resnet
            .as_ref()
            .expect("network_type: ResNet requires a `resnet:` section in the config");
        ResNetConfig {
            obs_channels: resnet.obs_channels,
            channels: resnet.channels,
            n_blocks: resnet.n_blocks,
            board_height: resnet.board_height,
            board_width: resnet.board_width,
            action_space: mz_conf.action_space,
            fc_hidden_size: resnet.fc_hidden_size,
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
        let hidden = self.unflatten(hidden, self.channels);
        let (hidden_state, reward) = self.dynamics.forward(hidden, action, action_size);
        (self.flatten(hidden_state), reward)
    }

    fn predict(&self, hidden: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let hidden = self.unflatten(hidden, self.channels);
        self.prediction.forward(hidden)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;

    use super::*;
    use crate::networks::MuZeroNets;

    type MyBackend = Wgpu<f32, i32>;
    type MyDevice = burn::backend::wgpu::WgpuDevice;

    fn test_nets(device: &MyDevice) -> ResNets<MyBackend> {
        ResNetConfig {
            obs_channels: 3,
            channels: 8,
            n_blocks: 2,
            board_height: 4,
            board_width: 4,
            action_space: 5,
            fc_hidden_size: 16,
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
        assert_eq!(reward.dims(), [2, 1]);

        let (value, policy) = nets.predict(hidden);
        assert_eq!(value.dims(), [2, 1]);
        assert_eq!(policy.dims(), [2, 5]);
    }

    #[test]
    fn policy_is_normalized_and_root_reward_zero() {
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

        let rows = policy.into_data().to_vec::<f32>().unwrap();
        for row in rows.chunks(5) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "policy row sums to {sum}");
        }
    }
}
