use burn::{
    Tensor,
    module::Module,
    nn::{BatchNorm, Linear, Relu, conv::Conv2d},
    tensor::{IndexingUpdateOp, Int, backend::Backend},
};

use crate::networks::resnet::resblock::ResBlock;

#[derive(Module, Debug)]
pub struct ResNetDynamics<B: Backend> {
    pub(super) fuse: Conv2d<B>,
    pub(super) fuse_bn: BatchNorm<B>,
    pub(super) blocks: Vec<ResBlock<B>>,
    pub(super) reward_conv: Conv2d<B>,
    pub(super) reward_bn: BatchNorm<B>,
    pub(super) reward_fc1: Linear<B>,
    pub(super) reward_fc2: Linear<B>,
    pub(super) relu: Relu,
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
