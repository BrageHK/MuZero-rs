use burn::{
    Tensor,
    module::Module,
    nn::{BatchNorm, Relu, conv::Conv2d},
    tensor::backend::Backend,
};

use crate::networks::resnet::tower_block::TowerBlock;

#[derive(Module, Debug)]
pub struct ResNetRepresentation<B: Backend> {
    pub(super) stem: Conv2d<B>,
    pub(super) stem_bn: BatchNorm<B>,
    pub(super) blocks: Vec<TowerBlock<B>>,
    pub(super) relu: Relu,
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
