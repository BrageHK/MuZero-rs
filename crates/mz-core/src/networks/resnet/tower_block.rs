use burn::{Tensor, module::Module, tensor::backend::Backend};

use crate::networks::resnet::gpool_resblock::GPoolResBlock;
use crate::networks::resnet::resblock::ResBlock;

/// One slot in a residual tower: either a plain block or a KataGo-style
/// global-pooling block. Mixing both types in one `Vec` requires this enum
/// since a Burn `Module` field needs a single concrete type.
#[derive(Module, Debug)]
pub enum TowerBlock<B: Backend> {
    Plain(ResBlock<B>),
    GPool(GPoolResBlock<B>),
}

impl<B: Backend> TowerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            TowerBlock::Plain(b) => b.forward(x),
            TowerBlock::GPool(b) => b.forward(x),
        }
    }
}

/// Builds a tower's block stack, inserting a `GPoolResBlock` every
/// `gpool_every`-th block (1-indexed position). `gpool_every == 0` disables
/// global pooling entirely, producing an all-`Plain` stack identical to the
/// pre-global-pooling architecture.
pub fn build_tower<B: Backend>(
    channels: usize,
    n_blocks: usize,
    gpool_every: usize,
    gpool_channels: usize,
    device: &B::Device,
) -> Vec<TowerBlock<B>> {
    use crate::networks::resnet::gpool_resblock::GPoolResBlockConfig;
    use crate::networks::resnet::resblock::ResBlockConfig;

    let c_pool = if gpool_channels == 0 {
        (channels / 4).max(4)
    } else {
        gpool_channels
    };

    (0..n_blocks)
        .map(|i| {
            let is_gpool = gpool_every > 0 && (i + 1) % gpool_every == 0;
            if is_gpool {
                let c_reg = channels - c_pool;
                TowerBlock::GPool(
                    GPoolResBlockConfig::new(channels, c_reg, c_pool, channels).init(device),
                )
            } else {
                TowerBlock::Plain(ResBlockConfig::new(channels, channels, channels).init(device))
            }
        })
        .collect()
}
