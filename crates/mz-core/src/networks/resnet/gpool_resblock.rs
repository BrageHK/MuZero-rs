use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::backend::Backend,
};

use crate::networks::resnet::pooling::global_pool_mean_max;

/// KataGo's global-pooling-bias residual block (arXiv:1902.10565 §3.3/A.3),
/// adapted to this codebase's post-activation `ResBlock` cadence rather than
/// KataGo's own pre-activation trunk. `conv1`'s output is split into a
/// regular branch (`c_reg` channels) and a pooling branch (`c_pool`
/// channels, carved out of the block's own width, not added on top); the
/// pooling branch is globally pooled and projected to a per-channel bias
/// that's added into the (still raw/pre-norm) regular branch before it
/// continues through the rest of the block. The skip connection requires
/// `d_input == d_output`, same constraint as `ResBlockConfig`.
#[derive(Config, Debug)]
pub struct GPoolResBlockConfig {
    pub d_input: usize,
    pub c_reg: usize,
    pub c_pool: usize,
    pub d_output: usize,
}

#[derive(Module, Debug)]
pub struct GPoolResBlock<B: Backend> {
    conv1: Conv2d<B>,
    pool_bn: BatchNorm<B>,
    gpool_fc: Linear<B>,
    reg_bn: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    relu: Relu,
    c_reg: usize,
    c_pool: usize,
}

impl GPoolResBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GPoolResBlock<B> {
        GPoolResBlock {
            conv1: Conv2dConfig::new([self.d_input, self.c_reg + self.c_pool], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool_bn: BatchNormConfig::new(self.c_pool).init(device),
            gpool_fc: LinearConfig::new(2 * self.c_pool, self.c_reg).init(device),
            reg_bn: BatchNormConfig::new(self.c_reg).init(device),
            conv2: Conv2dConfig::new([self.c_reg, self.d_output], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn2: BatchNormConfig::new(self.d_output).init(device),
            relu: Relu,
            c_reg: self.c_reg,
            c_pool: self.c_pool,
        }
    }
}

impl<B: Backend> GPoolResBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let skip = input.clone();

        let y = self.conv1.forward(input);
        let y_reg = y.clone().narrow(1, 0, self.c_reg);
        let y_pool = y.narrow(1, self.c_reg, self.c_pool);

        let g = self.relu.forward(self.pool_bn.forward(y_pool));
        let pooled = global_pool_mean_max(g);
        let [n, c, _, _] = y_reg.dims();
        let bias = self.gpool_fc.forward(pooled).reshape([n, c, 1, 1]);

        let x = self.relu.forward(self.reg_bn.forward(y_reg + bias));
        let z = self.bn2.forward(self.conv2.forward(x));
        self.relu.forward(z + skip)
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    type MyBackend = NdArray<f32>;

    #[test]
    fn preserves_shape_square_board() {
        let device = Default::default();
        let block = GPoolResBlockConfig::new(8, 4, 4, 8).init::<MyBackend>(&device);
        let input = Tensor::<MyBackend, 4>::zeros([2, 8, 5, 5], &device);
        assert_eq!(block.forward(input).dims(), [2, 8, 5, 5]);
    }

    #[test]
    fn preserves_shape_non_square_board() {
        let device = Default::default();
        let block = GPoolResBlockConfig::new(8, 4, 4, 8).init::<MyBackend>(&device);
        let input = Tensor::<MyBackend, 4>::zeros([2, 8, 3, 7], &device);
        assert_eq!(block.forward(input).dims(), [2, 8, 3, 7]);
    }

    #[test]
    fn preserves_shape_degenerate_1x1_board() {
        let device = Default::default();
        let block = GPoolResBlockConfig::new(8, 4, 4, 8).init::<MyBackend>(&device);
        let input = Tensor::<MyBackend, 4>::zeros([2, 8, 1, 1], &device);
        assert_eq!(block.forward(input).dims(), [2, 8, 1, 1]);
    }
}
