use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    tensor::backend::Backend,
};

const REDUCED_CHANNELS: usize = 16;
const POOL_SIZE: usize = 6;

#[derive(Module, Debug)]
pub struct ConvProjection<B: Backend> {
    reduce: Conv2d<B>,
    reduce_norm: BatchNorm<B>,
    pool: AdaptiveAvgPool2d,
    proj1: Linear<B>,
    proj1_norm: BatchNorm<B>,
    proj2: Linear<B>,
    proj2_norm: BatchNorm<B>,
    pred1: Linear<B>,
    pred1_norm: BatchNorm<B>,
    pred2: Linear<B>,
    relu: Relu,
}

impl<B: Backend> ConvProjection<B> {
    pub fn project(&self, hidden: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self
            .relu
            .forward(self.reduce_norm.forward(self.reduce.forward(hidden)));
        let x = self.pool.forward(x);
        let n = x.dims()[0] as i32;
        let x = x.reshape([n, -1]);
        let x = self
            .relu
            .forward(self.proj1_norm.forward(self.proj1.forward(x)));
        self.proj2_norm.forward(self.proj2.forward(x))
    }

    pub fn predict(&self, projection: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self
            .relu
            .forward(self.pred1_norm.forward(self.pred1.forward(projection)));
        self.pred2.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ConvProjectionConfig {
    pub channels: usize,
    pub board_height: usize,
    pub board_width: usize,
    pub proj_hidden: usize,
    pub proj_out: usize,
    pub pred_hidden: usize,
}

impl ConvProjectionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvProjection<B> {
        let pool_h = self.board_height.clamp(1, POOL_SIZE);
        let pool_w = self.board_width.clamp(1, POOL_SIZE);
        let flat = REDUCED_CHANNELS * pool_h * pool_w;

        ConvProjection {
            reduce: Conv2dConfig::new([self.channels, REDUCED_CHANNELS], [1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            reduce_norm: BatchNormConfig::new(REDUCED_CHANNELS).init(device),
            pool: AdaptiveAvgPool2dConfig::new([pool_h, pool_w]).init(),
            proj1: LinearConfig::new(flat, self.proj_hidden).init(device),
            proj1_norm: BatchNormConfig::new(self.proj_hidden).init(device),
            proj2: LinearConfig::new(self.proj_hidden, self.proj_out).init(device),
            proj2_norm: BatchNormConfig::new(self.proj_out).init(device),
            pred1: LinearConfig::new(self.proj_out, self.pred_hidden).init(device),
            pred1_norm: BatchNormConfig::new(self.pred_hidden).init(device),
            pred2: LinearConfig::new(self.pred_hidden, self.proj_out).init(device),
            relu: Relu,
        }
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    type MyBackend = NdArray<f32>;

    #[test]
    fn conv_projection_shapes() {
        let device = Default::default();
        for (h, w) in [(84, 84), (8, 8), (3, 3)] {
            let model =
                ConvProjectionConfig::new(32, h, w, 256, 64, 128).init::<MyBackend>(&device);
            let hidden = Tensor::<MyBackend, 4>::zeros([4, 32, h, w], &device);

            let projection = model.project(hidden);
            assert_eq!(projection.dims(), [4, 64]);
            assert_eq!(model.predict(projection).dims(), [4, 64]);
        }
    }
}
