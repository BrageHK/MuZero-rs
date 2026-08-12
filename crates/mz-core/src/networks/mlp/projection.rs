use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu},
    tensor::backend::Backend,
};

#[derive(Module, Debug)]
pub struct MlpProjection<B: Backend> {
    proj1: Linear<B>,
    proj1_norm: BatchNorm<B>,
    proj2: Linear<B>,
    proj2_norm: BatchNorm<B>,
    proj3: Linear<B>,
    proj3_norm: BatchNorm<B>,
    pred1: Linear<B>,
    pred1_norm: BatchNorm<B>,
    pred2: Linear<B>,
    relu: Relu,
}

impl<B: Backend> MlpProjection<B> {
    pub fn project(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self
            .relu
            .forward(self.proj1_norm.forward(self.proj1.forward(hidden)));
        let x = self
            .relu
            .forward(self.proj2_norm.forward(self.proj2.forward(x)));
        self.proj3_norm.forward(self.proj3.forward(x))
    }

    pub fn predict(&self, projection: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self
            .relu
            .forward(self.pred1_norm.forward(self.pred1.forward(projection)));
        self.pred2.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct MlpProjectionConfig {
    pub latent_size: usize,
    pub proj_hidden: usize,
    pub proj_out: usize,
    pub pred_hidden: usize,
}

impl MlpProjectionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MlpProjection<B> {
        MlpProjection {
            proj1: LinearConfig::new(self.latent_size, self.proj_hidden).init(device),
            proj1_norm: BatchNormConfig::new(self.proj_hidden).init(device),
            proj2: LinearConfig::new(self.proj_hidden, self.proj_hidden).init(device),
            proj2_norm: BatchNormConfig::new(self.proj_hidden).init(device),
            proj3: LinearConfig::new(self.proj_hidden, self.proj_out).init(device),
            proj3_norm: BatchNormConfig::new(self.proj_out).init(device),
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
    fn mlp_projection_shapes() {
        let device = Default::default();
        let model = MlpProjectionConfig::new(32, 256, 64, 128).init::<MyBackend>(&device);
        let hidden = Tensor::<MyBackend, 2>::zeros([4, 32], &device);

        let projection = model.project(hidden);
        assert_eq!(projection.dims(), [4, 64]);
        assert_eq!(model.predict(projection).dims(), [4, 64]);
    }
}
