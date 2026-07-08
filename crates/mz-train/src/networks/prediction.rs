use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{activation::softmax, backend::Backend},
};

#[derive(Module, Debug)]
pub struct PredictionModel<B: Backend> {
    backbone: Vec<Linear<B>>,
    value1: Linear<B>,
    value2: Linear<B>,
    policy1: Linear<B>,
    policy2: Linear<B>,
    relu: Relu,
}

impl<B: Backend> PredictionModel<B> {
    /// Returns (value, policy)
    pub fn forward(&self, hidden: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = hidden;
        for layer in &self.backbone {
            x = self.relu.forward(layer.forward(x));
        }

        let value = self.relu.forward(self.value1.forward(x.clone()));
        let value = self.value2.forward(value);

        let policy = self.relu.forward(self.policy1.forward(x));
        let policy = self.policy2.forward(policy);
        let policy = softmax(policy, 1);

        (value, policy)
    }
}

#[derive(Config, Debug)]
pub struct PredictionModelConfig {
    pub fc_hidden_size: usize,
    pub hidden_size: usize,
    pub action_space: usize,
    pub n_layers: usize,
}

impl PredictionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PredictionModel<B> {
        assert!(
            self.n_layers >= 1,
            "prediction backbone needs at least 1 layer"
        );
        let mut backbone = Vec::with_capacity(self.n_layers);
        backbone.push(LinearConfig::new(self.hidden_size, self.fc_hidden_size).init(device));
        for _ in 0..self.n_layers - 1 {
            backbone.push(LinearConfig::new(self.fc_hidden_size, self.fc_hidden_size).init(device));
        }

        PredictionModel {
            backbone,
            value1: LinearConfig::new(self.fc_hidden_size, self.fc_hidden_size).init(device),
            value2: LinearConfig::new(self.fc_hidden_size, 1).init(device),
            policy1: LinearConfig::new(self.fc_hidden_size, self.fc_hidden_size).init(device),
            policy2: LinearConfig::new(self.fc_hidden_size, self.action_space).init(device),
            relu: Relu,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;

    use super::*;

    type MyBackend = Wgpu<f32, i32>;

    #[test]
    fn forward_pass() {
        use burn::tensor::Float;

        let device = Default::default();

        let model = PredictionModelConfig::new(16, 8, 2, 2).init::<MyBackend>(&device);
        let t1 = Tensor::<MyBackend, 2, Float>::from_floats(
            [[1.0, 2.0, 0.5, 4.0, 0.0, 0.0, 0.0, 0.0]],
            &device,
        );
        let (value, policy) = model.forward(t1);
        assert_eq!(value.dims(), [1, 1]);
        assert_eq!(policy.dims(), [1, 2]);
    }
}
