use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{IndexingUpdateOp, Int, backend::Backend},
};

#[derive(Module, Debug)]
pub struct DynamicModelMLP<B: Backend> {
    backbone: Vec<Linear<B>>,
    reward_head1: Linear<B>,
    reward_head2: Linear<B>,
    hidden_head1: Linear<B>,
    hidden_head2: Linear<B>,
    relu: Relu,
}

impl<B: Backend> DynamicModelMLP<B> {
    /// Returns (hidden_state, reward_logits). reward_logits is a categorical
    /// distribution over the reward support (see `support`) for single-player
    /// envs, or a plain scalar column (width 1) for board games.
    pub fn forward(
        &self,
        hidden: Tensor<B, 2>,
        action: Tensor<B, 1, Int>,
        action_size: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Built from raw floats on `hidden`'s device (not via an int->float
        // cast) so it lands in the same autodiff-wrapped-or-not bucket as
        // `hidden` — burn-dispatch's int->float cast never carries the
        // Autodiff wrapper, which makes `Tensor::cat` below panic when
        // `hidden` is autodiff-tracked (e.g. training on the rocm backend).
        let batch_size = hidden.dims()[0];
        let device = hidden.device();
        let action_one_hot = Tensor::<B, 2>::zeros([batch_size, action_size], &device).scatter(
            1,
            action.reshape([batch_size, 1]),
            Tensor::<B, 2>::ones([batch_size, 1], &device),
            IndexingUpdateOp::Add,
        );

        let mut x = Tensor::cat(vec![hidden, action_one_hot], 1);
        for layer in &self.backbone {
            x = self.relu.forward(layer.forward(x));
        }

        let reward = self.relu.forward(self.reward_head1.forward(x.clone()));
        let reward = self.reward_head2.forward(reward);

        let hidden_state = self.relu.forward(self.hidden_head1.forward(x));
        let hidden_state = self.hidden_head2.forward(hidden_state);

        (hidden_state, reward)
    }
}

#[derive(Config, Debug)]
pub struct DynamicModelConfig {
    pub hidden_input: usize,
    pub fc_hidden_size: usize,
    pub hidden_output: usize,
    pub n_layers: usize,
    pub reward_support: usize,
}

impl DynamicModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DynamicModelMLP<B> {
        assert!(
            self.n_layers >= 1,
            "dynamic backbone needs at least 1 layer"
        );
        let mut backbone = Vec::with_capacity(self.n_layers);
        backbone.push(LinearConfig::new(self.hidden_input, self.fc_hidden_size).init(device));
        for _ in 0..self.n_layers - 1 {
            backbone.push(LinearConfig::new(self.fc_hidden_size, self.fc_hidden_size).init(device));
        }

        DynamicModelMLP {
            backbone,

            reward_head1: LinearConfig::new(self.fc_hidden_size, self.fc_hidden_size).init(device),
            reward_head2: LinearConfig::new(self.fc_hidden_size, self.reward_support).init(device),

            hidden_head1: LinearConfig::new(self.fc_hidden_size, self.fc_hidden_size).init(device),
            hidden_head2: LinearConfig::new(self.fc_hidden_size, self.hidden_output).init(device),

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
    fn forward_shapes() {
        let device = Default::default();
        let model = DynamicModelConfig::new(8 + 2, 16, 8, 3, 7).init::<MyBackend>(&device);
        let hidden = Tensor::<MyBackend, 2>::zeros([3, 8], &device);
        let action = Tensor::<MyBackend, 1, Int>::from_data([0, 1, 0], &device);
        let (hidden_state, reward) = model.forward(hidden, action, 2);
        assert_eq!(hidden_state.dims(), [3, 8]);
        assert_eq!(reward.dims(), [3, 7]);
    }
}
