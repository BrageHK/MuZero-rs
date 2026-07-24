use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Int, backend::Backend},
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
    /// distribution over the reward support (see `support`).
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
        // convert() first: the backend's int repr may be I32 or I64.
        let action_idx = action.into_data().convert::<i64>().to_vec::<i64>().unwrap();
        let mut one_hot = vec![0f32; batch_size * action_size];
        for (row, &a) in action_idx.iter().enumerate() {
            one_hot[row * action_size + a as usize] = 1.0;
        }
        let action_one_hot = Tensor::<B, 1>::from_floats(one_hot.as_slice(), &device)
            .reshape([batch_size, action_size]);

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

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;

    use super::*;

    type MyBackend = Wgpu<f32, i32>;

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
