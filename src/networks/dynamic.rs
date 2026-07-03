use burn::{
    Tensor, module::Module, nn::{Linear, LinearConfig, Relu}, tensor::{Int, backend::Backend},
};

#[derive(Module, Debug)]
pub struct DynamicModel<B: Backend> {
    backbone1: Linear<B>,
    backbone2: Linear<B>,
    backbone3: Linear<B>,
    reward_head1: Linear<B>,
    reward_head2: Linear<B>,
    hidden_head1: Linear<B>,
    hidden_head2: Linear<B>,
    relu: Relu,
}

impl<B: Backend> DynamicModel<B> {
    /// Returns (hidden_state, reward)
    pub fn forward(
        &self,
        obs: Tensor<B, 2>,
        action: Tensor<B, 1, Int>,
        action_size: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let action_one_hot: Tensor<B, 2, Int> = action.one_hot(action_size);

        let x = Tensor::cat(vec![obs, action_one_hot.float()], 1);
        let x = self.relu.forward(self.backbone1.forward(x));
        let x = self.relu.forward(self.backbone2.forward(x));
        let x = self.relu.forward(self.backbone3.forward(x));

        let reward = self.relu.forward(self.reward_head1.forward(x.clone()));
        let reward = self.reward_head2.forward(reward);

        let hidden_state = self.relu.forward(self.hidden_head1.forward(x));
        let hidden_state = self.hidden_head2.forward(hidden_state);

        (hidden_state, reward)
    }
}

#[derive(Debug)]
pub struct DynamicModelConfig {
    pub hidden_input: usize,
    pub fc_hidden_size: usize,
    pub hidden_output: usize,
}

impl DynamicModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DynamicModel<B> {
        DynamicModel {
            backbone1: LinearConfig::new(self.hidden_input, self.fc_hidden_size).init(device),
            backbone2: LinearConfig::new(self.fc_hidden_size, self.fc_hidden_size).init(device),
            backbone3: LinearConfig::new(self.fc_hidden_size, self.fc_hidden_size).init(device),

            reward_head1: LinearConfig::new(self.fc_hidden_size, self.fc_hidden_size).init(device),
            reward_head2: LinearConfig::new(self.fc_hidden_size, 1).init(device),

            hidden_head1: LinearConfig::new(self.fc_hidden_size, self.fc_hidden_size).init(device),
            hidden_head2: LinearConfig::new(self.fc_hidden_size, self.hidden_output).init(device),

            relu: Relu,
        }
    }
}