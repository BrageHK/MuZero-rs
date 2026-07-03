use burn::{
    Tensor,
    module::Module,
    nn::{Linear, LinearConfig, Tanh},
    tensor::{Int, backend::Backend},
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
    tanh: Tanh,
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
        let x = self.backbone1.forward(x);
        let x = self.backbone2.forward(x);
        let x = self.backbone3.forward(x);

        let reward = self.reward_head1.forward(x.clone());
        let reward = self.reward_head2.forward(reward);
        let reward = self.tanh.forward(reward);

        let hidden_state = self.hidden_head1.forward(x);
        let hidden_state = self.hidden_head2.forward(hidden_state);

        (hidden_state, reward.tanh())
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

            tanh: Tanh,
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use burn::backend::Wgpu;

//     use super::*;

//     type MyBackend = Wgpu<f32, i32>;

//     #[test]
//     fn forward_pass() {
//         use burn::tensor::Float;

//         let device = Default::default();

//         let model = DynamicModelConfig::new().init::<MyBackend>(&device);
//         let t1 = Tensor::<MyBackend, 2, Float>::from_floats([[1.0, 2.0, 0.5, 4.0]], &device);
//         let output = model.forward(t1);
//         println!("{:?}", &output);
//         //assert!();
//     }
// }
