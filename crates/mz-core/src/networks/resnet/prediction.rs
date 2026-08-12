use burn::{
    Tensor,
    module::Module,
    nn::{BatchNorm, Linear, Relu, conv::Conv2d},
    tensor::backend::Backend,
};

#[derive(Module, Debug)]
pub struct ResNetPrediction<B: Backend> {
    pub(super) policy_conv: Conv2d<B>,
    pub(super) policy_bn: BatchNorm<B>,
    pub(super) policy_fc: Linear<B>,
    pub(super) value_conv: Conv2d<B>,
    pub(super) value_bn: BatchNorm<B>,
    pub(super) value_fc1: Linear<B>,
    pub(super) value_fc2: Linear<B>,
    pub(super) relu: Relu,
}

impl<B: Backend> ResNetPrediction<B> {
    /// Returns (value_logits, policy_logits)
    pub fn forward(&self, hidden: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let n = hidden.dims()[0] as i32;

        let policy = self.relu.forward(
            self.policy_bn
                .forward(self.policy_conv.forward(hidden.clone())),
        );
        let policy = self.policy_fc.forward(policy.reshape([n, -1]));

        let value = self
            .relu
            .forward(self.value_bn.forward(self.value_conv.forward(hidden)));
        let value = self
            .relu
            .forward(self.value_fc1.forward(value.reshape([n, -1])));
        let value = self.value_fc2.forward(value);

        (value, policy)
    }
}
