use burn::{
    Tensor,
    module::Module,
    nn::{BatchNorm, Linear, Relu, conv::Conv2d},
    tensor::backend::Backend,
};

use crate::networks::resnet::pooling::global_pool_mean_max;

/// KataGo-style global-pooling-bias structure for the policy head (A.4): a
/// small parallel conv is globally pooled and projected to a per-channel
/// bias added into the (unrelated, fixed-width) spatial policy conv, purely
/// additive — it never changes `policy_fc`'s input shape.
#[derive(Module, Debug)]
pub struct PolicyGPool<B: Backend> {
    pub(super) g_conv: Conv2d<B>,
    pub(super) g_bn: BatchNorm<B>,
    pub(super) gpool_fc: Linear<B>,
    pub(super) relu: Relu,
}

#[derive(Module, Debug)]
pub struct ResNetPrediction<B: Backend> {
    pub(super) policy_conv: Conv2d<B>,
    pub(super) policy_bn: BatchNorm<B>,
    pub(super) policy_fc: Linear<B>,
    pub(super) policy_gpool: Option<PolicyGPool<B>>,
    pub(super) value_conv: Conv2d<B>,
    pub(super) value_bn: BatchNorm<B>,
    pub(super) value_fc1: Linear<B>,
    pub(super) value_fc2: Linear<B>,
    /// True when `value_conv`/`value_fc1` were sized for the pooled
    /// (KataGo A.5, resolution-agnostic) path rather than the original
    /// flatten(h*w) path.
    pub(super) value_pooled: bool,
    pub(super) relu: Relu,
}

impl<B: Backend> ResNetPrediction<B> {
    /// Returns (value_logits, policy_logits)
    pub fn forward(&self, hidden: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let n = hidden.dims()[0] as i32;

        let mut policy = self.policy_conv.forward(hidden.clone());
        if let Some(gpool) = &self.policy_gpool {
            let g = gpool
                .relu
                .forward(gpool.g_bn.forward(gpool.g_conv.forward(hidden.clone())));
            let pooled = global_pool_mean_max(g);
            let [pn, pc, _, _] = policy.dims();
            let bias = gpool.gpool_fc.forward(pooled).reshape([pn, pc, 1, 1]);
            policy = policy + bias;
        }
        let policy = self.relu.forward(self.policy_bn.forward(policy));
        let policy = self.policy_fc.forward(policy.reshape([n, -1]));

        let value = self
            .relu
            .forward(self.value_bn.forward(self.value_conv.forward(hidden)));
        let value_input = if self.value_pooled {
            global_pool_mean_max(value)
        } else {
            value.reshape([n, -1])
        };
        let value = self.relu.forward(self.value_fc1.forward(value_input));
        let value = self.value_fc2.forward(value);

        (value, policy)
    }
}
