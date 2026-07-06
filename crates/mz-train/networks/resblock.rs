use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{self, BatchNormConfig, PaddingConfig2d, Relu, conv::Conv2dConfig},
    tensor::backend::Backend,
};

/// The skip connection requires `d_input == d_output`.
#[derive(Config, Debug)]
pub struct ResBlockConfig {
    pub d_input: usize,
    pub d_hidden: usize,
    pub d_output: usize,
}

#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: nn::conv::Conv2d<B>,
    bn1: nn::BatchNorm<B>,
    conv2: nn::conv::Conv2d<B>,
    bn2: nn::BatchNorm<B>,
    activation: nn::Relu,
}

impl ResBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResBlock<B> {
        ResBlock {
            conv1: Conv2dConfig::new([self.d_input, self.d_hidden], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn1: BatchNormConfig::new(self.d_hidden).init(device),
            conv2: Conv2dConfig::new([self.d_hidden, self.d_output], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn2: BatchNormConfig::new(self.d_output).init(device),
            activation: Relu,
        }
    }
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        // Commnets are from AplhaGo Paper
        let skip = input.clone();
        // (1) A convolution of 256 filters of kernel size 3×3 with stride 1
        let x = self.conv1.forward(input);
        // (2) Batch normalization
        let x = self.bn1.forward(x);
        // (3) A rectifier nonlinearity
        let x = self.activation.forward(x);
        // (4) A convolution of 256 filters of kernel size 3×3 with stride 1
        let x = self.conv2.forward(x);
        // (5) Batch normalization
        let x = self.bn2.forward(x);
        // (6) A skip connection that adds the input to the block
        let x = x + skip;
        // (7) A rectifier nonlinearity
        self.activation.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;

    use super::*;

    type MyBackend = Wgpu<f32, i32>;

    #[test]
    fn preserves_shape() {
        let device = Default::default();
        let block = ResBlockConfig::new(8, 8, 8).init::<MyBackend>(&device);
        let input = Tensor::<MyBackend, 4>::zeros([2, 8, 5, 5], &device);
        let output = block.forward(input);
        assert_eq!(output.dims(), [2, 8, 5, 5]);
    }
}
