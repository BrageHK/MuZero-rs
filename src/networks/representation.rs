use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::backend::Backend,
};

#[derive(Module, Debug)]
pub struct RepresentationModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    relu: Relu,
}

impl<B: Backend> RepresentationModel<B> {
    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.relu.forward(self.linear1.forward(obs));
        let x = self.relu.forward(self.linear2.forward(x));
        self.linear3.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct RepresentationModelConfig {
    pub hidden_size: usize,
    pub fc_hidden_size: usize,
    pub input_size: usize,
}

impl RepresentationModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RepresentationModel<B> {
        RepresentationModel {
            linear1: LinearConfig::new(self.input_size, self.fc_hidden_size).init(device),
            linear2: LinearConfig::new(self.fc_hidden_size, self.fc_hidden_size).init(device),
            linear3: LinearConfig::new(self.fc_hidden_size, self.hidden_size).init(device),
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

        let model = RepresentationModelConfig::new(8, 16, 4).init::<MyBackend>(&device);
        let t1 = Tensor::<MyBackend, 2, Float>::from_floats([[1.0, 2.0, 0.5, 4.0]], &device);
        let output = model.forward(t1);
        println!("{:?}", &output);
        //assert!();
    }
}
