use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::backend::Backend,
};

#[derive(Module, Debug)]
pub struct RepresentationModel<B: Backend> {
    layers: Vec<Linear<B>>,
    relu: Relu,
}

impl<B: Backend> RepresentationModel<B> {
    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let last = self.layers.len() - 1;
        let mut x = obs;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i != last {
                x = self.relu.forward(x);
            }
        }
        x
    }
}

#[derive(Config, Debug)]
pub struct RepresentationModelConfig {
    pub hidden_size: usize,
    pub fc_hidden_size: usize,
    pub input_size: usize,
    pub n_layers: usize,
}

impl RepresentationModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RepresentationModel<B> {
        assert!(self.n_layers >= 1, "representation needs at least 1 layer");
        let mut layers = Vec::with_capacity(self.n_layers);
        if self.n_layers == 1 {
            layers.push(LinearConfig::new(self.input_size, self.hidden_size).init(device));
        } else {
            layers.push(LinearConfig::new(self.input_size, self.fc_hidden_size).init(device));
            for _ in 0..self.n_layers - 2 {
                layers
                    .push(LinearConfig::new(self.fc_hidden_size, self.fc_hidden_size).init(device));
            }
            layers.push(LinearConfig::new(self.fc_hidden_size, self.hidden_size).init(device));
        }
        RepresentationModel { layers, relu: Relu }
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

        let model = RepresentationModelConfig::new(8, 16, 4, 3).init::<MyBackend>(&device);
        let t1 = Tensor::<MyBackend, 2, Float>::from_floats([[1.0, 2.0, 0.5, 4.0]], &device);
        let output = model.forward(t1);
        assert_eq!(output.dims(), [1, 8]);
    }

    #[test]
    fn n_layers_shapes() {
        let device = Default::default();
        for n_layers in [1, 2, 5] {
            let model =
                RepresentationModelConfig::new(8, 16, 4, n_layers).init::<MyBackend>(&device);
            let obs = Tensor::<MyBackend, 2>::zeros([3, 4], &device);
            assert_eq!(model.forward(obs).dims(), [3, 8]);
        }
    }
}