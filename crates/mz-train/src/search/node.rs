use burn::{Tensor, tensor::backend::Backend};

pub struct Node<B: Backend> {
    pub visits: usize,
    pub action: usize,
    pub hidden_state: Option<Tensor<B, 1>>,
    pub cumulative_value: f32,
    pub reward: f32,
    pub children: Vec<usize>,
    pub policy: f32,
}
