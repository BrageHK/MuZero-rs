use burn::{Tensor, tensor::backend::Backend};

/// KataGo-style global pooling (arXiv:1902.10565 §3.3/A.2), reduced to two
/// terms since this codebase trains one network per fixed board size — the
/// paper's 3rd term (mean scaled by board width) exists purely to let a
/// single net generalize across board sizes at inference time, which doesn't
/// apply here. `[N,C,H,W] -> [N,2C]`: per-channel mean over (H,W) concatenated
/// with per-channel max over (H,W).
pub fn global_pool_mean_max<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 2> {
    let [n, c, _, _] = x.dims();
    let mean = x.clone().mean_dims(&[2, 3]).reshape([n, c]);
    let max = x.max_dims(&[2, 3]).reshape([n, c]);
    Tensor::cat(vec![mean, max], 1)
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    type MyBackend = NdArray<f32>;

    #[test]
    fn mean_and_max_per_channel() {
        let device = Default::default();
        // channel 0: constant 2.0 everywhere except one outlier at 9.0
        // channel 1: constant -1.0 everywhere except one outlier at 5.0
        let data: [[[[f32; 2]; 2]; 2]; 1] =
            [[[[2.0, 2.0], [2.0, 9.0]], [[-1.0, -1.0], [-1.0, 5.0]]]];
        let x = Tensor::<MyBackend, 4>::from_data(data, &device);

        let pooled = global_pool_mean_max(x);
        assert_eq!(pooled.dims(), [1, 4]);

        let values = pooled.into_data().to_vec::<f32>().unwrap();
        let expected_mean_0 = (2.0 + 2.0 + 2.0 + 9.0) / 4.0;
        let expected_mean_1 = (-1.0 - 1.0 - 1.0 + 5.0) / 4.0;
        assert!((values[0] - expected_mean_0).abs() < 1e-5);
        assert!((values[1] - expected_mean_1).abs() < 1e-5);
        assert!((values[2] - 9.0).abs() < 1e-5);
        assert!((values[3] - 5.0).abs() < 1e-5);
    }
}
