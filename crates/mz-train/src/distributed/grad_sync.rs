//! Flattens/unflattens `GradientsParams` to/from a plain `Vec<f32>` for the
//! coordinator/worker gRPC wire format.
//!
//! Wire format carries only the flat floats, never `ParamId`s: those are
//! per-process and would not match across machines. Both sides instead derive
//! parameter order and shape by walking the same statically-typed module
//! (`Module::visit`), which is deterministic for a fixed network architecture.

use std::marker::PhantomData;

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

struct FlattenVisitor<'a, B: Backend> {
    grads: &'a GradientsParams,
    out: Vec<f32>,
    _marker: PhantomData<B>,
}

impl<B: Backend> ModuleVisitor<B> for FlattenVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let value = param.val();
        let grad: Tensor<B, D> = self
            .grads
            .get(param.id)
            .unwrap_or_else(|| Tensor::zeros(value.shape(), &value.device()));
        let data = grad.into_data().convert::<f32>();
        self.out.extend(
            data.to_vec::<f32>()
                .expect("gradient tensor should convert to f32"),
        );
    }
}

/// Flattens every gradient tensor in `module`'s traversal order into one
/// contiguous `Vec<f32>`. `module` must be the same backend `GradientsParams`
/// was extracted for (typically `agent.valid()`, i.e. the inner/non-autodiff
/// module).
pub fn flatten_grads<B: Backend, M: Module<B>>(grads: &GradientsParams, module: &M) -> Vec<f32> {
    let mut visitor = FlattenVisitor {
        grads,
        out: Vec::new(),
        _marker: PhantomData,
    };
    module.visit(&mut visitor);
    visitor.out
}

struct UnflattenVisitor<'a, B: Backend> {
    flat: &'a [f32],
    cursor: usize,
    grads: GradientsParams,
    _marker: PhantomData<B>,
}

impl<B: Backend> ModuleVisitor<B> for UnflattenVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let value = param.val();
        let shape = value.shape();
        let n = shape.num_elements();
        let slice = &self.flat[self.cursor..self.cursor + n];
        self.cursor += n;
        let data = TensorData::new(slice.to_vec(), shape);
        let tensor = Tensor::<B, D>::from_data(data, &value.device());
        self.grads.register::<B, D>(param.id, tensor);
    }
}

/// Inverse of [`flatten_grads`]: rebuilds a `GradientsParams` keyed by this
/// process's own `ParamId`s, taking shapes and traversal order from `module`
/// (must be architecturally identical to whatever produced `flat`, which is
/// guaranteed since every replica in a distributed run shares the same config).
pub fn unflatten_grads<B: Backend, M: Module<B>>(flat: &[f32], module: &M) -> GradientsParams {
    let mut visitor = UnflattenVisitor {
        flat,
        cursor: 0,
        grads: GradientsParams::new(),
        _marker: PhantomData,
    };
    module.visit(&mut visitor);
    assert_eq!(
        visitor.cursor,
        flat.len(),
        "flat gradient buffer length does not match module parameter count; \
         are both sides running the same network_type/config?"
    );
    visitor.grads
}

/// Elementwise-sums `others` into `base` (in place) and divides by `count`.
/// `others` may contain fewer entries than `count - 1` if a straggling worker's
/// submission never arrived within the sync timeout.
pub fn average_into(base: &mut [f32], others: &[Vec<f32>], count: usize) {
    for other in others {
        for (b, o) in base.iter_mut().zip(other.iter()) {
            *b += *o;
        }
    }
    let count = count.max(1) as f32;
    for b in base.iter_mut() {
        *b /= count;
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::backend::NdArray;
    use burn::optim::GradientsParams;

    use super::*;
    use crate::mz_config::MuZeroConfig;
    use crate::mz_config::NetworkType;
    use crate::networks::mlp::MlpNets;

    type TestB = NdArray<f32>;

    fn agent() -> MlpNets<TestB> {
        let conf = MuZeroConfig {
            network_type: NetworkType::Linear,
            obs_dim: 4,
            action_space: 3,
            is_twoplayer: false,
            ..Default::default()
        };
        let device = Default::default();
        conf.init(&device)
    }

    #[test]
    fn roundtrip_preserves_values() {
        let net = agent();

        // Build a GradientsParams with a distinct, known value per parameter
        // (its element count, as an f32) so a mismatch in traversal order or
        // shape slicing would be caught by the final equality check.
        struct Fill<'a> {
            grads: &'a mut GradientsParams,
        }
        impl ModuleVisitor<TestB> for Fill<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TestB, D>>) {
                let value = param.val();
                let shape = value.shape();
                let n = shape.num_elements();
                let filled = Tensor::<TestB, D>::from_data(
                    TensorData::new(vec![n as f32; n], shape),
                    &value.device(),
                );
                self.grads.register::<TestB, D>(param.id, filled);
            }
        }
        let mut grads = GradientsParams::new();
        net.visit(&mut Fill { grads: &mut grads });

        let flat = flatten_grads(&grads, &net);
        assert!(!flat.is_empty());

        let restored = unflatten_grads(&flat, &net);
        let flat_again = flatten_grads(&restored, &net);
        assert_eq!(flat, flat_again);
    }

    #[test]
    fn average_into_divides_by_count() {
        let mut base = vec![1.0, 2.0, 3.0];
        let others = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
        average_into(&mut base, &others, 3);
        assert_eq!(base, vec![1.0, 2.0, 3.0]);
    }
}
