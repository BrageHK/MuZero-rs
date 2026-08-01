use burn::tensor::backend::Backend;

pub use mz_core::networks::*;

use crate::mz_config::MuZeroConfig;

/// `mz_core::networks::nets_to_backend` taking the full training config, so call
/// sites keep passing `&MuZeroConfig`.
pub fn nets_to_backend<B1: Backend, B2: Backend, N1: MuZeroNets<B1>, N2: MuZeroNets<B2>>(
    nets: &N1,
    mz_conf: &MuZeroConfig,
    device: &B2::Device,
) -> N2 {
    mz_core::networks::nets_to_backend(nets, &mz_conf.net_config(), device)
}
