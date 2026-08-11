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

/// ```ignore
/// use mz_rs::with_net;
///
/// with_net!(mz_conf, Net => {
///     // `Net<B>` is the family picked by `network_type`, for any backend B
///     let agent: Net<TrainB> = mz_conf.init_agent(&train_device);
/// });
/// ```
#[macro_export]
macro_rules! with_net {
    ($mz_conf:expr, $N:ident => $body:expr) => {
        match $mz_conf.network_type {
            $crate::mz_config::NetworkType::Linear => {
                type $N<B> = $crate::agent::MlpNets<B>;
                $body
            }
            $crate::mz_config::NetworkType::ResNet => {
                type $N<B> = $crate::networks::resnet::ResNets<B>;
                $body
            }
        }
    };
}
