use burn::DispatchDevice;
use serde::{Deserialize, Serialize};

/// Backend selected at runtime from config. A variant is only available when
/// the matching cargo feature compiled that backend in (see `[features]` in
/// Cargo.toml). picking a backend that wasn't compiled in panics with a hint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendChoice {
    /// Best compiled-in backend: cuda > metal > rocm > vulkan > wgpu > libtorch > flex > ndarray.
    #[default]
    Auto,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    Wgpu,
    Flex,
    NdArray,
    /// LibTorch on CPU.
    LibTorch,
    /// LibTorch on MPS (macOS) or CUDA (elsewhere).
    #[serde(rename = "libtorch-gpu")]
    LibTorchGpu,
}

/// Oof but works
pub fn select_device(choice: BackendChoice) -> DispatchDevice {
    match choice {
        BackendChoice::Auto => DispatchDevice::default(),

        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => DispatchDevice::Cuda(Default::default()),

        #[cfg(feature = "rocm")]
        BackendChoice::Rocm => DispatchDevice::Rocm(Default::default()),

        #[cfg(all(feature = "metal", not(feature = "vulkan")))]
        BackendChoice::Metal => DispatchDevice::Metal(Default::default()),

        #[cfg(all(feature = "vulkan", not(feature = "metal")))]
        BackendChoice::Vulkan => DispatchDevice::Vulkan(Default::default()),

        #[cfg(all(feature = "wgpu", not(feature = "metal"), not(feature = "vulkan")))]
        BackendChoice::Wgpu => DispatchDevice::Wgpu(Default::default()),

        // flex is a base burn feature, always compiled in
        BackendChoice::Flex => DispatchDevice::Flex(burn::backend::flex::FlexDevice),

        #[cfg(feature = "ndarray")]
        BackendChoice::NdArray => DispatchDevice::NdArray(Default::default()),

        #[cfg(feature = "tch")]
        BackendChoice::LibTorch => DispatchDevice::LibTorch(Default::default()),

        #[cfg(all(feature = "tch", target_os = "macos"))]
        BackendChoice::LibTorchGpu => {
            DispatchDevice::LibTorch(burn::backend::libtorch::LibTorchDevice::Mps)
        }

        #[cfg(all(feature = "tch", not(target_os = "macos")))]
        BackendChoice::LibTorchGpu => {
            DispatchDevice::LibTorch(burn::backend::libtorch::LibTorchDevice::Cuda(0))
        }

        #[allow(unreachable_patterns)]
        other => panic!(
            "backend {other:?} not compiled in — rebuild with the matching cargo feature, \
             e.g. `cargo build --features cuda`"
        ),
    }
}
