pub mod chess;
pub mod chess_mamba;
pub mod othello;

use burn::tensor::backend::BackendTypes;

#[cfg(feature = "flex")]
pub type Be = burn::backend::Flex;
#[cfg(all(feature = "ndarray", not(feature = "flex")))]
pub type Be = burn::backend::NdArray;
#[cfg(all(feature = "tch", not(feature = "flex"), not(feature = "ndarray")))]
pub type Be = burn::backend::LibTorch;
#[cfg(all(feature = "metal", not(feature = "flex"), not(feature = "ndarray"), not(feature = "tch")))]
pub type Be = burn::backend::Metal;
#[cfg(all(
    feature = "vulkan",
    not(feature = "flex"),
    not(feature = "ndarray"),
    not(feature = "tch"),
    not(feature = "metal")
))]
pub type Be = burn::backend::Vulkan;

pub type Device = <Be as BackendTypes>::Device;

pub async fn init_backend(_device: &Device) {}
