pub mod chess;
pub mod othello;

use burn::tensor::backend::BackendTypes;

#[cfg(feature = "flex")]
pub type Be = burn::backend::Flex;
#[cfg(all(feature = "ndarray", not(feature = "flex")))]
pub type Be = burn::backend::NdArray;

pub type Device = <Be as BackendTypes>::Device;

pub async fn init_backend(_device: &Device) {}
