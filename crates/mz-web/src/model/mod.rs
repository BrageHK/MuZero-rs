pub mod chess;
pub mod chess_mamba;
pub mod othello;

use burn::tensor::backend::BackendTypes;

// `Be`: the shared, feature-selected backend used by chess/chess_mamba and
// by the native benches (`examples/bench_mcts*.rs` swap it via
// `--features <x> --no-default-features`). Othello always runs on `CpuBe`
// below instead, regardless of which arm wins here -- see model/othello.rs.
#[cfg(feature = "webgpu")]
pub type Be = burn::backend::WebGpu;
#[cfg(all(feature = "flex", not(feature = "webgpu")))]
pub type Be = burn::backend::Flex;
#[cfg(all(feature = "ndarray", not(feature = "webgpu"), not(feature = "flex")))]
pub type Be = burn::backend::NdArray;
#[cfg(all(
    feature = "tch",
    not(feature = "webgpu"),
    not(feature = "flex"),
    not(feature = "ndarray")
))]
pub type Be = burn::backend::LibTorch;
#[cfg(all(
    feature = "metal",
    not(feature = "webgpu"),
    not(feature = "flex"),
    not(feature = "ndarray"),
    not(feature = "tch")
))]
pub type Be = burn::backend::Metal;
#[cfg(all(
    feature = "vulkan",
    not(feature = "webgpu"),
    not(feature = "flex"),
    not(feature = "ndarray"),
    not(feature = "tch"),
    not(feature = "metal")
))]
pub type Be = burn::backend::Vulkan;

pub type Device = <Be as BackendTypes>::Device;

/// WebGPU cannot be set up synchronously in the browser, so the adapter
/// request has to be awaited before the first tensor op. No-op for every
/// other backend `Be` might resolve to.
pub async fn init_backend(device: &Device) {
    #[cfg(feature = "webgpu")]
    burn::backend::wgpu::init_setup_async::<burn::backend::wgpu::graphics::WebGpu>(
        device,
        Default::default(),
    )
    .await;
    #[cfg(not(feature = "webgpu"))]
    let _ = device;
}

// Othello's dedicated CPU backend: fast for its batch-of-1 MLP regardless of
// which GPU backend `Be` is set to for chess. Unconditional (not behind a
// feature) so Othello keeps working under e.g. `--no-default-features
// --features tch`, which the native chess benches use to swap `Be` around.
pub type CpuBe = burn::backend::Flex;
pub type CpuDevice = <CpuBe as BackendTypes>::Device;

thread_local! {
    static SHARED_DEVICE: std::cell::RefCell<Option<Device>> = const { std::cell::RefCell::new(None) };
}

/// `ChessGame` and `ChessMambaBot` both run on `Be` and, in `worker.js`,
/// live in the same process -- cubecl's wgpu client registry panics ("a
/// server is still registered for device") if `Device::default()` +
/// `init_backend` each stand up their own runtime for what's the same
/// underlying GPU device. Both `create_*` functions call this instead, so
/// only the first ever registers a client; the rest reuse it.
pub(crate) async fn shared_device() -> Device {
    if let Some(device) = SHARED_DEVICE.with(|cell| cell.borrow().clone()) {
        return device;
    }
    let device = Device::default();
    init_backend(&device).await;
    SHARED_DEVICE.with(|cell| *cell.borrow_mut() = Some(device.clone()));
    device
}
