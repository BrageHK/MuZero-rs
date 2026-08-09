use burn::DispatchDevice;
use serde::{Deserialize, Serialize};

use crate::{mz_config::TemperatureSchedule, replay_buffer::ReplayBuffer};

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
             e.g. `cargo build --features {other:?}`"
        ),
    }
}

pub fn tau_for_step(schedule: &[TemperatureSchedule], step: usize) -> f32 {
    for entry in schedule {
        match entry.step {
            Some(threshold) if step <= threshold => return entry.tau,
            None => return entry.tau,
            _ => {}
        }
    }
    schedule.last().map(|e| e.tau).unwrap_or(1.0)
}

/// Linear warmup, then the MuZero pseudocode's exponential decay:
/// `lr = lr_init * lr_decay_rate ** (step / lr_decay_steps)`.
pub fn lr_for_step(
    base_lr: f64,
    warmup_steps: usize,
    decay_rate: f64,
    decay_steps: usize,
    step: usize,
) -> f64 {
    let warmed = if warmup_steps == 0 || step >= warmup_steps {
        base_lr
    } else {
        base_lr * (step + 1) as f64 / warmup_steps as f64
    };
    if decay_steps == 0 {
        warmed
    } else {
        warmed * decay_rate.powf(step as f64 / decay_steps as f64)
    }
}

pub struct QNormalization {
    q_max: f32,
    q_min: f32,
}

impl QNormalization {
    pub fn from_known_bounds(lower: f32, upper: f32) -> Self {
        QNormalization { 
            q_max: upper,
            q_min: lower
        }
    }

    pub fn update(&mut self, value: f32) {
        self.q_max = self.q_max.max(value);
        self.q_min = self.q_min.min(value);
    }

    pub fn normalize(&self, value: f32) -> f32 {
        if self.q_max > self.q_min {
            (value - self.q_min) / (self.q_max - self.q_min)
        } else {
            value
        }
    }
}

impl Default for QNormalization {
    fn default() -> Self {
        QNormalization {
            q_max: f32::NEG_INFINITY,
            q_min: f32::INFINITY,
        }
    }
}

pub fn save_buffer(buffer: &ReplayBuffer, path: &str) {
    let bytes = rmp_serde::to_vec(&buffer.states).expect("Failed to serialize replay buffer");
    std::fs::write(path, bytes).expect("Failed to write replay buffer");
}

#[cfg(test)]
mod lr_tests {
    use super::lr_for_step;

    #[test]
    fn warmup_ramps_then_holds() {
        assert_eq!(lr_for_step(0.2, 0, 1.0, 0, 0), 0.2);
        assert_eq!(lr_for_step(0.2, 4, 1.0, 0, 0), 0.05);
        assert_eq!(lr_for_step(0.2, 4, 1.0, 0, 3), 0.2);
        assert_eq!(lr_for_step(0.2, 4, 1.0, 0, 99), 0.2);
    }

    #[test]
    fn decay_matches_muzero_pseudocode_formula() {
        assert_eq!(lr_for_step(0.1, 0, 0.1, 100, 0), 0.1);
        assert!((lr_for_step(0.1, 0, 0.1, 100, 100) - 0.01).abs() < 1e-9);
        assert!((lr_for_step(0.1, 0, 0.1, 100, 200) - 0.001).abs() < 1e-9);
    }

    #[test]
    fn zero_decay_steps_disables_decay() {
        assert_eq!(lr_for_step(0.1, 0, 0.1, 0, 100_000), 0.1);
    }
}
