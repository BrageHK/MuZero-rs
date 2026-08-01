use fastrand::Rng;

use crate::mz_config::MuZeroConfig;

const PAD: usize = 4;
const INTENSITY_SCALE: f32 = 0.05;

pub struct Augmenter {
    channels: usize,
    height: usize,
    width: usize,
    pad: usize,
    scratch: Vec<f32>,
    rng: Rng,
}

impl Augmenter {
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        let pad = PAD.min(height.min(width).saturating_sub(1));
        Augmenter {
            channels,
            height,
            width,
            pad,
            scratch: vec![0.0; channels * height * width],
            rng: Rng::new(),
        }
    }

    pub fn from_config(mz_conf: &MuZeroConfig) -> Option<Self> {
        if !mz_conf.augmentation {
            return None;
        }
        let (h, w) = (mz_conf.board_height, mz_conf.board_width);
        if h == 0 || w == 0 || !mz_conf.obs_dim.is_multiple_of(h * w) {
            return None;
        }
        Some(Augmenter::new(mz_conf.obs_dim / (h * w), h, w))
    }

    pub fn apply(&mut self, state: &mut [f32]) {
        assert_eq!(
            state.len(),
            self.channels * self.height * self.width,
            "augmenter shape does not match the observation"
        );

        if self.pad > 0 {
            let dy = self.rng.usize(0..=2 * self.pad);
            let dx = self.rng.usize(0..=2 * self.pad);
            shift(
                state,
                &mut self.scratch,
                (self.channels, self.height, self.width),
                self.pad,
                (dy, dx),
            );
            state.copy_from_slice(&self.scratch);
        }

        let scale = 1.0 + INTENSITY_SCALE * standard_normal(&mut self.rng).clamp(-2.0, 2.0);
        for v in state.iter_mut() {
            *v *= scale;
        }
    }
}

fn shift(
    src: &[f32],
    dst: &mut [f32],
    shape: (usize, usize, usize),
    pad: usize,
    offset: (usize, usize),
) {
    let (channels, height, width) = shape;
    let (dy, dx) = offset;
    for c in 0..channels {
        let plane = c * height * width;
        for y in 0..height {
            let sy = (y + dy).saturating_sub(pad).min(height - 1);
            for x in 0..width {
                let sx = (x + dx).saturating_sub(pad).min(width - 1);
                dst[plane + y * width + x] = src[plane + sy * width + sx];
            }
        }
    }
}

fn standard_normal(rng: &mut Rng) -> f32 {
    let u1 = rng.f32().max(f32::EPSILON);
    let u2 = rng.f32();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(c: usize, h: usize, w: usize) -> Vec<f32> {
        (0..c * h * w).map(|i| i as f32).collect()
    }

    #[test]
    fn centered_shift_is_identity() {
        let src = ramp(2, 5, 5);
        let mut dst = vec![0.0; src.len()];
        shift(&src, &mut dst, (2, 5, 5), 4, (4, 4));
        assert_eq!(dst, src);
    }

    #[test]
    fn offset_shift_replicates_edges() {
        let src: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        let mut dst = vec![0.0; 9];

        shift(&src, &mut dst, (1, 3, 3), 1, (0, 1));
        assert_eq!(dst, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        shift(&src, &mut dst, (1, 3, 3), 1, (2, 1));
        assert_eq!(dst, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn apply_preserves_length_and_stays_finite() {
        let mut aug = Augmenter::new(1, 8, 8);
        let mut state = ramp(1, 8, 8);
        for _ in 0..32 {
            aug.apply(&mut state);
            assert_eq!(state.len(), 64);
            assert!(state.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn constant_image_only_gets_rescaled() {
        let mut aug = Augmenter::new(1, 6, 6);
        let mut state = vec![2.0; 36];
        aug.apply(&mut state);

        let first = state[0];
        assert!(state.iter().all(|v| (v - first).abs() < 1e-6));
        assert!((first / 2.0 - 1.0).abs() <= 2.0 * INTENSITY_SCALE + 1e-6);
    }
}
