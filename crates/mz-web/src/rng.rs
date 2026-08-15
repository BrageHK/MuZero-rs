/// splitmix64. Self-contained so the wasm build needs no entropy source: the
/// seed comes from JS.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in the open interval (0, 1), so the log below never blows up.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    /// Gumbel(0, 1) via inverse transform.
    pub fn gumbel(&mut self) -> f32 {
        -(-self.next_f32().ln()).ln()
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        self.next_f32()
    }

    /// Uniform integer in [0, bound).
    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            slice.swap(i, self.below(i + 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gumbel_is_finite_and_roughly_centred() {
        let mut rng = Rng::new(12345);
        let samples: Vec<f32> = (0..20_000).map(|_| rng.gumbel()).collect();
        assert!(samples.iter().all(|g| g.is_finite()));
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        // Gumbel(0, 1) has mean = Euler-Mascheroni.
        assert!((mean - 0.5772).abs() < 0.05, "mean {mean}");
    }
}
