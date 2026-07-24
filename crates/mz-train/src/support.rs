const EPS: f32 = 1e-3;

pub fn transform(x: f32) -> f32 {
    x.signum() * ((x.abs() + 1.0).sqrt() - 1.0) + EPS * x
}

pub fn inverse_transform(x: f32) -> f32 {
    let inner = ((1.0 + 4.0 * EPS * (x.abs() + 1.0 + EPS)).sqrt() - 1.0) / (2.0 * EPS);
    x.signum() * (inner * inner - 1.0)
}

pub fn support_len(support_size: usize) -> usize {
    2 * support_size + 1
}

pub fn scalar_to_two_hot(value: f32, support_size: usize) -> Vec<f32> {
    let len = support_len(support_size);
    let mut out = vec![0.0f32; len];
    let n = support_size as f32;
    let x = transform(value).clamp(-n, n);
    let lower = x.floor();
    let upper_weight = x - lower;
    let lower_idx = (lower + n) as usize;
    out[lower_idx] += 1.0 - upper_weight;
    if upper_weight > 0.0 {
        out[lower_idx + 1] += upper_weight;
    }
    out
}

pub fn support_to_scalar(probs: &[f32], support_size: usize) -> f32 {
    let n = support_size as i64;
    let x = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| p * (i as i64 - n) as f32)
        .sum();
    inverse_transform(x)
}

pub fn logits_to_scalars(flat_logits: &[f32], batch: usize, support_size: usize) -> Vec<f32> {
    let len = support_len(support_size);
    (0..batch)
        .map(|b| {
            let row = &flat_logits[b * len..(b + 1) * len];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
            support_to_scalar(&probs, support_size)
        })
        .collect()
}

pub fn two_hot_batch(values: &[f32], support_size: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() * support_len(support_size));
    for &v in values {
        out.extend(scalar_to_two_hot(v, support_size));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_roundtrip() {
        for x in [-333.0, -12.5, -1.0, 0.0, 1.0, 7.3, 300.0] {
            let y = inverse_transform(transform(x));
            assert!((x - y).abs() < 1e-2, "roundtrip {x} -> {y}");
        }
    }

    #[test]
    fn two_hot_recovers_scalar() {
        let support_size = 50;
        for x in [-20.0, -1.5, 0.0, 1.0, 3.7, 18.2] {
            let two_hot = scalar_to_two_hot(x, support_size);
            let sum: f32 = two_hot.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "two-hot sums to {sum}");
            let recovered = support_to_scalar(&two_hot, support_size);
            assert!((x - recovered).abs() < 1e-2, "{x} -> {recovered}");
        }
    }
}
