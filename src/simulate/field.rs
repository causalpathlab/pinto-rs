//! Smooth random fields on spot coordinates, from random Fourier features:
//! `z(x) = Σ_j a_j cos(w_j·x / ℓ + b_j)` with `w_j ~ N(0, I)`,
//! `b_j ~ U(0, 2π)` and `a_j ~ N(0, 1)` approximates a Gaussian process with
//! the squared-exponential kernel `exp(−|x − y|² / 2ℓ²)` (Rahimi and Recht
//! 2007, "Random features for large-scale kernel machines", NeurIPS).
//!
//! Kept apart from the random Fourier features the lra estimator adjusts
//! with, so the simulator shares no statistics code with a method it tests.

use rand::rngs::SmallRng;
use rand::RngExt;
use rand_distr::{Distribution, StandardNormal};

/// Frequencies per field: enough that the field is close to Gaussian.
pub const N_FREQ: usize = 128;

/// A field with length scale `length_um` at `coords`, standardized to mean 0
/// and SD 1 over them.
pub fn rff_field(coords: &[(f64, f64)], length_um: f64, rng: &mut SmallRng) -> Vec<f64> {
    let freqs: Vec<[f64; 4]> = (0..N_FREQ)
        .map(|_| {
            let wx: f64 = StandardNormal.sample(rng);
            let wy: f64 = StandardNormal.sample(rng);
            let b = rng.random::<f64>() * std::f64::consts::TAU;
            let a: f64 = StandardNormal.sample(rng);
            [wx, wy, b, a]
        })
        .collect();
    let z: Vec<f64> = coords
        .iter()
        .map(|&(x, y)| {
            freqs
                .iter()
                .map(|&[wx, wy, b, a]| a * ((wx * x + wy * y) / length_um + b).cos())
                .sum()
        })
        .collect();
    let (m, sd) = mean_sd(&z);
    z.into_iter()
        .map(|v| if sd > 0.0 { (v - m) / sd } else { 0.0 })
        .collect()
}

/// Mean and population SD.
pub fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len().max(1) as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    (m, var.sqrt())
}
