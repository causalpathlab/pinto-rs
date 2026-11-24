//! The two-group model: the empirical null recovered from the bulk, the
//! local fdr monotone above it, null below it, and what a non-finite
//! statistic gets.

use crate::util::local_fdr::{two_groups, NormalNull};

/// `n` statistics at the quantiles of `N(mean, sd²)`, deterministic.
fn normal_quantiles(n: usize, mean: f64, sd: f64) -> Vec<f64> {
    use special::Error as _;
    (0..n)
        .map(|k| {
            let u = (k as f64 + 0.5) / n as f64;
            mean + sd * std::f64::consts::SQRT_2 * (2.0 * u - 1.0).inv_error()
        })
        .collect()
}

#[test]
fn the_empirical_null_is_the_bulk() {
    let mut z = normal_quantiles(950, 0.3, 1.2);
    z.extend(normal_quantiles(50, 4.0, 0.5));
    let null = NormalNull::empirical(&z).unwrap();
    assert!((null.mean - 0.3).abs() < 0.15, "{null:?}");
    assert!((null.sd - 1.2).abs() < 0.2, "{null:?}");
    assert!(
        NormalNull::empirical(&[1.0, 1.0, 1.0]).is_none(),
        "no spread"
    );
    assert!(NormalNull::empirical(&[0.0, 1.0]).is_none(), "too few");
}

#[test]
fn the_local_fdr_falls_away_from_the_null_above_it() {
    let mut z = normal_quantiles(950, 0.0, 1.0);
    z.extend(normal_quantiles(50, 4.0, 0.5));
    let fit = two_groups(&z, NormalNull::THEORETICAL);
    assert!(
        fit.null_share > 0.8 && fit.null_share <= 1.0,
        "{}",
        fit.null_share
    );
    let mut above: Vec<usize> = (0..z.len()).filter(|&i| z[i] > 0.0).collect();
    above.sort_by(|&a, &b| z[a].total_cmp(&z[b]));
    assert!(
        above.windows(2).all(|w| fit.lfdr[w[1]] <= fit.lfdr[w[0]]),
        "monotone above the null"
    );
    let at = |v: f64| {
        let i = (0..z.len())
            .min_by(|&a, &b| (z[a] - v).abs().total_cmp(&(z[b] - v).abs()))
            .unwrap();
        fit.lfdr[i]
    };
    let (centre, up) = (at(0.1), at(4.0));
    assert!(centre > 0.8 && up < 0.1, "{centre} {up}");
}

#[test]
fn the_lower_side_is_null() {
    let mut z = normal_quantiles(900, 0.0, 1.0);
    z.extend(normal_quantiles(50, -4.0, 0.5));
    let fit = two_groups(&z, NormalNull::THEORETICAL);
    assert!(z
        .iter()
        .zip(&fit.lfdr)
        .filter(|(v, _)| **v <= 0.0)
        .all(|(_, l)| *l == 1.0));
}

#[test]
fn nan_has_no_local_fdr_and_infinity_is_signal() {
    let mut z = normal_quantiles(200, 0.0, 1.0);
    z[7] = f64::NAN;
    z[9] = f64::INFINITY;
    // NaN has none; +∞ is signal, −∞ null.
    let above = two_groups(&z, NormalNull::THEORETICAL);
    z[9] = f64::NEG_INFINITY;
    let below = two_groups(&z, NormalNull::THEORETICAL);
    assert!(above.lfdr[7].is_nan());
    assert_eq!((above.lfdr[9], below.lfdr[9]), (0.0, 1.0));
    // Too few finite statistics to fit: no lfdr.
    let few = two_groups(&[1.0, f64::NAN], NormalNull::THEORETICAL);
    assert!(few.lfdr.iter().all(|l| l.is_nan()));
}
