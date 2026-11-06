//! The response's association with the exposure, with no outcome model: the
//! weighted contrast of the arms and the marginal correlation.

use super::crossfit::CrossFit;
use super::inference::{clustered_sd, Estimate};

/// The weighted mean response of the exposed minus that of the unexposed over
/// `rows`, with its tile-clustered variance. Weight 1 is the crude contrast,
/// what a co-expression style comparison would report on the same scale; the
/// overlap weight of a row's own arm ([`super::overlap::OverlapTerms::arm_weight`]) is inverse
/// probability weighting, with no outcome model.
pub fn arm_contrast(
    y: &[f64],
    t: &[bool],
    rows: &[u32],
    cf: &CrossFit,
    weight: impl Fn(usize) -> f64,
) -> Estimate {
    let (mut w1, mut s1, mut w0, mut s0) = (0.0, 0.0, 0.0, 0.0);
    let mut treated = 0usize;
    for &i in rows {
        let i = i as usize;
        let w = weight(i);
        if t[i] {
            treated += 1;
            w1 += w;
            s1 += w * y[i];
        } else {
            w0 += w;
            s0 += w * y[i];
        }
    }
    if w1 <= 1e-12 || w0 <= 1e-12 {
        return Estimate::nan(rows.len(), treated);
    }
    let (m1, m0) = (s1 / w1, s0 / w0);
    let mut sums = vec![0.0; cf.n_clusters()];
    for &i in rows {
        let i = i as usize;
        sums[cf.cluster[i] as usize] += if t[i] {
            weight(i) * (y[i] - m1) / w1
        } else {
            -weight(i) * (y[i] - m0) / w0
        };
    }
    let (se, df) = clustered_sd(&sums);
    Estimate {
        theta: m1 - m0,
        se,
        df,
        ..Estimate::nan(rows.len(), treated)
    }
}

/// Pearson's correlation of `y` and `x` over `rows`, with the tile-clustered
/// variance of its linearization: the marginal association, no threshold and
/// no adjustment. A constant `y` or `x` has none.
pub fn marginal_correlation(y: &[f64], x: &[f64], rows: &[u32], cf: &CrossFit) -> Estimate {
    let nan = Estimate::nan(rows.len(), 0);
    let n = rows.len() as f64;
    let mean = |v: &[f64]| rows.iter().map(|&i| v[i as usize]).sum::<f64>() / n;
    let (my, mx) = (mean(y), mean(x));
    let spread = |v: &[f64], m: f64| {
        (rows
            .iter()
            .map(|&i| (v[i as usize] - m).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
    };
    let (sy, sx) = (spread(y, my), spread(x, mx));
    if rows.is_empty() || sy <= 1e-12 || sx <= 1e-12 {
        return nan;
    }
    let scaled = |i: u32| ((y[i as usize] - my) / sy, (x[i as usize] - mx) / sx);
    let r = rows
        .iter()
        .map(|&i| {
            let (a, b) = scaled(i);
            a * b
        })
        .sum::<f64>()
        / n;
    let mut sums = vec![0.0; cf.n_clusters()];
    for &i in rows {
        let (a, b) = scaled(i);
        sums[cf.cluster[i as usize] as usize] += (a * b - 0.5 * r * (a * a + b * b)) / n;
    }
    let (se, df) = clustered_sd(&sums);
    Estimate {
        theta: r,
        se,
        df,
        ..nan
    }
}
