//! Kernel-density bandwidths: Silverman's rule of thumb and Sheather and
//! Jones' plug-in, as R's `bw.nrd0` and `bw.SJ` take them (SCTransform
//! sets its smoothing bandwidth from `bw.SJ`).

/// R's default quantile (type 7) of sorted `x` at `p`.
pub(super) fn quantile(sorted: &[f64], p: f64) -> f64 {
    let h = (sorted.len() as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// Silverman's rule of thumb (Silverman 1986, "Density Estimation for
/// Statistics and Data Analysis", Chapman and Hall, eq. 3.31):
/// `0.9 min(sd, IQR / 1.34) n^(-1/5)`.
pub fn silverman(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let iqr = quantile(&s, 0.75) - quantile(&s, 0.25);
    0.9 * sd.min(iqr / 1.34) * n.powf(-0.2)
}

/// Bins of the pairwise differences in [`sheather_jones`], as R's `bw.SJ`
/// takes by default.
pub const SJ_BINS: usize = 1000;

/// A sample binned for its pairwise sums (Scott and Sheather 1985,
/// "Kernel density estimation with binned data", Communications in
/// Statistics: Theory and Methods 14:1353; the binning of R's `bw.SJ`, `nb` equal-width bins over the range): each
/// value at its bin, and the unordered pairs of distinct values counted by
/// the distance in bins between them, so a pairwise sum costs at most `nb`
/// terms whatever the sample's size.
pub struct PairBins {
    pub n: f64,
    pub width: f64,
    /// `counts[k]`: pairs `k` bins apart.
    pub counts: Vec<f64>,
}

impl PairBins {
    pub fn new(x: &[f64], nb: usize) -> Self {
        let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let nb = nb.max(1);
        let width = ((hi - lo) / nb as f64).max(f64::MIN_POSITIVE);
        let mut occ = vec![0f64; nb];
        for &v in x {
            occ[(((v - lo) / width) as usize).min(nb - 1)] += 1.0;
        }
        let full: Vec<(usize, f64)> = occ
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0.0)
            .map(|(b, &c)| (b, c))
            .collect();
        let mut counts = vec![0f64; nb];
        for (i, &(bi, ci)) in full.iter().enumerate() {
            counts[0] += ci * (ci - 1.0) / 2.0;
            for &(bj, cj) in &full[i + 1..] {
                counts[bj - bi] += ci * cj;
            }
        }
        while counts.len() > 1 && counts.last() == Some(&0.0) {
            counts.pop();
        }
        PairBins {
            n: x.len() as f64,
            width,
            counts,
        }
    }

    /// `sum_i sum_j phi^(r)((x_i - x_j) / a)` over every ordered pair (the
    /// diagonal included), `r` 4 or 6, each difference at its bins'.
    pub fn pair_sum(&self, a: f64, r: u32) -> f64 {
        let off: f64 = self
            .counts
            .iter()
            .enumerate()
            .map(|(k, &c)| c * gauss_deriv(k as f64 * self.width / a, r))
            .sum();
        self.n * gauss_deriv(0.0, r) + 2.0 * off
    }
}

/// `phi^(r)(u)`, the `r`-th derivative of the standard normal density, `r`
/// 4 or 6.
pub fn gauss_deriv(u: f64, r: u32) -> f64 {
    let phi = (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let u2 = u * u;
    match r {
        4 => (u2 * u2 - 6.0 * u2 + 3.0) * phi,
        _ => (u2 * u2 * u2 - 15.0 * u2 * u2 + 45.0 * u2 - 15.0) * phi,
    }
}

/// Sheather and Jones' plug-in bandwidth, the solve-the-equation variant
/// (Sheather and Jones 1991, JRSS B 53:683,
/// doi:10.1111/j.2517-6161.1991.tb01857.x, section 3): `h = [R(K) / (n
/// S(alpha2(h)))]^(1/5)` with the Gaussian kernel, `alpha2(h) = 1.357
/// [S(a) / T(b)]^(1/7) h^(5/7)`, `a = 0.920 lambda n^(-1/7)`, `b = 0.912
/// lambda n^(-1/9)`, `lambda` the interquartile range, and the density
/// functionals `S`, `T` summed over all pairs on [`SJ_BINS`] bins
/// ([`PairBins`]). 0 when the interquartile range is 0 (as Silverman's
/// rule then gives); Silverman's rule when the functionals are not
/// positive.
pub fn sheather_jones(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let lambda = quantile(&s, 0.75) - quantile(&s, 0.25);
    if lambda <= 0.0 || n < 2.0 {
        return 0.0;
    }
    let bins = PairBins::new(x, SJ_BINS);
    let nn = n * (n - 1.0);
    let s_hat = |a: f64| bins.pair_sum(a, 4) / (nn * a.powi(5));
    let t_hat = |b: f64| -bins.pair_sum(b, 6) / (nn * b.powi(7));
    let a = 0.920 * lambda * n.powf(-1.0 / 7.0);
    let b = 0.912 * lambda * n.powf(-1.0 / 9.0);
    let (sa, tb) = (s_hat(a), t_hat(b));
    if !(sa > 0.0 && tb > 0.0) {
        return silverman(x);
    }
    let rk = 1.0 / (2.0 * std::f64::consts::PI.sqrt());
    let ratio = 1.357 * (sa / tb).powf(1.0 / 7.0);
    // g(log h) = log h - log of the right-hand side, solved by the
    // Illinois false position on a bracket around Silverman's value
    let g = |lh: f64| {
        let h = lh.exp();
        let sv = s_hat(ratio * h.powf(5.0 / 7.0));
        if sv <= 0.0 {
            return f64::INFINITY;
        }
        lh - 0.2 * (rk / (n * sv)).ln()
    };
    let h0 = 1.06 * lambda / 1.34 * n.powf(-0.2);
    let (mut lo, mut hi) = (h0.ln() - 3.0, h0.ln() + 3.0);
    let (mut glo, mut ghi) = (g(lo), g(hi));
    let mut tries = 0;
    while glo.signum() == ghi.signum() && tries < 10 {
        lo -= 3.0;
        hi += 3.0;
        glo = g(lo);
        ghi = g(hi);
        tries += 1;
    }
    if !(glo.is_finite() && ghi.is_finite()) || glo.signum() == ghi.signum() {
        return silverman(x);
    }
    let mut side = 0;
    for _ in 0..100 {
        let mid = (lo * ghi - hi * glo) / (ghi - glo);
        let gm = g(mid);
        if gm.abs() < 1e-10 || (hi - lo).abs() < 1e-10 {
            return mid.exp();
        }
        if gm.signum() == glo.signum() {
            lo = mid;
            glo = gm;
            if side == -1 {
                ghi /= 2.0;
            }
            side = -1;
        } else {
            hi = mid;
            ghi = gm;
            if side == 1 {
                glo /= 2.0;
            }
            side = 1;
        }
    }
    (0.5 * (lo + hi)).exp()
}
