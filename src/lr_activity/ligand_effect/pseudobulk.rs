//! Random pseudobulk: the effect estimated on small random groups of
//! receivers instead of single cells.
//!
//! A receiver's programme count is mostly zeros; summing a few comparable
//! receivers makes it dense. A unit is `m` receivers drawn at random from
//! those sharing a section, a cross-fitting tile, an identity bin, the
//! contact exposure arm and the ring exposure arm. Grouping never mixes
//! arms or identities, so every unit keeps one treatment and one
//! competence, and its confounders are its members' average. Units inherit
//! their tile's fold and variance cluster.
//!
//! The grouping is random, so one partition is one draw. The estimate is
//! averaged over independent draws and its variance combined by Rubin's
//! rules (within-draw plus `(1 + 1/R)` × between-draw), so the interval
//! also carries the randomness of the grouping. Spatial averaging over a
//! tissue stands in for repeated sampling, which is what licenses treating
//! random groups of comparable cells as replicates.
//!
//! Units hold [`UNIT_SIZE`] receivers, and the number of draws is chosen
//! per pair ([`enough_draws`]): draws continue until the Monte Carlo error
//! of the averaged estimate is small beside its sampling error. When cell
//! types are given, units never mix them either.

use super::estimate::{Exposure, Response};
use super::overlap::{CrossFit, Design, Estimate};
use super::tissue::{AdjustmentStep, Tissue};
use crate::util::common::*;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Receivers per unit.
pub const UNIT_SIZE: usize = 10;

/// Fewest draws: the between-draw variance needs at least two degrees of
/// freedom before the stopping rule can read it.
pub const MIN_DRAWS: usize = 3;

/// Most draws. Past this the grouping noise is large beside the sampling
/// noise, and more draws would only refine an interval that already
/// carries that spread (Rubin's between-draw term, Barnard–Rubin df).
pub const MAX_DRAWS: usize = 20;

/// Monte Carlo SE of the averaged estimate, as a share of its within-draw
/// SE, below which the draws stop. At 0.1 the finite number of draws adds
/// at most 1% to the variance of the average.
pub const MC_SE_RATIO: f64 = 0.1;

/// Whether the draws so far (one headline estimate per draw) are enough:
/// at least [`MIN_DRAWS`], and either [`MAX_DRAWS`] or a Monte Carlo SE
/// `sd(θ_d) / √R` below [`MC_SE_RATIO`] × `√mean(se_d²)`. Draws that were
/// not estimable are left out; with fewer than two estimable ones after
/// [`MIN_DRAWS`] the headline is not estimable and more draws would not
/// change that.
pub fn enough_draws(draws: &[Estimate]) -> bool {
    if draws.len() >= MAX_DRAWS {
        return true;
    }
    if draws.len() < MIN_DRAWS {
        return false;
    }
    let ok: Vec<&Estimate> = draws.iter().filter(|e| usable(e)).collect();
    if ok.len() < 2 {
        return true;
    }
    let r = ok.len() as f64;
    let mean = ok.iter().map(|e| e.theta).sum::<f64>() / r;
    let between = ok.iter().map(|e| (e.theta - mean).powi(2)).sum::<f64>() / (r - 1.0);
    let within = ok.iter().map(|e| e.se * e.se).sum::<f64>() / r;
    (between / r).sqrt() < MC_SE_RATIO * within.sqrt()
}

/// Most draws under `opts`: one on single cells, [`MAX_DRAWS`] under
/// [`AUTO_DRAWS`](super::estimate::AUTO_DRAWS) (stopped early by
/// [`stop_drawing`]), else the fixed count. `lra` and `lra-simulate` both
/// draw by this rule, so the benchmark measures the shipped estimator.
pub fn draw_limit(opts: &super::estimate::EffectOpts) -> usize {
    match (opts.pseudobulk > 1, opts.draws) {
        (false, _) => 1,
        (true, super::estimate::AUTO_DRAWS) => MAX_DRAWS,
        (true, d) => d,
    }
}

/// Whether automatic draws can stop, given the headline of each draw so
/// far ([`enough_draws`]); fixed draws run to [`draw_limit`].
pub fn stop_drawing(opts: &super::estimate::EffectOpts, headline: &[Estimate]) -> bool {
    opts.pseudobulk > 1 && opts.draws == super::estimate::AUTO_DRAWS && enough_draws(headline)
}

/// One random partition of the receivers into units.
pub struct Units {
    pub design: Design,
    pub steps: Vec<AdjustmentStep>,
    pub cf: CrossFit,
    /// The units inside each of the tissue's scopes ([`Tissue::scopes`]).
    pub scope_rows: Vec<Vec<u32>>,
    /// The pair's exposure side on the units.
    pub exposure: Exposure,
    /// Unit of each receiver.
    pub unit_of: Vec<u32>,
    /// Summed size factor of each unit.
    pub size_factor: Vec<f64>,
    /// First member (receiver row) and member count of each unit.
    pub lead: Vec<u32>,
    pub size: Vec<usize>,
}

impl Units {
    /// A per-receiver flag at the unit level (units are pure in identity
    /// bin, so competence is one value per unit).
    pub fn of_lead(&self, v: &[bool]) -> Vec<bool> {
        self.lead.iter().map(|&i| v[i as usize]).collect()
    }

    fn mean_by_unit(&self, v: &[f64]) -> Vec<f64> {
        let s = sum_by_unit(v, &self.unit_of, self.size.len());
        s.iter()
            .zip(&self.size)
            .map(|(s, &m)| s / m as f64)
            .collect()
    }

    /// A programme's response side on these units: the response from the
    /// members' summed counts of `targets`, spillover and bleed as the
    /// members' means of `r`, the same programme per receiver.
    pub fn response_of(&self, r: &Response, targets: &[(f64, &[(u32, f32)])]) -> Response {
        Response {
            y: unit_response(targets, &self.unit_of, &self.size_factor),
            spill: self.mean_by_unit(&r.spill),
            bleed: self.mean_by_unit(&r.bleed),
        }
    }
}

/// Assign each receiver to a unit of about `m` (between `m/2` and `3m/2`)
/// inside its stratum. Strata smaller than `m` form one unit.
pub fn partition(tissue: &Tissue, f: &Exposure, m: usize, seed: u64) -> (Vec<u32>, usize) {
    let n = tissue.n();
    // Section, tile, identity bin, cell type, contact arm, ring arm.
    type Stratum = (u32, u32, usize, u32, bool, bool);
    let mut strata: HashMap<Stratum, Vec<u32>> = HashMap::default();
    for i in 0..n {
        let key = (
            tissue.section[i],
            tissue.cf.cluster[i],
            tissue.identity_bin[i],
            tissue.cell_type.as_ref().map_or(u32::MAX, |c| c[i]),
            f.t[i],
            f.t_ring[i],
        );
        strata.entry(key).or_default().push(i as u32);
    }
    // Visit strata in a fixed order so a seed fixes the partition.
    let mut keys: Vec<_> = strata.keys().copied().collect();
    keys.sort_unstable();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut unit_of = vec![u32::MAX; n];
    let mut next = 0u32;
    for key in keys {
        let members = strata.get_mut(&key).expect("key from map");
        members.shuffle(&mut rng);
        let k = (members.len() / m.max(1)).max(1);
        // `k` units, sizes differing by at most one.
        for (j, &i) in members.iter().enumerate() {
            unit_of[i as usize] = next + (j * k / members.len()) as u32;
        }
        next += k as u32;
    }
    (unit_of, next as usize)
}

pub fn sum_by_unit(v: &[f64], unit_of: &[u32], n_units: usize) -> Vec<f64> {
    let mut s = vec![0.0; n_units];
    for (&u, &x) in unit_of.iter().zip(v) {
        s[u as usize] += x;
    }
    s
}

/// The signed programme response of each unit from its members' summed
/// counts: `Σ_g w_g log1p(Σ x_g / Σ sf) / Σ |w_g|`.
pub fn unit_response(
    targets: &[(f64, &[(u32, f32)])],
    unit_of: &[u32],
    size_factor: &[f64],
) -> Vec<f64> {
    let n_units = size_factor.len();
    let total: f64 = targets.iter().map(|(w, _)| w.abs()).sum::<f64>().max(1e-12);
    let mut y = vec![0.0; n_units];
    let mut gene = vec![0.0; n_units];
    let mut touched: Vec<usize> = Vec::new();
    for (w, nz) in targets {
        for &(r, x) in *nz {
            let u = unit_of[r as usize] as usize;
            if gene[u] == 0.0 {
                touched.push(u);
            }
            gene[u] += x as f64;
        }
        for &u in &touched {
            y[u] += w * (gene[u] / size_factor[u]).ln_1p();
            gene[u] = 0.0;
        }
        touched.clear();
    }
    for v in &mut y {
        *v /= total;
    }
    y
}

/// Build one partition's units and the pair's exposure side on them:
/// treatment from each unit's first member (units are pure in it), own
/// ligand from the members' summed `ligand_counts` (per receiver, as the
/// estimator sees them), the niche as the members' mean.
pub fn draw(tissue: &Tissue, f: &Exposure, ligand_counts: &[f64], m: usize, seed: u64) -> Units {
    let (unit_of, n_units) = partition(tissue, f, m, seed);
    let mut first = vec![u32::MAX; n_units];
    let mut size = vec![0usize; n_units];
    for (i, &u) in unit_of.iter().enumerate() {
        if first[u as usize] == u32::MAX {
            first[u as usize] = i as u32;
        }
        size[u as usize] += 1;
    }
    let lead = |u: usize| first[u] as usize;

    let size_factor = sum_by_unit(&tissue.size_factor, &unit_of, n_units);
    let cf = CrossFit::new(
        (0..n_units).map(|u| tissue.cf.fold[lead(u)]).collect(),
        tissue.cf.k,
        (0..n_units).map(|u| tissue.cf.cluster[lead(u)]).collect(),
    );
    let section: Vec<u32> = (0..n_units).map(|u| tissue.section[lead(u)]).collect();
    let cell_type: Option<Vec<u32>> = tissue
        .cell_type
        .as_ref()
        .map(|c| (0..n_units).map(|u| c[lead(u)]).collect());
    let scopes: Vec<_> = tissue.scopes.iter().map(|(_, s)| *s).collect();
    let scope_rows = super::tissue::scope_rows(&scopes, &section, cell_type.as_deref());

    // Members' mean confounders, accumulated in f64 a design column at a
    // time (the matrix is column-major).
    let p = tissue.design.p();
    let mut acc = vec![0.0f64; n_units * p];
    for j in 0..p {
        let col = tissue.design.x.column(j);
        for (&u, &v) in unit_of.iter().zip(col.as_slice()) {
            acc[u as usize * p + j] += v as f64;
        }
    }
    // Members' means, then the unit's pooled size factor and member count:
    // the pooled log response depends on pooled depth (a bigger unit has
    // less Jensen shrinkage at the same rate), and unit sizes can differ
    // between arms, so both are confounders at the unit level.
    let x = DMatrix::<f32>::from_fn(n_units, p + 2, |u, j| {
        if j < p {
            (acc[u * p + j] / size[u] as f64) as f32
        } else if j == p {
            size_factor[u].ln() as f32
        } else {
            (size[u] as f64).ln() as f32
        }
    });
    drop(acc);
    let design = Design::new(x, &cf);
    let steps = tissue
        .steps
        .iter()
        .map(|s| AdjustmentStep {
            name: s.name,
            design: design.prefix(s.design.p()),
        })
        .collect();
    let lig = sum_by_unit(ligand_counts, &unit_of, n_units);
    let niche = sum_by_unit(&f.niche, &unit_of, n_units);
    let exposure = Exposure {
        tau: f.tau,
        t: (0..n_units).map(|u| f.t[lead(u)]).collect(),
        t_ring: (0..n_units).map(|u| f.t_ring[lead(u)]).collect(),
        own: (0..n_units)
            .map(|u| (lig[u] / size_factor[u]).ln_1p())
            .collect(),
        niche: (0..n_units).map(|u| niche[u] / size[u] as f64).collect(),
    };
    Units {
        design,
        steps,
        cf,
        scope_rows,
        exposure,
        unit_of,
        size_factor,
        lead: first,
        size,
    }
}

/// Whether a draw's estimate can enter Rubin's rules: a finite estimate
/// and SE and a positive df. A NaN df (fewer than two variance clusters)
/// makes the draw not estimable; it never reads as infinite df.
pub fn usable(e: &Estimate) -> bool {
    e.theta.is_finite() && e.se.is_finite() && e.df > 0.0
}

/// Rubin's rules over draws: mean estimate, variance = mean within-draw
/// variance + `(1 + 1/R)` × between-draw variance. Draws that were not
/// estimable (see [`usable`]) are left out; none left gives NaN. Degrees of freedom by
/// Barnard & Rubin (1999, Biometrika 86:948), with the draws' smallest df
/// as the complete-data df: the between-draw spread, estimated from `R`
/// draws, costs degrees of freedom in proportion to its share of the
/// variance.
pub fn combine(draws: &[Estimate]) -> Estimate {
    let ok: Vec<&Estimate> = draws.iter().filter(|e| usable(e)).collect();
    let Some(first) = ok.first() else {
        let e = draws
            .first()
            .copied()
            .unwrap_or_else(|| Estimate::nan(0, 0));
        // Every draw in one variance cluster: say so, as each draw did.
        let one_cluster = !draws.is_empty()
            && draws
                .iter()
                .all(|d| d.df == super::estimate::ONE_CLUSTER_DF);
        let nan = Estimate::nan(e.n, e.n_treated);
        let df = if one_cluster { e.df } else { nan.df };
        return Estimate { df, ..nan };
    };
    let r = ok.len() as f64;
    let mean = |f: &dyn Fn(&Estimate) -> f64| ok.iter().map(|e| f(e)).sum::<f64>() / r;
    let theta = mean(&|e| e.theta);
    let within = mean(&|e| e.se * e.se);
    let between = if ok.len() > 1 {
        ok.iter().map(|e| (e.theta - theta).powi(2)).sum::<f64>() / (r - 1.0)
    } else {
        0.0
    };
    let total = within + (1.0 + 1.0 / r) * between;
    let df_com = ok.iter().map(|e| e.df).fold(f64::INFINITY, f64::min);
    let lambda = if total > 0.0 {
        (1.0 + 1.0 / r) * between / total
    } else {
        0.0
    };
    let df_obs = if df_com.is_finite() {
        (df_com + 1.0) / (df_com + 3.0) * df_com * (1.0 - lambda)
    } else {
        f64::INFINITY
    };
    let df = if ok.len() < 2 {
        df_com
    } else if lambda <= 0.0 {
        df_obs
    } else {
        let df_old = (r - 1.0) / (lambda * lambda);
        df_old * df_obs / (df_old + df_obs)
    };
    Estimate {
        theta,
        se: total.sqrt(),
        n: first.n,
        n_treated: first.n_treated,
        mean_h: mean(&|e| e.mean_h),
        ess: mean(&|e| e.ess),
        df,
    }
}
