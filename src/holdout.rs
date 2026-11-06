//! `pinto predict` on an `lc` (or `dsvd`) model: held-out scoring.
//!
//! A model fitted without some cells (a donor's files left out, or a region
//! split off with `data-beans split`) is asked how well its community profiles
//! `λ_gk` explain those cells. lc assigns EDGES, so the unit scored is a cell
//! pair of the held-out sample's own graph: its counts `x_i + x_j` against the
//! profiles, as the fit saw its training pairs (`--holdout-unit cell` scores
//! cells instead).
//!
//! Each unit's counts are split in two. One part fits the unit's community
//! proportions `π` by EM; the other is scored as a multinomial over the scored
//! features, `Σ_g x_g log p(g)` with `p(g) ∝ Σ_k w_k δ_g p̃_k(g)` renormalised
//! over them. Two ways to split:
//!
//! - document completion (the default): binomial thinning,
//!   `--holdout-fit-fraction` of every count to the fitting part;
//! - `--ablate-features FILE`: the listed features are hidden from the fit and
//!   exactly they are scored, as `senna predict --ablate-features` does, so the
//!   same file gives the same protocol in both.
//!
//! The scores sit beside the references `senna predict` and cage's `pinto
//! predict` report under the same names: the null (`eval_null_*`, the
//! count-weighted feature composition of the TRAINING data, `--null-from`), and
//! the ceiling (the held-out sample's own composition: it knows the sample's
//! marginal, but is one profile for every unit, so a per-unit mixture can
//! score above it).
//! The pooled K = 1 profile (the communities' profiles weighted by their
//! training edge shares, `eval_pooled_k1_*`) isolates what the communities add
//! over one profile. A non-spatial model (an lc fit without `-c`) scored here on
//! the same sample is scored on the same pairs.
//!
//! lc fits its profiles on raw counts, so the held-out sample is on their
//! scale; what it lacks is its own per-feature batch shift `δ`. That is
//! estimated per batch from the FITTING parts only (`--holdout-delta-iters`
//! rounds of `δ_g ← δ_g (obs_g + 1) / (pred_g + 1)`), so the scored parts never
//! inform it. Hidden features under `--ablate-features` keep `δ = 1`: nothing in
//! the fit sees them.

use crate::util::common::*;
use crate::util::profile_match::em_proportions;
use clap::{Args, ValueEnum};
use legume_numeric::matrix::agreement::{agreement_from_rate, PROB_FLOOR};
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution};
use serde::{Deserialize, Serialize};

mod predict;
pub use predict::predict_profiles;

#[derive(ValueEnum, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum HeldOut {
    /// Whole donors were left out of the model's fit
    #[default]
    WholeDonor,
    /// Some of a donor's cells were left out, others were fitted
    Cells,
}

#[derive(ValueEnum, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    /// Cell pairs of the sample's graph, as lc assigns them
    #[default]
    Pair,
    /// Single cells
    Cell,
}

/// How a unit's counts were split, as the manifest names it.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SplitKind {
    Thinning,
    FeatureAblation,
}

#[derive(Args, Debug, Clone, PartialEq)]
#[command(next_help_heading = "Held-out scoring (lc / dsvd models)")]
pub struct HoldoutArgs {
    #[arg(
        long,
        value_enum,
        default_value_t = HeldOut::WholeDonor,
        help = "What the model's fit left out of this sample: whole donors, or some cells",
        long_help = "What the model's fit left out, as recorded in the manifest:\n\
                     whole-donor when every cell of this sample's donors was left\n\
                     out (their files), cells when the model fitted other cells of\n\
                     the same donors (a region split, `data-beans split --grid`)."
    )]
    pub held_out: HeldOut,

    #[arg(
        long,
        value_enum,
        default_value_t = Unit::Pair,
        help = "The unit scored: cell pairs (as lc assigns them) or cells"
    )]
    pub holdout_unit: Unit,

    #[arg(
        long,
        default_value_t = 0.5,
        value_parser = open_fraction,
        help = "Share of each unit's counts that fits its proportions; the rest is scored"
    )]
    pub holdout_fit_fraction: f64,

    #[arg(
        long,
        value_name = "FILE",
        help = "Hide these features from the fit and score exactly them (one name per line)",
        long_help = "Instead of thinning every count, hide the listed features from\n\
                     the fit of each unit's proportions and score exactly them, as\n\
                     `senna predict --ablate-features` does: pass the same file to\n\
                     both for one protocol. Implies --eval-features on the same file\n\
                     unless that is given."
    )]
    pub ablate_features: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 3,
        help = "Rounds of the held-out batch shift δ, from the fitting parts (0 = none)",
        long_help = "Per round: fit every unit's proportions under the current δ,\n\
                     then set δ_g ← δ_g (obs_g + 1) / (pred_g + 1) per batch, over the\n\
                     FITTING parts only, so the scored parts never inform it. Same\n\
                     default as `senna predict --delta-iters`. 0 applies the model's\n\
                     profiles unshifted."
    )]
    pub holdout_delta_iters: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "EM iterations per unit",
        hide = true
    )]
    pub holdout_em_iters: usize,
}

fn open_fraction(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|e| format!("{e}"))?;
    if v > 0.0 && v < 1.0 {
        Ok(v)
    } else {
        Err(format!("{v} is not in (0, 1)"))
    }
}

/// A per-count total is NaN when nothing was scored. JSON has no NaN, so it
/// is written as `null`; this reads `null` back as NaN, so such a manifest
/// still loads.
mod nan_as_null {
    use serde::{Deserialize, Deserializer};

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
    }
}

/// The manifest's `holdout` block.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HoldoutInfo {
    /// The model's prefix, as given.
    pub model: String,
    /// The model's command (`lc`, `dsvd`).
    pub model_command: String,
    pub held_out: HeldOut,
    pub unit: Unit,
    pub split: SplitKind,
    /// Under thinning: the share of each count that fits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit_fraction: Option<f64>,
    pub seed: u64,
    pub n_units: usize,
    /// Units with counts in both parts: the ones scored.
    pub n_scored: usize,
    pub n_features_matched: usize,
    /// Features the likelihood is over.
    pub n_features_scored: usize,
    /// Under feature ablation: the features hidden from the fit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_features_ablated: Option<usize>,
    /// Where the null came from: `--null-from`, the model's own training data,
    /// or (with a warning) the held-out sample, which makes it a second ceiling.
    pub null_from: String,
    pub delta_iters: usize,
    /// Counts in the scored parts.
    #[serde(deserialize_with = "nan_as_null::deserialize")]
    pub eval_count: f64,
    #[serde(deserialize_with = "nan_as_null::deserialize")]
    pub eval_llik_per_count: f64,
    #[serde(deserialize_with = "nan_as_null::deserialize")]
    pub eval_null_llik_per_count: f64,
    #[serde(deserialize_with = "nan_as_null::deserialize")]
    pub eval_pooled_k1_llik_per_count: f64,
    #[serde(deserialize_with = "nan_as_null::deserialize")]
    pub eval_ceiling_llik_per_count: f64,
    /// Means over units with a value (`--eval-features` only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_spearman: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_pearson_log1p: Option<f64>,
    /// The same totals per section (the `@sample` of the cells' names), so a
    /// run holding out several donors keeps them apart.
    #[serde(default)]
    pub per_section: Vec<SectionScore>,
    pub notes: Vec<String>,
}

/// One section's held-out totals.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SectionScore {
    pub section: String,
    pub n_scored: usize,
    #[serde(deserialize_with = "nan_as_null::deserialize")]
    pub eval_count: f64,
    #[serde(deserialize_with = "nan_as_null::deserialize")]
    pub eval_llik_per_count: f64,
    #[serde(deserialize_with = "nan_as_null::deserialize")]
    pub eval_null_llik_per_count: f64,
    #[serde(deserialize_with = "nan_as_null::deserialize")]
    pub eval_pooled_k1_llik_per_count: f64,
    #[serde(deserialize_with = "nan_as_null::deserialize")]
    pub eval_ceiling_llik_per_count: f64,
}

/// Sums over scored units: the overall totals and each section's.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    pub n: usize,
    pub count: f64,
    pub llik: f64,
    pub null: f64,
    pub pooled: f64,
    pub ceiling: f64,
}

impl Totals {
    /// Add a unit; one without scored counts is left out.
    pub fn add(&mut self, s: &Scored) {
        if s.n_test <= 0.0 {
            return;
        }
        self.n += 1;
        self.count += s.n_test;
        self.llik += s.loglik;
        self.null += s.loglik_null;
        self.pooled += s.loglik_pooled;
        self.ceiling += s.loglik_ceiling;
    }

    /// `x` per scored count; NaN with none.
    pub fn per_count(&self, x: f64) -> f64 {
        if self.count > 0.0 {
            x / self.count
        } else {
            f64::NAN
        }
    }
}

/// Per-section totals of the scored units, sections in order of appearance.
pub fn per_section(sections: &[&str], scored: &[Scored]) -> Vec<SectionScore> {
    let mut order: Vec<&str> = Vec::new();
    let mut sums: HashMap<&str, Totals> = HashMap::default();
    for (&sec, s) in sections.iter().zip(scored) {
        if s.n_test <= 0.0 {
            continue;
        }
        sums.entry(sec)
            .or_insert_with(|| {
                order.push(sec);
                Totals::default()
            })
            .add(s);
    }
    order
        .into_iter()
        .map(|sec| {
            let t = sums[sec];
            SectionScore {
                section: sec.to_string(),
                n_scored: t.n,
                eval_count: t.count,
                eval_llik_per_count: t.per_count(t.llik),
                eval_null_llik_per_count: t.per_count(t.null),
                eval_pooled_k1_llik_per_count: t.per_count(t.pooled),
                eval_ceiling_llik_per_count: t.per_count(t.ceiling),
            }
        })
        .collect()
}

/// A unit's counts on the matched window: (matched column, count).
pub type Counts = Vec<(usize, f64)>;

/// A unit's fitting part and scored part, widened for EM and the score.
pub type UnitParts = (Counts, Counts);

/// A cell's counts as stored for the whole run: (matched column, count) in
/// 8 bytes, half of [`Counts`]. Counts arrive as f32, so this loses nothing
/// below 2^24 per cell and feature; a thinned part above that rounds as f32.
pub type Compact = Vec<(u32, f32)>;

/// A cell's fitting part and scored part.
pub type CellParts = (Compact, Compact);

/// One cell's counts split once into its fitting and scored parts, on the
/// matched window. `nonzeros` is the cell's (data row, count) in row order.
///
/// Under thinning every count gives `Bin(x, frac)` to the fitting part and the
/// rest to the scored part, drawn for EVERY nonzero row from the cell's own
/// stream, matched or not. So the split depends only on the data and the seed:
/// two models scored on one sample, whatever features each has, split it
/// alike. A cell is split once, and a pair's parts are its two cells' parts,
/// so no count is in one unit's scored part and another's fitting part.
/// Counts are rounded first. Under feature ablation the fit and scored
/// features take a cell's counts as they are.
pub fn split_cell(
    nonzeros: impl Iterator<Item = (usize, f32)>,
    row_to_matched: &[Option<usize>],
    split: &Split,
    fit: &[bool],
    scored: &[bool],
    seed: u64,
) -> CellParts {
    let mut fit_part: Compact = Vec::new();
    let mut test_part: Compact = Vec::new();
    match *split {
        Split::Thinning(frac) => {
            let mut rng = SmallRng::seed_from_u64(seed);
            for (row, v) in nonzeros {
                let n = f64::from(v).round().max(0.0) as u64;
                if n == 0 {
                    continue;
                }
                let a = Binomial::new(n, frac)
                    .expect("frac in (0, 1)")
                    .sample(&mut rng);
                if let Some(s) = row_to_matched[row] {
                    if a > 0 {
                        fit_part.push((s as u32, a as f32));
                    }
                    if a < n && scored[s] {
                        test_part.push((s as u32, (n - a) as f32));
                    }
                }
            }
        }
        Split::Ablate => {
            for (row, v) in nonzeros {
                if let Some(s) = row_to_matched[row] {
                    if fit[s] {
                        fit_part.push((s as u32, v));
                    }
                    if scored[s] {
                        test_part.push((s as u32, v));
                    }
                }
            }
        }
    }
    (fit_part, test_part)
}

/// How a unit's counts are split, and what is scored.
pub enum Split {
    Thinning(f64),
    /// `hidden[g]`: the feature is kept out of the fit and is what is scored.
    Ablate,
}

/// The model's profiles under one batch's `δ`, in the forms the fit and the
/// score need.
pub struct BatchProfiles {
    /// `[K, G]` `δ_g p̃_k(g)`, each row renormalised over every matched feature.
    pub q: DMatrix<f64>,
    /// `[K, G]` `q` renormalised over the FIT features (zero elsewhere): what EM
    /// fits on.
    pub p_fit: DMatrix<f64>,
    /// `m_k = Σ_{g fit} q_k(g)`: converts fitted proportions back to weights on `q`.
    pub fit_mass: Vec<f64>,
    /// `s_k = Σ_{g scored} q_k(g)`: the scored window's normaliser.
    pub scored_mass: Vec<f64>,
}

impl BatchProfiles {
    pub fn new(p_t: &DMatrix<f64>, delta: &[f64], fit: &[bool], scored: &[bool]) -> Self {
        let (k, g) = (p_t.nrows(), p_t.ncols());
        let mut q = DMatrix::<f64>::zeros(k, g);
        for c in 0..k {
            let mut total = 0.0;
            for j in 0..g {
                let v = p_t[(c, j)] * delta[j];
                q[(c, j)] = v;
                total += v;
            }
            if total > 0.0 {
                q.row_mut(c).scale_mut(1.0 / total);
            }
        }
        let mass = |on: &[bool]| -> Vec<f64> {
            (0..k)
                .map(|c| (0..g).filter(|&j| on[j]).map(|j| q[(c, j)]).sum())
                .collect()
        };
        let (fit_mass, scored_mass) = (mass(fit), mass(scored));
        let mut p_fit = DMatrix::<f64>::zeros(k, g);
        for c in 0..k {
            if fit_mass[c] > 0.0 {
                for j in (0..g).filter(|&j| fit[j]) {
                    p_fit[(c, j)] = q[(c, j)] / fit_mass[c];
                }
            }
        }
        BatchProfiles {
            q,
            p_fit,
            fit_mass,
            scored_mass,
        }
    }

    /// Weights on `q` from proportions fitted on `p_fit`: `w_k ∝ π_k / m_k`.
    pub fn weights(&self, pi: &[f64]) -> Vec<f64> {
        let mut w: Vec<f64> = pi
            .iter()
            .zip(&self.fit_mass)
            .map(|(&p, &m)| if m > 0.0 { p / m } else { 0.0 })
            .collect();
        let z: f64 = w.iter().sum();
        if z > 0.0 {
            w.iter_mut().for_each(|x| *x /= z);
        }
        w
    }

    /// `Σ x log p(g)` with `p(g) = Σ_k w_k q_k(g) / Σ_k w_k s_k`: the mixture
    /// renormalised over the scored features.
    pub fn loglik(&self, test: &[(usize, f64)], w: &[f64]) -> f64 {
        let z: f64 = w.iter().zip(&self.scored_mass).map(|(a, b)| a * b).sum();
        if z <= 0.0 {
            return f64::NAN;
        }
        test.iter()
            .map(|&(slot, x)| {
                let r: f64 = self.q.column(slot).iter().zip(w).map(|(q, w)| q * w).sum();
                x * (r / z).max(f64::from(PROB_FLOOR)).ln()
            })
            .sum()
    }

    /// The mixture's rate over `slots`, renormalised over them.
    pub fn rate_over(&self, slots: &[usize], w: &[f64]) -> Vec<f32> {
        let r: Vec<f64> = slots
            .iter()
            .map(|&j| self.q.column(j).iter().zip(w).map(|(q, w)| q * w).sum())
            .collect();
        let z: f64 = r.iter().sum::<f64>().max(1e-300);
        r.iter().map(|&v| (v / z) as f32).collect()
    }
}

/// One unit's result.
#[derive(Debug, Clone, Default)]
pub struct Scored {
    pub n_fit: f64,
    pub n_test: f64,
    pub loglik: f64,
    pub loglik_null: f64,
    pub loglik_pooled: f64,
    pub loglik_ceiling: f64,
    pub spearman: f32,
    pub pearson_log1p: f32,
    /// Weights on the profiles, fitted on the fitting part.
    pub pi: Vec<f64>,
}

/// Everything fixed across units.
pub struct Plan<'a> {
    pub split: Split,
    /// Log-probabilities over the scored features, per slot (NaN off them).
    pub log_null: &'a [f64],
    pub log_ceiling: &'a [f64],
    pub shares: &'a [f64],
    /// The scored slots, in order, when per-unit agreement is wanted.
    pub agreement: Option<Agreement>,
    pub iters: usize,
}

/// The slots per-unit agreement is scored on, and each slot's place among
/// them (`u32::MAX` off them): built once, not per unit.
pub struct Agreement {
    pub slots: Vec<usize>,
    pub at: Vec<u32>,
}

impl Agreement {
    pub fn new(slots: Vec<usize>, g: usize) -> Self {
        let mut at = vec![u32::MAX; g];
        for (i, &j) in slots.iter().enumerate() {
            at[j] = i as u32;
        }
        Agreement { slots, at }
    }
}

/// Counts with each slot once, its counts summed: a pair's two cells repeat
/// the features both express, and EM and the score then work on each once.
fn merged(mut v: Counts) -> Counts {
    v.sort_unstable_by_key(|o| o.0);
    v.dedup_by(|next, kept| {
        if next.0 == kept.0 {
            kept.1 += next.1;
            true
        } else {
            false
        }
    });
    v
}

/// A stored part as EM and the score read it.
pub fn widen<'a>(part: impl Iterator<Item = &'a (u32, f32)>) -> Counts {
    part.map(|&(s, x)| (s as usize, f64::from(x))).collect()
}

/// The fitting part of a cell's parts.
pub fn fit_side(p: &CellParts) -> &Compact {
    &p.0
}

/// The scored part of a cell's parts.
pub fn test_side(p: &CellParts) -> &Compact {
    &p.1
}

/// One side of a unit (`fit_side` or `test_side`): its cell's, or the sum of
/// its two cells', each slot once.
pub fn unit_side(
    parts: &[CellParts],
    i: usize,
    j: Option<usize>,
    side: fn(&CellParts) -> &Compact,
) -> Counts {
    merged(widen(
        std::iter::once(i)
            .chain(j)
            .flat_map(|c| side(&parts[c]).iter()),
    ))
}

/// A unit's parts: its fitting part and its scored part.
pub fn unit_parts(parts: &[CellParts], i: usize, j: Option<usize>) -> UnitParts {
    (
        unit_side(parts, i, j, fit_side),
        unit_side(parts, i, j, test_side),
    )
}

/// One unit, under one batch's profiles: fit on its fitting part, score its
/// scored part under the fitted mixture and every reference.
pub fn score_unit(
    fit: &[(usize, f64)],
    test: &[(usize, f64)],
    prof: &BatchProfiles,
    plan: &Plan,
) -> Scored {
    let k = prof.q.nrows();
    let mut pi = vec![0.0; k];
    let mut out = Scored {
        n_fit: fit.iter().map(|o| o.1).sum(),
        n_test: test.iter().map(|o| o.1).sum(),
        spearman: f32::NAN,
        pearson_log1p: f32::NAN,
        ..Default::default()
    };
    if !em_proportions(fit, &prof.p_fit, plan.iters, &mut pi) {
        out.n_test = 0.0;
        out.pi = pi;
        return out;
    }
    // Weights on the full profiles, also for a unit with nothing to score, so
    // every unit's propensity is on one scale.
    let w = prof.weights(&pi);
    if out.n_test <= 0.0 {
        out.n_test = 0.0;
        out.pi = w;
        return out;
    }
    out.loglik = prof.loglik(test, &w);
    out.loglik_pooled = prof.loglik(test, plan.shares);
    let table = |t: &[f64]| test.iter().map(|&(j, x)| x * t[j]).sum::<f64>();
    out.loglik_null = table(plan.log_null);
    out.loglik_ceiling = table(plan.log_ceiling);
    if let Some(ag) = &plan.agreement {
        let mut observed = vec![0f32; ag.slots.len()];
        for &(j, x) in test {
            if let Some(i) = ag.at.get(j).filter(|&&i| i != u32::MAX) {
                observed[*i as usize] += x as f32;
            }
        }
        let a = agreement_from_rate(&observed, &prof.rate_over(&ag.slots, &w));
        out.spearman = a.spearman;
        out.pearson_log1p = a.pearson_log1p;
    }
    out.pi = w;
    out
}

/// A cell's thinning seed: the run's seed and the cell's index, mixed so
/// neighbouring cells do not get neighbouring streams.
pub fn cell_seed(seed: u64, cell: usize) -> u64 {
    legume_numeric::matrix::rand_util::mix_seed(seed, cell as u64)
}

/// `log(c_g / Σ_{scored} c)` per slot, NaN off the scored features.
pub fn log_composition(totals: &[f64], scored: &[bool]) -> Vec<f64> {
    let z: f64 = totals
        .iter()
        .zip(scored)
        .filter(|(_, &s)| s)
        .map(|(t, _)| t)
        .sum::<f64>()
        .max(1e-300);
    totals
        .iter()
        .zip(scored)
        .map(|(&t, &s)| {
            if s {
                (t / z).max(f64::from(PROB_FLOOR)).ln()
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// One δ round's update from the fitting parts' observed and predicted totals:
/// `δ ← δ (obs + 1) / (pred + 1)` on the fit features, clamped to
/// `[0.01, 100]` and scaled to mean 1 over them.
pub fn update_delta(delta: &mut [f64], obs: &[f64], pred: &[f64], fit: &[bool]) {
    for j in (0..delta.len()).filter(|&j| fit[j]) {
        delta[j] = (delta[j] * (obs[j] + 1.0) / (pred[j] + 1.0)).clamp(0.01, 100.0);
    }
    let n = fit.iter().filter(|&&f| f).count().max(1) as f64;
    let mean: f64 = (0..delta.len())
        .filter(|&j| fit[j])
        .map(|j| delta[j])
        .sum::<f64>()
        / n;
    if mean > 0.0 {
        for j in (0..delta.len()).filter(|&j| fit[j]) {
            delta[j] /= mean;
        }
    }
}

#[cfg(test)]
#[path = "tests/holdout.rs"]
mod tests;
