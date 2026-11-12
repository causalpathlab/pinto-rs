//! Drawing counts and running one simulated pair.

use super::calibrate::{marker_bins, n_bins, Producers, SimBase, MARKER_LIFT};
use super::fields::{confounding, SimShared};
use super::row::SimRow;
use super::scenario::Scenario;
use crate::lr_activity::ligand_effect::association::marginal_correlation;
use crate::lr_activity::ligand_effect::averaging::abf_weights;
use crate::lr_activity::ligand_effect::baselines::{baselines, Baselines};
use crate::lr_activity::ligand_effect::dose::{choose_threshold, eta_squared, exposed, Exposure};
use crate::lr_activity::ligand_effect::effects::Sensitivity;
use crate::lr_activity::ligand_effect::fits::{indicator, Rows};
use crate::lr_activity::ligand_effect::options::EffectOpts;
use crate::lr_activity::ligand_effect::report::{report, ReportOpts};
use crate::lr_activity::ligand_effect::response::{expected_bleed, Response};
use crate::lr_activity::ligand_effect::rows::EffectRow;
use crate::lr_activity::ligand_effect::thresholds::{
    over_thresholds, thresholds, ThresholdAverage,
};
use crate::lr_activity::ligand_effect::tissue::Tissue;
use crate::lr_activity::ligand_effect::verdict::Verdict;
use legume_numeric::matrix::rand_util::mix_seed;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Poisson, StandardNormal};
use special::Error as _;

/// What multiplies a gene's bin mean at each row, for genes of positive then
/// of negative weight: the field `f` at strength `beta` (normalized to keep the
/// mean) and the marker types' lift.
pub fn sign_lifts(f: &[f64], beta: f64, marker: &[bool], bin: &[usize]) -> [Vec<f64>; 2] {
    [1.0, -1.0].map(|s| {
        let norm = f.iter().map(|z| (s * beta * z).exp()).sum::<f64>() / f.len().max(1) as f64;
        (0..f.len())
            .map(|i| {
                let m = if marker[bin[i]] { s * MARKER_LIFT } else { 0.0 };
                (s * beta * f[i] + m).exp() / norm
            })
            .collect()
    })
}

/// The coefficient on a standardized field that gives it `pve` of a
/// logistic model's latent variance, whose noise is the logistic's `π²/3`.
pub fn logit_coef(pve: f64) -> f64 {
    let pve = pve.clamp(0.0, 1.0 - 1e-9);
    (pve * std::f64::consts::PI.powi(2) / 3.0 / (1.0 - pve)).sqrt()
}

/// The coefficient on a standardized field that gives it `|pve|` of a log
/// rate's variance, carrying the sign of `pve`.
pub fn log_rate_coef(pve: f64) -> f64 {
    pve.signum() * pve.abs().sqrt()
}

/// Per-cell, per-gene log-rate noise of variance `var`, mean-preserving:
/// the multiplier `e^{√var ε − var/2}`.
fn noise_lift(rng: &mut SmallRng, var: f64) -> f64 {
    let e: f64 = StandardNormal.sample(rng);
    (var.sqrt() * e - 0.5 * var).exp()
}

/// The Poisson quantile at `u`: exact by summing the pmf for small rates,
/// normal with continuity correction above 30.
fn poisson_quantile(u: f64, lam: f64) -> f64 {
    if lam <= 0.0 {
        return 0.0;
    }
    if lam > 30.0 {
        let z = std::f64::consts::SQRT_2 * (2.0 * u - 1.0).clamp(-0.999_999, 0.999_999).inv_error();
        return (lam + lam.sqrt() * z + 0.5).floor().max(0.0);
    }
    let mut k = 0.0;
    let mut p = (-lam).exp();
    let mut cdf = p;
    while cdf < u && k < 1000.0 {
        k += 1.0;
        p *= lam / k;
        cdf += p;
    }
    k
}

fn poisson(rng: &mut SmallRng, rate: f64) -> f64 {
    if rate > 0.0 {
        Poisson::new(rate).map(|p| p.sample(rng)).unwrap_or(0.0)
    } else {
        0.0
    }
}

/// Mean of `a − b` over `subset`; NaN for an empty subset.
fn sample_effect(a: &[f64], b: &[f64], subset: &[bool]) -> f64 {
    let (mut s, mut k) = (0.0, 0usize);
    for i in (0..a.len()).filter(|&i| subset[i]) {
        s += a[i] - b[i];
        k += 1;
    }
    if k == 0 {
        f64::NAN
    } else {
        s / k as f64
    }
}

/// What stays fixed through one run of a scenario.
struct Run<'a> {
    tissue: &'a Tissue,
    base: &'a SimBase<'a>,
    scen: &'a Scenario,
    opts: &'a EffectOpts,
    /// The run's seed; each random stream mixes in its own salt.
    seed: u64,
    /// The hidden field.
    f: Vec<f64>,
}

/// Who truly makes ligand, and what is detected of it.
struct TrueLigand {
    /// The identity bins of the marker type; none without a marker scenario.
    marker: Vec<bool>,
    /// The true rate of each row, 0 for a non-producer.
    rate: Vec<f64>,
    /// The detected counts.
    counts: Vec<f64>,
}

fn draw_ligand(run: &Run<'_>, shared: &SimShared<'_>, rng: &mut SmallRng) -> TrueLigand {
    let (tissue, scen, f) = (run.tissue, run.scen, &run.f);
    let n = tissue.n();
    let (sf, bin) = (&tissue.size_factor, &tissue.identity_bin);
    let mut cal = shared.producers.clone();
    let marker = if scen.marker {
        marker_bins(&cal)
    } else {
        vec![false; n_bins(tissue)]
    };
    if scen.marker {
        for (c, &m) in cal.iter_mut().zip(&marker) {
            if !m {
                c.p = 0.0;
            }
        }
    }
    let alpha = logit_coef(scen.ligand_field_pve);
    let rate: Vec<f64> = (0..n)
        .map(|i| {
            let Producers { p, r, .. } = cal[bin[i]];
            let pi = if p <= 0.0 {
                0.0
            } else {
                let q = p.clamp(1e-9, 1.0 - 1e-9);
                let eta = (q / (1.0 - q)).ln() + alpha * f[i];
                1.0 / (1.0 + (-eta).exp())
            };
            if rng.random::<f64>() < pi {
                r
            } else {
                0.0
            }
        })
        .collect();
    let counts: Vec<f64> = (0..n).map(|i| poisson(rng, sf[i] * rate[i])).collect();
    TrueLigand {
        marker,
        rate,
        counts,
    }
}

/// What receivers respond to: the latent dose.
struct LatentExposure {
    /// `T*`, the true contact exposure.
    t: Vec<bool>,
    /// How much of the planted effect each receiver's true dose reaches.
    dose_share: Vec<f64>,
}

fn latent_exposure(run: &Run<'_>, rate: &[f64]) -> LatentExposure {
    let (tissue, scen) = (run.tissue, run.scen);
    let dose = tissue.contacts.sum(rate);
    let tau = choose_threshold(&dose, run.opts);
    let t = exposed(&dose, tau);
    let dose_share: Vec<f64> = if scen.dose_response {
        let mut pos: Vec<f64> = dose.iter().copied().filter(|&v| v > 0.0).collect();
        pos.sort_by(f64::total_cmp);
        let (q, order) = (scen.dose_midpoint, scen.dose_order);
        let k = pos
            .get(((pos.len() as f64 - 1.0) * q) as usize)
            .copied()
            .unwrap_or(1.0);
        dose.iter()
            .map(|&v| {
                let (x, m) = (v.powi(order), k.powi(order));
                x / (x + m).max(1e-300)
            })
            .collect()
    } else {
        vec![1.0; tissue.n()]
    };
    LatentExposure { t, dose_share }
}

/// What the estimator sees: the detected ligand's exposure, and the baselines.
struct SeenByEstimator {
    exposure: Exposure,
    baselines: Baselines,
    baseline_seconds: f64,
}

/// The detected counts; the estimator's own exposure rule on them; and the
/// co-expression baselines on the same counts.
fn seen_by_estimator(run: &Run<'_>, ligand_counts: &[f64]) -> SeenByEstimator {
    let (tissue, base, opts) = (run.tissue, run.base, run.opts);
    let exposure = Exposure::build(tissue, ligand_counts, opts);

    // The receptor is the real one: the hidden field never touches it, so
    // competence is a fixed set of identity bins.
    let clock = std::time::Instant::now();
    let coexpression_baselines = baselines(
        tissue,
        ligand_counts,
        base.receptor_counts,
        mix_seed(run.seed, 8),
    );
    let baseline_seconds = clock.elapsed().as_secs_f64();
    SeenByEstimator {
        exposure,
        baselines: coexpression_baselines,
        baseline_seconds,
    }
}

/// Who responds, whom the estimator calls competent, and how the effect
/// falls on them.
struct Receivers {
    responds: Vec<bool>,
    comp: Vec<bool>,
    /// The share of the planted effect per identity bin.
    u_bin: Vec<f64>,
}

fn who_responds(run: &Run<'_>, ligand: &TrueLigand, competent: &[bool]) -> Receivers {
    let (tissue, scen) = (run.tissue, run.scen);
    let n = tissue.n();
    let (bin, marker) = (&tissue.identity_bin, &ligand.marker);
    let responds: Vec<bool> = (0..n)
        .map(|i| {
            if scen.marker {
                marker[bin[i]]
            } else {
                competent[i]
            }
        })
        .collect();
    let comp: Vec<bool> = (0..n)
        .map(|i| competent[i] || (scen.marker && marker[bin[i]]))
        .collect();
    let nb = n_bins(tissue);
    let u_bin: Vec<f64> = if scen.heterogeneous {
        let mut r = SmallRng::seed_from_u64(mix_seed(run.seed, 5));
        (0..nb).map(|_| 0.5 + r.random::<f64>()).collect()
    } else {
        vec![1.0; nb]
    };
    Receivers {
        responds,
        comp,
        u_bin,
    }
}

/// One programme gene's counts: observed, and potential in `T*`.
struct GeneCounts {
    obs: Vec<f64>,
    c1: Vec<f64>,
    c0: Vec<f64>,
}

impl GeneCounts {
    fn new(n: usize) -> Self {
        Self {
            obs: vec![0.0; n],
            c1: vec![0.0; n],
            c0: vec![0.0; n],
        }
    }
}

/// Draw gene `g`'s counts from one shared uniform per receiver, so that
/// potential counts differ by the effect alone. `planted` is each receiver's
/// log-rate lift under `T*` before the gene's sign (`θ · u_bin · dose share`),
/// zero for a receiver that does not respond.
fn draw_gene(
    run: &Run<'_>,
    g: usize,
    lifts: &[Vec<f64>; 2],
    latent: &LatentExposure,
    planted: &[f64],
    c: &mut GeneCounts,
    rng: &mut SmallRng,
) {
    let (tissue, base, scen) = (run.tissue, run.base, run.scen);
    let (sf, bin) = (&tissue.size_factor, &tissue.identity_bin);
    let ng = base.target_weights.len();
    let t_star = &latent.t;
    let w = base.target_weights[g];
    let s = if w < 0.0 { -1.0 } else { 1.0 };
    let lift = &lifts[(w < 0.0) as usize];
    let noise_var = log_rate_noise(scen.effect_pve, scen.response_field_pve);
    for i in 0..tissue.n() {
        let b = bin[i];
        let noise = noise_lift(rng, noise_var);
        let r0 = sf[i] * base.target_bin_means[b * ng + g] * lift[i] * noise;
        let u: f64 = rng.random();
        let th = s * planted[i];
        let q = |l: f64| poisson_quantile(u, r0 * l.exp());
        let hi = q(th);
        let lo = if th == 0.0 { hi } else { q(0.0) };
        c.c1[i] = hi;
        c.c0[i] = lo;
        c.obs[i] = if t_star[i] { hi } else { lo };
    }
}

/// The programme's responses: as observed (`y_seen`), and potential in `T*`
/// (`y1`, `y0`), with the estimator's bleed covariate.
struct ProgrammeResponses {
    y_seen: Vec<f64>,
    y1: Vec<f64>,
    y0: Vec<f64>,
    bleed: Vec<f64>,
}

impl ProgrammeResponses {
    fn new(n: usize) -> Self {
        Self {
            y_seen: vec![0.0; n],
            y1: vec![0.0; n],
            y0: vec![0.0; n],
            bleed: vec![0.0; n],
        }
    }

    /// Accumulate a gene of weight `w`.
    fn add_gene(&mut self, run: &Run<'_>, w: f64, bin_cells: &[usize], c: &GeneCounts) {
        let tissue = run.tissue;
        let (sf, bin) = (&tissue.size_factor, &tissue.identity_bin);
        // The bleed covariate from the observed counts, as
        // [`super::super::response::programme_columns`] reads them: bin means,
        // then expected neighbour counts.
        let mut m = vec![0.0; bin_cells.len()];
        for (i, s) in sf.iter().enumerate() {
            let obs = c.obs[i] / s;
            self.y_seen[i] += w * obs.ln_1p();
            self.y1[i] += w * (c.c1[i] / s).ln_1p();
            self.y0[i] += w * (c.c0[i] / s).ln_1p();
            m[bin[i]] += obs;
        }
        for (mb, &k) in m.iter_mut().zip(bin_cells) {
            *mb /= k.max(1) as f64;
        }
        for (o, x) in self
            .bleed
            .iter_mut()
            .zip(expected_bleed(&m, bin, &tissue.contacts, sf))
        {
            *o += w * x;
        }
    }

    fn divide_by(&mut self, total_w: f64) {
        for v in [
            &mut self.y_seen,
            &mut self.y1,
            &mut self.y0,
            &mut self.bleed,
        ] {
            for x in v.iter_mut() {
                *x /= total_w;
            }
        }
    }
}

/// The programme, gene by gene.
fn draw_programme(
    run: &Run<'_>,
    lifts: &[Vec<f64>; 2],
    latent: &LatentExposure,
    receivers: &Receivers,
    rng: &mut SmallRng,
) -> ProgrammeResponses {
    let (tissue, weights) = (run.tissue, run.base.target_weights);
    let n = tissue.n();
    let mut bin_cells = vec![0usize; n_bins(tissue)];
    for &b in tissue.identity_bin.iter() {
        bin_cells[b] += 1;
    }
    let bin = &tissue.identity_bin;
    let x: Vec<f64> = (0..n)
        .map(|i| {
            if latent.t[i] {
                receivers.u_bin[bin[i]] * latent.dose_share[i]
            } else {
                0.0
            }
        })
        .collect();
    let theta = effect_coef(run.scen.effect_pve, &x, &receivers.responds, bin);
    let share = |i: usize, size: f64| {
        if receivers.responds[i] {
            size * receivers.u_bin[bin[i]]
        } else {
            0.0
        }
    };
    let planted: Vec<f64> = (0..n)
        .map(|i| share(i, theta) * latent.dose_share[i])
        .collect();
    let mut counts = GeneCounts::new(n);
    let mut out = ProgrammeResponses::new(n);
    for (g, &w) in weights.iter().enumerate() {
        draw_gene(run, g, lifts, latent, &planted, &mut counts, rng);
        out.add_gene(run, w, &bin_cells, &counts);
    }
    let total_w: f64 = weights.iter().map(|w| w.abs()).sum::<f64>().max(1e-12);
    out.divide_by(total_w);
    out
}

/// Share of competent receivers whose seen exposure is not their true exposure.
fn misclassified_share(comp: &[bool], seen: &[bool], truth: &[bool]) -> f64 {
    let k = comp.iter().filter(|&&c| c).count();
    let miss = (0..comp.len())
        .filter(|&i| comp[i] && seen[i] != truth[i])
        .count();
    miss as f64 / k.max(1) as f64
}

/// Log-rate noise of a programme gene: what the effect's and the field's
/// shares leave of its variance.
fn log_rate_noise(effect_pve: f64, field_pve: f64) -> f64 {
    (1.0 - effect_pve - field_pve.abs()).max(0.0)
}

/// The effect's log-rate coefficient: `√effect_pve` over the standard
/// deviation of the exposure term `x` (true exposure × the bin's share × the
/// dose share) across responding receivers, centred within identity bin, so
/// it explains `effect_pve` of a gene's log-rate variance beyond the cell type.
pub fn effect_coef(effect_pve: f64, x: &[f64], responds: &[bool], bin: &[usize]) -> f64 {
    if effect_pve <= 0.0 {
        return 0.0;
    }
    let nb = bin.iter().copied().max().map_or(0, |b| b + 1);
    let (mut sum, mut cnt) = (vec![0.0; nb], vec![0usize; nb]);
    for i in (0..x.len()).filter(|&i| responds[i]) {
        sum[bin[i]] += x[i];
        cnt[bin[i]] += 1;
    }
    let (mut ss, mut n) = (0.0, 0usize);
    for i in (0..x.len()).filter(|&i| responds[i]) {
        ss += (x[i] - sum[bin[i]] / cnt[bin[i]] as f64).powi(2);
        n += 1;
    }
    let var = ss / n.max(1) as f64;
    if var > 0.0 {
        (effect_pve / var).sqrt()
    } else {
        0.0
    }
}

/// `effect` contrasted between the seen arms `t` over the competent receivers,
/// each arm's mean under `weight`. An empty arm has no contrast.
pub fn seen_contrast(effect: &[f64], t: &[bool], comp: &[bool], weight: &[f64]) -> f64 {
    let (mut num, mut den) = ([0.0; 2], [0.0; 2]);
    for i in (0..effect.len()).filter(|&i| comp[i]) {
        let arm = t[i] as usize;
        num[arm] += weight[i] * effect[i];
        den[arm] += weight[i];
    }
    if den[0] > 0.0 && den[1] > 0.0 {
        num[1] / den[1] - num[0] / den[0]
    } else {
        f64::NAN
    }
}

/// [`SimRow::truth`] and [`SimRow::truth_exposed`].
fn effect_truths(
    y1: &[f64],
    y0: &[f64],
    t_star: &[bool],
    comp: &[bool],
    dose: &[f64],
    avg: &ThresholdAverage,
    opts: &EffectOpts,
) -> (f64, f64) {
    let averaged = if opts.threshold.is_some() {
        1
    } else {
        avg.points.len()
    };
    let realized: Vec<f64> = (0..y1.len())
        .map(|i| if t_star[i] { y1[i] - y0[i] } else { 0.0 })
        .collect();
    let parts: Vec<_> = avg.points[..averaged].iter().map(|p| p.1).collect();
    let (weights, _) = abf_weights(&parts, opts.abf_prior_scale);
    let truth = if weights.is_empty() {
        f64::NAN
    } else {
        weights
            .iter()
            .map(|&(k, w)| {
                let t = exposed(dose, avg.points[k].0);
                w * seen_contrast(&realized, &t, comp, &avg.arm_weights[k])
            })
            .sum()
    };
    let among: Vec<bool> = (0..y1.len()).map(|i| comp[i] && t_star[i]).collect();
    (truth, sample_effect(y1, y0, &among))
}

/// The verdict `lra` would give this run alone.
fn run_verdict(
    tissue: &Tissue,
    rows: Rows<'_>,
    seen: &SeenByEstimator,
    avg: &ThresholdAverage,
) -> Verdict {
    let e = avg.scopes[0];
    let mut judged = EffectRow::pending(
        e,
        (seen.baselines.coexpr_log_or, seen.baselines.coexpr_se),
        eta_squared(
            &tissue.neighbour_bin,
            &indicator(&seen.exposure.leaned_on(e.abf)),
            &rows.scope_rows[0],
        ),
    );
    report(std::slice::from_mut(&mut judged), &ReportOpts::default());
    judged.verdict
}

/// One run of `scen` on `base`, with the pair's shared ingredients
/// computed once ([`super::SimShared`]); see the module doc.
pub fn simulate(
    tissue: &Tissue,
    base: &SimBase<'_>,
    shared: &SimShared<'_>,
    scen: &Scenario,
    opts: &EffectOpts,
    seed: u64,
) -> anyhow::Result<SimRow> {
    let start = std::time::Instant::now();
    let opts = &EffectOpts {
        keep_arm_weights: true,
        ..*opts
    };
    let n = tissue.n();
    let bin = &tissue.identity_bin;
    let nb = n_bins(tissue);
    let ng = base.target_weights.len();
    scen.validate()?;
    anyhow::ensure!(ng > 0, "the programme has no genes");
    anyhow::ensure!(
        base.target_bin_means.len() == nb * ng,
        "programme bin means are {} long, expected {nb} bins × {ng} genes",
        base.target_bin_means.len()
    );
    let seed = mix_seed(seed, scen.rep as u64);
    let f = confounding(tissue, base, shared, scen, mix_seed(seed, 1))?;
    let run = Run {
        tissue,
        base,
        scen,
        opts,
        seed,
        f,
    };

    // The ligand and the programme draw from this one stream, ligand first.
    let mut rng = SmallRng::seed_from_u64(mix_seed(seed, 2));
    let ligand = draw_ligand(&run, shared, &mut rng);
    let latent = latent_exposure(&run, &ligand.rate);
    let seen = seen_by_estimator(&run, &ligand.counts);
    let exposure = &seen.exposure;
    let receivers = who_responds(&run, &ligand, run.base.competent);
    let comp = &receivers.comp;

    let beta = log_rate_coef(scen.response_field_pve);
    let lifts = sign_lifts(&run.f, beta, &ligand.marker, bin);
    let ProgrammeResponses {
        y_seen,
        y1,
        y0,
        bleed,
    } = draw_programme(&run, &lifts, &latent, &receivers, &mut rng);
    let misclassified = misclassified_share(comp, &exposure.t, &latent.t);

    let response = Response::new(tissue, y_seen, &bleed);

    // The estimands over the exposure thresholds, as `lra` reads them; a fixed
    // threshold has no others here, no curve being read.
    let levels = std::slice::from_ref(comp);
    let taus = if opts.threshold.is_none() {
        thresholds(exposure, levels, opts)
    } else {
        Vec::new()
    };
    let rows = Rows::cells(tissue).pooled();
    let avg = over_thresholds(
        rows,
        exposure,
        &taus,
        std::slice::from_ref(&response),
        levels,
        Sensitivity::STEPS,
        opts,
    )?;
    let e = avg.scopes[0];
    let (truth, truth_exposed) =
        effect_truths(&y1, &y0, &latent.t, comp, &exposure.dose, &avg, opts);
    let comp_rows: Vec<u32> = (0..n as u32).filter(|&i| comp[i as usize]).collect();
    let log_dose: Vec<f64> = exposure.dose.iter().map(|d| d.ln_1p()).collect();
    let marginal_cor = marginal_correlation(&response.y, &log_dose, &comp_rows, rows.cf);
    Ok(SimRow {
        ligand: base.ligand.into(),
        receptor: base.receptor.into(),
        scenario: *scen,
        truth,
        truth_exposed,
        estimate: e.competent,
        marginal_cor,
        unadjusted_diff: e.unadjusted_diff,
        ipw: e.ipw,
        outcome: e.outcome.theta,
        incompetent: e.incompetent,
        steps: e.steps,
        contrast: e.effect,
        verdict: run_verdict(tissue, rows, &seen, &avg),
        misclassified,
        baselines: seen.baselines,
        seconds: start.elapsed().as_secs_f64(),
        baseline_seconds: seen.baseline_seconds,
    })
}
