//! Drawing counts and running one simulated pair.

use super::calibrate::{marker_bins, n_bins, Producers, SimBase, MARKER_LIFT};
use super::fields::{confounding, shift_field, SimShared};
use super::row::SimRow;
use super::scenario::Scenario;
use crate::lr_activity::ligand_effect::baselines::baselines;
use crate::lr_activity::ligand_effect::dose::{choose_threshold, exposed, Exposure};
use crate::lr_activity::ligand_effect::effects::Sensitivity;
use crate::lr_activity::ligand_effect::exposure::competence;
use crate::lr_activity::ligand_effect::fits::{indicator, Rows};
use crate::lr_activity::ligand_effect::options::EffectOpts;
use crate::lr_activity::ligand_effect::overlap::marginal_correlation;
use crate::lr_activity::ligand_effect::report::{eta_squared, report};
use crate::lr_activity::ligand_effect::response::{expected_bleed, Response};
use crate::lr_activity::ligand_effect::rows::EffectRow;
use crate::lr_activity::ligand_effect::thresholds::{over_thresholds, thresholds};
use crate::lr_activity::ligand_effect::tissue::Tissue;
use crate::lr_activity::ligand_effect::INTERNALS;
use legume_numeric::matrix::rand_util::mix_seed;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Poisson, StandardNormal};
use rayon::prelude::*;
use special::Error as _;

/// A background programme matched gene by gene to the real one: the same
/// weights and bin means, no planted effect, the field's push on the response
/// scaled by `share`, and the `generic` lift of each row. Counts come from
/// their own stream, so the background changes nothing else in the run.
#[allow(clippy::too_many_arguments)]
pub fn simulate_background(
    tissue: &Tissue,
    base: &SimBase<'_>,
    lifts: &[Vec<f64>; 2],
    kappa: f64,
    seed: u64,
) -> Vec<Vec<(u32, f32)>> {
    let n = tissue.n();
    let (sf, bin) = (&tissue.size_factor, &tissue.identity_bin);
    let ng = base.target_weights.len();
    let mut rng = SmallRng::seed_from_u64(seed);
    base.target_weights
        .iter()
        .enumerate()
        .map(|(g, &w)| {
            let lift = &lifts[(w < 0.0) as usize];
            let counts: Vec<f64> = (0..n)
                .map(|i| {
                    poisson(
                        &mut rng,
                        sf[i] * base.target_bin_means[bin[i] * ng + g] * lift[i],
                    )
                })
                .collect();
            if kappa > 0.0 {
                let bled = tissue.contacts.sum(&counts);
                nonzeros(counts.iter().zip(&bled).map(|(c, b)| c + kappa * b))
            } else {
                nonzeros(counts.into_iter())
            }
        })
        .collect()
}

/// What multiplies a gene's bin mean at each row, for genes of positive then
/// of negative weight: the field `f` at strength `beta` (normalized to keep the
/// mean), the marker types' lift and the shift `generic`. Real and background genes
/// are generated through this alike.
pub fn sign_lifts(
    f: &[f64],
    beta: f64,
    generic: &[f64],
    marker: &[bool],
    bin: &[usize],
) -> [Vec<f64>; 2] {
    [1.0, -1.0].map(|s| {
        let norm = f.iter().map(|z| (s * beta * z).exp()).sum::<f64>() / f.len().max(1) as f64;
        (0..f.len())
            .map(|i| {
                let m = if marker[bin[i]] { s * MARKER_LIFT } else { 0.0 };
                (s * beta * f[i] + generic[i] + m).exp() / norm
            })
            .collect()
    })
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

/// Non-zero entries of a count vector.
fn nonzeros(c: impl Iterator<Item = f64>) -> Vec<(u32, f32)> {
    c.enumerate()
        .filter(|(_, v)| *v > 0.0)
        .map(|(i, v)| (i as u32, v as f32))
        .collect()
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
    let n = tissue.n();
    let sf = &tissue.size_factor;
    let bin = &tissue.identity_bin;
    let nb = n_bins(tissue);
    let ng = base.target_weights.len();
    anyhow::ensure!(ng > 0, "the programme has no genes");
    anyhow::ensure!(
        base.target_bin_means.len() == nb * ng,
        "programme bin means are {} long, expected {nb} bins × {ng} genes",
        base.target_bin_means.len()
    );
    let seed = mix_seed(seed, scen.rep as u64);
    let f = confounding(tissue, base, shared, scen, mix_seed(seed, 1))?;

    // Who truly makes ligand, and what is detected of it.
    let mut cal = shared.producers.clone();
    let marker = if scen.marker {
        marker_bins(&cal)
    } else {
        vec![false; nb]
    };
    if scen.marker {
        for (c, &m) in cal.iter_mut().zip(&marker) {
            if !m {
                c.p = 0.0;
            }
        }
    }
    // The generic shift's hidden trait, and each row's contacts' mean of it.
    let trait_z: Vec<f64> = if scen.generic_shift != 0.0 {
        let mut r = SmallRng::seed_from_u64(mix_seed(seed, 10));
        (0..n).map(|_| StandardNormal.sample(&mut r)).collect()
    } else {
        vec![0.0; n]
    };
    let trait_near = tissue.contacts.mean(&trait_z);
    let mut rng = SmallRng::seed_from_u64(mix_seed(seed, 2));
    let lam_true: Vec<f64> = (0..n)
        .map(|i| {
            let Producers { p, r, .. } = cal[bin[i]];
            let pi = if p <= 0.0 {
                0.0
            } else {
                let q = p.clamp(1e-9, 1.0 - 1e-9);
                let eta = (q / (1.0 - q)).ln()
                    + scen.confound_ligand * f[i]
                    + scen.generic_shift * trait_z[i];
                1.0 / (1.0 + (-eta).exp())
            };
            if rng.random::<f64>() < pi {
                r
            } else {
                0.0
            }
        })
        .collect();
    let ligand: Vec<f64> = (0..n)
        .map(|i| poisson(&mut rng, sf[i] * lam_true[i]))
        .collect();

    // What receivers respond to: the latent dose.
    let a_star = tissue.contacts.sum(&lam_true);
    let tau_star = choose_threshold(&a_star, opts);
    let t_star = exposed(&a_star, tau_star);
    let r_star = exposed(&tissue.ring_sum(&lam_true), tau_star);
    // How much of the planted effect each receiver's true dose reaches.
    let dose_share: Vec<f64> = if scen.dose_response {
        let mut pos: Vec<f64> = a_star.iter().copied().filter(|&v| v > 0.0).collect();
        pos.sort_by(f64::total_cmp);
        let (q, order) = (scen.dose_midpoint, scen.dose_order);
        let k = pos
            .get(((pos.len() as f64 - 1.0) * q) as usize)
            .copied()
            .unwrap_or(1.0);
        a_star
            .iter()
            .map(|&v| {
                let (x, m) = (v.powi(order), k.powi(order));
                x / (x + m).max(1e-300)
            })
            .collect()
    } else {
        vec![1.0; n]
    };

    // What the estimator sees: detected counts, possibly displaced,
    // possibly with neighbours' ligand bled into the own-ligand covariate.
    let lnorm: Vec<f64> = (0..n).map(|i| ligand[i] / sf[i]).collect();
    let shifted = scen.shift > 0.0;
    let seen = if shifted {
        shift_field(tissue, &lnorm, scen.shift, mix_seed(seed, 3))
    } else {
        lnorm
    };
    // The estimator's own exposure rule on the counts it sees.
    let seen_counts: Vec<f64> = (0..n).map(|i| seen[i] * sf[i]).collect();
    let mut exposure = Exposure::build(tissue, &seen_counts, opts);

    // The receptor the estimator and the baselines see, and competence read
    // from it by the estimator's own rule.
    let sim_receptor;
    let (receptor, competent): (&[f64], Vec<bool>) = if scen.confound_receptor {
        sim_receptor = simulate_receptor(
            tissue,
            base.receptor_counts,
            &f,
            scen.confound_response,
            mix_seed(seed, 6),
        );
        let detected: Vec<bool> = sim_receptor.iter().map(|&r| r > 0.0).collect();
        (&sim_receptor, competence(bin, &detected, opts.competence()))
    } else {
        (base.receptor_counts, base.competent.to_vec())
    };
    let clock = std::time::Instant::now();
    let coexpression_baselines = baselines(tissue, &seen_counts, receptor, mix_seed(seed, 8));
    let baseline_seconds = clock.elapsed().as_secs_f64();
    let kappa = scen.spillover;
    let own_seen = if kappa > 0.0 {
        let bled = tissue.contacts.sum(&seen_counts);
        (0..n)
            .map(|i| (seen_counts[i] + kappa * bled[i]) / sf[i])
            .collect()
    } else {
        seen
    };

    // Who responds, and whom the estimator calls competent.
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
    let u_bin: Vec<f64> = if scen.heterogeneous {
        let mut r = SmallRng::seed_from_u64(mix_seed(seed, 5));
        (0..nb).map(|_| 0.5 + r.random::<f64>()).collect()
    } else {
        vec![1.0; nb]
    };

    // The programme, gene by gene: observed counts (as measured, with
    // spillover) and potential counts, in T* at the realized ring
    // exposure and in the ring exposure at the realized T*, accumulated
    // into responses. Potential counts carry the realized bleed, so the
    // truths are on the measured scale.
    let total_w: f64 = base
        .target_weights
        .iter()
        .map(|w| w.abs())
        .sum::<f64>()
        .max(1e-12);
    let beta = scen.confound_response;
    let generic: Vec<f64> = (0..n)
        .map(|i| {
            if comp[i] {
                scen.generic_shift * trait_near[i]
            } else {
                0.0
            }
        })
        .collect();
    let lifts = sign_lifts(&f, beta, &generic, &marker, bin);
    let ring_on = scen.ring_effect != 0.0;
    let (mut y_seen, mut y1, mut y0) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut bleed = vec![0.0; n];
    let (mut yd1, mut yd0) = if ring_on {
        (vec![0.0; n], vec![0.0; n])
    } else {
        (Vec::new(), Vec::new())
    };
    let (mut c_obs, mut c1, mut c0) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut d1, mut d0) = if ring_on {
        (vec![0.0; n], vec![0.0; n])
    } else {
        (Vec::new(), Vec::new())
    };
    let mut seen_g = vec![0.0; n];
    let mut bin_cells = vec![0usize; nb];
    for &b in bin.iter() {
        bin_cells[b] += 1;
    }
    for (g, &w) in base.target_weights.iter().enumerate() {
        let s = if w < 0.0 { -1.0 } else { 1.0 };
        let lift = &lifts[(w < 0.0) as usize];
        for i in 0..n {
            let b = bin[i];
            let r0 = sf[i] * base.target_bin_means[b * ng + g] * lift[i];
            let u: f64 = rng.random();
            let (th, thr) = if responds[i] {
                (
                    s * scen.effect * u_bin[b] * dose_share[i],
                    s * scen.ring_effect * u_bin[b],
                )
            } else {
                (0.0, 0.0)
            };
            let q = |l: f64| poisson_quantile(u, r0 * l.exp());
            let ring_lift = if r_star[i] { thr } else { 0.0 };
            let hi = q(th + ring_lift);
            let lo = if th == 0.0 { hi } else { q(ring_lift) };
            c1[i] = hi;
            c0[i] = lo;
            c_obs[i] = if t_star[i] { hi } else { lo };
            if ring_on {
                let contact = if t_star[i] { th } else { 0.0 };
                // The count at the ring exposure r* does not have.
                let other = if thr == 0.0 {
                    c_obs[i]
                } else {
                    q(contact + if r_star[i] { 0.0 } else { thr })
                };
                (d1[i], d0[i]) = if r_star[i] {
                    (c_obs[i], other)
                } else {
                    (other, c_obs[i])
                };
            }
        }
        // Bleed from the neighbours' observed counts, shared by every
        // potential count of the receiver.
        let bled: Vec<f64> = if kappa > 0.0 {
            tissue
                .contacts
                .sum(&c_obs)
                .into_iter()
                .map(|b| kappa * b)
                .collect()
        } else {
            vec![0.0; n]
        };
        for i in 0..n {
            seen_g[i] = c_obs[i] + bled[i];
            c1[i] += bled[i];
            c0[i] += bled[i];
            y_seen[i] += w * (seen_g[i] / sf[i]).ln_1p();
            y1[i] += w * (c1[i] / sf[i]).ln_1p();
            y0[i] += w * (c0[i] / sf[i]).ln_1p();
            if ring_on {
                yd1[i] += w * ((d1[i] + bled[i]) / sf[i]).ln_1p();
                yd0[i] += w * ((d0[i] + bled[i]) / sf[i]).ln_1p();
            }
        }
        // The bleed covariate from the measured counts, as
        // [`super::super::response::programme_columns`] reads them: bin means,
        // then expected neighbour counts.
        let mut m = vec![0.0; nb];
        for i in 0..n {
            m[bin[i]] += seen_g[i] / sf[i];
        }
        for (mb, &c) in m.iter_mut().zip(&bin_cells) {
            *mb /= c.max(1) as f64;
        }
        for (o, x) in bleed
            .iter_mut()
            .zip(expected_bleed(&m, bin, &tissue.contacts, sf))
        {
            *o += w * x;
        }
    }
    drop((c_obs, c1, c0, d1, d0, seen_g));
    for v in [
        &mut y_seen,
        &mut y1,
        &mut y0,
        &mut yd1,
        &mut yd0,
        &mut bleed,
    ] {
        for x in v.iter_mut() {
            *x /= total_w;
        }
    }
    let ring_truth = if shifted || !ring_on {
        0.0
    } else {
        sample_effect(&yd1, &yd0, &comp)
    };
    drop((yd1, yd0));
    let misclassified = {
        let k = comp.iter().filter(|&&c| c).count();
        let miss = (0..n)
            .filter(|&i| comp[i] && exposure.t[i] != t_star[i])
            .count();
        miss as f64 / k.max(1) as f64
    };

    exposure.own = own_seen.iter().map(|v| v.ln_1p()).collect();
    let response = Response {
        spill: tissue.contacts.mean(&y_seen),
        y: y_seen,
        bleed,
    };
    let background_lifts = sign_lifts(
        &f,
        beta * scen.background_confounding,
        &generic,
        &marker,
        bin,
    );
    let make_backgrounds = || -> Vec<Response> {
        (0..opts.background_draws)
            .into_par_iter()
            .map(|k| {
                let seed = mix_seed(mix_seed(seed, 9), k as u64);
                let nz = simulate_background(tissue, base, &background_lifts, kappa, seed);
                let genes: Vec<(f64, &[(u32, f32)])> = base
                    .target_weights
                    .iter()
                    .zip(&nz)
                    .map(|(&w, v)| (w, v.as_slice()))
                    .collect();
                Response::build(tissue, &genes)
            })
            .collect()
    };

    // The estimands over the exposure thresholds, as `lra` reads them; a fixed
    // threshold has no others here, no curve being read.
    let levels = std::slice::from_ref(&comp);
    let taus = if opts.threshold.is_none() {
        thresholds(&exposure, levels, opts)
    } else {
        Vec::new()
    };
    let rows = Rows::cells(tissue).pooled();
    let avg = over_thresholds(
        rows,
        &exposure,
        &taus,
        std::slice::from_ref(&response),
        make_backgrounds,
        levels,
        Sensitivity {
            nospill: kappa > 0.0,
            steps: true,
        },
        opts,
    )?;
    let e = avg.scopes[0];
    // The displaced field the estimator sees has no effect.
    let (truth, truth_weighted) = if shifted {
        (0.0, 0.0)
    } else {
        let (mut num, mut den) = (0.0, 0.0);
        for i in (0..avg.h.len()).filter(|&i| comp[i]) {
            num += avg.h[i] * (y1[i] - y0[i]);
            den += avg.h[i];
        }
        let weighted = if den > 0.0 { num / den } else { f64::NAN };
        (sample_effect(&y1, &y0, &comp), weighted)
    };
    let comp_rows: Vec<u32> = (0..n as u32).filter(|&i| comp[i as usize]).collect();
    let log_dose: Vec<f64> = exposure.dose.iter().map(|d| d.ln_1p()).collect();
    let marginal_cor = marginal_correlation(&response.y, &log_dose, &comp_rows, rows.cf);
    let mut judged = EffectRow::pending(
        e,
        (
            coexpression_baselines.coexpr_log_or,
            coexpression_baselines.coexpr_se,
        ),
        eta_squared(
            &tissue.neighbour_bin,
            &indicator(&exposure.leaned_on(e.abf)),
            &rows.scope_rows[0],
        ),
        (avg.background, avg.over_background),
        ng,
    );
    report(std::slice::from_mut(&mut judged), &INTERNALS.report_opts());
    Ok(SimRow {
        ligand: base.ligand.into(),
        receptor: base.receptor.into(),
        scenario: *scen,
        truth,
        truth_weighted,
        estimate: e.competent,
        marginal_cor,
        crude: e.crude,
        ipw: e.ipw,
        outcome: e.outcome,
        incompetent: e.incompetent,
        ring: e.ring,
        ring_truth,
        no_spill: e.nospill,
        steps: e.steps,
        contrast: e.effect,
        over_ring: e.over_ring,
        background: avg.background,
        over_background: avg.over_background,
        verdict: judged.verdict,
        misclassified,
        baselines: coexpression_baselines,
        seconds: start.elapsed().as_secs_f64(),
        baseline_seconds,
    })
}

/// The receptor driven by the hidden field: `Poisson(sf · ρ_b · e^{γ f} /
/// mean e^{γ f})`, `ρ_b` the real receptor's mean count / size factor in
/// identity bin `b`.
fn simulate_receptor(tissue: &Tissue, real: &[f64], f: &[f64], gamma: f64, seed: u64) -> Vec<f64> {
    let n = tissue.n();
    let (sf, bin) = (&tissue.size_factor, &tissue.identity_bin);
    let nb = n_bins(tissue);
    let (mut rho, mut cells) = (vec![0.0; nb], vec![0usize; nb]);
    for i in 0..n {
        rho[bin[i]] += real[i] / sf[i];
        cells[bin[i]] += 1;
    }
    for (r, &c) in rho.iter_mut().zip(&cells) {
        *r /= c.max(1) as f64;
    }
    let norm = f.iter().map(|z| (gamma * z).exp()).sum::<f64>() / n.max(1) as f64;
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|i| poisson(&mut rng, sf[i] * rho[bin[i]] * (gamma * f[i]).exp() / norm))
        .collect()
}
