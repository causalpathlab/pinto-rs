//! Scoring LR pairs in one stratum against their sample permutations.

use super::pseudobulk::CollapseOut;
use super::MIN_SAMPLES_PER_STRATUM;
use crate::lr_activity::orientation::CommunityStrata;
use crate::lr_activity::outputs::LrActivityRow;
use crate::util::common::*;
use legume_numeric::matrix::rand_util::mix_seed;
use legume_numeric::param::traits::Inference;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Floor to avoid div-by-zero when a sample has no presence in a community.
const EPS: f32 = 1e-8;

/// Weighted covariance between `l` and `r` using sample weights `w`.
/// Preferred over correlation here: (1) preserves absolute magnitude of
/// L-R coupling so pairs of different scales separate cleanly under
/// restandardization (correlation is bounded and piles up at the median);
/// (2) tied-zero samples (zero-inflated pseudobulks) contribute 0 to the
/// running sum instead of the spurious ±1 inflation that correlation
/// suffers when zero-patterns co-occur. Returns NaN when weights sum to 0.
pub(crate) fn weighted_cov(l: &[f32], r: &[f32], w: &[f32]) -> f32 {
    let mut sw = 0.0f32;
    let mut sl = 0.0f32;
    let mut sr = 0.0f32;
    for k in 0..l.len() {
        sw += w[k];
        sl += w[k] * l[k];
        sr += w[k] * r[k];
    }
    if sw <= EPS {
        return f32::NAN;
    }
    let ml = sl / sw;
    let mr = sr / sw;
    let mut cov = 0.0f32;
    for k in 0..l.len() {
        cov += w[k] * (l[k] - ml) * (r[k] - mr);
    }
    cov / sw
}

/// Robust median of a slice; `None` when empty. NaN-tolerant ordering.
fn robust_median(v: &[f32]) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    let mut buf = v.to_vec();
    let mid = buf.len() / 2;
    let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    let (_, m, _) = buf.select_nth_unstable_by(mid, cmp);
    Some(*m)
}

/// The Efron-Tibshirani restandardization scale for a stratum, or `None`
/// when the stratum cannot calibrate one.
///
/// `z_re` asks whether a pair stands out against the OTHER pairs in its
/// stratum, so the cross-pair spread must be commensurate with a single
/// pair's own permutation noise. When every pair sits in one tight clump the
/// clump's width is a shared noise floor, and dividing by it manufactures
/// astronomical significance for trivial values: a real run produced
/// `z_re = 12.3, p_re = 1e-34` on a pair whose own permutation p was 0.42.
///
/// The guard is scale-RELATIVE — `1.4826 * MAD >= 0.1 * median(null_sd)` —
/// never an absolute floor, which its predecessor was: an absolute floor
/// both passes the failure above (its MAD sat just over the constant) and
/// wrongly flags healthy strata measured in small units. The 0.1 sits an
/// order of magnitude below every healthy stratum measured (0.32-1.98 on
/// the reference run) and an order above the failure case (~0.02).
pub(crate) fn restandardization_scale(stats: &[f32], null_sds: &[f32]) -> Option<f32> {
    let med = robust_median(stats)?;
    let abs_dev: Vec<f32> = stats.iter().map(|&v| (v - med).abs()).collect();
    let mad = robust_median(&abs_dev)?;
    // 1.4826 is the standard normal-consistent MAD scaling.
    let sigma_emp = 1.4826 * mad;
    let noise = robust_median(null_sds)?;
    (sigma_emp.is_finite() && noise.is_finite() && sigma_emp >= 0.1 * noise).then_some(sigma_emp)
}

/// The weighted covariance, or `None` when the pair carries no testable
/// hypothesis in this stratum.
///
/// `None` when the statistic is non-finite (no usable weight at all) or lands
/// on bit-exact zero. Exact zero is not a measured null: it means no weighted
/// sample deviated on both sides at once (a constant side, or two genes
/// detected in disjoint samples), or the few real contributions cancelled to
/// the last bit, which is what near-floor genes produce in practice. Such a
/// pair cannot reject at any threshold, so scoring it would pay multiplicity
/// in the Westfall-Young family, drag the robust spread that scales every
/// other pair's `z_re`, and ship a row that reads as a measured null rather
/// than as an absent measurement.
///
/// An earlier guard keyed on the observed statistic AND the null spread both
/// being zero, which never fired: the null is built from fresh posterior
/// draws that fluctuate even where the observed posterior mean is flat.
///
/// A merely SMALL statistic stays in, deliberately — that is a magnitude
/// question, and a threshold here would be a silent power cut rather than an
/// exclusion of the untestable.
pub(crate) fn testable_weighted_cov(l: &[f32], r: &[f32], w: &[f32]) -> Option<f32> {
    let t = weighted_cov(l, r, w);
    if t.is_finite() && t != 0.0 {
        Some(t)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn score_pairs_for_stratum(
    batch_label: &str,
    stratum: u32,
    samples_in_stratum: &[usize],
    sample_propbin: &[usize],
    real_pairs: &[(Box<str>, Box<str>, usize, usize)],
    gene_to_local: &HashMap<usize, usize>,
    gene_names: &[Box<str>],
    collapse: &CollapseOut,
    fisher_lr: &[f32],
    n_perm: usize,
    base_seed: u64,
    stratum_id: usize,
    shuffle_within_batch: bool,
    sample_batch_label: &[Box<str>],
    shuffle_stratify_dim: usize,
    strata_map: &CommunityStrata,
) -> Vec<LrActivityRow> {
    let c = stratum as usize;
    let n_s = samples_in_stratum.len();
    if n_s < MIN_SAMPLES_PER_STRATUM {
        return Vec::new();
    }

    let w_pair: Vec<f32> = samples_in_stratum
        .iter()
        .map(|&s| collapse.denom[c][s])
        .collect();

    // Permutation buckets: rows of `bucket_idx` give the position-indices
    // (into the stratum's sample list) that may be shuffled together.
    // Pooled strata always shuffle within batch (to preserve batch-level
    // confounders); per-batch strata are unconstrained on the batch axis.
    // When `shuffle_stratify_dim > 0`, we additionally subgroup by the top
    // bits of the propensity binary code so the cell-population marginal
    // is preserved across permutations.
    let strat_mask: usize = if shuffle_stratify_dim == 0 {
        0
    } else {
        (1usize << shuffle_stratify_dim) - 1
    };
    let bucket_idx: Vec<Vec<usize>> = {
        let mut buckets: HashMap<(Box<str>, usize), Vec<usize>> = HashMap::default();
        for (k, &s) in samples_in_stratum.iter().enumerate() {
            let batch_key: Box<str> = if shuffle_within_batch {
                sample_batch_label[s].clone()
            } else {
                Box::from("_")
            };
            let strat_key = sample_propbin[s] & strat_mask;
            buckets.entry((batch_key, strat_key)).or_default().push(k);
        }
        buckets.into_values().collect()
    };

    // Pre-generate K *shared* sample permutations for this stratum. Same
    // shuffle σ_k applied to every pair → preserves cross-pair dependence
    // (shared genes, batch confounders) for Westfall-Young joint
    // inference. Single seeded RNG for determinism.
    // Each stored shuffle is a permutation of GLOBAL sample ids, in stratum
    // position order. The buckets shuffle positions, and the final map takes
    // every position through `samples_in_stratum`, so consumers gather from
    // the rate matrices directly and no position-space index survives to be
    // confused with a global one (which has been a live bug here once).
    let shared_shuffles: Vec<Vec<usize>> = {
        let mut rng = SmallRng::seed_from_u64(base_seed);
        (0..n_perm)
            .map(|_| {
                let mut perm: Vec<usize> = (0..n_s).collect();
                for bucket in &bucket_idx {
                    if bucket.len() < 2 {
                        continue;
                    }
                    let mut vals: Vec<usize> = bucket.iter().map(|&p| perm[p]).collect();
                    vals.shuffle(&mut rng);
                    for (slot_pos, &out_pos) in bucket.iter().enumerate() {
                        perm[out_pos] = vals[slot_pos];
                    }
                }
                perm.into_iter().map(|p| samples_in_stratum[p]).collect()
            })
            .collect()
    };

    // Per-pair fixed quantities. Untestable pairs (see
    // `testable_weighted_cov`) are compacted away here, so nothing downstream
    // carries a placeholder for them: not the permutation loop, not the WY
    // family, not the restandardization moments.
    struct PairCtx {
        l_local: usize,
        r_local: usize,
        lname: Box<str>,
        rname: Box<str>,
        li: usize,
        ri: usize,
        t_obs: f32,
        /// Product of the two per-gene Fisher weights. The observed side has
        /// them baked into the log-mean matrices; under the bilinear
        /// `weighted_cov` the same scaling reaches a permuted statistic as
        /// this one multiply, instead of rescaling a whole draw matrix per
        /// permutation.
        w_lr: f32,
    }
    let pair_ctx: Vec<PairCtx> = real_pairs
        .par_iter()
        .filter_map(|(lname, rname, li, ri)| -> Option<PairCtx> {
            let l_local = *gene_to_local.get(li)?;
            let r_local = *gene_to_local.get(ri)?;
            let l_vec: Vec<f32> = samples_in_stratum
                .iter()
                .map(|&s| collapse.log_mean[c][(l_local, s)])
                .collect();
            let r_vec: Vec<f32> = samples_in_stratum
                .iter()
                .map(|&s| collapse.log_mean[c][(r_local, s)])
                .collect();
            let t_obs = testable_weighted_cov(&l_vec, &r_vec, &w_pair)?;
            let w_lr = fisher_lr[l_local] * fisher_lr[r_local];
            Some(PairCtx {
                l_local,
                r_local,
                lname: lname.clone(),
                rname: rname.clone(),
                li: *li,
                ri: *ri,
                t_obs,
                w_lr,
            })
        })
        .collect();

    // Build per-pair null vectors `t_perm[i]`.
    //
    // Parallel over PERMUTATIONS, sequential over pairs within each. The
    // Westfall-Young guarantee constrains sharing WITHIN a permutation —
    // every pair must see the same shuffle and the same posterior draw — and
    // says nothing about the order permutations are computed in. Successive
    // permutations carry no state either: the draw seed is a pure function of
    // `k`, and the shuffles were all materialised up front.
    //
    // This is the better axis to parallelise on: far more permutations than
    // cores, so the granularity is coarse, where the old shape re-entered a
    // parallel region once per permutation (the draws themselves were
    // internally parallel, so the win is fewer region entries and coarser
    // work items, not idle cores recovered). Draw memory is bounded by the
    // worker count, since only in-flight draws are alive.
    //
    // `collect` on an indexed parallel iterator preserves order, which the WY
    // step below relies on: `min_p[k]` must refer to the same permutation for
    // every pair.

    let t_perm_per_pair: Vec<Vec<f32>> = {
        let t_per_k: Vec<Vec<f32>> = shared_shuffles
            .par_iter()
            .enumerate()
            .map(|(k, sigma)| {
                // Seeded per (stratum, permutation, role) through the shared
                // avalanche mixer, so nearby permutation indices map to
                // well-separated streams. (The shuffles beside this were
                // already deterministic; the posterior draw once was not, and
                // `--seed` did not reach the null.)
                let draw_seed = mix_seed(base_seed, k as u64);
                // One draw serves both genes: entries of the posterior sample
                // are independent across (gene, sample) cells, so the ligand
                // and receptor rows carry independent noise from one matrix.
                let log_draw = collapse.gamma[c]
                    .posterior_log_sample(draw_seed)
                    .expect("posterior_log_sample failed");
                // No Fisher rescale of the draw matrices here: `weighted_cov`
                // is bilinear, so the per-gene weights reach each statistic as
                // the precomputed `w_lr` scalar below — the same algebra the
                // observed side uses by baking the weights into the log-means.
                pair_ctx
                    .iter()
                    .map(|pc| {
                        let l_perm: Vec<f32> =
                            sigma.iter().map(|&g| log_draw[(pc.l_local, g)]).collect();
                        let r_obs: Vec<f32> = samples_in_stratum
                            .iter()
                            .map(|&s| log_draw[(pc.r_local, s)])
                            .collect();
                        pc.w_lr * weighted_cov(&l_perm, &r_obs, &w_pair)
                    })
                    .collect()
            })
            .collect();
        // Transpose [k][pair] -> [pair][k]; the k order is what `min_p`
        // indexes, so it must survive.
        // NOT `vec![Vec::with_capacity(..); n]`: that clones the prototype,
        // and a Vec clone copies contents (none) rather than capacity, so
        // every slot but the last would start at zero and regrow ~10 times.
        let mut t_per_pair: Vec<Vec<f32>> = (0..pair_ctx.len())
            .map(|_| Vec::with_capacity(n_perm))
            .collect();
        for t_k in t_per_k {
            for (i, t) in t_k.into_iter().enumerate() {
                t_per_pair[i].push(t);
            }
        }
        t_per_pair
    };

    // Per-pair: aggregate t_perm into stats and build the row.
    let pair_results: Vec<(LrActivityRow, Vec<f32>)> = pair_ctx
        .into_par_iter()
        .zip(t_perm_per_pair.into_par_iter())
        .filter_map(|(pc, t_perm)| {
            // Single pass over t_perm collects all per-pair null moments.
            let mut n_finite = 0usize;
            let mut sum = 0.0f32;
            let mut sumsq = 0.0f32;
            let mut n_gt = 0u32;
            let mut n_eq = 0u32;
            for &v in &t_perm {
                if !v.is_finite() {
                    continue;
                }
                n_finite += 1;
                sum += v;
                sumsq += v * v;
                if v > pc.t_obs {
                    n_gt += 1;
                } else if v == pc.t_obs {
                    n_eq += 1;
                }
            }
            if n_finite < n_perm.div_ceil(2) {
                return None;
            }
            let n_f = n_finite as f32;
            let mu = sum / n_f;
            let var = (sumsq - n_f * mu * mu) / (n_f - 1.0).max(1.0);
            let sd_raw = var.max(0.0).sqrt();

            // A null with no spread cannot standardize anything: flag NaN
            // for the per-pair stats rather than divide by (near-)zero.
            // This guards the NULL axis only — an untestable OBSERVED
            // statistic never gets this far (`testable_weighted_cov`
            // compacted those pairs away before the permutation loop). No
            // magnitude floor here for the same reason as there: a small
            // real spread is still a distribution.
            let degenerate = !sd_raw.is_finite() || sd_raw <= 0.0;

            let p_emp = if degenerate {
                f32::NAN
            } else {
                (n_gt as f32 + 0.5 * n_eq as f32 + 0.5) / (n_f + 1.0)
            };

            let (z, p_z) = if degenerate {
                (f32::NAN, f32::NAN)
            } else {
                let zv = (pc.t_obs - mu) / sd_raw;
                (zv, one_sided_p_z(zv))
            };

            let row = LrActivityRow {
                batch: Box::from(batch_label),
                // The REAL link community id — no longer a stratum index
                // hiding under this name.
                community: strata_map.community(c) as i32,
                ligand: pc.lname.clone(),
                receptor: pc.rname.clone(),
                ligand_resolved: gene_names[pc.li].clone(),
                receptor_resolved: gene_names[pc.ri].clone(),
                n_samples: n_s as i32,
                stat_obs: pc.t_obs,
                null_mean: mu,
                null_sd: sd_raw,
                z,
                p_empirical: p_emp,
                p_z,
                z_re: f32::NAN,
                p_re: f32::NAN,
                fwer_wy: f32::NAN,
                stratum_id,
            };
            Some((row, t_perm))
        })
        .collect();

    let (mut rows, perms_per_pair): (Vec<LrActivityRow>, Vec<Vec<f32>>) =
        pair_results.into_iter().unzip();

    // Efron-Tibshirani restandardization: re-center / re-scale stat_obs
    // against the across-pair empirical bulk in this stratum, using robust
    // moments (median, MAD). Untestable pairs never got a row, so every row
    // with a finite z participates. The scale is guarded relative to the
    // pairs' own permutation noise — see `restandardization_scale`.
    let finite: Vec<&LrActivityRow> = rows
        .iter()
        .filter(|r| r.stat_obs.is_finite() && r.z.is_finite())
        .collect();
    let stats: Vec<f32> = finite.iter().map(|r| r.stat_obs).collect();
    let null_sds: Vec<f32> = finite.iter().map(|r| r.null_sd).collect();
    match restandardization_scale(&stats, &null_sds) {
        Some(sigma_emp) => {
            let med = robust_median(&stats).expect("stats nonempty when scale exists");
            for r in rows.iter_mut() {
                if r.stat_obs.is_finite() && r.z.is_finite() {
                    let zr = (r.stat_obs - med) / sigma_emp;
                    r.z_re = zr;
                    // Two-sided p so the null is properly Uniform(0,1)
                    // regardless of any residual skew in the stat_obs
                    // distribution; Storey's π₀ estimator stays calibrated.
                    // Sign restriction (active LR = positive z_re) is
                    // applied at the reporting layer.
                    let p_two = 2.0 * one_sided_p_z(zr.abs());
                    r.p_re = p_two.min(1.0);
                }
            }
        }
        None => {
            if !stats.is_empty() {
                warn!(
                    "stratum {}: cross-pair spread is too small against the pairs' own \
                     permutation noise to calibrate restandardization; z_re and p_re \
                     withheld (NaN) for its {} pairs",
                    stratum_id,
                    stats.len()
                );
            }
        }
    }

    // Westfall-Young single-step minP: `min_p[k] = min_i p_perm[k, i]` is
    // the null distribution of "the most significant pair under shuffle
    // k". Adjusted p = (1 + #{k : min_p[k] ≤ p_obs[i]}) / (K + 1).
    // Degenerate pairs don't enter min_p — they can't be most significant.
    if !perms_per_pair.is_empty() && !perms_per_pair[0].is_empty() {
        let k_perm = perms_per_pair[0].len();
        let k_perm_f = k_perm as f32;
        let mut min_p: Vec<f32> = vec![1.0f32; k_perm];
        let mut order: Vec<usize> = Vec::with_capacity(k_perm);
        for (row, t_perm) in rows.iter().zip(perms_per_pair.iter()) {
            if !row.stat_obs.is_finite() || !row.z.is_finite() {
                continue;
            }
            order.clear();
            order.extend(0..k_perm);
            order.sort_by(|&a, &b| {
                t_perm[b]
                    .partial_cmp(&t_perm[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (pos, &k) in order.iter().enumerate() {
                let p_perm = (pos + 1) as f32 / k_perm_f;
                if p_perm < min_p[k] {
                    min_p[k] = p_perm;
                }
            }
        }

        for (row, t_perm) in rows.iter_mut().zip(perms_per_pair.iter()) {
            if !row.stat_obs.is_finite() || !row.z.is_finite() {
                continue;
            }
            let n_ge = t_perm
                .iter()
                .filter(|&&v| v.is_finite() && v >= row.stat_obs)
                .count();
            let p_obs = (n_ge as f32 + 1.0) / (k_perm_f + 1.0);
            let n_le = min_p.iter().filter(|&&v| v <= p_obs).count();
            let fwer_wy = (n_le as f32 + 1.0) / (k_perm_f + 1.0);
            row.fwer_wy = fwer_wy.min(1.0);
        }
    }
    rows
}

/// One-sided upper-tail Gaussian p-value: P(Z >= z) for standard normal.
#[inline]
fn one_sided_p_z(z: f32) -> f32 {
    // erfc(z/√2) / 2; numerically stable for large |z|.
    let p = super::super::ligand_effect::inference::upper_normal_p(z as f64);
    (p as f32).clamp(0.0, 1.0)
}
