//! Delete-one-tile jackknife of a scope's effect, nuisance fits included.
//!
//! The clustered variance of [`super::overlap::subset_estimate`] is that of
//! the influence function, with the cross-fitted nuisances (propensity and
//! outcome models) taken as known. On a scope inside a few tiles, a region
//! of one donor say, those models' own error is not small, and it is shared
//! by every receiver of a held-out fold: on `pinto sim new`
//! donor-confounded-donor-only-cell-type a region's null z had an SD near
//! 1.4. Deleting each tile in turn and refitting everything on the rest
//! counts that error (Efron & Stein 1981, Ann Stat 9:586; a
//! delete-one-cluster jackknife, as in MacKinnon, Nielsen & Webb 2023,
//! J Econometrics 232:272).
//!
//! The tiles deleted are those of the sections the scope lies in: tiles of
//! other sections reach the scope's estimate only through the shared
//! slopes of the nuisance models, a small part. Each replicate is the pair
//! at its headline threshold, without the sensitivity fits, so the
//! jackknife is read as a ratio to the clustered SE of the same estimate,
//! which then scales the reported one ([`Jackknife::widen`]). Replicates
//! are independent and run in parallel; each owns only its fold Grams
//! ([`super::crossfit::Design::leaving_out`]).
//!
//! `--region-jackknife no-refit`, the default, keeps the full nuisance fits
//! and leaves the tile out of the averages alone; every replicate is read
//! from one pass of the fits, as scopes of their own. On that scenario's
//! one-donor region it cut the nulls rejected from 10.5% to 7.6% at a small
//! loss of power (0.84 to 0.79). `refit` counts the nuisance error and is
//! conservative on regions in few tiles (0% of nulls rejected there, power
//! 0.13 at effect PVE 0.02), at one refit per tile.
//!
//! A replicate's fits are exact, but cheap: the outcome ridge is solved on
//! the full Grams less the tile's ([`super::crossfit::Design::leaving_out`]),
//! and each fold's logistic propensity starts from the full fit's solution,
//! which differs from the replicate's by one tile's worth of data, so its
//! Newton steps converge in a few. This is the approximate leave-out of
//! Giordano, Stephenson, Liu, Jordan & Broderick (2019, "A Swiss army
//! infinitesimal jackknife", AISTATS, PMLR 89:1139) and Rad & Maleki (2020,
//! "A scalable estimate of the out-of-sample prediction error via
//! approximate leave-one-out cross-validation", J R Stat Soc B 82:965),
//! one Newton step from the full solution, iterated to the replicate's
//! optimum: the problem is strictly convex, so the start changes the steps,
//! not the answer.

use super::effects::{complement, ScopeEffects, Sensitivity};
use super::fits::Rows;
use super::inference::Estimate;
use super::overlap::FoldStarts;
use super::thresholds::over_thresholds;
pub use super::thresholds::PairFit;
use super::tissue::Tissue;
use crate::lr_activity::args::RegionJackknife;
use crate::util::common::*;

/// A scope's headline p below which its region rows are jackknifed: above
/// it a wider SE could not make the row significant, so skipping it changes
/// no call ([`Jackknife::widen`] only widens).
pub const JACKKNIFE_P: f64 = 0.2;

/// Jackknife SE over the clustered SE at the headline threshold, for the
/// competent effect and for its contrast with the rest; NaN where it was
/// not computed (no estimate, or a replicate without one).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Jackknife {
    pub competent: f64,
    pub versus_rest: f64,
    /// Tiles deleted, one replicate each.
    pub replicates: usize,
}

impl Jackknife {
    pub const NONE: Self = Self {
        competent: f64::NAN,
        versus_rest: f64::NAN,
        replicates: 0,
    };

    /// `e` with its competent effect's SE, the negative-control contrast's
    /// (competent less incompetent, which the verdict tests the effect
    /// against) and the contrast with the rest scaled up by the ratios,
    /// never down: the jackknife adds the nuisance error the clustered SE
    /// leaves out, and a ratio below 1 is its own noise.
    pub fn widen(&self, e: &mut ScopeEffects) {
        let scale = |x: &mut Estimate, r: f64| {
            if r.is_finite() && r > 1.0 {
                x.se *= r;
            }
        };
        scale(&mut e.competent, self.competent);
        scale(&mut e.effect, self.competent);
        scale(&mut e.versus_rest, self.versus_rest);
    }

    /// Whether replicates ran but gave no ratio: deleting some tile left an
    /// arm thin, so the row keeps its clustered SE, likely too small.
    pub fn failed(&self) -> bool {
        self.replicates > 0 && !self.competent.is_finite()
    }
}

/// The delete-one-tile jackknife of the scope with rows `inside` (ascending),
/// on the tissue `rows` reads.
pub fn jackknife(
    rows: Rows<'_>,
    tissue: &Tissue,
    inside: &[u32],
    fit: &PairFit<'_>,
) -> anyhow::Result<Jackknife> {
    let pooled = &rows.scope_rows[0];
    let headline =
        |r: Rows<'_>, init: FoldStarts| over_thresholds(r, fit, &[], Sensitivity::NONE, init);
    let tiles = tiles_of(tissue, rows, inside);
    if tiles.len() < 2 {
        return Ok(Jackknife::NONE);
    }
    // The full estimate, and each replicate's effect and contrast with the
    // rest.
    let (full, competent, versus_rest) = match fit.opts.region_jackknife {
        RegionJackknife::Refit => {
            let sr = vec![pooled.clone(), inside.to_vec()];
            let full_fit = headline(
                Rows {
                    steps: &[],
                    scope_rows: &sr,
                    share: None,
                    ..rows
                },
                FoldStarts::default(),
            )?;
            let full = full_fit.scopes[1];
            if !full.competent.theta.is_finite() {
                return Ok(Jackknife::NONE);
            }
            let reps: Vec<(f64, f64)> = tiles
                .par_iter()
                .map(|out| {
                    let sr = vec![complement(pooled, out), complement(inside, out)];
                    let out: Vec<usize> = out.iter().map(|&i| i as usize).collect();
                    let cf = rows.cf.leaving_out(&out);
                    let design = rows.design.leaving_out(rows.cf, &out);
                    let r = Rows {
                        design: &design,
                        steps: &[],
                        cf: &cf,
                        scope_rows: &sr,
                        share: None,
                        deleted: &[],
                    };
                    headline(r, full_fit.starts.clone()).map(|a| {
                        let e = a.scopes[1];
                        (e.competent.theta, e.versus_rest.theta)
                    })
                })
                .collect::<anyhow::Result<_>>()?;
            // Every tile moves the refitted nuisances, so every one counts.
            let (c, v) = reps.into_iter().unzip();
            (full, c, v)
        }
        // The nuisance fits do not change, so one pass of them reads the
        // full estimate and every replicate: the region less each tile and
        // the rest less it, as scopes of their own; a replicate's contrast
        // is their difference. A tile holding none of the region's rows
        // leaves its estimate as it was, so the effect counts only the
        // tiles it lies in; the contrast with the rest counts them all.
        RegionJackknife::NoRefit => {
            let rest = complement(pooled, inside);
            let mut sr = vec![pooled.clone(), inside.to_vec()];
            for out in &tiles {
                sr.push(complement(inside, out));
                sr.push(complement(&rest, out));
            }
            let all = headline(
                Rows {
                    steps: &[],
                    scope_rows: &sr,
                    share: None,
                    ..rows
                },
                FoldStarts::default(),
            )?;
            let full = all.scopes[1];
            if !full.competent.theta.is_finite() {
                return Ok(Jackknife::NONE);
            }
            let (mut c, mut v) = (Vec::new(), Vec::new());
            for (out, [less, rest_less]) in tiles.iter().zip(all.scopes[2..].as_chunks::<2>().0) {
                let a = less.competent.theta;
                if out.iter().any(|i| inside.binary_search(i).is_ok()) {
                    c.push(a);
                }
                v.push(a - rest_less.competent.theta);
            }
            (full, c, v)
        }
    };
    Ok(Jackknife {
        competent: jackknife_se(&competent) / full.competent.se,
        versus_rest: jackknife_se(&versus_rest) / full.versus_rest.se,
        replicates: tiles.len(),
    })
}

/// The rows of each tile of the sections `inside` lies in, ascending, one
/// list per tile.
fn tiles_of(tissue: &Tissue, rows: Rows<'_>, inside: &[u32]) -> Vec<Vec<u32>> {
    let sections: HashSet<u32> = inside.iter().map(|&i| tissue.section[i as usize]).collect();
    let mut by_tile: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for &i in &rows.scope_rows[0] {
        if sections.contains(&tissue.section[i as usize]) {
            by_tile
                .entry(rows.cf.cluster[i as usize])
                .or_default()
                .push(i);
        }
    }
    by_tile.into_values().collect()
}

/// Each estimator's delete-one-tile jackknife SE and degrees of freedom, of
/// the whole estimate as reported: averaged over exposure thresholds; NaN
/// where it was not computed.
///
/// The jackknife is the no-refit one of [`jackknife`]: the nuisance fits of
/// the full data are kept and each tile is left out of the averages alone,
/// every replicate read from one pass of the fits as a scope of its own. A
/// replicate is the scope less one tile carried through everything the
/// estimate is made of ([`super::thresholds::over_thresholds`]): each
/// exposure threshold's estimate, Rubin's mean over the competence and
/// programme-weighting choices, and the threshold average with its
/// approximate-Bayes-factor weights recomputed on the replicate. With `G`
/// replicates `θ̄₍₋g₎`, `SE² = (G − 1)/G Σ_g (θ̄₍₋g₎ − mean)²`, and the t
/// reference's df is `G_eff − 1`, `G_eff` counting the replicates the
/// variance rests on as [`super::inference::effective_clusters`] counts
/// clusters. The jackknife of Rubin's mean leaves out its between-choice
/// term: the choices are not resampled, so the SE is the sampling error of
/// the reported estimate alone. The tiles deleted are those of the sections
/// the scope lies in. Holding the fits fixed leaves out the error of the
/// outcome model's coefficients, which is the outcome regression's sampling
/// error, so its replicates move them by the tile's exact leave-out ridge
/// shift ([`super::overlap::OutcomeInfluence`]), a tile outside the scope
/// included; the other estimators are first-order insensitive to the fits
/// (the augmented one) or have none to move (the unadjusted difference),
/// and the overlap weights' propensity is held fixed. The same leave-out
/// gives the outcome regression its SE at each threshold
/// ([`super::overlap::outcome_estimate`]), which weighs the thresholds of its
/// average: with the fits taken as known that SE was far too small, and
/// the weights went nearly all to the threshold of largest z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EstimatorJackknife {
    /// `(se, df)` of the headline, `effect_dr` (doubly robust).
    pub competent: (f64, f64),
    /// Overlap weights alone, `effect_ipw`.
    pub ipw: (f64, f64),
    /// The outcome regression alone, `effect_outcome`.
    pub outcome: (f64, f64),
    /// `effect_unadj`.
    pub unadj: (f64, f64),
    /// Tiles deleted, one replicate each.
    pub replicates: usize,
}

impl EstimatorJackknife {
    pub const NONE: Self = Self {
        competent: (f64::NAN, f64::NAN),
        ipw: (f64::NAN, f64::NAN),
        outcome: (f64::NAN, f64::NAN),
        unadj: (f64::NAN, f64::NAN),
        replicates: 0,
    };

    /// `e` with each estimator's SE and df replaced by the jackknife's, up or
    /// down; where a replicate had no estimate (deleting a tile left an arm
    /// thin) the clustered SE stays. The headline is left to the region rows'
    /// [`Jackknife`] when `headline` is false.
    pub fn apply(&self, e: &mut ScopeEffects, headline: bool) {
        let set = |x: &mut Estimate, (se, df): (f64, f64)| {
            if se.is_finite() && df > 0.0 {
                (x.se, x.df) = (se, df);
            }
        };
        if headline {
            set(&mut e.competent, self.competent);
        }
        set(&mut e.ipw, self.ipw);
        set(&mut e.outcome, self.outcome);
        set(&mut e.unadj, self.unadj);
    }
}

/// [`EstimatorJackknife`] of each scope of `rows` named by `scopes` (indices
/// into `rows.scope_rows`, the pooled scope first in `rows`), from one pass
/// of the fits over the headline and `taus`, as the estimate was read,
/// whose scopes are those scopes and each less each of its tiles.
pub fn estimator_jackknife(
    rows: Rows<'_>,
    tissue: &Tissue,
    scopes: &[usize],
    fit: &PairFit<'_>,
    taus: &[(f64, f64)],
) -> anyhow::Result<Vec<EstimatorJackknife>> {
    let pooled = &rows.scope_rows[0];
    let mut sr = vec![pooled.clone()];
    let mut deleted = vec![None];
    // Per scope: its replicates' span in `sr`, after the scope itself.
    let mut spans = Vec::with_capacity(scopes.len());
    for &k in scopes {
        let inside = &rows.scope_rows[k];
        let parent = sr.len();
        sr.push(inside.clone());
        deleted.push(None);
        for out in tiles_of(tissue, rows, inside) {
            sr.push(complement(inside, &out));
            deleted.push(Some((parent, rows.cf.cluster[out[0] as usize])));
        }
        spans.push(parent + 1..sr.len());
    }
    let all = over_thresholds(
        Rows {
            steps: &[],
            scope_rows: &sr,
            share: None,
            deleted: &deleted,
            ..rows
        },
        fit,
        taus,
        Sensitivity::NONE,
        FoldStarts::default(),
    )?;
    Ok(spans
        .into_iter()
        .map(|span| {
            let reps = &all.scopes[span];
            if reps.len() < 2 {
                return EstimatorJackknife::NONE;
            }
            let of = |pick: fn(&ScopeEffects) -> Estimate| {
                super::inference::jackknife_sd(
                    &reps.iter().map(|r| pick(r).theta).collect::<Vec<_>>(),
                )
            };
            EstimatorJackknife {
                competent: of(|e| e.competent),
                ipw: of(|e| e.ipw),
                outcome: of(|e| e.outcome),
                unadj: of(|e| e.unadj),
                replicates: reps.len(),
            }
        })
        .collect())
}
