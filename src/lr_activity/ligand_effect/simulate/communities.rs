//! Synthetic communities, so the boundary design can be tested on layouts no
//! one tissue offers. Labels come from the coordinates and from nothing the
//! ligand or the programme does, so they cannot show collider bias.
//! [`boundary_estimate`] is the comparison at the boundaries they draw.

use super::scenario::CommunityLayout;
use crate::lr_activity::ligand_effect::contacts::Contacts;
use crate::lr_activity::ligand_effect::crossfit::{CrossFit, Design};
use crate::lr_activity::ligand_effect::dose::{choose_threshold, exposed, Exposure};
use crate::lr_activity::ligand_effect::effects::Sensitivity;
use crate::lr_activity::ligand_effect::fits::Rows;
use crate::lr_activity::ligand_effect::inference::Estimate;
use crate::lr_activity::ligand_effect::options::EffectOpts;
use crate::lr_activity::ligand_effect::response::Response;
use crate::lr_activity::ligand_effect::thresholds::{
    over_thresholds, thresholds, ThresholdAverage,
};
use crate::lr_activity::ligand_effect::tissue::Tissue;
use crate::util::common::HashMap;
use legume_numeric::matrix::rand_util::mix_seed;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Each row's community: the nearest of `number` random cells in its section.
/// The first community's edge moves out until it holds the `dominant` share of
/// the section, and a `mixing` share of rows then takes another label at
/// random, which scatters the boundaries through the tissue.
pub fn draw_communities(tissue: &Tissue, layout: &CommunityLayout, seed: u64) -> Vec<u32> {
    let n = tissue.n();
    let k = layout.number;
    let mut rng = SmallRng::seed_from_u64(mix_seed(seed, 1));
    let mut labels = vec![0u32; n];
    let at = |i: usize| (tissue.coords[i].0 as f64, tissue.coords[i].1 as f64);
    for s in 0..tissue.unit.len() {
        let rows: Vec<usize> = (0..n)
            .filter(|&i| tissue.section[i] as usize == s)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let centres: Vec<(f64, f64)> = (0..k)
            .map(|_| at(rows[rng.random_range(0..rows.len())]))
            .collect();
        let nearest_other: Vec<(usize, f64)> = rows
            .iter()
            .map(|&i| {
                let (x, y) = at(i);
                let dist = |c: &(f64, f64)| (x - c.0).hypot(y - c.1);
                centres[1..]
                    .iter()
                    .enumerate()
                    .map(|(j, c)| (j + 1, dist(c)))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("at least two communities")
            })
            .collect();
        let margin: Vec<f64> = rows
            .iter()
            .zip(&nearest_other)
            .map(|(&i, &(_, d))| {
                let (x, y) = at(i);
                (x - centres[0].0).hypot(y - centres[0].1) - d
            })
            .collect();
        let edge = if layout.dominant > 0.0 {
            let mut sorted = margin.clone();
            let q = ((rows.len() - 1) as f64 * layout.dominant) as usize;
            *sorted.select_nth_unstable_by(q, f64::total_cmp).1
        } else {
            0.0
        };
        for ((&i, &(other, _)), &m) in rows.iter().zip(&nearest_other).zip(&margin) {
            labels[i] = if m <= edge { 0 } else { other as u32 };
        }
    }
    for l in labels.iter_mut() {
        if rng.random::<f64>() < layout.mixing {
            let step = rng.random_range(1..k) as u32;
            *l = (*l + step) % k as u32;
        }
    }
    labels
}

/// `v` summed over each row's contacts in another community.
pub fn between_sum(contacts: &Contacts, labels: &[u32], v: &[f64]) -> Vec<f64> {
    contacts
        .adj
        .iter()
        .enumerate()
        .map(|(i, a)| {
            a.iter()
                .filter(|&&j| labels[j as usize] != labels[i])
                .map(|&j| v[j as usize])
                .sum()
        })
        .collect()
}

/// Whether each row touches another community.
pub fn on_boundary(contacts: &Contacts, labels: &[u32]) -> Vec<bool> {
    contacts
        .adj
        .iter()
        .enumerate()
        .map(|(i, a)| a.iter().any(|&j| labels[j as usize] != labels[i]))
        .collect()
}

/// The boundary comparison's fits, and what its truth reads.
pub(super) struct Boundary {
    /// Competent receivers touching another community.
    pub(super) receivers: Vec<bool>,
    /// The seen dose over contacts between communities.
    pub(super) dose: Vec<f64>,
    /// The fits on the boundary rows, their arm weights put back on every row
    /// (zero away from a boundary).
    pub(super) avg: ThresholdAverage,
}

/// The effect of ligand reaching a receiver from another community, among the
/// competent receivers that touch one: those with that ligand against those
/// without, with the ligand from their own community and their community as
/// covariates.
///
/// Every nuisance fit, and the exposure threshold, reads the rows touching
/// another community alone. A cell away from every boundary differs from one
/// at a boundary in whatever being at a boundary brings, and a fit pooled over
/// both learns that difference from the neighbourhood covariates, which then
/// carry it into the comparison.
///
/// `lnorm` is the seen ligand over size factor and `x` the exposure read from
/// it over every contact.
#[allow(clippy::too_many_arguments)]
pub(super) fn boundary_estimate(
    tissue: &Tissue,
    labels: &[u32],
    lnorm: &[f64],
    x: &Exposure,
    comp: &[bool],
    response: &Response,
    rows: Rows<'_>,
    opts: &EffectOpts,
) -> anyhow::Result<Boundary> {
    let between = between_sum(&tissue.contacts, labels, lnorm);
    let all = tissue.contacts.sum(lnorm);
    let touching = on_boundary(&tissue.contacts, labels);
    let receivers: Vec<bool> = comp.iter().zip(&touching).map(|(&c, &b)| c && b).collect();
    let idx: Vec<usize> = (0..touching.len()).filter(|&i| touching[i]).collect();
    let pick = |v: &[f64]| idx.iter().map(|&i| v[i]).collect::<Vec<f64>>();

    let dose = pick(&between);
    let within = idx
        .iter()
        .map(|&i| (all[i] - between[i]).max(0.0).ln_1p())
        .collect();
    let k = labels.iter().copied().max().map_or(0, |l| l + 1);
    let mut covariates = vec![within];
    covariates.extend((1..k).map(|c| idx.iter().map(|&i| (labels[i] == c) as u8 as f64).collect()));
    let tau = choose_threshold(&dose, opts);
    let bx = Exposure {
        tau,
        t: exposed(&dose, tau),
        own: pick(&x.own),
        niche: pick(&x.niche),
        dose,
        covariates,
    };
    let r = Response {
        y: pick(&response.y),
        spill: pick(&response.spill),
        bleed: pick(&response.bleed),
    };
    let level = vec![idx.iter().map(|&i| receivers[i]).collect::<Vec<bool>>()];

    // The folds and the variance clusters of the rows kept, clusters renumbered.
    let mut renumber = HashMap::default();
    let cluster: Vec<u32> = idx
        .iter()
        .map(|&i| {
            let next = renumber.len() as u32;
            *renumber.entry(rows.cf.cluster[i]).or_insert(next)
        })
        .collect();
    let fold = idx.iter().map(|&i| rows.cf.fold[i]).collect();
    let cf = CrossFit::new(fold, rows.cf.k, cluster);
    let design = Design::new(rows.design.x.select_rows(idx.iter()), &cf);
    let pooled = vec![(0..idx.len() as u32).collect::<Vec<u32>>()];
    let sub = Rows {
        design: &design,
        steps: &[],
        cf: &cf,
        scope_rows: &pooled,
    };
    let taus = if opts.threshold.is_none() {
        thresholds(&bx, &level, opts)
    } else {
        Vec::new()
    };
    let mut avg = over_thresholds(
        sub,
        &bx,
        &taus,
        std::slice::from_ref(&r),
        Vec::new,
        &level,
        Sensitivity::NONE,
        opts,
    )?;
    for w in avg.arm_weights.iter_mut() {
        let mut full = vec![0.0; touching.len()];
        for (&i, &v) in idx.iter().zip(w.iter()) {
            full[i] = v;
        }
        *w = full;
    }
    Ok(Boundary {
        receivers,
        dose: between,
        avg,
    })
}

/// The boundary comparison's estimate and the truth it targets.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryResult {
    pub estimate: Estimate,
    pub truth: f64,
}

impl BoundaryResult {
    /// No layout, no boundary comparison.
    pub(super) fn none() -> Self {
        Self {
            estimate: Estimate::nan(0, 0),
            truth: f64::NAN,
        }
    }
}
