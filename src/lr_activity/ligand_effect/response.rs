//! The response programme and the response-side covariates read from it.

use super::regression::standardize;
use super::tissue::Tissue;
use rayon::prelude::*;

/// A programme: `(weight, non-zero counts by row)` for each gene.
pub type Targets<'a> = [(f64, &'a [(u32, f32)])];

/// The response side of a pair's per-row quantities: rebuilt for each
/// programme (each weighting, the background).
pub struct Response {
    pub y: Vec<f64>,
    /// The contact-neighbour mean response, standardized.
    pub spill: Vec<f64>,
    /// The neighbours' expected programme counts on the receiver's scale
    /// ([`programme_columns`]), standardized.
    pub bleed: Vec<f64>,
}

impl Response {
    /// The response `y` with its spillover and the given bleed, both
    /// standardized once here for every fit that reads them.
    pub fn new(tissue: &Tissue, y: Vec<f64>, bleed: &[f64]) -> Self {
        Self {
            spill: standardize(&tissue.contacts.mean(&y)),
            bleed: standardize(bleed),
            y,
        }
    }

    pub fn build(tissue: &Tissue, targets: &Targets<'_>) -> Self {
        let (genes, weights) = genes_and_weights(targets);
        Self::build_many(tissue, &genes, &[weights])
            .pop()
            .expect("one weighting")
    }

    /// One response side per weighting of the same genes, each gene read
    /// once; a zero weight leaves the gene out of that weighting.
    pub fn build_many(
        tissue: &Tissue,
        genes: &[&[(u32, f32)]],
        weightings: &[Vec<f64>],
    ) -> Vec<Self> {
        let sf = &tissue.size_factor;
        let (ys, bleeds) = programme_columns(
            genes,
            weightings,
            &tissue.identity_bin,
            &tissue.contacts,
            sf,
        );
        ys.into_iter()
            .zip(bleeds)
            .map(|(y, bleed)| Self::new(tissue, y, &bleed))
            .collect()
    }
}

/// One gene's non-zero counts by row.
type Gene<'t> = &'t [(u32, f32)];

/// A programme's genes and weights, apart.
pub(super) fn genes_and_weights<'t>(targets: &Targets<'t>) -> (Vec<Gene<'t>>, Vec<f64>) {
    targets.iter().map(|&(w, nz)| (nz, w)).unzip()
}

/// The response and the bleed of each weighting of `genes`, in one pass
/// over the genes: each gene's `log1p` counts and expected bleed are
/// computed once and added to every weighting that gives it a non-zero
/// weight.
///
/// The response is the signed weighted mean of `log1p(x / size factor)`
/// over the programme, accumulated over non-zeros only (a zero count
/// contributes `log1p(0) = 0`).
///
/// The bleed is what contact neighbours' programme transcripts are expected to add to
/// each receiver's response if segmentation handed it all of them: per
/// gene, the neighbours' expected counts (their size factor times their
/// identity bin's mean count / size factor) summed and normalized by the
/// receiver's size factor, through the programme's signed `log1p` mean.
///
/// Expected rather than observed counts: a neighbour's observed programme
/// is partly its own response to senders it shares with the receiver, so
/// adjusting for it would absorb a real effect. The expectation keeps what
/// drives bleed and is not downstream of exposure: how many contacts,
/// their depth and identity, and the receiver's depth.
pub fn programme_columns(
    genes: &[&[(u32, f32)]],
    weightings: &[Vec<f64>],
    identity_bin: &[usize],
    contacts: &super::contacts::Contacts,
    sf: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = sf.len();
    let nb = identity_bin.iter().copied().max().map_or(0, |b| b + 1);
    let mut cells = vec![0usize; nb];
    for &b in identity_bin {
        cells[b] += 1;
    }
    let mut ys = vec![vec![0.0; n]; weightings.len()];
    let mut bleeds = vec![vec![0.0; n]; weightings.len()];
    let mut m = vec![0.0; nb];
    for (g, nz) in genes.iter().enumerate() {
        let on: Vec<usize> = (0..weightings.len())
            .filter(|&k| weightings[k][g] != 0.0)
            .collect();
        if on.is_empty() {
            continue;
        }
        m.fill(0.0);
        for &(r, v) in *nz {
            let r = r as usize;
            let x = v as f64 / sf[r];
            m[identity_bin[r]] += x;
            let l = x.ln_1p();
            for &k in &on {
                ys[k][r] += weightings[k][g] * l;
            }
        }
        for (mb, &c) in m.iter_mut().zip(&cells) {
            *mb /= c.max(1) as f64;
        }
        let b = expected_bleed(&m, identity_bin, contacts, sf);
        for &k in &on {
            let w = weightings[k][g];
            for (o, x) in bleeds[k].iter_mut().zip(&b) {
                *o += w * x;
            }
        }
    }
    for (k, w) in weightings.iter().enumerate() {
        let total: f64 = w.iter().map(|w| w.abs()).sum::<f64>().max(1e-12);
        for v in ys[k].iter_mut().chain(bleeds[k].iter_mut()) {
            *v /= total;
        }
    }
    (ys, bleeds)
}

/// One gene's `log1p(Σ_contacts sf_j m_{bin(j)} / sf_i)`, `m` being the
/// gene's mean count / size factor per identity bin.
pub fn expected_bleed(
    m: &[f64],
    identity_bin: &[usize],
    contacts: &super::contacts::Contacts,
    sf: &[f64],
) -> Vec<f64> {
    contacts
        .adj
        .par_iter()
        .enumerate()
        .map(|(i, adj)| {
            let e: f64 = adj
                .iter()
                .map(|&j| sf[j as usize] * m[identity_bin[j as usize]])
                .sum();
            (e / sf[i]).ln_1p()
        })
        .collect()
}
