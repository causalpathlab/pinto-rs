//! The hidden spatial fields and how they enter a scenario.

use super::calibrate::{Producers, SimBase};
use super::scenario::{ConfoundShape, ConfounderKind, Scenario};
use crate::lr_activity::ligand_effect::confounders::fourier_features;
use crate::lr_activity::ligand_effect::regression::standardize;
use crate::lr_activity::ligand_effect::tissue::Tissue;
use crate::util::common::HashMap;
use legume_numeric::matrix::rand_util::mix_seed;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Random Fourier features at `scale` contact lengths.
fn fourier_field(tissue: &Tissue, scale: f64, seed: u64) -> Vec<f64> {
    let d = 64;
    let feats = fourier_features(
        &tissue.coords,
        &tissue.section,
        &tissue.unit,
        scale,
        d,
        seed,
    );
    let mut rng = SmallRng::seed_from_u64(mix_seed(seed, 7));
    let a: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
    let z: Vec<f64> = (0..tissue.n())
        .map(|i| feats.iter().zip(&a).map(|(f, a)| f[i] * a).sum())
        .collect();
    standardize(&z)
}

/// A sum of Gaussian bumps of width `scale` contact lengths at random
/// cells, with random signs and amplitudes in `[0.5, 1.5]`: about two per
/// width² of tissue (a cell per contact length²), cut at three widths.
fn bump_field(tissue: &Tissue, scale: f64, seed: u64) -> Vec<f64> {
    let n = tissue.n();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut z = vec![0.0; n];
    for s in 0..tissue.unit.len() {
        let rows: Vec<usize> = (0..n)
            .filter(|&i| tissue.section[i] as usize == s)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let w = scale * tissue.unit[s];
        let cut = 3.0 * w;
        let key = |x: f64, y: f64| ((x / cut).floor() as i64, (y / cut).floor() as i64);
        let k = ((2.0 * rows.len() as f64 / (scale * scale)).round() as usize).max(2);
        // Bumps `[x, y, signed amplitude]` by grid square.
        let mut grid: HashMap<(i64, i64), Vec<[f64; 3]>> = HashMap::default();
        for _ in 0..k {
            let c = rows[rng.random_range(0..rows.len())];
            let (x, y) = tissue.coords[c];
            let sign = if rng.random::<bool>() { 1.0 } else { -1.0 };
            let amp = sign * (0.5 + rng.random::<f64>());
            grid.entry(key(x as f64, y as f64))
                .or_default()
                .push([x as f64, y as f64, amp]);
        }
        for &i in &rows {
            let (x, y) = (tissue.coords[i].0 as f64, tissue.coords[i].1 as f64);
            let (gx, gy) = key(x, y);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for &[bx, by, amp] in grid.get(&(gx + dx, gy + dy)).into_iter().flatten() {
                        let d2 = (x - bx).powi(2) + (y - by).powi(2);
                        if d2 < cut * cut {
                            z[i] += amp * (-d2 / (2.0 * w * w)).exp();
                        }
                    }
                }
            }
        }
    }
    standardize(&z)
}

/// A real gene's `log1p(count / sf)`, smoothed by a Gaussian of width
/// `scale` contact lengths (the bump field's kernel), cut at three widths,
/// within each section.
fn gene_field(tissue: &Tissue, counts: &[f64], scale: f64) -> Vec<f64> {
    let n = tissue.n();
    let sf = &tissue.size_factor;
    let v: Vec<f64> = (0..n).map(|i| (counts[i] / sf[i]).ln_1p()).collect();
    let mut z = vec![0.0; n];
    for s in 0..tissue.unit.len() {
        let w = scale * tissue.unit[s];
        let cut = 3.0 * w;
        let key = |i: usize| {
            let (x, y) = tissue.coords[i];
            (
                (x as f64 / cut).floor() as i64,
                (y as f64 / cut).floor() as i64,
            )
        };
        let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::default();
        for i in (0..n).filter(|&i| tissue.section[i] as usize == s) {
            grid.entry(key(i)).or_default().push(i);
        }
        let rows: Vec<usize> = grid.values().flatten().copied().collect();
        let smoothed: Vec<(usize, f64)> = rows
            .par_iter()
            .map(|&i| {
                let (x, y) = (tissue.coords[i].0 as f64, tissue.coords[i].1 as f64);
                let (gx, gy) = key(i);
                let (mut num, mut den) = (0.0, 0.0);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for &j in grid.get(&(gx + dx, gy + dy)).into_iter().flatten() {
                            let d2 = (x - tissue.coords[j].0 as f64).powi(2)
                                + (y - tissue.coords[j].1 as f64).powi(2);
                            if d2 < cut * cut {
                                let k = (-d2 / (2.0 * w * w)).exp();
                                num += k * v[j];
                                den += k;
                            }
                        }
                    }
                }
                (i, num / den)
            })
            .collect();
        for (i, x) in smoothed {
            z[i] = x;
        }
    }
    standardize(&z)
}

/// What every scenario of a pair shares: its calibrated producers
/// ([`super::calibrate_producers`]) and the real-gene field at each scale the
/// grid uses ([`super::ConfounderKind::RealGene`]), computed once.
pub struct SimShared<'a> {
    pub producers: Vec<Producers>,
    /// `(scale, field)` ([`gene_fields`]); a scale missing here is
    /// computed when a scenario needs it.
    pub gene_fields: &'a [(f64, Vec<f64>)],
}

/// The real-gene field at each distinct scale; empty without a gene.
pub fn gene_fields(tissue: &Tissue, gene: Option<&[f64]>, scales: &[f64]) -> Vec<(f64, Vec<f64>)> {
    let Some(g) = gene else {
        return Vec::new();
    };
    let mut out: Vec<(f64, Vec<f64>)> = Vec::new();
    for &scale in scales {
        if !out.iter().any(|(s, _)| *s == scale) {
            out.push((scale, gene_field(tissue, g, scale)));
        }
    }
    out
}

/// The scenario's hidden field through its shape, standardized: `f(Z)` per
/// row, so a field PVE sets its share of variance whatever the shape.
pub(super) fn confounding(
    tissue: &Tissue,
    base: &SimBase<'_>,
    shared: &SimShared<'_>,
    scen: &Scenario,
    seed: u64,
) -> anyhow::Result<Vec<f64>> {
    let n = tissue.n();
    if scen.ligand_field_pve == 0.0 && scen.response_field_pve == 0.0 {
        return Ok(vec![0.0; n]);
    }
    let z = match scen.confounder {
        ConfounderKind::Fourier => fourier_field(tissue, scen.scale, seed),
        ConfounderKind::Bumps => bump_field(tissue, scen.scale, seed),
        ConfounderKind::RealGene => {
            let g = base
                .confounder_gene
                .ok_or_else(|| anyhow::anyhow!("no real gene given for the confounder"))?;
            match shared.gene_fields.iter().find(|(s, _)| *s == scen.scale) {
                Some((_, z)) => z.clone(),
                None => gene_field(tissue, g, scen.scale),
            }
        }
    };
    let shaped: Vec<f64> = (0..n)
        .map(|i| match scen.shape {
            ConfoundShape::Linear => z[i],
            ConfoundShape::Quadratic => z[i] * z[i] - 1.0,
        })
        .collect();
    Ok(standardize(&shaped))
}
