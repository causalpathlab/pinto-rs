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

/// A real gene's `log1p(count / sf)`, smoothed by `scale / 2` rounds of
/// averaging each row with its contacts.
fn gene_field(tissue: &Tissue, counts: &[f64], scale: f64) -> Vec<f64> {
    let n = tissue.n();
    let sf = &tissue.size_factor;
    let mut v: Vec<f64> = (0..n).map(|i| (counts[i] / sf[i]).ln_1p()).collect();
    for _ in 0..((scale / 2.0).round() as usize).max(1) {
        let s = tissue.contacts.sum(&v);
        for i in 0..n {
            v[i] = (v[i] + s[i]) / (1 + tissue.contacts.degree(i)) as f64;
        }
    }
    standardize(&v)
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

/// The scenario's hidden field through its shape: `f(Z)` per row.
pub(super) fn confounding(
    tissue: &Tissue,
    base: &SimBase<'_>,
    shared: &SimShared<'_>,
    scen: &Scenario,
    seed: u64,
) -> anyhow::Result<Vec<f64>> {
    let n = tissue.n();
    if scen.confound_ligand == 0.0 && scen.confound_response == 0.0 {
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
    Ok((0..n)
        .map(|i| match scen.shape {
            ConfoundShape::Linear => z[i],
            ConfoundShape::Quadratic => z[i] * z[i] - 1.0,
        })
        .collect())
}

/// `v` read at each row's point plus an offset of `shift` contact lengths
/// in a random direction: the value of the nearest row in the same
/// section. A grid of one contact length per section, searched in growing
/// square rings until no closer row can lie outside.
pub(super) fn shift_field(tissue: &Tissue, v: &[f64], shift: f64, seed: u64) -> Vec<f64> {
    let n = tissue.n();
    let phi = SmallRng::seed_from_u64(seed).random::<f64>() * std::f64::consts::TAU;
    let (dx, dy) = (phi.cos(), phi.sin());
    let cell_of = |s: u32, x: f64, y: f64| {
        let g = tissue.unit[s as usize];
        (s, (x / g).floor() as i64, (y / g).floor() as i64)
    };
    let mut grid: HashMap<(u32, i64, i64), Vec<u32>> = HashMap::default();
    for i in 0..n {
        let (x, y) = tissue.coords[i];
        grid.entry(cell_of(tissue.section[i], x as f64, y as f64))
            .or_default()
            .push(i as u32);
    }
    // Far enough to cross any section from any point near it.
    let max_ring = 1 + (grid.len() as f64).sqrt() as i64 * 4;
    (0..n)
        .map(|i| {
            let s = tissue.section[i];
            let g = tissue.unit[s as usize];
            let (x, y) = tissue.coords[i];
            let (qx, qy) = (x as f64 + shift * g * dx, y as f64 + shift * g * dy);
            let (_, cx, cy) = cell_of(s, qx, qy);
            let mut best = (f64::INFINITY, i);
            for r in 0..=max_ring {
                for gx in (cx - r)..=(cx + r) {
                    for gy in (cy - r)..=(cy + r) {
                        if (gx - cx).abs() != r && (gy - cy).abs() != r {
                            continue;
                        }
                        for &j in grid.get(&(s, gx, gy)).into_iter().flatten() {
                            let (jx, jy) = tissue.coords[j as usize];
                            let d2 = (jx as f64 - qx).powi(2) + (jy as f64 - qy).powi(2);
                            if d2 < best.0 {
                                best = (d2, j as usize);
                            }
                        }
                    }
                }
                if best.0 <= (r as f64 * g).powi(2) {
                    break;
                }
            }
            v[best.1]
        })
        .collect()
}
