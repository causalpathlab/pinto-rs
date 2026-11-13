//! The log rate of every gene in every spot, from terms scaled to their
//! proportion of variance explained (PVE):
//!
//! ```text
//! log μ[g,i,c] = a[g] + log s[i] + T[g,c] + D[g,d] + X[g,c,d] + S[g,s] + E[g,i]
//! μ[g,i]       = Σ_c w[i,c] · exp(log μ[g,i,c])
//! ```
//!
//! `T` is the cell-type effect, `D` donor-wide nuisance, `X` donor by type,
//! `S` per section, `E` spot noise; `i` spot, `c` state, `d` donor, `s`
//! section, `w` the spot's mixture. Each term is scaled so its variance
//! over spots (at each spot's dominant state) is `pve · V`, `V` being
//! `log_rate_var`, and `E` takes `1 − Σ pve`: for a pure spot the shares
//! are exact, the convention `pinto lra-simulate` uses. `T` is also centred,
//! its level belonging to `a`. The nuisance terms are only scaled, so a
//! clean donor or section keeps exactly 0. Genes outside every programme
//! have no `T` and carry the other terms at the same absolute variance, so
//! nuisance is genome-wide.

use super::field::mean_sd;
use super::panel::Section;
use super::params::SimParams;
use super::rng::{rng, Stream};
use super::truth::Truth;
use rand::rngs::SmallRng;
use rand::seq::index::sample;
use rand::RngExt;
use rand_distr::{Distribution, StandardNormal};

fn normal(r: &mut SmallRng) -> f64 {
    StandardNormal.sample(r)
}

fn normals(mut r: SmallRng, n: usize) -> Vec<f64> {
    (0..n).map(|_| normal(&mut r)).collect()
}

/// The log-rate terms, in the order of [`Terms`].
#[derive(Clone, Copy, Debug)]
pub enum Term {
    Type,
    Donor,
    Inter,
    Section,
    Noise,
}

/// Which terms are on, indexed by [`Term`].
pub type Terms = [bool; 5];

/// The terms a run keeps: all, or none of the nuisance with `nuisance_off`.
pub fn active(p: &SimParams) -> Terms {
    let n = !p.nuisance_off;
    [true, n, n, n, true]
}

pub struct Genes {
    pub names: Vec<Box<str>>,
    /// Baseline log rate per unit depth.
    pub a: Vec<f64>,
    /// NB dispersion: variance = μ + dispersion · μ².
    pub dispersion: Vec<f64>,
    /// `programme[c][g]`: state `c`'s programme, before scaling.
    pub programme: Vec<Vec<f64>>,
    /// The first state whose programme holds gene `g`, if any.
    pub programme_of: Vec<Option<usize>>,
    /// The copula factor gene `g` loads on.
    pub factor: Vec<usize>,
}

/// `b` rotated toward a random direction so its cosine with `b` is `cosine`,
/// at `b`'s norm. The direction is a random programme of `k` genes, made
/// orthogonal to `b`.
pub fn rotate_to_cosine(b: &[f64], cosine: f64, k: usize, r: &mut SmallRng) -> Vec<f64> {
    let n = b.len();
    let norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let u: Vec<f64> = b.iter().map(|v| v / norm).collect();
    let mut xi = vec![0.0; n];
    for g in sample(r, n, k.min(n)) {
        xi[g] = 1.0;
    }
    let proj: f64 = xi.iter().zip(&u).map(|(x, y)| x * y).sum();
    xi.iter_mut().zip(&u).for_each(|(x, y)| *x -= proj * y);
    let xn = xi
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    let sin = (1.0 - cosine * cosine).max(0.0).sqrt();
    u.iter()
        .zip(&xi)
        .map(|(ui, xv)| norm * (cosine * ui + sin * xv / xn))
        .collect()
}

pub fn draw_genes(p: &SimParams, truth: &Truth) -> Genes {
    let mut r = rng(p.seed, Stream::GeneBaselines, &[]);
    let a = (0..p.n_genes)
        .map(|_| p.log_mean + p.log_mean_sd * normal(&mut r))
        .collect();
    let dispersion = (0..p.n_genes)
        .map(|_| p.dispersion * (p.dispersion_sd * normal(&mut r)).exp())
        .collect();
    let mut r = rng(p.seed, Stream::Programmes, &[]);
    let mut programme: Vec<Vec<f64>> = Vec::with_capacity(truth.n_states());
    for st in &truth.states {
        let prog = if st.owner.is_none() {
            let mut v = vec![0.0; p.n_genes];
            for g in sample(&mut r, p.n_genes, p.programme_genes) {
                v[g] = 1.0;
            }
            v
        } else {
            // a donor-only state: the first shared state's programme, rotated
            rotate_to_cosine(&programme[0], p.patch_cosine, p.programme_genes, &mut r)
        };
        programme.push(prog);
    }
    let programme_of: Vec<Option<usize>> = (0..p.n_genes)
        .map(|g| (0..programme.len()).find(|&c| programme[c][g] > 0.0))
        .collect();
    let mut r = rng(p.seed, Stream::Loading, &[]);
    let factor = programme_of
        .iter()
        .map(|c| c.map_or_else(|| r.random_range(0..p.copula_rank), |c| c % p.copula_rank))
        .collect();
    Genes {
        names: (0..p.n_genes)
            .map(|g| format!("GENE{}", g + 1).into())
            .collect(),
        a,
        dispersion,
        programme,
        programme_of,
        factor,
    }
}

/// Nuisance draws before scaling: standard normal, then 0 for clean donors
/// and sections. Each gene, and each gene and state, has its own stream.
pub struct Nuisance {
    /// `donor[g][d]`
    pub donor: Vec<Vec<f64>>,
    /// `inter[g][c][d]`
    pub inter: Vec<Vec<Vec<f64>>>,
    /// `section[g][s]`
    pub section: Vec<Vec<f64>>,
}

pub fn draw_nuisance(p: &SimParams, n_states: usize) -> Nuisance {
    let clean_d: Vec<bool> = (0..p.n_donors)
        .map(|d| p.clean_donors.contains(&d))
        .collect();
    let clean_s: Vec<bool> = (0..p.n_sections())
        .map(|s| p.clean_sections.contains(&s))
        .collect();
    let masked = |v: Vec<f64>, clean: &[bool]| -> Vec<f64> {
        v.into_iter()
            .zip(clean)
            .map(|(x, &c)| if c { 0.0 } else { x })
            .collect()
    };
    let nd = p.n_donors;
    Nuisance {
        donor: (0..p.n_genes)
            .map(|g| masked(normals(rng(p.seed, Stream::Donor, &[g]), nd), &clean_d))
            .collect(),
        inter: (0..p.n_genes)
            .map(|g| {
                (0..n_states)
                    .map(|c| masked(normals(rng(p.seed, Stream::Inter, &[g, c]), nd), &clean_d))
                    .collect()
            })
            .collect(),
        section: (0..p.n_genes)
            .map(|g| {
                masked(
                    normals(rng(p.seed, Stream::Section, &[g]), p.n_sections()),
                    &clean_s,
                )
            })
            .collect(),
    }
}

/// Spot depths per section, log-normal around 1.
pub fn draw_depth(p: &SimParams, sections: &[Section]) -> Vec<Vec<f64>> {
    (0..sections.len())
        .map(|s| {
            normals(rng(p.seed, Stream::Depth, &[s]), sections[s].n())
                .into_iter()
                .map(|z| (p.depth_sd * z).exp())
                .collect()
        })
        .collect()
}

/// Spot noise of gene `g` in section `s`, standard normal per spot.
pub fn spot_noise(p: &SimParams, s: usize, g: usize, n: usize) -> Vec<f64> {
    normals(rng(p.seed, Stream::Noise, &[s, g]), n)
}

/// Per gene: the type term's `(shift, scale)` and the nuisance terms'
/// scales, so each scaled term has variance `pve · V` over spots.
pub struct Scaling {
    pub ty: Vec<(f64, f64)>,
    pub donor: Vec<f64>,
    pub inter: Vec<f64>,
    pub section: Vec<f64>,
    pub noise: f64,
}

pub fn scaling(
    p: &SimParams,
    sections: &[Section],
    truth: &Truth,
    genes: &Genes,
    nu: &Nuisance,
) -> Scaling {
    use rayon::prelude::*;
    let v = p.log_rate_var;
    let scale = |pve: f64, sd: f64| if sd > 0.0 { (pve * v).sqrt() / sd } else { 0.0 };
    // a term's values over every spot, at the spot's dominant state
    let over_spots = |f: &dyn Fn(usize, usize, usize, usize) -> f64| -> Vec<f64> {
        sections
            .iter()
            .enumerate()
            .flat_map(|(s, sec)| (0..sec.n()).map(move |i| (s, sec.donor, i)))
            .map(|(s, d, i)| f(s, d, i, truth.dominant[s][i]))
            .collect()
    };
    let per_gene: Vec<[f64; 5]> = (0..p.n_genes)
        .into_par_iter()
        .map(|g| {
            let (mt, st) = mean_sd(&over_spots(&|_, _, _, c| genes.programme[c][g]));
            let (_, sd) = mean_sd(&over_spots(&|_, d, _, _| nu.donor[g][d]));
            let (_, sx) = mean_sd(&over_spots(&|_, d, _, c| nu.inter[g][c][d]));
            let (_, ss) = mean_sd(&over_spots(&|s, _, _, _| nu.section[g][s]));
            [
                mt,
                scale(p.pve_type, st),
                scale(p.pve_donor, sd),
                scale(p.pve_inter, sx),
                scale(p.pve_section, ss),
            ]
        })
        .collect();
    Scaling {
        ty: per_gene.iter().map(|t| (t[0], t[1])).collect(),
        donor: per_gene.iter().map(|t| t[2]).collect(),
        inter: per_gene.iter().map(|t| t[3]).collect(),
        section: per_gene.iter().map(|t| t[4]).collect(),
        noise: (p.pve_noise() * v).sqrt(),
    }
}

/// The scaled terms of one gene in one section: per state, the type and
/// donor-by-type terms; for the section, the donor and section terms; per
/// spot, the noise.
pub struct Components {
    pub a: f64,
    pub ty: Vec<f64>,
    pub inter: Vec<f64>,
    pub donor: f64,
    pub section: f64,
    pub noise: Vec<f64>,
}

impl Components {
    /// The log rate per spot without depth, from the terms in `on`:
    /// `log Σ_c w[i,c] exp(...)`, the terms outside the mixture added.
    pub fn log_rate(&self, on: Terms, mixture: &[Vec<f64>]) -> Vec<f64> {
        let w = |t: Term| if on[t as usize] { 1.0 } else { 0.0 };
        let e: Vec<f64> = self
            .ty
            .iter()
            .zip(&self.inter)
            .map(|(t, x)| (w(Term::Type) * t + w(Term::Inter) * x).exp())
            .collect();
        let outside = self.a + w(Term::Donor) * self.donor + w(Term::Section) * self.section;
        mixture
            .iter()
            .zip(&self.noise)
            .map(|(wi, &z)| {
                let inside: f64 = wi.iter().zip(&e).map(|(a, b)| a * b).sum();
                outside + w(Term::Noise) * z + inside.ln()
            })
            .collect()
    }
}
