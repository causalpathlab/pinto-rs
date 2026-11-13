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
//! section, `w` the spot's mixture. Each term has variance `pve · V` over
//! spots (at each spot's dominant state), `V` being `log_rate_var`, and `E`
//! takes `1 − Σ pve`, the convention `pinto lra-simulate` uses. The type
//! term is scaled per gene, exactly, and centred (its level belongs to `a`).
//! A nuisance term is a few draws per gene, so it gets one scale from its
//! expected variance `Σ w (1 − w)` over the groups that are not clean (`w`
//! their shares of the spots): `pve · V` in expectation over genes, with a
//! gene's terms depending on its own draws alone. Nuisance terms are not
//! centred, so a clean donor or section keeps exactly 0. Genes outside every
//! programme have no `T` and carry the other terms at the same variance, so
//! nuisance is genome-wide.

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

/// The type term's `(shift, scale)` per gene, and one scale per nuisance
/// term, so each term has variance `pve · V` over spots: exactly for the
/// type term of each gene, and in expectation for a nuisance term.
pub struct Scaling {
    pub ty: Vec<(f64, f64)>,
    pub donor: f64,
    pub inter: f64,
    pub section: f64,
    pub noise: f64,
}

/// Expected variance over spots of a term that is iid N(0, 1) per group,
/// the groups holding shares `w` of the spots (a clean group's share set to
/// 0, its term being 0): `E[Σ w z² − (Σ w z)²] = Σ_g w_g (1 − w_g)`.
fn expected_variance(w: &[f64]) -> f64 {
    w.iter().map(|x| x * (1.0 - x)).sum()
}

pub fn scaling(p: &SimParams, sections: &[Section], truth: &Truth, genes: &Genes) -> Scaling {
    let v = p.log_rate_var;
    let n_states = truth.n_states();
    // shares of the spots by state, and by donor, section and (state, donor)
    // with clean groups at 0
    let n: f64 = sections.iter().map(|s| s.n()).sum::<usize>() as f64;
    let mut by_state = vec![0.0; n_states];
    let mut by_donor = vec![0.0; p.n_donors];
    let mut by_section = vec![0.0; sections.len()];
    let mut by_state_donor = vec![0.0; n_states * p.n_donors];
    for (s, sec) in sections.iter().enumerate() {
        let share = sec.n() as f64 / n;
        let donor_on = !p.clean_donors.contains(&sec.donor);
        if !p.clean_sections.contains(&s) {
            by_section[s] = share;
        }
        if donor_on {
            by_donor[sec.donor] += share;
        }
        for &c in &truth.dominant[s] {
            by_state[c] += 1.0 / n;
            if donor_on {
                by_state_donor[c * p.n_donors + sec.donor] += 1.0 / n;
            }
        }
    }
    let scale = |pve: f64, var: f64| {
        if var > 0.0 {
            (pve * v / var).sqrt()
        } else {
            0.0
        }
    };
    // the type term per gene, from the states' shares of the spots
    let ty = (0..p.n_genes)
        .map(|g| {
            let mean: f64 = (0..n_states)
                .map(|c| by_state[c] * genes.programme[c][g])
                .sum();
            let var: f64 = (0..n_states)
                .map(|c| by_state[c] * (genes.programme[c][g] - mean).powi(2))
                .sum();
            (mean, scale(p.pve_type, var))
        })
        .collect();
    Scaling {
        ty,
        donor: scale(p.pve_donor, expected_variance(&by_donor)),
        inter: scale(p.pve_inter, expected_variance(&by_state_donor)),
        section: scale(p.pve_section, expected_variance(&by_section)),
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
