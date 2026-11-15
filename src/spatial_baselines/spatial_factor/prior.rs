//! The sparse factor prior both engines share: the graph Laplacian `L` of
//! the per-section spatial graph, its spectrum (exact for small sections,
//! stochastic Lanczos quadrature otherwise), and conjugate-gradient solves
//! of the systems `a_s I + b L` the engines meet, `a_s` constant within a
//! section `s`.
//!
//! The prior of a factor is `z ~ N(0, v R(rho))`, `R(rho) = (I + rho
//! L)^{-1} / cbar(rho)`, `cbar(rho) = tr((I + rho L)^{-1}) / N`: the
//! regularised-Laplacian kernel on a graph (Smola and Kondor 2003, COLT,
//! doi:10.1007/978-3-540-45167-9_12) as a proper Gaussian Markov random
//! field (Rue and Held 2005, "Gaussian Markov Random Fields", Chapman and
//! Hall/CRC, doi:10.1201/9780203492024), scaled so its mean marginal
//! variance is one. No `N × N` matrix is formed: every determinant and
//! trace is a sum `sum_i f(lambda_i(L_s))` over each section's eigenvalues.

use super::kit::sym_eig_desc;
use crate::spatial_baselines::graph::Neighbours;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// Steps of a conjugate-gradient solve at most.
pub const CG_MAX_ITER: usize = 500;

/// Rows above which one matrix-vector product runs in parallel.
const PAR_ROWS: usize = 8192;

/// The Laplacian `L = D - W` of a symmetric weighted graph whose edges stay
/// within sections, as compressed rows.
#[derive(Debug, Clone)]
pub struct Laplacian {
    offsets: Vec<usize>,
    index: Vec<u32>,
    weight: Vec<f64>,
    /// Weighted degree of each node, `D`.
    pub degree: Vec<f64>,
    /// Each node's section.
    pub section: Vec<usize>,
    /// The nodes of each section, in order.
    pub sections: Vec<Vec<usize>>,
    /// Each node's place in its section's list.
    place: Vec<usize>,
}

impl Laplacian {
    /// From symmetric neighbour lists (each edge listed at both ends), each
    /// edge of weight one.
    pub fn from_neighbours(nb: &Neighbours, section: &[usize], n_sections: usize) -> Self {
        let lists = nb
            .index
            .iter()
            .enumerate()
            .map(|(i, js)| js.iter().filter(|&&j| j != i).map(|&j| (j, 1.0)).collect());
        Self::from_lists(lists.collect(), section, n_sections)
    }

    /// From undirected weighted edges `(a, b)`, each listed once (a
    /// super-graph's), on `n` nodes.
    pub fn from_edges(
        n: usize,
        edges: &[(usize, usize)],
        weight: &[f64],
        section: &[usize],
        n_sections: usize,
    ) -> Self {
        let mut lists: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (&(a, b), &w) in edges.iter().zip(weight) {
            if a != b {
                lists[a].push((b, w));
                lists[b].push((a, w));
            }
        }
        Self::from_lists(lists, section, n_sections)
    }

    fn from_lists(lists: Vec<Vec<(usize, f64)>>, section: &[usize], n_sections: usize) -> Self {
        let n = lists.len();
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        let edges: usize = lists.iter().map(Vec::len).sum();
        let mut index = Vec::with_capacity(edges);
        let mut weight = Vec::with_capacity(edges);
        let mut degree = Vec::with_capacity(n);
        for (i, l) in lists.into_iter().enumerate() {
            let mut d = 0.0;
            for (j, w) in l {
                debug_assert_eq!(section[i], section[j], "an edge across sections");
                index.push(j as u32);
                weight.push(w);
                d += w;
            }
            degree.push(d);
            offsets.push(index.len());
        }
        let mut sections = vec![Vec::new(); n_sections];
        let mut place = vec![0; n];
        for (i, &s) in section.iter().enumerate() {
            place[i] = sections[s].len();
            sections[s].push(i);
        }
        Laplacian {
            place,
            offsets,
            index,
            weight,
            degree,
            section: section.to_vec(),
            sections,
        }
    }

    pub fn n(&self) -> usize {
        self.degree.len()
    }

    /// Stored off-diagonal entries (each edge twice).
    pub fn nnz(&self) -> usize {
        self.index.len()
    }

    /// Mean weighted degree.
    pub fn mean_degree(&self) -> f64 {
        self.degree.iter().sum::<f64>() / self.n().max(1) as f64
    }

    /// Total edge weight, each undirected edge once.
    pub fn total_weight(&self) -> f64 {
        self.degree.iter().sum::<f64>() / 2.0
    }

    /// `(L x)_i`.
    fn row(&self, i: usize, x: &[f64]) -> f64 {
        let r = self.offsets[i]..self.offsets[i + 1];
        let off: f64 = self.index[r.clone()]
            .iter()
            .zip(&self.weight[r])
            .map(|(&j, &w)| w * x[j as usize])
            .sum();
        self.degree[i] * x[i] - off
    }

    /// `y = L x`.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        if x.len() >= PAR_ROWS {
            y.par_iter_mut()
                .enumerate()
                .for_each(|(i, yi)| *yi = self.row(i, x));
        } else {
            for (i, yi) in y.iter_mut().enumerate() {
                *yi = self.row(i, x);
            }
        }
    }

    /// `x^T L x`.
    pub fn quad(&self, x: &[f64]) -> f64 {
        (0..self.n()).map(|i| x[i] * self.row(i, x)).sum()
    }

    /// Section `s`'s block `L_s` as a dense matrix (small sections only).
    fn dense_block(&self, s: usize) -> DMatrix<f64> {
        let cells = &self.sections[s];
        let k = cells.len();
        let mut m = DMatrix::zeros(k, k);
        for (a, &i) in cells.iter().enumerate() {
            m[(a, a)] = self.degree[i];
            for e in self.offsets[i]..self.offsets[i + 1] {
                let b = self.place[self.index[e] as usize];
                m[(a, b)] -= self.weight[e];
            }
        }
        m
    }

    /// `y = L_s x` on section `s`'s nodes, in their order.
    fn apply_block(&self, s: usize, x: &[f64], y: &mut [f64]) {
        for (a, &i) in self.sections[s].iter().enumerate() {
            let mut v = self.degree[i] * x[a];
            for e in self.offsets[i]..self.offsets[i + 1] {
                v -= self.weight[e] * x[self.place[self.index[e] as usize]];
            }
            y[a] = v;
        }
    }
}

/// How the spectrum of each section's Laplacian is held.
#[derive(Debug, Clone, Copy)]
pub struct SpectrumOpts {
    /// Sections of at most this many nodes take their exact eigenvalues.
    pub exact_eig_max: usize,
    /// Rademacher probes per larger section.
    pub probes: usize,
    /// Lanczos steps per probe.
    pub steps: usize,
    pub seed: u64,
}

/// Quadrature nodes and weights per section: `sum_i f(lambda_i(L_s))` is
/// `sum_j w_j f(theta_j)` over section `s`'s nodes, exactly (the
/// eigenvalues, weight one) or by stochastic Lanczos quadrature (Ubaru,
/// Chen and Saad 2017, SIAM J. Matrix Anal. Appl. 38:1075,
/// doi:10.1137/16M1104974; the trace estimator of Hutchinson 1989,
/// Commun. Stat. Simul. Comput. 18:1059, doi:10.1080/03610918908812806,
/// republished in 1990 as 19:433). The same nodes serve every parameter
/// value, so every estimate is a deterministic function of the seed.
#[derive(Debug, Clone)]
pub struct Spectrum {
    pub nodes: Vec<Vec<(f64, f64)>>,
    /// Whether each section's nodes are its exact eigenvalues.
    pub exact: Vec<bool>,
}

impl Spectrum {
    pub fn new(lap: &Laplacian, o: &SpectrumOpts) -> Self {
        let per: Vec<(Vec<(f64, f64)>, bool)> = (0..lap.sections.len())
            .into_par_iter()
            .map(|s| {
                let k = lap.sections[s].len();
                if k == 0 {
                    (Vec::new(), true)
                } else if k <= o.exact_eig_max.max(1) {
                    let (vals, _) = sym_eig_desc(&lap.dense_block(s));
                    (vals.into_iter().map(|v| (v.max(0.0), 1.0)).collect(), true)
                } else {
                    (slq(lap, s, o), false)
                }
            })
            .collect();
        let (nodes, exact) = per.into_iter().unzip();
        Spectrum { nodes, exact }
    }

    /// `sum_i f(lambda_i(L_s))`.
    pub fn sum(&self, s: usize, f: impl Fn(f64) -> f64) -> f64 {
        self.nodes[s].iter().map(|&(t, w)| w * f(t)).sum()
    }

    /// `sum_s sum_i f(lambda_i(L_s))`.
    pub fn total(&self, f: impl Fn(f64) -> f64) -> f64 {
        (0..self.nodes.len()).map(|s| self.sum(s, &f)).sum()
    }

    /// Nodes over all sections.
    #[cfg(test)]
    pub fn n(&self) -> f64 {
        self.total(|_| 1.0)
    }

    /// `cbar(rho) = tr((I + rho L)^{-1}) / N` and `d log cbar / d log rho`.
    pub fn cbar(&self, rho: f64, n: usize) -> (f64, f64) {
        let c = self.total(|t| 1.0 / (1.0 + rho * t)) / n as f64;
        let dc = -self.total(|t| t / (1.0 + rho * t).powi(2)) / n as f64;
        (c, rho * dc / c)
    }
}

/// Stochastic Lanczos quadrature nodes of section `s`: `o.probes`
/// Rademacher probes, `o.steps` Lanczos steps each (no
/// reorthogonalisation), each tridiagonal's eigenvalues the nodes and
/// `N_s / probes` times the squared first entries of its eigenvectors the
/// weights.
fn slq(lap: &Laplacian, s: usize, o: &SpectrumOpts) -> Vec<(f64, f64)> {
    let k = lap.sections[s].len();
    let probes = o.probes.max(1);
    (0..probes)
        .into_par_iter()
        .flat_map_iter(|p| {
            let mut rng = SmallRng::seed_from_u64(
                o.seed
                    ^ (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(((s as u64) << 20) | (p as u64 + 1))),
            );
            let scale = 1.0 / (k as f64).sqrt();
            let mut v: Vec<f64> = (0..k)
                .map(|_| if rng.random::<bool>() { scale } else { -scale })
                .collect();
            let mut v_prev = vec![0.0; k];
            let mut w = vec![0.0; k];
            let (mut alpha, mut beta) = (Vec::new(), Vec::new());
            let mut b_prev = 0.0;
            for _ in 0..o.steps.max(1).min(k) {
                lap.apply_block(s, &v, &mut w);
                let a: f64 = w.iter().zip(&v).map(|(x, y)| x * y).sum();
                for i in 0..k {
                    w[i] -= a * v[i] + b_prev * v_prev[i];
                }
                alpha.push(a);
                let b = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                if b < 1e-12 {
                    break;
                }
                beta.push(b);
                for i in 0..k {
                    v_prev[i] = v[i];
                    v[i] = w[i] / b;
                }
                b_prev = b;
            }
            let m = alpha.len();
            let t = DMatrix::from_fn(m, m, |i, j| {
                if i == j {
                    alpha[i]
                } else if i + 1 == j || j + 1 == i {
                    beta[i.min(j)]
                } else {
                    0.0
                }
            });
            let (vals, vecs) = sym_eig_desc(&t);
            let wscale = k as f64 / probes as f64;
            (0..m)
                .map(|j| (vals[j].max(0.0), wscale * vecs[(0, j)].powi(2)))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Solve `(a_s I + b L) X = B` column by column by conjugate gradients
/// (Hestenes and Stiefel 1952, J. Res. Natl. Bur. Stand. 49:409,
/// doi:10.6028/jres.049.044) with the Jacobi preconditioner, `a_s` the
/// shift of each node's section (all positive), from `x0` when given, to
/// relative residual `tol`, at most [`CG_MAX_ITER`] steps (a warning when
/// reached). `B` and `X` are `N × columns`. The solution and the most
/// steps any column took.
pub fn cg_solve(
    lap: &Laplacian,
    shift: &[f64],
    b: f64,
    rhs: &DMatrix<f64>,
    x0: Option<&DMatrix<f64>>,
    tol: f64,
) -> (DMatrix<f64>, usize) {
    let n = lap.n();
    let diag: Vec<f64> = (0..n)
        .map(|i| shift[lap.section[i]] + b * lap.degree[i])
        .collect();
    let mut x = match x0 {
        Some(x0) if x0.shape() == rhs.shape() => x0.clone(),
        _ => DMatrix::zeros(n, rhs.ncols()),
    };
    let steps: Vec<usize> = x
        .as_mut_slice()
        .par_chunks_mut(n.max(1))
        .zip(rhs.as_slice().par_chunks(n.max(1)))
        .map(|(xc, bc)| cg_column(lap, shift, b, &diag, bc, xc, tol))
        .collect();
    let most = steps.into_iter().max().unwrap_or(0);
    if most >= CG_MAX_ITER {
        log::warn!("conjugate gradients: {CG_MAX_ITER} steps reached before tolerance {tol:e}");
    }
    (x, most)
}

fn cg_column(
    lap: &Laplacian,
    shift: &[f64],
    b: f64,
    diag: &[f64],
    rhs: &[f64],
    x: &mut [f64],
    tol: f64,
) -> usize {
    let n = rhs.len();
    let apply = |v: &[f64], out: &mut [f64]| {
        lap.apply(v, out);
        for i in 0..n {
            out[i] = shift[lap.section[i]] * v[i] + b * out[i];
        }
    };
    let bnorm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    if bnorm == 0.0 {
        x.iter_mut().for_each(|v| *v = 0.0);
        return 0;
    }
    let mut ax = vec![0.0; n];
    apply(x, &mut ax);
    let mut r: Vec<f64> = rhs.iter().zip(&ax).map(|(b, a)| b - a).collect();
    let mut z: Vec<f64> = r.iter().zip(diag).map(|(r, d)| r / d).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let mut ap = vec![0.0; n];
    for it in 0..CG_MAX_ITER {
        let rnorm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rnorm <= tol * bnorm {
            return it;
        }
        apply(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            z[i] = r[i] / diag[i];
        }
        let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    CG_MAX_ITER
}
