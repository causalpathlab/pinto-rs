//! The expression the factor models read: MEFISTO's kept implicit, a
//! sparse matrix on the counts' own pattern, less a few rank-one terms,
//! rows scaled,
//!
//! `Y = D (S - sum_j a_j b_j^T)` (genes × cells),
//!
//! so `Y V` and `Y^T U` cost `O(nnz + J (m + n))` a column and no dense
//! genes × cells matrix is ever held; SpatialPCA's held as dense rows of
//! f32, as SCTransform's residuals are dense:
//!
//! - [`DenseRows::sct`] (SpatialPCA): SCTransform's Pearson residuals
//!   ([`super::sct`]), clipped at `+-sqrt(N)`, then each gene standardised
//!   (mean 0, sample SD 1), the paper's input;
//! - [`SparseOp::pearson`]: analytic Pearson residuals of a Poisson model
//!   (Lause, Berens and Kobak 2021, Genome Biology 22:258,
//!   doi:10.1186/s13059-021-02451-7), for tests;
//! - [`SparseOp::log_centred`] (MEFISTO): Seurat's `LogNormalize`,
//!   `ln(1 + 1e4 x / lib)`, each gene centred within each section (a
//!   MEFISTO group).
//!
//! The library `lib` is a cell's total over all genes of the counts as read
//! (δ-divided when the input has a δ), taken before the genes are
//! selected.

use super::kit::Dense;
use crate::util::common::log_cp10k;
use nalgebra_sparse::CscMatrix;
use rayon::prelude::*;

/// Each cell's total count over every gene.
pub fn library(counts: &CscMatrix<f32>) -> Vec<f64> {
    counts
        .col_iter()
        .map(|c| c.values().iter().map(|&v| v as f64).sum())
        .collect()
}

/// The implicit expression matrix (see the module documentation).
#[derive(Debug, Clone)]
pub struct SparseOp {
    pub m: usize,
    pub n: usize,
    // S by gene
    g_ptr: Vec<usize>,
    g_cell: Vec<u32>,
    g_val: Vec<f32>,
    // S by cell
    c_ptr: Vec<usize>,
    c_gene: Vec<u32>,
    c_val: Vec<f32>,
    /// The rank-one terms `(a_j, b_j)`, `a_j` over genes, `b_j` over cells.
    terms: Vec<(Vec<f64>, Vec<f64>)>,
    /// The row scaling `D`.
    scale: Vec<f64>,
}

/// The expression as dense f32 rows (`m` genes × `n` cells, row-major).
#[derive(Debug, Clone)]
pub struct DenseRows {
    pub m: usize,
    pub n: usize,
    rows: Vec<f32>,
}

impl SparseOp {
    /// From `S` as per-cell `(gene, value)` lists over `m` genes.
    fn from_cells(m: usize, cells: Vec<Vec<(u32, f32)>>) -> Self {
        let n = cells.len();
        let mut c_ptr = Vec::with_capacity(n + 1);
        c_ptr.push(0);
        let nnz: usize = cells.iter().map(Vec::len).sum();
        let mut c_gene = Vec::with_capacity(nnz);
        let mut c_val = Vec::with_capacity(nnz);
        for c in cells {
            for (g, v) in c {
                c_gene.push(g);
                c_val.push(v);
            }
            c_ptr.push(c_gene.len());
        }
        let mut g_ptr = vec![0usize; m + 1];
        for &g in &c_gene {
            g_ptr[g as usize + 1] += 1;
        }
        for g in 0..m {
            g_ptr[g + 1] += g_ptr[g];
        }
        let mut next = g_ptr.clone();
        let mut g_cell = vec![0u32; nnz];
        let mut g_val = vec![0f32; nnz];
        for c in 0..n {
            for e in c_ptr[c]..c_ptr[c + 1] {
                let slot = &mut next[c_gene[e] as usize];
                g_cell[*slot] = c as u32;
                g_val[*slot] = c_val[e];
                *slot += 1;
            }
        }
        SparseOp {
            m,
            n,
            g_ptr,
            g_cell,
            g_val,
            c_ptr,
            c_gene,
            c_val,
            terms: Vec::new(),
            scale: vec![1.0; m],
        }
    }

    /// Each gene's `(sum_c S_gc, sum_c S_gc^2, sum_c S_gc b_c)` for a cell
    /// vector `b`.
    fn row_sums(&self, b: &[f64]) -> Vec<(f64, f64, f64)> {
        (0..self.m)
            .into_par_iter()
            .map(|g| {
                let (mut s, mut ss, mut sb) = (0.0, 0.0, 0.0);
                for e in self.g_ptr[g]..self.g_ptr[g + 1] {
                    let v = self.g_val[e] as f64;
                    s += v;
                    ss += v * v;
                    sb += v * b[self.g_cell[e] as usize];
                }
                (s, ss, sb)
            })
            .collect()
    }

    /// [`log_cp10k`] of `counts` (the selected genes × cells) with all-gene
    /// libraries `lib` (a cell with none stores nothing), each gene
    /// centred within each section (`section[c]`, `0..n_sections`).
    pub fn log_centred(
        counts: &CscMatrix<f32>,
        lib: &[f64],
        section: &[usize],
        n_sections: usize,
    ) -> Self {
        let m = counts.nrows();
        let cells: Vec<Vec<(u32, f32)>> = counts
            .col_iter()
            .enumerate()
            .map(|(c, col)| {
                if lib[c] <= 0.0 {
                    return Vec::new();
                }
                col.row_indices()
                    .iter()
                    .zip(col.values())
                    .map(|(&g, &x)| (g as u32, log_cp10k(f64::from(x), lib[c]) as f32))
                    .collect()
            })
            .collect();
        let mut op = Self::from_cells(m, cells);
        let mut size = vec![0f64; n_sections];
        for &s in section {
            size[s] += 1.0;
        }
        for (s, &k) in size.iter().enumerate() {
            let ind: Vec<f64> = section
                .iter()
                .map(|&t| f64::from(u8::from(t == s)))
                .collect();
            let sums = op.row_sums(&ind);
            let mean: Vec<f64> = sums.iter().map(|t| t.2 / k.max(1.0)).collect();
            op.terms.push((mean, ind));
        }
        op
    }

    /// `Y V`, `V` cells × k: genes × k.
    pub fn apply(&self, v: &Dense) -> Dense {
        let k = v.ncols();
        let n = self.n;
        let vs = v.as_slice();
        let proj: Vec<Vec<f64>> = self
            .terms
            .iter()
            .map(|(_, b)| {
                (0..k)
                    .map(|j| {
                        b.iter()
                            .zip(&vs[j * n..(j + 1) * n])
                            .map(|(x, y)| x * y)
                            .sum()
                    })
                    .collect()
            })
            .collect();
        let rows: Vec<Vec<f64>> = (0..self.m)
            .into_par_iter()
            .map(|g| {
                let mut out = vec![0f64; k];
                for e in self.g_ptr[g]..self.g_ptr[g + 1] {
                    let (c, s) = (self.g_cell[e] as usize, self.g_val[e] as f64);
                    for (j, o) in out.iter_mut().enumerate() {
                        *o += s * vs[j * n + c];
                    }
                }
                for ((a, _), pj) in self.terms.iter().zip(&proj) {
                    for (o, p) in out.iter_mut().zip(pj) {
                        *o -= a[g] * p;
                    }
                }
                out.iter_mut().for_each(|o| *o *= self.scale[g]);
                out
            })
            .collect();
        Dense::from_fn(self.m, k, |g, j| rows[g][j])
    }

    /// `Y^T U`, `U` genes × k: cells × k.
    pub fn apply_t(&self, u: &Dense) -> Dense {
        let k = u.ncols();
        let m = self.m;
        let us: Vec<f64> = (0..k)
            .flat_map(|j| (0..m).map(move |g| (j, g)))
            .map(|(j, g)| self.scale[g] * u[(g, j)])
            .collect();
        let proj: Vec<Vec<f64>> = self
            .terms
            .iter()
            .map(|(a, _)| {
                (0..k)
                    .map(|j| {
                        a.iter()
                            .zip(&us[j * m..(j + 1) * m])
                            .map(|(x, y)| x * y)
                            .sum()
                    })
                    .collect()
            })
            .collect();
        let mut out = Dense::zeros(self.n, k);
        let n = self.n;
        let cols: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|c| {
                let mut o = vec![0f64; k];
                for e in self.c_ptr[c]..self.c_ptr[c + 1] {
                    let (g, s) = (self.c_gene[e] as usize, self.c_val[e] as f64);
                    for (j, oj) in o.iter_mut().enumerate() {
                        *oj += s * us[j * m + g];
                    }
                }
                for ((_, b), pj) in self.terms.iter().zip(&proj) {
                    for (oj, p) in o.iter_mut().zip(pj) {
                        *oj -= b[c] * p;
                    }
                }
                o
            })
            .collect();
        for (c, o) in cols.into_iter().enumerate() {
            for (j, v) in o.into_iter().enumerate() {
                out[(c, j)] = v;
            }
        }
        out
    }

    /// `sum_{c : section[c] = s} Y_gc^2` for each gene and section, genes ×
    /// sections, in closed form from one pass over each gene's entries:
    /// with `Y_gc = D_g (S_gc - sum_j a_jg b_jc)`, the sum over a section
    /// is `D_g^2 [sum S^2 - 2 sum_j a_jg sum S b_j + sum_{j,j'} a_jg a_j'g
    /// sum b_j b_j']`, each sum over the section's cells.
    pub fn section_sumsq(&self, section: &[usize], n_sections: usize) -> Dense {
        let t = self.terms.len();
        // sum_{c in s} b_j[c] b_j'[c], per section
        let mut bb = vec![vec![0f64; t * t]; n_sections];
        for (c, &s) in section.iter().enumerate() {
            for j in 0..t {
                let bj = self.terms[j].1[c];
                if bj == 0.0 {
                    continue;
                }
                for jj in 0..t {
                    bb[s][j * t + jj] += bj * self.terms[jj].1[c];
                }
            }
        }
        let rows: Vec<Vec<f64>> = (0..self.m)
            .into_par_iter()
            .map(|g| {
                let mut ss = vec![0f64; n_sections];
                let mut sb = vec![0f64; n_sections * t];
                for e in self.g_ptr[g]..self.g_ptr[g + 1] {
                    let c = self.g_cell[e] as usize;
                    let (v, s) = (self.g_val[e] as f64, section[c]);
                    ss[s] += v * v;
                    for (j, (_, b)) in self.terms.iter().enumerate() {
                        sb[s * t + j] += v * b[c];
                    }
                }
                let a: Vec<f64> = self.terms.iter().map(|(a, _)| a[g]).collect();
                (0..n_sections)
                    .map(|s| {
                        let mut v = ss[s];
                        for j in 0..t {
                            v -= 2.0 * a[j] * sb[s * t + j];
                            for jj in 0..t {
                                v += a[j] * a[jj] * bb[s][j * t + jj];
                            }
                        }
                        (self.scale[g] * self.scale[g] * v).max(0.0)
                    })
                    .collect()
            })
            .collect();
        Dense::from_fn(self.m, n_sections, |g, s| rows[g][s])
    }

    /// `Y_s V` for each section `s` (`Y` with the other sections' cells
    /// zeroed), genes × k each, in one pass over the entries.
    pub fn apply_by_section(&self, v: &Dense, section: &[usize], n_sections: usize) -> Vec<Dense> {
        let k = v.ncols();
        let n = self.n;
        let vs = v.as_slice();
        // sum_{c in s} b_j[c] v[c, :], per term, section and column
        let proj: Vec<Vec<f64>> = self
            .terms
            .iter()
            .map(|(_, b)| {
                let mut p = vec![0f64; n_sections * k];
                for j in 0..k {
                    for (c, (&bc, &s)) in b.iter().zip(section).enumerate() {
                        p[s * k + j] += bc * vs[j * n + c];
                    }
                }
                p
            })
            .collect();
        let rows: Vec<Vec<f64>> = (0..self.m)
            .into_par_iter()
            .map(|g| {
                let mut out = vec![0f64; n_sections * k];
                for e in self.g_ptr[g]..self.g_ptr[g + 1] {
                    let (c, x) = (self.g_cell[e] as usize, self.g_val[e] as f64);
                    let o = &mut out[section[c] * k..(section[c] + 1) * k];
                    for (j, oj) in o.iter_mut().enumerate() {
                        *oj += x * vs[j * n + c];
                    }
                }
                for ((a, _), p) in self.terms.iter().zip(&proj) {
                    for (o, pj) in out.iter_mut().zip(p) {
                        *o -= a[g] * pj;
                    }
                }
                out.iter_mut().for_each(|o| *o *= self.scale[g]);
                out
            })
            .collect();
        (0..n_sections)
            .map(|s| Dense::from_fn(self.m, k, |g, j| rows[g][s * k + j]))
            .collect()
    }

    /// Row `c` of `Y^T U_s` for each cell `c`, `s` its section: `u` one
    /// genes × k matrix per section, cells × k, in one pass over the
    /// entries.
    pub fn apply_t_by_section(&self, u: &[Dense], section: &[usize]) -> Dense {
        let k = u.first().map_or(0, |x| x.ncols());
        let m = self.m;
        let n_sections = u.len();
        // the scaled U_s, section by section, column-major within each
        let us: Vec<Vec<f64>> = u
            .iter()
            .map(|us| {
                (0..k)
                    .flat_map(|j| (0..m).map(move |g| (j, g)))
                    .map(|(j, g)| self.scale[g] * us[(g, j)])
                    .collect()
            })
            .collect();
        let proj: Vec<Vec<f64>> = self
            .terms
            .iter()
            .map(|(a, _)| {
                (0..n_sections)
                    .flat_map(|s| (0..k).map(move |j| (s, j)))
                    .map(|(s, j)| {
                        a.iter()
                            .zip(&us[s][j * m..(j + 1) * m])
                            .map(|(x, y)| x * y)
                            .sum()
                    })
                    .collect()
            })
            .collect();
        let cols: Vec<Vec<f64>> = (0..self.n)
            .into_par_iter()
            .map(|c| {
                let s = section[c];
                let mut o = vec![0f64; k];
                for e in self.c_ptr[c]..self.c_ptr[c + 1] {
                    let (g, x) = (self.c_gene[e] as usize, self.c_val[e] as f64);
                    for (j, oj) in o.iter_mut().enumerate() {
                        *oj += x * us[s][j * m + g];
                    }
                }
                for ((_, b), p) in self.terms.iter().zip(&proj) {
                    for (j, oj) in o.iter_mut().enumerate() {
                        *oj -= b[c] * p[s * k + j];
                    }
                }
                o
            })
            .collect();
        Dense::from_fn(self.n, k, |c, j| cols[c][j])
    }
}

impl DenseRows {
    /// The matrix from its rows (`m` genes × `n` cells, row-major).
    pub fn from_rows(m: usize, n: usize, rows: Vec<f32>) -> Self {
        assert_eq!(rows.len(), m * n);
        DenseRows { m, n, rows }
    }

    /// SCTransform's residuals of `counts` (all genes × cells, libraries
    /// `lib` over them) on the genes `genes`, each gene standardised to
    /// mean 0 and sample SD 1 (a gene with no variance a row of zeros);
    /// the models' report.
    pub fn sct(
        counts: &CscMatrix<f32>,
        lib: &[f64],
        genes: &[usize],
        seed: u64,
    ) -> (Self, super::sct::SctReport) {
        let n = counts.ncols();
        let by_gene = super::sct::ByGene::from_csc(counts);
        let depths = super::sct::Depths::new(lib);
        let (models, report) = super::sct::sctransform(&by_gene, &depths, genes, seed);
        let mut flat = vec![0f32; genes.len() * n];
        flat.par_chunks_mut(n.max(1))
            .zip(genes.par_iter().zip(&models))
            .for_each(|(row, (&g, model))| {
                let mut z = super::sct::residuals(by_gene.gene(g), &depths.t, model);
                standardise(&mut z);
                row.iter_mut().zip(z).for_each(|(r, v)| *r = v as f32);
            });
        (Self::from_rows(genes.len(), n, flat), report)
    }

    fn row(&self, g: usize) -> &[f32] {
        &self.rows[g * self.n..(g + 1) * self.n]
    }

    /// `Y V`, `V` cells × k: genes × k.
    pub fn apply(&self, v: &Dense) -> Dense {
        let (k, n) = (v.ncols(), self.n);
        let vs = v.as_slice();
        let out: Vec<Vec<f64>> = (0..self.m)
            .into_par_iter()
            .map(|g| {
                let row = self.row(g);
                (0..k)
                    .map(|j| {
                        row.iter()
                            .zip(&vs[j * n..(j + 1) * n])
                            .map(|(&y, &x)| y as f64 * x)
                            .sum()
                    })
                    .collect()
            })
            .collect();
        Dense::from_fn(self.m, k, |g, j| out[g][j])
    }

    /// `Y^T U`, `U` genes × k: cells × k, a block of cells a task.
    pub fn apply_t(&self, u: &Dense) -> Dense {
        const BLOCK: usize = 256;
        let (k, n) = (u.ncols(), self.n);
        let blocks: Vec<Vec<f64>> = (0..n.div_ceil(BLOCK))
            .into_par_iter()
            .map(|b| {
                let cells = b * BLOCK..((b + 1) * BLOCK).min(n);
                let mut acc = vec![0f64; cells.len() * k];
                for g in 0..self.m {
                    for (r, &y) in self.row(g)[cells.clone()].iter().enumerate() {
                        let y = y as f64;
                        for j in 0..k {
                            acc[r * k + j] += y * u[(g, j)];
                        }
                    }
                }
                acc
            })
            .collect();
        Dense::from_fn(n, k, |c, j| blocks[c / BLOCK][(c % BLOCK) * k + j])
    }

    /// `sum_{c : section[c] = s} Y_gc^2` for each gene and section.
    pub fn section_sumsq(&self, section: &[usize], n_sections: usize) -> Dense {
        let rows: Vec<Vec<f64>> = (0..self.m)
            .into_par_iter()
            .map(|g| {
                let mut acc = vec![0f64; n_sections];
                for (&y, &s) in self.row(g).iter().zip(section) {
                    acc[s] += (y as f64).powi(2);
                }
                acc
            })
            .collect();
        Dense::from_fn(self.m, n_sections, |g, s| rows[g][s])
    }

    /// `tr(Y Y^T)`.
    pub fn frob2(&self, section: &[usize], n_sections: usize) -> f64 {
        self.section_sumsq(section, n_sections).sum()
    }
}

/// `z` to mean 0 and sample SD 1 (all zeros when it has no variance).
pub fn standardise(z: &mut [f64]) {
    let n = z.len() as f64;
    let mean = z.iter().sum::<f64>() / n;
    let var = z.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let scale = if var > 1e-12 { 1.0 / var.sqrt() } else { 0.0 };
    z.iter_mut().for_each(|v| *v = (*v - mean) * scale);
}

/// Rows `genes` of `counts`, in that order.
pub fn select_rows(counts: &CscMatrix<f32>, genes: &[usize]) -> CscMatrix<f32> {
    let mut at = vec![None; counts.nrows()];
    for (j, &g) in genes.iter().enumerate() {
        at[g] = Some(j);
    }
    let mut offsets = Vec::with_capacity(counts.ncols() + 1);
    offsets.push(0);
    let mut rows = Vec::new();
    let mut values = Vec::new();
    for col in counts.col_iter() {
        let mut kept: Vec<(usize, f32)> = col
            .row_indices()
            .iter()
            .zip(col.values())
            .filter_map(|(&g, &v)| at[g].map(|j| (j, v)))
            .collect();
        kept.sort_unstable_by_key(|e| e.0);
        for (j, v) in kept {
            rows.push(j);
            values.push(v);
        }
        offsets.push(rows.len());
    }
    CscMatrix::try_from_csc_data(genes.len(), counts.ncols(), offsets, rows, values)
        .expect("sorted rows within each column")
}
