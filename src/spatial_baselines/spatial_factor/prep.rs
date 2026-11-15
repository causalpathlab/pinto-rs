//! The expression the factor models read, kept implicit: a sparse matrix on
//! the counts' own pattern, less a few rank-one terms, rows scaled,
//!
//! `Y = D (S - sum_j a_j b_j^T)` (genes × cells),
//!
//! so `Y V` and `Y^T U` cost `O(nnz + J (m + n))` a column and no dense
//! genes × cells matrix is ever held. Two recipes, one per method:
//!
//! - [`ResidualOp::pearson`] (SpatialPCA): analytic Pearson residuals of a
//!   Poisson model with a gene's share of each cell's library as its mean
//!   (Lause, Berens and Kobak 2021, Genome Biology 22:258,
//!   doi:10.1186/s13059-021-02451-7, the `theta = infinity` limit), clipped
//!   at `+-sqrt(N)`, then each gene standardised (mean 0, sample SD 1);
//! - [`ResidualOp::log_centred`] (MEFISTO): Seurat's `LogNormalize`,
//!   `ln(1 + 1e4 x / lib)`, each gene centred within each section (a
//!   MEFISTO group).
//!
//! The library `lib` is a cell's total over all genes of the counts as read
//! (δ-divided when the input has a δ), taken before the genes are
//! selected.

use super::kit::Dense;
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
pub struct ResidualOp {
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
    /// Entries the negative clip added off the counts' pattern.
    pub clipped_zeros: usize,
}

impl ResidualOp {
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
        ResidualOp {
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
            clipped_zeros: 0,
        }
    }

    #[cfg(test)]
    /// A dense genes × cells matrix as the operator, every entry stored
    /// (a library entry point for expression normalised elsewhere).
    pub fn from_dense(y: &Dense) -> Self {
        let (m, n) = y.shape();
        let cells = (0..n)
            .map(|c| (0..m).map(|g| (g as u32, y[(g, c)] as f32)).collect())
            .collect();
        Self::from_cells(m, cells)
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

    /// Analytic Pearson residuals of `counts` (the selected genes × cells)
    /// with all-gene libraries `lib`, clipped at `+-sqrt(N)`, each gene
    /// standardised to mean 0 and sample SD 1 (a gene with no variance is
    /// a row of zeros).
    pub fn pearson(counts: &CscMatrix<f32>, lib: &[f64]) -> Self {
        let (m, n) = (counts.nrows(), counts.ncols());
        let total: f64 = lib.iter().sum();
        let mut row = vec![0f64; m];
        for (g, _, &v) in counts.triplet_iter() {
            row[g] += v as f64;
        }
        let p: Vec<f64> = row.iter().map(|r| r / total.max(1e-300)).collect();
        let a: Vec<f64> = p.iter().map(|v| v.sqrt()).collect();
        let b: Vec<f64> = lib.iter().map(|v| v.sqrt()).collect();
        let clip = (n as f64).sqrt();
        // cells by library, largest first: those where a zero count's
        // residual -sqrt(mu) passes -sqrt(N)
        let mut by_lib: Vec<usize> = (0..n).collect();
        by_lib.sort_by(|&i, &j| lib[j].total_cmp(&lib[i]));
        let mut extra: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut clipped_zeros = 0;
        for (g, &pg) in p.iter().enumerate() {
            if pg <= 0.0 {
                continue;
            }
            for &c in by_lib.iter().take_while(|&&c| pg * lib[c] > n as f64) {
                extra[c].push(g as u32);
                clipped_zeros += 1;
            }
        }
        let cells: Vec<Vec<(u32, f32)>> = counts
            .col_iter()
            .zip(extra)
            .enumerate()
            .map(|(c, (col, ex))| {
                let mut out: Vec<(u32, f32)> = Vec::with_capacity(col.nnz() + ex.len());
                let mut have = std::collections::HashSet::new();
                for (&g, &x) in col.row_indices().iter().zip(col.values()) {
                    let mu = p[g] * lib[c];
                    if mu <= 0.0 {
                        continue;
                    }
                    let sm = mu.sqrt();
                    let r = (x as f64 / sm - sm).clamp(-clip, clip);
                    out.push((g as u32, (r + sm) as f32));
                    if !ex.is_empty() {
                        have.insert(g as u32);
                    }
                }
                for g in ex {
                    if have.contains(&g) {
                        continue;
                    }
                    let sm = (p[g as usize] * lib[c]).sqrt();
                    out.push((g, (sm - clip) as f32));
                }
                out.sort_unstable_by_key(|e| e.0);
                out
            })
            .collect();
        let mut op = Self::from_cells(m, cells);
        op.clipped_zeros = clipped_zeros;
        if clipped_zeros > 0 {
            log::info!("pearson residuals: {clipped_zeros} zero counts clipped at -sqrt(N)");
        }
        let sums = op.row_sums(&b);
        let (sb, sbb) = (b.iter().sum::<f64>(), b.iter().map(|v| v * v).sum::<f64>());
        let nf = n as f64;
        let mut mean = vec![0f64; m];
        let mut scale = vec![0f64; m];
        for g in 0..m {
            let (s, ss, sx) = sums[g];
            let sum = s - a[g] * sb;
            let sq = ss - 2.0 * a[g] * sx + a[g] * a[g] * sbb;
            mean[g] = sum / nf;
            let var = (sq - nf * mean[g] * mean[g]) / (nf - 1.0).max(1.0);
            scale[g] = if var > 1e-12 { 1.0 / var.sqrt() } else { 0.0 };
        }
        op.terms = vec![(a, b), (mean, vec![1.0; n])];
        op.scale = scale;
        op
    }

    /// `ln(1 + 1e4 x / lib)` of `counts` (the selected genes × cells) with
    /// all-gene libraries `lib` (a cell with none is all zeros), each gene
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
                    .map(|(&g, &x)| (g as u32, (1e4 * x as f64 / lib[c]).ln_1p() as f32))
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

    /// Gene `g`'s row of `Y`, dense over the cells.
    pub fn gene_row(&self, g: usize) -> Vec<f64> {
        let mut out = vec![0f64; self.n];
        for e in self.g_ptr[g]..self.g_ptr[g + 1] {
            out[self.g_cell[e] as usize] = self.g_val[e] as f64;
        }
        for (a, b) in &self.terms {
            for (o, bc) in out.iter_mut().zip(b) {
                *o -= a[g] * bc;
            }
        }
        out.iter_mut().for_each(|o| *o *= self.scale[g]);
        out
    }

    /// `sum_{c : section[c] = s} Y_gc^2` for each gene and section, genes ×
    /// sections, for rank-one terms that are each a cell indicator times a
    /// gene vector (the log recipe's centring) or none.
    pub fn section_sumsq(&self, section: &[usize], n_sections: usize) -> Dense {
        let rows: Vec<Vec<f64>> = (0..self.m)
            .into_par_iter()
            .map(|g| {
                let mut acc = vec![0f64; n_sections];
                // the row, dense by section, in one pass over its entries
                let mut row = vec![0f64; self.n];
                for e in self.g_ptr[g]..self.g_ptr[g + 1] {
                    row[self.g_cell[e] as usize] = self.g_val[e] as f64;
                }
                for (a, b) in &self.terms {
                    for (r, bc) in row.iter_mut().zip(b) {
                        *r -= a[g] * bc;
                    }
                }
                for (c, r) in row.iter().enumerate() {
                    acc[section[c]] += (self.scale[g] * r).powi(2);
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

    /// `Y` as a dense genes × cells matrix (the reference modes, which hold
    /// it as the originals do).
    pub fn dense(&self) -> Dense {
        let rows: Vec<Vec<f64>> = (0..self.m)
            .into_par_iter()
            .map(|g| self.gene_row(g))
            .collect();
        Dense::from_fn(self.m, self.n, |g, c| rows[g][c])
    }
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
