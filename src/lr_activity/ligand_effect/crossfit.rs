//! Cross-fitting folds and the shared design matrix with its per-fold Grams.

use nalgebra::DMatrix;
use rayon::prelude::*;
use std::sync::Arc;

/// Ridge penalty on the standardized confounders; the intercept is free.
pub const DEFAULT_RIDGE: f64 = 1.0;

/// The fold of a row left out of every fold, training and held out alike:
/// a jackknife replicate's deleted tile ([`CrossFit::leaving_out`]).
pub const LEFT_OUT: u8 = u8::MAX;

/// Fold assignment and clusters for cross-fitting and the variance.
pub struct CrossFit {
    /// Fold of each row, `0..k`.
    pub fold: Vec<u8>,
    pub k: usize,
    /// Cluster of each row for the variance (a spatial tile).
    pub cluster: Vec<u32>,
    n_clusters: usize,
    /// Training and held-out rows of each fold, ascending.
    pub(super) train: Vec<Vec<usize>>,
    pub(super) test: Vec<Vec<usize>>,
}

impl CrossFit {
    pub fn new(fold: Vec<u8>, k: usize, cluster: Vec<u32>) -> Self {
        let n_clusters = cluster.iter().copied().max().map_or(0, |c| c as usize + 1);
        let mut test: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &f) in fold.iter().enumerate() {
            if f != LEFT_OUT {
                test[f as usize].push(i);
            }
        }
        let train = (0..k)
            .map(|f| {
                (0..fold.len())
                    .filter(|&i| fold[i] as usize != f && fold[i] != LEFT_OUT)
                    .collect()
            })
            .collect();
        Self {
            fold,
            k,
            cluster,
            n_clusters,
            train,
            test,
        }
    }

    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// The same folds and clusters with the rows `out` in none of them:
    /// neither trained on nor predicted.
    pub fn leaving_out(&self, out: &[usize]) -> Self {
        let mut fold = self.fold.clone();
        for &i in out {
            fold[i] = LEFT_OUT;
        }
        Self::new(fold, self.k, self.cluster.clone())
    }
}

/// The confounder matrix with one Gram matrix per held-out fold.
///
/// Column 0 is the intercept; every other column is standardized, so one
/// ridge penalty means the same thing for every confounder.
pub struct Design {
    /// Rows are receivers, column-major `n × p`, stored in f32 to halve the
    /// largest allocation of the run; every sum over it is taken in f64.
    /// Shared, so a jackknife replicate ([`Design::leaving_out`]) owns only
    /// its Grams.
    pub x: Arc<DMatrix<f32>>,
    /// `gram[k] = Σ_{fold(i) ≠ k} x_i x_iᵀ`: the training Gram for fold `k`.
    gram: Vec<DMatrix<f64>>,
}

impl Design {
    /// Standardize the non-intercept columns and precompute the per-fold
    /// training Grams. A constant column is zeroed rather than divided by 0.
    pub fn new(mut x: DMatrix<f32>, cf: &CrossFit) -> Self {
        let n = x.nrows().max(1) as f64;
        for j in 1..x.ncols() {
            let mut col = x.column_mut(j);
            let mean = col.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = col.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            let sd = var.sqrt();
            for v in col.iter_mut() {
                *v = if sd > 1e-12 {
                    ((*v as f64 - mean) / sd) as f32
                } else {
                    0.0
                };
            }
        }
        if x.ncols() > 0 {
            x.column_mut(0).fill(1.0);
        }
        let gram = fold_grams(&x, cf);
        Self {
            x: Arc::new(x),
            gram,
        }
    }

    pub fn p(&self) -> usize {
        self.x.ncols()
    }

    /// The design with the rows `out` taken out of every training Gram, for
    /// the folds `cf` (the ones it was built with) and
    /// [`CrossFit::leaving_out`]`(out)`: each Gram less those rows' outer
    /// products, its standardization kept.
    pub fn leaving_out(&self, cf: &CrossFit, out: &[usize]) -> Self {
        let p = self.p();
        // Each fold's Gram over its rows left out, by the blocked kernel;
        // fold k's training Gram loses every other fold's.
        let mut by_fold = vec![Vec::new(); self.gram.len()];
        for &i in out {
            by_fold[cf.fold[i] as usize].push(i);
        }
        let mut buf = Vec::new();
        let mut parts = Vec::with_capacity(by_fold.len());
        for rows in &by_fold {
            gather_rows(&self.x, p, rows, &mut buf);
            parts.push(super::regression::weighted_gram(
                &buf,
                p,
                &vec![1.0; rows.len()],
            ));
        }
        let mut total = DMatrix::<f64>::zeros(p, p);
        for part in &parts {
            total += part;
        }
        let gram = self
            .gram
            .iter()
            .zip(&parts)
            .map(|(g, own)| g - (&total - own))
            .collect();
        Self {
            x: Arc::clone(&self.x),
            gram,
        }
    }
}

/// A borrowed design: the first `p` columns of `x` with their per-fold
/// training Grams. The full [`Design`] and each [`DesignPrefix`] lend one.
#[derive(Clone, Copy)]
pub struct DesignView<'a> {
    pub x: &'a DMatrix<f32>,
    pub p: usize,
    pub(super) gram: &'a [DMatrix<f64>],
}

/// The first `p` columns of a [`Design`] as a nested adjustment set. It
/// owns only its `p × p` Gram per fold (the Gram of a column subset is the
/// submatrix of the full Gram) and borrows the matrix, so a step costs
/// `k p²` numbers rather than another `n × p` copy.
pub struct DesignPrefix {
    p: usize,
    gram: Vec<DMatrix<f64>>,
}

impl Design {
    pub fn view(&self) -> DesignView<'_> {
        DesignView {
            x: &self.x,
            p: self.p(),
            gram: &self.gram,
        }
    }

    /// The nested design on columns `0..p` (column 0, the intercept,
    /// always included).
    pub fn prefix(&self, p: usize) -> DesignPrefix {
        let p = p.clamp(1, self.p());
        DesignPrefix {
            p,
            gram: self
                .gram
                .iter()
                .map(|g| g.view((0, 0), (p, p)).clone_owned())
                .collect(),
        }
    }
}

impl DesignPrefix {
    pub fn p(&self) -> usize {
        self.p
    }

    /// This prefix over `design`, the design it was cut from.
    pub fn view<'a>(&'a self, design: &'a Design) -> DesignView<'a> {
        debug_assert!(self.p <= design.p());
        DesignView {
            x: &design.x,
            p: self.p,
            gram: &self.gram,
        }
    }
}

/// Rows `rows` of the first `p` columns of `x`, row-major in f64, read
/// column by column: `x` is column-major, so a row read strides by `n`.
pub(super) fn gather_rows(x: &DMatrix<f32>, p: usize, rows: &[usize], buf: &mut Vec<f64>) {
    buf.clear();
    buf.resize(rows.len() * p, 0.0);
    for j in 0..p {
        let col = x.column(j);
        let col = col.as_slice();
        for (r, &i) in rows.iter().enumerate() {
            buf[r * p + j] = col[i] as f64;
        }
    }
}

/// Rows per [`gather_rows`] block: small enough to stay in cache.
pub(super) const ROW_BLOCK: usize = 256;

/// `Σ_{i in fold k} x_i x_iᵀ` for each fold, then training = total − fold.
/// Accumulated over row chunks in f64, rows gathered a block at a time.
fn fold_grams(x: &DMatrix<f32>, cf: &CrossFit) -> Vec<DMatrix<f64>> {
    let (n, p) = x.shape();
    let rows: Vec<usize> = (0..n).collect();
    let per_fold = rows
        .par_chunks(4096)
        .map(|chunk| {
            let mut acc = vec![DMatrix::<f64>::zeros(p, p); cf.k];
            let mut buf = Vec::new();
            for block in chunk.chunks(ROW_BLOCK) {
                gather_rows(x, p, block, &mut buf);
                for (xi, &i) in buf.chunks_exact(p.max(1)).zip(block) {
                    if cf.fold[i] == LEFT_OUT {
                        continue;
                    }
                    let g = &mut acc[cf.fold[i] as usize];
                    for a in 0..p {
                        let xa = xi[a];
                        if xa == 0.0 {
                            continue;
                        }
                        for b in a..p {
                            g[(a, b)] += xa * xi[b];
                        }
                    }
                }
            }
            acc
        })
        .reduce(
            || vec![DMatrix::<f64>::zeros(p, p); cf.k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );
    let mut per_fold = per_fold;
    for g in &mut per_fold {
        for a in 0..p {
            for b in 0..a {
                g[(a, b)] = g[(b, a)];
            }
        }
    }
    let mut total = DMatrix::<f64>::zeros(p, p);
    for g in &per_fold {
        total += g;
    }
    per_fold.into_iter().map(|g| &total - g).collect()
}
