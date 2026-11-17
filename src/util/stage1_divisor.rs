//! The stage-1 divisor a stage-2 fit reads (`--offset <dir>`): the tables
//! `pinto residual --residual divide` writes ([`Stage1Tables`]), and those
//! tables aligned to a fit's own cells and rows ([`Stage1Divisor`]).
//!
//! The stage-1 expectation of row `g` in cell `i` of batch `b` is
//!
//! `μ_gi = n_i Σ_k θ_ik p_{k,b}(g)`,
//!
//! and the divisor of its count is `d_gi = μ_gi / (n_i p̄_g)
//! = Σ_k θ_ik p_{k,b}(g) / p̄_g`, `p̄_g = Σ_i μ_gi / Σ_i n_i` the row's
//! pooled share, so `d` is on the count scale with an `n`-weighted mean of
//! 1 over the cells, as a batch effect `δ` is. Neither `μ` nor `d` is ever
//! stored per entry: each is one `K`-long dot product, taken where a count
//! is read (at its nonzeros) from `θ` (`n × K`) and the profiles
//! (`B × G × K`).

use crate::util::common::LIBRARY_SCALE;
use crate::util::common::*;
use crate::util::memory::{ensure_fits, human_bytes, Budget};
use legume_numeric::matrix::parquet::{
    peek_parquet_field_names, read_table_columns, write_named_table, Column,
};
use legume_numeric::matrix::utils::generate_minibatch_intervals;
use std::path::{Path, PathBuf};

/// Per cell: `n`, its batch and `θ` (`{dir}/stage1.cells.parquet`).
pub const CELLS_TABLE: &str = "stage1.cells.parquet";
/// Per row: `p_{k,b}` (`{dir}/stage1.profiles.parquet`).
pub const PROFILES_TABLE: &str = "stage1.profiles.parquet";
/// Per row: `p̄`, `φ` and `σ²_b` (`{dir}/stage1.features.parquet`).
pub const FEATURES_TABLE: &str = "stage1.features.parquet";

fn theta_column(k: usize) -> String {
    format!("theta_{k}")
}

fn profile_column(k: usize, batch: &str) -> String {
    format!("p{k}@{batch}")
}

fn sigma2_column(batch: &str) -> String {
    format!("sigma2@{batch}")
}

/// What `pinto residual --residual divide` writes: the stage-1 model's
/// low-rank pieces, from which `μ` is rebuilt wherever it is needed.
#[derive(Debug, Clone, PartialEq)]
pub struct Stage1Tables {
    /// The cells, as the stage-1 run named them.
    pub cells: Vec<Box<str>>,
    /// Each cell's total `n_i` over the rows.
    pub n: Vec<f32>,
    /// Each cell's batch, an index into [`Self::batches`].
    pub batch: Vec<u32>,
    /// `θ`, `[cells × K]` row-major.
    pub theta: Vec<f32>,
    pub k: usize,
    /// The batches the profiles were adapted in (one under `--adapt none`).
    pub batches: Vec<Box<str>>,
    /// The rows, as the stage-1 run read them.
    pub features: Vec<Box<str>>,
    /// `p_{k,b}(g)`, `[batches × rows × K]` row-major.
    pub profiles: Vec<f32>,
    /// `p̄_g`.
    pub pbar: Vec<f32>,
    /// `φ_g`, the negative-binomial overdispersion by the method of moments.
    pub phi: Vec<f32>,
    /// `σ²_{gb}`, `[rows × batches]` row-major: the mean of `μ + φ_g μ²`
    /// over the batch's cells.
    pub sigma2: Vec<f32>,
}

impl Stage1Tables {
    /// The three tables under `dir`, in the order they are written.
    pub fn paths(dir: &str) -> [String; 3] {
        [CELLS_TABLE, PROFILES_TABLE, FEATURES_TABLE].map(|t| in_dir(dir, t))
    }

    pub fn write(&self, dir: &str) -> anyhow::Result<()> {
        let [cells, profiles, features] = Self::paths(dir);
        let (k, g, b) = (self.k, self.features.len(), self.batches.len());
        let batch: Vec<Box<str>> = self
            .batch
            .iter()
            .map(|&i| self.batches[i as usize].clone())
            .collect();
        let thetas: Vec<Vec<f32>> = (0..k)
            .map(|t| {
                (0..self.cells.len())
                    .map(|i| self.theta[i * k + t])
                    .collect()
            })
            .collect();
        let names: Vec<String> = (0..k).map(theta_column).collect();
        let mut columns = vec![
            ("batch".into(), Column::Str(&batch)),
            ("n".into(), Column::F32(&self.n)),
        ];
        columns.extend(
            names
                .iter()
                .zip(&thetas)
                .map(|(n, v)| (n.as_str().into(), Column::F32(v))),
        );
        write_named_table(&cells, "cell", &self.cells, &columns)?;

        let mut profile_cols: Vec<(String, Vec<f32>)> = Vec::with_capacity(k * b);
        for (bi, bn) in self.batches.iter().enumerate() {
            for t in 0..k {
                let col = (0..g)
                    .map(|r| self.profiles[(bi * g + r) * k + t])
                    .collect();
                profile_cols.push((profile_column(t, bn), col));
            }
        }
        let columns: Vec<(Box<str>, Column)> = profile_cols
            .iter()
            .map(|(n, v)| (n.as_str().into(), Column::F32(v)))
            .collect();
        write_named_table(&profiles, "feature", &self.features, &columns)?;

        let sigmas: Vec<(String, Vec<f32>)> = self
            .batches
            .iter()
            .enumerate()
            .map(|(bi, bn)| {
                (
                    sigma2_column(bn),
                    (0..g).map(|r| self.sigma2[r * b + bi]).collect(),
                )
            })
            .collect();
        let mut columns: Vec<(Box<str>, Column)> = vec![
            ("pbar".into(), Column::F32(&self.pbar)),
            ("phi".into(), Column::F32(&self.phi)),
        ];
        columns.extend(
            sigmas
                .iter()
                .map(|(n, v)| (n.as_str().into(), Column::F32(v))),
        );
        write_named_table(&features, "feature", &self.features, &columns)
    }

    /// The tables under `dir`, the cells' numeric columns checked against
    /// `budget` before they are read (`--offset <dir>`).
    pub fn read(dir: &str, budget: &Budget) -> anyhow::Result<Self> {
        let [cells_path, profiles_path, features_path] = Self::paths(dir);
        for p in [&cells_path, &profiles_path, &features_path] {
            anyhow::ensure!(
                Path::new(p).exists(),
                "--offset {dir}: {p} not found; give the folder `pinto residual --residual \
                 divide` wrote"
            );
        }
        let fields = peek_parquet_field_names(&cells_path)?;
        let k = (0..)
            .take_while(|&t| fields.iter().any(|f| f.as_ref() == theta_column(t)))
            .count();
        anyhow::ensure!(k > 0, "{cells_path}: no theta_0 column");
        let (strs, _) = read_table_columns(&cells_path, &["cell", "batch"], &[])?;
        let [cells, batch_names]: [Vec<Box<str>>; 2] = strs
            .try_into()
            .map_err(|_| anyhow::anyhow!("{cells_path}: expected cell and batch columns"))?;
        // the reader holds the numbers twice as f64 while it reads them
        ensure_fits(
            "reading the stage-1 divisor",
            cells.len() * (k + 1) * (2 * 8 + 4),
            budget,
            "fit fewer cells",
        )?;
        let mut numeric: Vec<String> = vec!["n".into()];
        numeric.extend((0..k).map(theta_column));
        let refs: Vec<&str> = numeric.iter().map(String::as_str).collect();
        let (_, nums) = read_table_columns(&cells_path, &[], &refs)?;
        let n: Vec<f32> = nums[0].iter().map(|&x| x as f32).collect();
        let mut theta = vec![0.0f32; cells.len() * k];
        for (t, col) in nums[1..].iter().enumerate() {
            for (i, &x) in col.iter().enumerate() {
                theta[i * k + t] = x as f32;
            }
        }
        drop(nums);
        let mut batches: Vec<Box<str>> = Vec::new();
        let batch: Vec<u32> = crate::util::ablation::dense_ids(batch_names.iter())
            .into_iter()
            .collect();
        for (i, &b) in batch.iter().enumerate() {
            if b as usize == batches.len() {
                batches.push(batch_names[i].clone());
            }
        }

        let (strs, _) = read_table_columns(&features_path, &["feature"], &[])?;
        let features = strs.into_iter().next().unwrap_or_default();
        let mut numeric: Vec<String> = vec!["pbar".into(), "phi".into()];
        numeric.extend(batches.iter().map(|b| sigma2_column(b)));
        let refs: Vec<&str> = numeric.iter().map(String::as_str).collect();
        let (_, nums) = read_table_columns(&features_path, &[], &refs)?;
        let g = features.len();
        let nb = batches.len();
        let pbar = nums[0].iter().map(|&x| x as f32).collect();
        let phi = nums[1].iter().map(|&x| x as f32).collect();
        let mut sigma2 = vec![0.0f32; g * nb];
        for (bi, col) in nums[2..].iter().enumerate() {
            for (r, &x) in col.iter().enumerate() {
                sigma2[r * nb + bi] = x as f32;
            }
        }

        let (strs, _) = read_table_columns(&profiles_path, &["feature"], &[])?;
        anyhow::ensure!(
            strs.first().map(Vec::as_slice) == Some(&features[..]),
            "{profiles_path} and {features_path} list different rows"
        );
        let mut numeric: Vec<String> = Vec::with_capacity(nb * k);
        for b in &batches {
            numeric.extend((0..k).map(|t| profile_column(t, b)));
        }
        let refs: Vec<&str> = numeric.iter().map(String::as_str).collect();
        let (_, nums) = read_table_columns(&profiles_path, &[], &refs)?;
        let mut profiles = vec![0.0f32; nb * g * k];
        for (c, col) in nums.iter().enumerate() {
            let (bi, t) = (c / k, c % k);
            for (r, &x) in col.iter().enumerate() {
                profiles[(bi * g + r) * k + t] = x as f32;
            }
        }
        Ok(Self {
            cells,
            n,
            batch,
            theta,
            k,
            batches,
            features,
            profiles,
            pbar,
            phi,
            sigma2,
        })
    }
}

/// The smallest divisor `d` is allowed (its mean is about 1): a count
/// stage 1 expected a tenth as often as average is amplified at most
/// tenfold, a shrinkage of the rate toward the stage-1 expectation where
/// that expectation is near zero.
pub const D_FLOOR: f32 = 0.1;

/// The stage-1 divisor on a fit's own cells and rows: what `--offset`
/// divides the counts by ([`Self::at`]), and the pieces of `μ` the fit's
/// other uses of it read ([`Self::mu`], [`Self::profile`]).
#[derive(Debug, Clone)]
pub struct Stage1Divisor {
    /// The folder it was read from, canonical: what the manifest records.
    pub dir: PathBuf,
    k: usize,
    n_rows: usize,
    n_batches: usize,
    /// `[cells × K]`, on the fit's cells.
    theta: Vec<f32>,
    n: Vec<f32>,
    batch: Vec<u32>,
    /// `p_{k,b}(g)`, `[batches × fit rows × K]`.
    p: Vec<f32>,
    /// `p̄_g` per fit row (`d` divides by it).
    pbar: Vec<f32>,
    /// `σ_{gb}`, `[fit rows × batches]`.
    sigma: Vec<f32>,
}

impl Stage1Divisor {
    /// The tables of `dir` on a fit's `cells` and `rows`, each found by
    /// name; a cell or row of the fit the stage-1 tables lack is refused.
    pub fn load(dir: &str, cells: &[Box<str>], rows: &[Box<str>]) -> anyhow::Result<Self> {
        let budget = Budget::now()?;
        let tables = Stage1Tables::read(dir, &budget)?;
        let canonical =
            std::fs::canonicalize(dir).map_err(|e| anyhow::anyhow!("--offset {dir}: {e}"))?;
        let held =
            cells.len() * (tables.k + 2) * 4 + 2 * tables.batches.len() * rows.len() * tables.k * 4;
        ensure_fits(
            "the stage-1 divisor",
            held,
            &budget,
            "fit fewer cells or rows",
        )?;
        let me = Self::align(&tables, cells, rows)
            .map_err(|e| anyhow::anyhow!("--offset {dir}: {e:#}"))?;
        info!(
            "Stage-1 divisor from {}: {} communities, {} batch(es), {} held",
            canonical.display(),
            me.k,
            me.n_batches,
            human_bytes(held)
        );
        Ok(Self {
            dir: canonical,
            ..me
        })
    }

    /// `tables` on `cells` and `rows`, by name.
    pub fn align(
        tables: &Stage1Tables,
        cells: &[Box<str>],
        rows: &[Box<str>],
    ) -> anyhow::Result<Self> {
        let k = tables.k;
        let cell_at: HashMap<&str, usize> = tables
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_ref(), i))
            .collect();
        let row_at: HashMap<&str, usize> = tables
            .features
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_ref(), i))
            .collect();
        let (mut theta, mut n, mut batch) = (
            Vec::with_capacity(cells.len() * k),
            Vec::with_capacity(cells.len()),
            Vec::with_capacity(cells.len()),
        );
        for c in cells {
            let &i = cell_at.get(c.as_ref()).ok_or_else(|| {
                anyhow::anyhow!(
                    "cell {c} of the data is not in the stage-1 cells table: a stage-2 fit \
                     reads the stage-1 run's own data files"
                )
            })?;
            theta.extend_from_slice(&tables.theta[i * k..(i + 1) * k]);
            n.push(tables.n[i]);
            batch.push(tables.batch[i]);
        }
        let of_row: Vec<usize> = rows
            .iter()
            .map(|r| {
                row_at.get(r.as_ref()).copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "row {r} of the data is not in the stage-1 tables: read the data \
                         as the stage-1 run did (its --feature-name-mode)"
                    )
                })
            })
            .collect::<anyhow::Result<_>>()?;
        let (g0, nb) = (tables.features.len(), tables.batches.len());
        let mut p = Vec::with_capacity(nb * rows.len() * k);
        let pbar: Vec<f32> = of_row.iter().map(|&r| tables.pbar[r]).collect();
        let mut sigma = Vec::with_capacity(rows.len() * nb);
        for b in 0..nb {
            for &r in &of_row {
                p.extend_from_slice(&tables.profiles[(b * g0 + r) * k..(b * g0 + r + 1) * k]);
            }
        }
        for &r in &of_row {
            sigma.extend(
                tables.sigma2[r * nb..(r + 1) * nb]
                    .iter()
                    .map(|&s| s.max(0.0).sqrt()),
            );
        }
        Ok(Self {
            dir: PathBuf::new(),
            k,
            n_rows: rows.len(),
            n_batches: nb,
            theta,
            n,
            batch,
            p,
            pbar,
            sigma,
        })
    }

    /// Only the cells `keep` marks, in order (cells cut after the graph).
    pub fn select_cells(&mut self, keep: &[bool]) {
        let k = self.k;
        let kept: Vec<usize> = (0..keep.len()).filter(|&i| keep[i]).collect();
        self.theta = kept
            .iter()
            .flat_map(|&i| self.theta[i * k..(i + 1) * k].iter().copied())
            .collect();
        self.n = kept.iter().map(|&i| self.n[i]).collect();
        self.batch = kept.iter().map(|&i| self.batch[i]).collect();
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n_batches(&self) -> usize {
        self.n_batches
    }

    pub fn n_cells(&self) -> usize {
        self.n.len()
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Cell `i`'s `θ_i`.
    #[inline]
    pub fn theta(&self, cell: usize) -> &[f32] {
        &self.theta[cell * self.k..(cell + 1) * self.k]
    }

    /// Cell `i`'s total `n_i`.
    #[inline]
    pub fn total(&self, cell: usize) -> f32 {
        self.n[cell]
    }

    /// Cell `i`'s batch.
    #[inline]
    pub fn batch_of(&self, cell: usize) -> usize {
        self.batch[cell] as usize
    }

    /// `p_{·,b}(g)`, the `K` communities' shares of row `g` in batch `b`.
    #[inline]
    pub fn profile(&self, batch: usize, row: usize) -> &[f32] {
        let at = (batch * self.n_rows + row) * self.k;
        &self.p[at..at + self.k]
    }

    /// Adds cell `cell`'s stage-1 mixture `n_i θ_i` to `out` (`K` long):
    /// a unit's expectation is its summed mixture through `p_b`.
    #[inline]
    pub fn add_mixture(&self, cell: usize, out: &mut [f32]) {
        let n = self.total(cell);
        out.iter_mut()
            .zip(self.theta(cell))
            .for_each(|(x, &t)| *x += n * t);
    }

    /// `p̄_g`, row `g`'s pooled share.
    #[inline]
    pub fn pbar(&self, row: usize) -> f32 {
        self.pbar[row]
    }

    /// `σ_{gb}`, the Pearson scale of row `g` in batch `b`.
    #[inline]
    pub fn sigma(&self, row: usize, batch: usize) -> f32 {
        self.sigma[row * self.n_batches + batch]
    }

    /// `d_gi = Σ_k θ_ik p_{k,b}(g) / p̄_g`, floored at [`D_FLOOR`]: where
    /// stage 1 expects almost nothing, one stray count would otherwise be
    /// blown up by `1/d` and dominate the cell.
    #[inline]
    pub fn at(&self, cell: usize, row: usize) -> f32 {
        let pbar = self.pbar[row];
        if pbar <= 0.0 {
            return D_FLOOR;
        }
        (dot(self.theta(cell), self.profile(self.batch_of(cell), row)) / pbar).max(D_FLOOR)
    }

    /// `μ_gi = n_i Σ_k θ_ik p_{k,b}(g)`: what the tests rebuild `μ` with
    /// (a fit reads `d`, or `μ` through its factorised sums).
    #[cfg(test)]
    #[inline]
    pub fn mu(&self, cell: usize, row: usize) -> f32 {
        self.total(cell) * dot(self.theta(cell), self.profile(self.batch_of(cell), row))
    }

    /// `Σ_i π_ic d_gi`, `[rows × communities]`: each row's expected mass
    /// in each community of `prop` (`[cells × communities]`) relative to the
    /// stage-1 expectation, the denominator of the rate `λ̂_gc = D_gc / M_gc`
    /// with `D_gc = Σ_i π_ic y_gi`. It factorises through `θ`:
    /// `Σ_b Σ_t p_{t,b}(g) A_b[t, c] / p̄_g`, `A_b = Σ_{i∈b} θ_iᵀ π_i` (`K ×
    /// communities` per batch), so it costs one pass over the cells and no
    /// pass over the rows × cells.
    pub fn expected_mass(&self, prop: &Mat) -> Mat {
        let (k, kk) = (self.k, prop.ncols());
        assert_eq!(prop.nrows(), self.n_cells(), "one propensity row per cell");
        // per-batch `K × communities` sums, folded over cell chunks; the
        // propensities read by row once, not strided per entry
        let prop_t = prop.transpose();
        let zero = || vec![nalgebra::DMatrix::<f64>::zeros(k, kk); self.n_batches];
        let a = (0..self.n_cells())
            .into_par_iter()
            .fold(zero, |mut a, i| {
                let ab = &mut a[self.batch_of(i)];
                let pi = prop_t.column(i);
                for (t, &th) in self.theta(i).iter().enumerate() {
                    if th != 0.0 {
                        for c in 0..kk {
                            ab[(t, c)] += f64::from(th) * f64::from(pi[c]);
                        }
                    }
                }
                a
            })
            .reduce(zero, |mut x, y| {
                for (xb, yb) in x.iter_mut().zip(y) {
                    *xb += yb;
                }
                x
            });
        let mut out = Mat::zeros(self.n_rows, kk);
        for (b, ab) in a.iter().enumerate() {
            let at = b * self.n_rows * k;
            let p_kg = Mat::from_column_slice(k, self.n_rows, &self.p[at..at + self.n_rows * k]);
            out += p_kg.transpose() * ab.map(|x| x as f32);
        }
        for (g, mut row) in out.row_iter_mut().enumerate() {
            let pbar = self.pbar[g];
            row *= if pbar > 0.0 { 1.0 / pbar } else { 0.0 };
        }
        out
    }

    /// Each column of `x` (column `j` being cell `cell_of_column(j)`)
    /// divided by its `d_gi` as numeric's `AdjustByDivisionOp` divides by a
    /// batch effect: the self-normalising Poisson ratio
    /// ([`Self::ratio_divide`]), each cell back at [`LIBRARY_SCALE`]: the one
    /// division every `--offset` read takes.
    pub fn divide_columns(
        &self,
        x: &mut nalgebra_sparse::CscMatrix<f32>,
        cell_of_column: impl Fn(usize) -> usize,
    ) {
        for (j, mut col) in x.col_iter_mut().enumerate() {
            let (rows, vals) = col.rows_and_values_mut();
            self.ratio_divide(rows, vals, cell_of_column(j));
        }
    }

    /// One cell's values (`rows`, `vals`) divided by `d` as numeric's
    /// adjust-by-division divides (the self-normalising Poisson ratio
    /// `y / (λ d)`, `λ = Σ y / Σ d`: the cell's scale is absorbed), then
    /// scaled back to a total of [`LIBRARY_SCALE`], the count scale every
    /// `--offset` read is on.
    pub fn ratio_divide(&self, rows: &[usize], vals: &mut [f32], cell: usize) {
        // the ratio reads each divisor twice: take it once
        let d: Vec<f32> = rows.iter().map(|&g| self.at(cell, g)).collect();
        adjust_by_poisson_ratio(vals, |k| d[k]);
        let total: f32 = vals.iter().sum();
        if total > 0.0 {
            let s = LIBRARY_SCALE / total;
            vals.iter_mut().for_each(|v| *v *= s);
        }
    }

    /// Each batch's profiles through `basis` (`[rows × m]`): `[K × m]` per
    /// batch, `P_b[k] = Σ_g p_{k,b}(g) basis[g, ·]`, so a cell's expected
    /// profile in that space is `n_i θ_iᵀ P_b` ([`Self::projected_mu`]).
    pub fn projected_profiles(&self, basis: &Mat) -> Vec<Mat> {
        assert_eq!(basis.nrows(), self.n_rows, "one basis row per data row");
        (0..self.n_batches)
            .map(|b| {
                let at = b * self.n_rows * self.k;
                // [K × rows], column g the K shares of row g
                let p_kg = Mat::from_column_slice(
                    self.k,
                    self.n_rows,
                    &self.p[at..at + self.n_rows * self.k],
                );
                &p_kg * basis
            })
            .collect()
    }

    /// Cell `i`'s expected profile through the basis of `projected`
    /// ([`Self::projected_profiles`]), added into `out` (`m` long).
    pub fn add_projected_mu(&self, projected: &[Mat], cell: usize, out: &mut [f32]) {
        let pb = &projected[self.batch_of(cell)];
        let n = self.total(cell);
        for (t, &th) in self.theta(cell).iter().enumerate() {
            let w = n * th;
            if w != 0.0 {
                for (o, &x) in out.iter_mut().zip(pb.row(t).iter()) {
                    *o += w * x;
                }
            }
        }
    }

    /// The per-cell random projection preprocessing takes, on the counts
    /// divided by `d` (pinto divides, then projects: the shared projection
    /// in data-beans takes no per-cell divisor). The same transform and
    /// basis as data-beans' (`log1p`, unit columns, a seeded Gaussian
    /// basis with `row_weights` scaling or zeroing its rows, then each
    /// coordinate standardised over the cells and clamped to ±4), with no
    /// batch correction: a stage-2 fit takes no batch effect.
    pub fn project_columns(
        &self,
        data: &SparseIoVec,
        target_dim: usize,
        row_weights: Option<&[f32]>,
    ) -> anyhow::Result<data_beans::alg::random_projection::RandColProjOut> {
        use data_beans::alg::random_projection::{RandColProjOut, DEFAULT_PROJECTION_SEED};
        let (nrows, ncols) = (data.num_rows(), data.num_columns());
        anyhow::ensure!(
            nrows == self.n_rows && ncols == self.n_cells(),
            "the stage-1 divisor is {} rows over {} cells, the data {nrows} over {ncols}",
            self.n_rows,
            self.n_cells()
        );
        let mut basis_dk = Mat::rnorm_seeded(nrows, target_dim, DEFAULT_PROJECTION_SEED);
        if let Some(w) = row_weights {
            anyhow::ensure!(w.len() == nrows, "one projection weight per row");
            for (r, &w) in w.iter().enumerate() {
                if w <= 0.0 {
                    basis_dk.row_mut(r).fill(0.0);
                } else if (w - 1.0).abs() > 1e-6 {
                    basis_dk.row_mut(r).scale_mut(w);
                }
            }
        }
        let basis_kd = basis_dk.transpose();
        info!("Per-cell random projection of the counts divided by the stage-1 divisor...");
        let jobs = generate_minibatch_intervals(ncols, nrows, None);
        // each job fills its own columns of `proj`, held once
        let mut proj = Mat::zeros(target_dim, ncols);
        let mut slots: Vec<(usize, &mut [f32])> = Vec::with_capacity(jobs.len());
        let mut rest = proj.as_mut_slice();
        for &(lb, ub) in &jobs {
            let (mine, tail) = rest.split_at_mut((ub - lb) * target_dim);
            slots.push((lb, mine));
            rest = tail;
        }
        slots
            .into_par_iter()
            .try_for_each(|(lb, out)| -> anyhow::Result<()> {
                let ub = lb + out.len() / target_dim.max(1);
                let mut x = data.read_columns_csc(lb..ub)?;
                self.divide_columns(&mut x, |j| lb + j);
                for v in x.values_mut() {
                    *v = v.ln_1p();
                }
                for (col, out) in x.col_iter().zip(out.chunks_mut(target_dim.max(1))) {
                    let norm = col
                        .values()
                        .iter()
                        .map(|v| v * v)
                        .sum::<f32>()
                        .sqrt()
                        .max(1e-8);
                    for (&i, &v) in col.row_indices().iter().zip(col.values()) {
                        for (o, &b) in out.iter_mut().zip(basis_kd.column(i).iter()) {
                            *o += v / norm * b;
                        }
                    }
                }
                Ok(())
            })?;
        proj.scale_columns_inplace();
        if proj.max() > 4.0 || proj.min() < -4.0 {
            proj.iter_mut().for_each(|x| *x = x.clamp(-4.0, 4.0));
            proj.scale_columns_inplace();
        }
        Ok(RandColProjOut {
            basis: basis_dk,
            proj,
        })
    }
}

/// The options that only set up a batch-effect estimate, refused beside
/// `--offset {dir}`: a stage-2 fit takes no batch effect, the stage-1
/// profiles having been adapted per batch already. `--ignore-batch` asks for
/// what is done anyway.
pub fn refuse_batch_options(
    dir: &str,
    batch: &crate::util::input::BatchArgs,
    collapse: &crate::util::input::CollapseArgs,
) -> anyhow::Result<()> {
    let given = [
        ("--batch-files", batch.batch_files.is_some()),
        ("--auto-batch", batch.auto_batch),
        ("--cnv-clones", collapse.cnv_clones.is_some()),
    ];
    if let Some((flag, _)) = given.iter().find(|(_, given)| *given) {
        anyhow::bail!(
            "{flag} sets up a batch-effect estimate, and a stage-2 fit takes no batch \
             effect (--offset {dir}): the stage-1 divisor already carries each batch's \
             community profiles, and what is left in a batch is its own local structure. \
             Drop {flag}"
        );
    }
    Ok(())
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
