//! The genes a run needs, read once and kept sparse.
//!
//! Ligands, receptors and every programme's targets are read in one pass
//! over the matrix, keeping only their non-zero counts on receiver rows,
//! and every cell's total count is taken in the same pass. A dense
//! genes × cells block would be the run's largest allocation by far
//! (thousands of target genes × a million cells); the non-zeros are a few
//! percent of it.

use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use legume_numeric::matrix::utils::generate_minibatch_intervals;

/// Non-zero counts per gene, as `(row, count)` sorted by row.
pub struct GeneStore {
    nz: Vec<Vec<(u32, f32)>>,
}

impl GeneStore {
    /// Read `genes` (gene-axis indices) for the rows `row_of_cell` maps
    /// (`usize::MAX` = not a receiver), and `log1p(total count)` for every
    /// cell. Splice channels of one gene are summed.
    pub fn read(
        data: &SparseIoVec,
        axis: &FeatureAxis,
        genes: &[usize],
        row_of_cell: &[usize],
        block_size: Option<usize>,
    ) -> anyhow::Result<(Self, Vec<f32>)> {
        let n_cells = data.num_columns();
        let mut local_of_gene: HashMap<usize, u32> = HashMap::default();
        for (k, &g) in genes.iter().enumerate() {
            local_of_gene.insert(g, k as u32);
        }
        let local_of_row: Vec<u32> = (0..data.num_rows())
            .map(|r| {
                local_of_gene
                    .get(&axis.feature_of_row(r))
                    .copied()
                    .unwrap_or(u32::MAX)
            })
            .collect();
        let jobs = generate_minibatch_intervals(n_cells, data.num_rows(), block_size);
        type Block = (usize, Vec<f32>, Vec<(u32, u32, f32)>);
        let pb = new_progress_bar(jobs.len() as u64).with_message("reading gene blocks");
        let blocks: Vec<Block> = jobs
            .par_iter()
            .map(|&(lb, ub)| -> anyhow::Result<Block> {
                let x = data.read_columns_csc(lb..ub)?;
                let mut depth = vec![0.0f32; ub - lb];
                let mut kept = Vec::new();
                for (c, col) in x.col_iter().enumerate() {
                    let cell = lb + c;
                    depth[c] = col.values().iter().sum::<f32>().ln_1p();
                    let row = row_of_cell[cell];
                    if row == usize::MAX {
                        continue;
                    }
                    for (&r, &v) in col.row_indices().iter().zip(col.values()) {
                        let g = local_of_row[r];
                        if g != u32::MAX && v != 0.0 {
                            kept.push((g, row as u32, v));
                        }
                    }
                }
                pb.inc(1);
                Ok((lb, depth, kept))
            })
            .collect::<anyhow::Result<_>>()?;
        pb.finish_and_clear();
        let mut log1p_depth = vec![0.0f32; n_cells];
        let mut nz: Vec<Vec<(u32, f32)>> = vec![Vec::new(); genes.len()];
        for (lb, depth, kept) in blocks {
            log1p_depth[lb..lb + depth.len()].copy_from_slice(&depth);
            for (g, row, v) in kept {
                nz[g as usize].push((row, v));
            }
        }
        for v in &mut nz {
            v.sort_unstable_by_key(|&(r, _)| r);
            // Splice channels of one gene arrive as separate entries.
            v.dedup_by(|b, a| {
                if a.0 == b.0 {
                    a.1 += b.1;
                    true
                } else {
                    false
                }
            });
        }
        Ok((Self { nz }, log1p_depth))
    }

    pub fn nonzeros(&self, local: usize) -> &[(u32, f32)] {
        &self.nz[local]
    }

    /// Gene `local` as a dense vector over `n` rows.
    pub fn dense(&self, local: usize, n: usize) -> Vec<f64> {
        let mut v = vec![0.0; n];
        for &(r, x) in &self.nz[local] {
            v[r as usize] = x as f64;
        }
        v
    }

    pub fn total(&self, local: usize) -> f64 {
        self.nz[local].iter().map(|&(_, x)| x as f64).sum()
    }
}
