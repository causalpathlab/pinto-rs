//! The exact unit-module term's counts when the trained unit is a cell.
//!
//! cage gathers a step's pseudobulk rows from a dense `[n_pb × G]` table.
//! On cells that table would be features × cells, so a step reads only its
//! drawn cells' sparse columns and pools their nonzero counts through the
//! membership, never as a dense units × features block.

use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use legume_numeric::candle::candle_core::Tensor;
use legume_numeric::candle::fast_index::gather_rows;

/// The cells a cage-cell module term draws from: the data, read sparse a
/// step at a time, and the fold of its rows onto features.
#[derive(Clone, Copy)]
pub(super) struct CellCounts<'a> {
    pub data: &'a SparseIoVec,
    pub feature_axis: &'a FeatureAxis,
}

impl CellCounts<'_> {
    /// `pooled[u, m] = Σ_g x[u, g] · π[g, m]`, `[cells.len() × M]` on `pi`'s
    /// device: the raw counts of `cells` (repeats allowed), their rows
    /// folded onto features, times the membership `pi` `[G × M]`.
    ///
    /// A sparse product: each nonzero count gathers its feature's membership
    /// row and adds it, scaled, to its cell's row, so the largest table is
    /// `[nonzeros × M]`. `pi` must carry no gradient (the term's target is
    /// detached).
    pub(super) fn pool(&self, cells: &[u32], pi: &Tensor) -> anyhow::Result<Tensor> {
        let x = self
            .data
            .read_columns_csc(cells.iter().map(|&c| c as usize))?;
        let (mut unit, mut feature) = (Vec::with_capacity(x.nnz()), Vec::with_capacity(x.nnz()));
        for (u, col) in x.col_iter().enumerate() {
            for &row in col.row_indices() {
                unit.push(u as u32);
                feature.push(self.feature_axis.feature_of_row(row) as u32);
            }
        }
        let (nnz, dev) = (unit.len(), pi.device());
        let pooled = Tensor::zeros((cells.len(), pi.dim(1)?), pi.dtype(), dev)?;
        if nnz == 0 {
            return Ok(pooled);
        }
        let counts = Tensor::from_slice(x.values(), (nnz, 1), dev)?;
        let rows =
            gather_rows(pi, &Tensor::from_vec(feature, nnz, dev)?)?.broadcast_mul(&counts)?;
        Ok(pooled.index_add(&Tensor::from_vec(unit, nnz, dev)?, &rows, 0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::CellCounts;
    use crate::test_support::{numbered, sparse_vec};
    use crate::util::feature_axis::FeatureAxis;
    use graph_embedding_util::loss::masked_membership;
    use legume_numeric::candle::candle_core::{Device, Tensor};

    const N_FEATURES: usize = 3;
    const N_MODULES: usize = 2;
    const N_CELLS: usize = 5;

    /// The pooled table of the drawn cells is the dense product geu's
    /// module term makes (`x · masked_membership(π, keep)`), with a
    /// feature's two channel rows folded, repeats drawn twice, and one row
    /// per drawn cell, never one per cell of the data.
    #[test]
    fn pooled_counts_equal_the_dense_product_of_the_drawn_cells() {
        let dir = tempfile::tempdir().unwrap();
        // two channel rows per feature, so the fold is not the identity
        let rows: Vec<Box<str>> = (0..N_FEATURES)
            .flat_map(|g| ["unspliced", "spliced"].map(|ch| format!("GENE{g}/count/{ch}").into()))
            .collect();
        let mut triplets = Vec::new();
        let mut dense = [0f32; N_CELLS * N_FEATURES];
        for c in 0..N_CELLS {
            for r in 0..rows.len() {
                let v = ((c * 5 + r * 3) % 4) as f32;
                if v > 0.0 {
                    triplets.push((r as u64, c as u64, v));
                    dense[c * N_FEATURES + r / 2] += v;
                }
            }
        }
        let data = sparse_vec(dir.path(), "x", &rows, &numbered("c", N_CELLS), &triplets);
        let axis = FeatureAxis::resolve(&rows).unwrap();
        assert_eq!(axis.n_features(), N_FEATURES);
        let counts = CellCounts {
            data: &data,
            feature_axis: &axis,
        };

        // exact zeros, as sparsemax leaves them
        let pi = [0.75f32, 0.25, 0.0, 1.0, 1.0, 0.0];
        let cells = [3u32, 0, 3, 4];
        let dev = Device::Cpu;
        let x: Vec<f32> = cells
            .iter()
            .flat_map(|&c| dense[c as usize * N_FEATURES..][..N_FEATURES].to_vec())
            .collect();
        let x = Tensor::from_vec(x, (cells.len(), N_FEATURES), &dev).unwrap();
        let pi_t = Tensor::from_slice(&pi, (N_FEATURES, N_MODULES), &dev).unwrap();

        for keep in [None, Some([1f32, 0.0, 1.0]), Some([0f32, 1.0, 0.0])] {
            let pi_masked = match keep {
                None => pi_t.clone(),
                Some(k) => {
                    let k = Tensor::from_slice(&k, (N_FEATURES, 1), &dev).unwrap();
                    masked_membership(&pi_t, &k).unwrap()
                }
            };
            let host = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let want = host(x.matmul(&pi_masked).unwrap());
            let got = counts.pool(&cells, &pi_masked).unwrap();
            assert_eq!(
                got.dims(),
                [cells.len(), N_MODULES],
                "one row per drawn cell"
            );
            for (i, (g, w)) in host(got).iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                    "{keep:?} [{i}]: {g} vs {w}"
                );
            }
            assert!(
                want.iter().any(|&w| w > 0.0),
                "{keep:?}: an empty reference"
            );
        }
    }
}
