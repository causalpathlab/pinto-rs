//! Which data file measured which feature, for data files whose feature
//! panels differ and are read as their union. A file that never measured
//! a feature is no evidence of zero there: a feature's mass is summed over
//! the cells of the files that measured it.

use crate::util::common::*;

/// Each file's measured rows, and each column's file.
pub struct Coverage {
    by_file: Vec<Vec<bool>>,
    file_of: Vec<u32>,
}

impl Coverage {
    /// `None` when every file measured every row of `data`, or when a
    /// cell merges several files (a cell has one panel here, and such a
    /// cell has several: its unmeasured rows then read as zeros).
    pub fn of(data: &SparseIoVec) -> Option<Self> {
        let by_file = data.row_coverage_by_backend()?;
        let file_of: Option<Vec<u32>> = (0..data.num_columns())
            .map(|c| data.column_source(c).map(|b| b as u32))
            .collect();
        let Some(file_of) = file_of else {
            log::warn!(
                "cells merge several data files: rows some file never measured read as zeros"
            );
            return None;
        };
        Some(Coverage { by_file, file_of })
    }

    pub fn n_files(&self) -> usize {
        self.by_file.len()
    }

    /// The file column `col` comes from.
    pub fn file(&self, col: usize) -> usize {
        self.file_of[col] as usize
    }

    /// Whether column `col`'s file measured row `row`; the viewer's gene
    /// maps ask it per cell.
    #[cfg(any(feature = "view", test))]
    pub fn measured(&self, row: usize, col: usize) -> bool {
        self.by_file[self.file(col)][row]
    }
}

/// Row `row`'s mass per group: the mass `n_bk` (files × groups) of the
/// files that measured it, or of every file without `coverage`.
pub fn measured_mass(n_bk: &Mat, coverage: Option<&Coverage>, row: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_bk.ncols()];
    for b in 0..n_bk.nrows() {
        if coverage.is_none_or(|c| c.by_file[b][row]) {
            for (o, &m) in out.iter_mut().zip(n_bk.row(b).iter()) {
                *o += m;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_feature_s_mass_is_that_of_the_files_that_measured_it() {
        // Group mass by file: file 0 holds [2, 0], file 1 [1, 3].
        let n_bk = Mat::from_row_slice(2, 2, &[2., 0., 1., 3.]);
        // GENE1 in both files, GENE2 only in file 0, GENE3 only in file 1.
        let cov = Coverage {
            by_file: vec![vec![true, true, false], vec![true, false, true]],
            file_of: vec![0, 1],
        };
        let mass = |row| measured_mass(&n_bk, Some(&cov), row);
        assert_eq!(
            (mass(0), mass(1), mass(2)),
            (vec![3., 3.], vec![2., 0.], vec![1., 3.])
        );
        // Every file measuring every feature: the whole mass.
        assert_eq!(measured_mass(&n_bk, None, 1), vec![3., 3.]);
        assert!(cov.measured(1, 0) && !cov.measured(1, 1));
    }
}
