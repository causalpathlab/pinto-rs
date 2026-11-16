//! The receivers as the analysis sees them: which cells they are, their
//! contacts and their size factors. [`super::tissue::Tissue::build`] takes
//! them from here, so `pinto view` (`frame`) draws a pair as the table saw it.

use super::contacts::Contacts;
use crate::util::common::{log_cp10k, LIBRARY_SCALE};

/// Rows of the analysis: cells with coordinates and at least one contact
/// to another such cell. A sender is itself in that set, so contacts stay
/// inside it.
fn receiver_cells(coords: &[Option<(f32, f32)>], pairs: &[(usize, usize)]) -> Vec<usize> {
    let mut degree = vec![0usize; coords.len()];
    for &(i, j) in pairs {
        if i != j && coords[i].is_some() && coords[j].is_some() {
            degree[i] += 1;
            degree[j] += 1;
        }
    }
    (0..coords.len()).filter(|&c| degree[c] > 0).collect()
}

/// Receivers (cells with coordinates and a contact to another such cell)
/// by data column, and each data column's row (`usize::MAX` for none).
pub fn receiver_rows(
    coords: &[Option<(f32, f32)>],
    pairs: &[(usize, usize)],
) -> (Vec<usize>, Vec<usize>) {
    let cell = receiver_cells(coords, pairs);
    let mut row_of = vec![usize::MAX; coords.len()];
    for (r, &c) in cell.iter().enumerate() {
        row_of[c] = r;
    }
    (cell, row_of)
}

/// [`receiver_rows`], and the contacts between rows.
pub fn receiver_contacts(
    coords: &[Option<(f32, f32)>],
    pairs: &[(usize, usize)],
) -> (Vec<usize>, Vec<usize>, Contacts) {
    let (cell, row_of) = receiver_rows(coords, pairs);
    let row_pairs: Vec<(usize, usize)> = pairs
        .iter()
        .filter(|&&(i, j)| row_of[i] != usize::MAX && row_of[j] != usize::MAX)
        .map(|&(i, j)| (row_of[i], row_of[j]))
        .collect();
    let contacts = Contacts::from_pairs(cell.len(), &row_pairs);
    (cell, row_of, contacts)
}

/// Each receiver's size factor, from `log1p(total count)`: its total count
/// (at least 1) over [`LIBRARY_SCALE`], the scale every method takes a unit
/// to before a log. It depends on the receiver alone, not on the other
/// receivers of the run, so a count over it is the count per
/// [`LIBRARY_SCALE`] of the cell's total.
pub fn size_factors(log_depth: &[f64]) -> Vec<f64> {
    let scale = f64::from(LIBRARY_SCALE);
    log_depth
        .iter()
        .map(|l| l.exp_m1().max(1.0) / scale)
        .collect()
}

/// `ln(1 + count / size factor)`, the size factor one of [`size_factors`]:
/// [`log_cp10k`] of the count in the cell's total, which every log of a
/// depth-normalized count here goes through.
pub fn log_normalized(count: f64, size_factor: f64) -> f64 {
    log_cp10k(count, size_factor * f64::from(LIBRARY_SCALE))
}
