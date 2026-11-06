//! The receivers as the analysis sees them: which cells they are, their
//! contacts, their size factors and each one's contact dose of a ligand.
//! [`super::tissue::Tissue::build`] takes its receivers, contacts and size
//! factors from here and the estimate its dose, so `pinto view`
//! ([`super::frame`]) draws a pair as the table saw it.

use super::exposure::{upper_median, Contacts};

/// Rows of the analysis: cells with coordinates and at least one contact
/// to another such cell. A sender is itself in that set, so contacts stay
/// inside it.
pub fn receiver_cells(coords: &[Option<(f32, f32)>], pairs: &[(usize, usize)]) -> Vec<usize> {
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

/// Each receiver's depth over the median depth, from `log1p(total count)`.
pub fn size_factors(log_depth: &[f64]) -> Vec<f64> {
    let depth: Vec<f64> = log_depth.iter().map(|l| l.exp_m1().max(1.0)).collect();
    let med = upper_median(&mut depth.clone()).unwrap_or(1.0).max(1.0);
    depth.iter().map(|d| d / med).collect()
}

/// Each row's ligand over its size factor, and its contact dose: that
/// summed over its contacts, which the exposure threshold reads.
pub fn contact_dose(
    contacts: &Contacts,
    size_factor: &[f64],
    ligand_counts: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let lnorm: Vec<f64> = ligand_counts
        .iter()
        .zip(size_factor)
        .map(|(c, s)| c / s)
        .collect();
    let a = contacts.sum(&lnorm);
    (lnorm, a)
}
