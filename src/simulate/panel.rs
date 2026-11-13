//! Donors and sections. Donors are planted here: a section's donor is set by
//! the panel, never read from real metadata.

use super::params::SimParams;

pub type Coords = Vec<(f64, f64)>;

pub struct Section {
    /// `S1`, `S2`, …: also the stem of the section's backend file.
    pub name: Box<str>,
    pub donor: usize,
    /// Spot names as the backend's column names.
    pub barcodes: Vec<Box<str>>,
    pub coords: Coords,
}

impl Section {
    pub fn n(&self) -> usize {
        self.coords.len()
    }
}

pub fn donor_label(d: usize) -> Box<str> {
    format!("D{}", d + 1).into()
}

/// The donor a [`donor_label`] names.
pub fn parse_donor_label(label: &str) -> Option<usize> {
    label
        .strip_prefix('D')
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .map(|n| n - 1)
}

/// A square grid, `side × side` spots `spacing_um` apart.
pub fn grid(side: usize, spacing_um: f64) -> Coords {
    (0..side * side)
        .map(|k| {
            (
                (k / side) as f64 * spacing_um,
                (k % side) as f64 * spacing_um,
            )
        })
        .collect()
}

/// Spot positions from a coordinate file's columns `names`. Only positions
/// are kept: a real file's spot names and other columns never reach output.
pub fn read_coords(path: &str, names: &[Box<str>]) -> anyhow::Result<Coords> {
    let m = crate::util::input::read_one_coord_file(path, &[], names, None)?;
    anyhow::ensure!(
        m.mat.ncols() >= 2,
        "{path}: fewer than two coordinate columns"
    );
    anyhow::ensure!(m.mat.nrows() >= 1, "{path}: no spots");
    Ok((0..m.mat.nrows())
        .map(|i| (m.mat[(i, 0)] as f64, m.mat[(i, 1)] as f64))
        .collect())
}

/// Sections `S1..` in donor order, `sections_per_donor` each; geometries are
/// used in turn, cycling when there are fewer than sections.
pub fn build_panel(p: &SimParams, geometries: &[Coords]) -> Vec<Section> {
    (0..p.n_sections())
        .map(|s| {
            let name: Box<str> = format!("S{}", s + 1).into();
            let coords = geometries[s % geometries.len()].clone();
            let barcodes = (0..coords.len())
                .map(|i| format!("{name}_{:05}", i + 1).into())
                .collect();
            Section {
                donor: s / p.sections_per_donor,
                name,
                barcodes,
                coords,
            }
        })
        .collect()
}
