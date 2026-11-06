//! The confounder matrix shared by every ligand-receptor pair.
//!
//! | Columns | Stands for |
//! |---|---|
//! | intercept, section indicators | batch, donor |
//! | random Fourier features of the coordinates | region, microenvironment: smooth fields |
//! | receiver identity (random projection) | cell type and state |
//! | contact-neighbour mean identity | neighbour composition |
//! | cell-type indicators (optional) | annotated cell type |
//! | receiver log depth, neighbour mean log depth | detection |
//!
//! The smooth fields stand in for unmeasured regional confounders because
//! contact exposure varies at the scale of one cell while they vary over
//! many (Gilbert, Datta, Casey & Ogburn, arXiv 2112.14946; Paciorek 2010).
//! Their length scales are therefore set well above one contact length;
//! a smooth at contact scale would absorb the exposure itself. Composition,
//! identity and depth vary at the fine scale, so they are measured
//! directly instead. Link communities are deliberately absent: they are
//! built from expression that includes the ligand and the targets, so
//! conditioning on them could open a collider path.
//!
//! The identity projection is computed with ligand, receptor and target
//! genes weighted out, so a receiver's identity is not partly its outcome.
//! Columns specific to one pair (own ligand, spillover, bleed, niche
//! ligand, receptor competence) are added per pair, not here.
//!
//! The groups are laid out in the order of the sequential adjustment
//! (section, smooth, identity, depth), so each adjustment step is a column
//! prefix of the one matrix and needs no copy of it.

use super::exposure::Contacts;
use legume_numeric::matrix::rand_util::mix_seed;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

/// Random Fourier features approximating a squared-exponential field of
/// length `scale × unit[section]`: `√2 cos(ω·s / ℓ + b)`, with `ω` standard
/// normal and `b` uniform on `[0, 2π)`. Frequencies are shared across
/// sections, scaled by each section's unit.
pub fn fourier_features(
    coords: &[(f32, f32)],
    section: &[u32],
    unit: &[f64],
    scale: f64,
    n_freq: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let phase = Uniform::new(0.0, std::f64::consts::TAU).expect("valid range");
    let freqs: Vec<(f64, f64, f64)> = (0..n_freq)
        .map(|_| {
            (
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                phase.sample(&mut rng),
            )
        })
        .collect();
    freqs
        .iter()
        .map(|&(wx, wy, b)| {
            coords
                .iter()
                .zip(section)
                .map(|(&(x, y), &s)| {
                    let l = scale * unit[s as usize];
                    std::f64::consts::SQRT_2 * ((wx * x as f64 + wy * y as f64) / l + b).cos()
                })
                .collect()
        })
        .collect()
}

pub struct ConfounderInputs<'a> {
    pub coords: &'a [(f32, f32)],
    pub section: &'a [u32],
    pub n_sections: usize,
    /// Median contact length per section.
    pub unit: &'a [f64],
    pub contacts: &'a Contacts,
    /// Identity projection, `K × n` (column per row).
    pub identity: &'a nalgebra::DMatrix<f32>,
    pub log_depth: &'a [f64],
    /// Optional cell-type code per row; one-hot columns (first level
    /// dropped) join the identity block.
    pub cell_type: Option<&'a [u32]>,
    /// Smooth length scales in contact lengths.
    pub scales: &'a [f64],
    pub n_freq: usize,
    pub seed: u64,
}

/// A named block of confounder columns.
#[derive(Clone, Debug)]
pub struct Group {
    pub name: String,
    pub cols: std::ops::Range<usize>,
}

/// Assemble the raw confounder matrix (unstandardized; [`super::crossfit::Design`]
/// standardizes it) and the column range of each group, in matrix order.
pub fn build(inp: &ConfounderInputs<'_>) -> (DMatrix<f32>, Vec<Group>) {
    let n = inp.coords.len();
    let k = inp.identity.nrows();
    let n_section_cols = inp.n_sections.saturating_sub(1);
    let levels: Vec<u32> = inp.cell_type.map_or_else(Vec::new, |ct| {
        let mut l = ct.to_vec();
        l.sort_unstable();
        l.dedup();
        l
    });
    let n_type_cols = levels.len().saturating_sub(1);
    let p = 1 + n_section_cols + inp.scales.len() * inp.n_freq + 2 * k + n_type_cols + 2;
    // Filled column by column, so the only full-size allocation is the
    // matrix itself.
    let mut x = DMatrix::<f32>::zeros(n, p);
    let mut at = 0usize;
    let mut put = |x: &mut DMatrix<f32>, col: &[f64]| {
        for (dst, &v) in x.column_mut(at).iter_mut().zip(col) {
            *dst = v as f32;
        }
        at += 1;
    };
    let mut groups: Vec<Group> = Vec::new();
    let close = |groups: &mut Vec<Group>, name: String, len: usize| {
        let start = groups.last().map_or(0, |g| g.cols.end);
        groups.push(Group {
            name,
            cols: start..start + len,
        });
    };

    put(&mut x, &vec![1.0; n]);
    close(&mut groups, "intercept".into(), 1);
    for b in 1..inp.n_sections {
        let col: Vec<f64> = inp
            .section
            .iter()
            .map(|&s| (s as usize == b) as u8 as f64)
            .collect();
        put(&mut x, &col);
    }
    close(&mut groups, "section".into(), n_section_cols);

    for (si, &scale) in inp.scales.iter().enumerate() {
        let f = fourier_features(
            inp.coords,
            inp.section,
            inp.unit,
            scale,
            inp.n_freq,
            mix_seed(inp.seed, si as u64),
        );
        for col in &f {
            put(&mut x, col);
        }
        close(&mut groups, format!("smooth@{scale}"), f.len());
    }

    for r in 0..k {
        let col: Vec<f64> = inp.identity.row(r).iter().map(|&v| v as f64).collect();
        put(&mut x, &col);
    }
    for r in 0..k {
        let col: Vec<f64> = inp.identity.row(r).iter().map(|&v| v as f64).collect();
        put(&mut x, &inp.contacts.mean(&col));
    }
    close(&mut groups, "identity".into(), k);
    close(&mut groups, "neighbour identity".into(), k);

    if let Some(ct) = inp.cell_type {
        for &level in levels.iter().skip(1) {
            let col: Vec<f64> = ct.iter().map(|&c| (c == level) as u8 as f64).collect();
            put(&mut x, &col);
        }
        close(&mut groups, "cell type".into(), n_type_cols);
    }

    put(&mut x, inp.log_depth);
    put(&mut x, &inp.contacts.mean(inp.log_depth));
    close(&mut groups, "depth".into(), 2);
    debug_assert_eq!(at, p);
    debug_assert_eq!(groups.last().map(|g| g.cols.end), Some(p));
    (x, groups)
}
