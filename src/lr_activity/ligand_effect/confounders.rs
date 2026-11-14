//! The confounder matrix shared by every ligand-receptor pair.
//!
//! | Columns | Stands for |
//! |---|---|
//! | intercept, section indicators | batch, donor |
//! | random Fourier features of the coordinates, per section | region: smooth fields |
//! | receiver identity (random projection) | cell type and state |
//! | contact-neighbour mean identity | neighbour composition |
//! | cell-type indicators (optional) | annotated cell type |
//! | share of each cell type among contacts (optional) | annotated sender cell type |
//! | region indicators within each section (optional) | a labelled region, e.g. a residual-fit patch |
//! | receiver log depth (piecewise linear), neighbour mean log depth | detection |
//!
//! The smooth fields stand in for unmeasured regional confounders because
//! contact exposure varies at the scale of one cell while they vary over
//! many (Gilbert, Datta, Casey & Ogburn, arXiv 2112.14946; Paciorek 2010).
//! Their length scales are therefore set well above one contact length;
//! a smooth at contact scale would absorb the exposure itself. Composition,
//! identity and depth vary at the fine scale, so they are measured
//! directly instead. Sections do not share a coordinate frame, so each has
//! smooth columns of its own, and the sections split the frequencies of one
//! ([`freq_per_section`]) so the matrix does not widen with them. Link
//! communities are deliberately absent: they are built from expression that
//! includes the ligand and the targets, so conditioning on them could open a
//! collider path.
//!
//! The identity projection is computed with ligand, receptor and target
//! genes weighted out, so a receiver's identity is not partly its outcome.
//! Columns specific to one pair (own ligand, spillover, bleed, niche
//! ligand, receptor competence) are added per pair, not here.
//!
//! The groups are laid out in the order of the sequential adjustment
//! (section, smooth, identity, depth), so each adjustment step is a column
//! prefix of the one matrix and needs no copy of it.

use super::contacts::{quantile_sorted, Contacts};
use legume_numeric::matrix::rand_util::mix_seed;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

/// Random Fourier features approximating a squared-exponential field of
/// length `scale × unit[section]`: `√2 cos(ω·s / ℓ + b)`, with `ω` standard
/// normal and `b` uniform on `[0, 2π)`. A feature runs over every section,
/// scaled by each section's unit; [`build`] keeps it within one at a time.
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

/// The fewest frequencies a section's smooth field is given.
const MIN_SECTION_FREQ: usize = 16;

/// Frequencies of each section's smooth field at one scale: `n_freq` split
/// across the sections, so the columns stay near `n_freq` per scale, and at
/// least [`MIN_SECTION_FREQ`] (or `n_freq`, if smaller) each.
pub fn freq_per_section(n_freq: usize, n_sections: usize) -> usize {
    (n_freq / n_sections.max(1))
        .max(MIN_SECTION_FREQ)
        .min(n_freq)
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
    /// dropped) and the share of each of those types among the row's
    /// contacts join the identity block.
    pub cell_type: Option<&'a [u32]>,
    /// Optional region code per row: an indicator per region within each
    /// section ([`region_levels`]), after the cell types. The smooth cannot
    /// follow a sharp region edge; these can.
    pub region: Option<&'a [u32]>,
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
    let region_cols = inp
        .region
        .map_or_else(Vec::new, |r| region_levels(inp.section, r));
    let n_fields = inp.n_sections.max(1);
    let n_freq = freq_per_section(inp.n_freq, n_fields);
    let p = 1
        + n_section_cols
        + inp.scales.len() * n_fields * n_freq
        + 2 * k
        + 2 * n_type_cols
        + region_cols.len()
        + DEPTH_COLS;
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
            n_freq,
            mix_seed(inp.seed, si as u64),
        );
        // A field of its own per section: each feature is kept within one
        // section and is zero, before standardizing, in the others.
        let mut within = vec![0.0; n];
        for field in 0..n_fields as u32 {
            for col in &f {
                for ((w, &v), &s) in within.iter_mut().zip(col).zip(inp.section) {
                    *w = if s == field { v } else { 0.0 };
                }
                put(&mut x, &within);
            }
        }
        close(&mut groups, format!("smooth@{scale}"), n_fields * n_freq);
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
        // The receiver's type, then the senders': the share of each type
        // among its contacts.
        let is =
            |level: u32| -> Vec<f64> { ct.iter().map(|&c| (c == level) as u8 as f64).collect() };
        for &level in &levels[1..] {
            put(&mut x, &is(level));
        }
        close(&mut groups, "cell type".into(), n_type_cols);
        for &level in &levels[1..] {
            put(&mut x, &inp.contacts.mean(&is(level)));
        }
        close(&mut groups, "neighbour cell type".into(), n_type_cols);
    }

    if let Some(r) = inp.region {
        for &(sec, level) in &region_cols {
            let col: Vec<f64> = inp
                .section
                .iter()
                .zip(r)
                .map(|(&s, &v)| (s == sec && v == level) as u8 as f64)
                .collect();
            put(&mut x, &col);
        }
        close(&mut groups, "region".into(), region_cols.len());
    }

    for col in depth_basis(inp.log_depth) {
        put(&mut x, &col);
    }
    put(&mut x, &inp.contacts.mean(inp.log_depth));
    close(&mut groups, "depth".into(), DEPTH_COLS);
    debug_assert_eq!(at, p);
    debug_assert_eq!(groups.last().map(|g| g.cols.end), Some(p));
    (x, groups)
}

/// The region indicator columns, as (section, region) pairs: within each
/// section, every region present but the section's most common one (the
/// section's intercept stands for it), in section then region order. A
/// region gets a column per section, so a region that differs between
/// donors is not forced to one shift.
pub fn region_levels(section: &[u32], region: &[u32]) -> Vec<(u32, u32)> {
    let mut count: std::collections::BTreeMap<(u32, u32), usize> = Default::default();
    for (&s, &r) in section.iter().zip(region) {
        *count.entry((s, r)).or_default() += 1;
    }
    let mut out = Vec::new();
    let mut at = count.iter().peekable();
    while let Some(&(&(s, _), _)) = at.peek() {
        let in_section: Vec<(u32, usize)> = std::iter::from_fn(|| {
            at.next_if(|((s2, _), _)| *s2 == s)
                .map(|(&(_, r), &n)| (r, n))
        })
        .collect();
        // The most common region, ties to the smaller code, is the reference.
        let reference = in_section
            .iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
            .map(|&(r, _)| r);
        out.extend(
            in_section
                .iter()
                .filter(|&&(r, _)| Some(r) != reference)
                .map(|&(r, _)| (s, r)),
        );
    }
    out
}

/// Knots of the receiver's depth basis, as quantiles of its log depth.
const DEPTH_KNOTS: [f64; 4] = [0.2, 0.4, 0.6, 0.8];

/// The depth group's width: the receiver's depth basis and the neighbours'
/// mean log depth.
pub const DEPTH_COLS: usize = DEPTH_KNOTS.len() + 2;

/// The receiver's log depth, piecewise linear: the log depth and a hinge
/// `max(0, d − q)` at each knot. Depth drives detection, so both the
/// response and the pair columns read off counts (own ligand, competence)
/// depend on it nonlinearly; one slope leaves a remainder that conditioning
/// on those pair columns ties to exposure.
pub(super) fn depth_basis(log_depth: &[f64]) -> Vec<Vec<f64>> {
    let mut sorted = log_depth.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let at = |q: f64| quantile_sorted(&sorted, q).unwrap_or(0.0);
    std::iter::once(log_depth.to_vec())
        .chain(DEPTH_KNOTS.iter().map(|&q| {
            let k = at(q);
            log_depth.iter().map(|&d| (d - k).max(0.0)).collect()
        }))
        .collect()
}
