//! Null draws of the contact effect, `pinto lra --null-permutations K`: the
//! same estimator on a ligand shuffled where it cannot reach the response,
//! written to their own table (`{table}.null.parquet`) for the reader to
//! compare the real statistic with. lra judges nothing by them.
//!
//! A draw shuffles the ligand's counts among the rows of one stratum, an
//! identity bin within a spatial tile (tiles lie within a section): the
//! ligand keeps its cell-type make-up and its coarse spatial pattern, and
//! only the fine alignment of contacts with responders breaks. Cells are
//! still not exchangeable across contacts, so this null is narrower than the
//! truth's; it is one comparison, beside N(0, 1) and others to come (decoy
//! ligands), none privileged.
//!
//! A draw is seeded by the run's seed, the ligand and the draw's index, so
//! the pairs of one ligand see one shuffle (and share its fits, as in the
//! real run), and a draw is the same wherever it runs.

use super::inference::Estimate;
use super::rows::EffectRow;
use super::tissue::Tissue;
use legume_numeric::matrix::rand_util::mix_seed;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Reads one estimator's estimate off a row.
type EstimatorOfRow = fn(&EffectRow) -> Estimate;

/// The estimators a null draw reports: the headline (doubly robust) and the
/// outcome regression alone, so either can be read as the headline.
pub const NULL_ESTIMATORS: [(&str, EstimatorOfRow); 2] = [
    ("dr", |r| r.effects.competent),
    ("outcome", |r| r.effects.outcome),
];

/// One estimate under one null draw: a row of `{table}.null.parquet`.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct NullRow {
    pub section: Box<str>,
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    /// `permutation`.
    pub null: Box<str>,
    pub draw: u32,
    /// A name of [`NULL_ESTIMATORS`].
    pub estimator: Box<str>,
    pub estimate: Estimate,
}

/// The seed of draw `draw` of `ligand` (its genes' names joined, as the pair
/// names it) under the run's `seed`. FNV-1a, which no Rust release changes.
pub fn draw_seed(seed: u64, ligand: &str, draw: u32) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in ligand.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    mix_seed(mix_seed(seed, h), u64::from(draw))
}

/// `counts` (one per tissue row) shuffled within each identity bin × tile.
pub fn permuted_within_strata(tissue: &Tissue, counts: &[f64], seed: u64) -> Vec<f64> {
    let mut strata: Vec<(usize, u32, u32)> = (0..counts.len())
        .map(|i| (tissue.identity_bin[i], tissue.cf.cluster[i], i as u32))
        .collect();
    strata.sort_unstable();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut out = counts.to_vec();
    for stratum in strata.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)) {
        let rows: Vec<usize> = stratum.iter().map(|s| s.2 as usize).collect();
        let mut values: Vec<f64> = rows.iter().map(|&i| counts[i]).collect();
        values.shuffle(&mut rng);
        for (&i, v) in rows.iter().zip(values) {
            out[i] = v;
        }
    }
    out
}

/// The null rows of one pair's `rows` under permutation draw `draw`: each
/// row's estimate by each of [`NULL_ESTIMATORS`].
pub fn null_rows(rows: &[EffectRow], draw: u32) -> Vec<NullRow> {
    rows.iter()
        .flat_map(|r| {
            NULL_ESTIMATORS.iter().map(move |(name, of)| NullRow {
                section: r.section.clone(),
                ligand: r.ligand.clone(),
                receptor: r.receptor.clone(),
                null: "permutation".into(),
                draw,
                estimator: (*name).into(),
                estimate: of(r),
            })
        })
        .collect()
}
