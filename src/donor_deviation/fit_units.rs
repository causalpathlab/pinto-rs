//! The units stage 2 is fitted on (`--pairs`): the stage-1 run's cell pairs
//! themselves, or its pb pairs, the cell pairs collapsed by the super-cells
//! of their two cells.
//!
//! A pb pair is every cell pair whose cells sit in the same two super-cells
//! (`{run}.cell_pb.parquet`, cage's finest coarsening), folded by
//! `data_beans::alg::cell_pairs::collapse_pairs`. A cell pair inside one
//! super-cell folds into that super-cell's self-loop `(a, a)`: a contact
//! within a super-cell is still a contact. Every cell pair is in exactly one
//! pb pair, and a pb pair lies in one section (a super-cell never spans
//! one, which is checked).
//!
//! A pb pair's counts are its cell pairs' pooled counts averaged (per cell
//! pair, so a self-loop holding a super-cell's many interior pairs and a
//! cross pair holding a few boundary pairs are on one scale, and neither
//! dominates a contrast set by its size); its state
//! coordinates and its embedding for matching are its cell pairs' means
//! (the embedding renormalized to unit length, as cell pairs' are).
//!
//! The deviation is linear in the state coordinates, so a fit on pb pairs
//! is read out on every cell pair at its own coordinates, at full
//! resolution, with the same coefficients.

use super::stage1_inputs::{PairCountsByGene, Stage1CellPairs};
use crate::util::common::*;
use crate::util::metadata::Run;
use legume_numeric::matrix::parquet::read_table_columns;

/// What `--pairs` fits on.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[clap(rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum FitUnitKind {
    /// Every cell pair.
    Cell,
    /// The cell pairs collapsed by their cells' super-cells, self-loops kept.
    Pb,
}

impl std::fmt::Display for FitUnitKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            FitUnitKind::Cell => "cell",
            FitUnitKind::Pb => "pb",
        })
    }
}

/// The units the contrast is fitted on. For `--pairs cell` the per-unit
/// tables are the cell pairs' own, borrowed, not copied.
pub struct FitUnits<'a> {
    /// Each unit's section.
    pub section_of_unit: std::borrow::Cow<'a, [Box<str>]>,
    /// `[units × embedding dims]`: where each unit's matched other-donor
    /// units are looked up.
    pub embedding_of_unit: std::borrow::Cow<'a, Mat>,
    /// `[units × coordinates_per_pair]`, row-major: each unit's state
    /// coordinates (intercept first).
    pub state_coordinates_of_unit: std::borrow::Cow<'a, [f32]>,
    /// Each gene's counts over the units.
    pub counts: PairCountsByGene,
    /// For every cell pair, the unit it is in (itself for `--pairs cell`).
    pub unit_of_cell_pair: Vec<usize>,
    /// Each pb pair's super-cells, `a-b` (`a <= b`); `None` for cell pairs.
    pub pb_pair_name_of_unit: Option<Vec<Box<str>>>,
}

impl<'a> FitUnits<'a> {
    pub fn number_of_units(&self) -> usize {
        self.section_of_unit.len()
    }

    /// The cell pairs as the units.
    pub fn cell_pairs(
        pairs: &'a Stage1CellPairs,
        counts: PairCountsByGene,
        cell_pair_state_coordinates: &'a [f32],
    ) -> Self {
        Self {
            section_of_unit: std::borrow::Cow::Borrowed(&pairs.section_of_pair),
            embedding_of_unit: std::borrow::Cow::Borrowed(&pairs.stage1_pair_embedding),
            state_coordinates_of_unit: std::borrow::Cow::Borrowed(cell_pair_state_coordinates),
            counts,
            unit_of_cell_pair: (0..pairs.number_of_pairs()).collect(),
            pb_pair_name_of_unit: None,
        }
    }

    /// The cell pairs collapsed into pb pairs by the run's super-cells.
    pub fn pb_pairs(
        run: &Run,
        pairs: &'a Stage1CellPairs,
        cell_pair_counts: &PairCountsByGene,
        cell_pair_state_coordinates: &[f32],
        coordinates_per_pair: usize,
    ) -> anyhow::Result<Self> {
        let super_cell_of_cell = read_super_cell_of_cell(run)?;
        // the pairs' cells as indices, each cell's super-cell its label
        let mut index_of_cell: HashMap<&str, usize> = HashMap::default();
        let mut super_cell_of_index: Vec<usize> = Vec::new();
        let mut cell_index_pairs: Vec<(usize, usize)> = Vec::with_capacity(pairs.number_of_pairs());
        for (left, right) in pairs
            .left_cell_of_pair
            .iter()
            .zip(&pairs.right_cell_of_pair)
        {
            let mut ends = [0usize; 2];
            for (end, cell) in ends.iter_mut().zip([left.as_ref(), right.as_ref()]) {
                *end = match index_of_cell.get(cell) {
                    Some(&i) => i,
                    None => {
                        let super_cell = *super_cell_of_cell.get(cell).ok_or_else(|| {
                            anyhow::anyhow!(
                                "cell {cell} of the stage-1 pairs has no super-cell in cell_pb"
                            )
                        })?;
                        let i = super_cell_of_index.len();
                        super_cell_of_index.push(super_cell);
                        index_of_cell.insert(cell, i);
                        i
                    }
                };
            }
            cell_index_pairs.push((ends[0], ends[1]));
        }
        let (pb_pair_set, unit_of_cell_pair) =
            data_beans::alg::cell_pairs::collapse_pairs(&cell_index_pairs, &super_cell_of_index);
        let number_of_units = pb_pair_set.len();
        let pb_pair_name_of_unit: Vec<Box<str>> = pb_pair_set
            .pairs
            .iter()
            .map(|&(a, b)| format!("{a}-{b}").into())
            .collect();

        // each unit's section, the same for all its cell pairs
        let mut section_of_unit: Vec<Option<Box<str>>> = vec![None; number_of_units];
        for (pair, &unit) in unit_of_cell_pair.iter().enumerate() {
            let section = &pairs.section_of_pair[pair];
            match &section_of_unit[unit] {
                None => section_of_unit[unit] = Some(section.clone()),
                Some(s) => anyhow::ensure!(
                    s == section,
                    "pb pair {} spans sections {s} and {section}: a super-cell crosses sections",
                    pb_pair_name_of_unit[unit]
                ),
            }
        }
        let section_of_unit: Vec<Box<str>> = section_of_unit
            .into_iter()
            .map(|s| s.expect("every unit has a cell pair"))
            .collect();

        let mut cell_pairs_in_unit = vec![0usize; number_of_units];
        for &unit in &unit_of_cell_pair {
            cell_pairs_in_unit[unit] += 1;
        }
        let embedding_of_unit = unit_length_mean_over_units(
            &pairs.stage1_pair_embedding,
            &unit_of_cell_pair,
            number_of_units,
        );
        let state_coordinates_of_unit = mean_flat_rows_over_units(
            cell_pair_state_coordinates,
            coordinates_per_pair,
            &unit_of_cell_pair,
            &cell_pairs_in_unit,
        );
        let counts =
            mean_counts_over_units(cell_pair_counts, &unit_of_cell_pair, &cell_pairs_in_unit);
        info!(
            "{} cell pairs in {number_of_units} pb pairs ({} self-loops)",
            pairs.number_of_pairs(),
            pb_pair_set.pairs.iter().filter(|(a, b)| a == b).count()
        );
        Ok(Self {
            section_of_unit: std::borrow::Cow::Owned(section_of_unit),
            embedding_of_unit: std::borrow::Cow::Owned(embedding_of_unit),
            state_coordinates_of_unit: std::borrow::Cow::Owned(state_coordinates_of_unit),
            counts,
            unit_of_cell_pair,
            pb_pair_name_of_unit: Some(pb_pair_name_of_unit),
        })
    }
}

/// Each cell's super-cell, from the run's `cell_pb.parquet` (`cell`, `pb`).
fn read_super_cell_of_cell(run: &Run) -> anyhow::Result<HashMap<Box<str>, usize>> {
    let path: String = run
        .recorded_output(run.meta.outputs.cell_pb.as_deref(), "cell_pb.parquet")?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{}: records no cell_pb; --pairs pb needs the stage-1 run's super-cells",
                run.manifest.display()
            )
        })?
        .to_string_lossy()
        .into_owned();
    // cage writes the super-cell id as text
    let (columns, _) = read_table_columns(&path, &["cell", "pb"], &[])?;
    columns[0]
        .iter()
        .zip(&columns[1])
        .map(|(cell, pb)| {
            let super_cell = pb.trim().parse::<usize>().map_err(|_| {
                anyhow::anyhow!("{path}: super-cell {pb:?} of cell {cell} is not an id")
            })?;
            Ok((cell.clone(), super_cell))
        })
        .collect()
}

/// Each unit's mean direction of its cell pairs' rows of `rows` (`[cell pairs
/// × dims]`): their sum scaled to unit length.
fn unit_length_mean_over_units(
    rows: &Mat,
    unit_of_cell_pair: &[usize],
    number_of_units: usize,
) -> Mat {
    let mut sums = Mat::zeros(number_of_units, rows.ncols());
    for (pair, &unit) in unit_of_cell_pair.iter().enumerate() {
        let mut row = sums.row_mut(unit);
        row += rows.row(pair);
    }
    legume_numeric::matrix::dense_mat_io::l2_normalize_rows_inplace(&mut sums);
    sums
}

/// Each unit's mean of its cell pairs' rows of a flat row-major
/// `[cell pairs × width]` vector.
pub(super) fn mean_flat_rows_over_units(
    rows: &[f32],
    width: usize,
    unit_of_cell_pair: &[usize],
    cell_pairs_in_unit: &[usize],
) -> Vec<f32> {
    let mut means = vec![0f32; cell_pairs_in_unit.len() * width];
    for (pair, &unit) in unit_of_cell_pair.iter().enumerate() {
        for (m, x) in means[unit * width..(unit + 1) * width]
            .iter_mut()
            .zip(&rows[pair * width..(pair + 1) * width])
        {
            *m += x / cell_pairs_in_unit[unit] as f32;
        }
    }
    means
}

/// Each gene's counts over the units: its cell pairs' pooled counts
/// averaged over the unit's `cell_pairs_in_unit` cell pairs.
pub(super) fn mean_counts_over_units(
    cell_pair_counts: &PairCountsByGene,
    unit_of_cell_pair: &[usize],
    cell_pairs_in_unit: &[usize],
) -> PairCountsByGene {
    let nonzero_pairs_of_gene = cell_pair_counts
        .nonzero_pairs_of_gene
        .par_iter()
        .map(|nonzero| {
            let mut summed: HashMap<u32, f32> = HashMap::default();
            for &(pair, count) in nonzero {
                *summed
                    .entry(unit_of_cell_pair[pair as usize] as u32)
                    .or_insert(0.0) += count;
            }
            let mut v: Vec<(u32, f32)> = summed
                .into_iter()
                .map(|(unit, total)| (unit, total / cell_pairs_in_unit[unit as usize] as f32))
                .collect();
            v.sort_unstable_by_key(|&(unit, _)| unit);
            v
        })
        .collect();
    PairCountsByGene {
        gene_names: cell_pair_counts.gene_names.clone(),
        nonzero_pairs_of_gene,
    }
}
