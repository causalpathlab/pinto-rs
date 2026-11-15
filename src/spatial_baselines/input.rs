//! The inputs every spatial baseline reads, through pinto's data-beans
//! reader ([`crate::util::input::read_data_with_coordinates`]), so cells are
//! named, ordered and split into sections exactly as `lc`, `dsvd` and `cage`
//! name, order and split them.

use super::memory::{delta_estimate_bytes, DELTA_SAMPLES_PER_BATCH};
use crate::util::ablation;
use crate::util::common::{HashSet, Mat};
use crate::util::input::{
    read_data_with_coordinates, SRTReadArgs, COORD_COLUMN_NAMES, PRELOAD_HELP,
};
use anyhow::Context;
use data_beans::aux::feature_names::FeatureNameKind;
use data_beans::convert::try_open_or_convert;
use data_beans::sparse_io::SparseIo;
use nalgebra_sparse::CscMatrix;

/// Data files, their coordinates, the output prefix and the seed.
#[derive(clap::Args, Debug, Clone)]
pub struct SpatialInputArgs {
    /// Expression data files (.zarr or .h5), comma separated; each file is a
    /// section
    #[arg(required = true, value_delimiter(','))]
    pub data_files: Vec<Box<str>>,

    /// Coordinate files, one per data file, comma separated (as `pinto lc -c`)
    #[arg(long = "coord", short = 'c', value_delimiter(','), required = true)]
    pub coord_files: Vec<Box<str>>,

    /// 0-based coordinate column indices (overrides --coord-column-names)
    #[arg(long = "coord-column-indices", value_delimiter(','))]
    pub coord_columns: Option<Vec<usize>>,

    /// Coordinate column names (as `pinto lc`)
    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        default_value = COORD_COLUMN_NAMES
    )]
    pub coord_column_names: Vec<Box<str>>,

    /// Batch label files, one per data file (as `pinto lc -b`): each
    /// (file, batch) is a section of its own
    #[arg(long, short = 'b', value_delimiter(','))]
    pub batch_files: Option<Vec<Box<str>>>,

    /// Header row index in coordinate files (default: detected)
    #[arg(long, hide = true)]
    pub coord_header_row: Option<usize>,

    #[arg(long, default_value_t = false, help = PRELOAD_HELP)]
    pub preload_data: bool,

    /// Run the original method: no batch effects (δ) are estimated or
    /// divided out. Without it, a multi-batch input's counts are divided by
    /// its own δ estimate (as cage estimates it) before the method's own
    /// preprocessing
    #[arg(long, default_value_t = false)]
    pub ignore_batch: bool,

    /// Output prefix
    #[arg(long, short, required = true)]
    pub out: Box<str>,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

/// Counts and coordinates of every cell, sections kept apart.
pub struct SpatialData {
    /// Raw counts, genes × cells.
    pub counts: CscMatrix<f32>,
    pub genes: Vec<Box<str>>,
    pub cells: Vec<Box<str>>,
    /// Spatial coordinates, cells × dimensions (no batch column).
    pub xy: Mat,
    /// The coordinate columns' names.
    pub coord_names: Vec<Box<str>>,
    /// Each cell's batch label, as the reader gives it.
    pub batches: Vec<Box<str>>,
    /// Each cell's section, 0..n_sections, in order of first appearance.
    pub section: Vec<usize>,
    pub n_sections: usize,
    /// The batch effects the counts were divided by, when they were.
    pub delta: Option<AppliedDelta>,
}

/// The δ a baseline's counts were divided by: estimated on the input
/// itself, as cage estimates it, and written to `{out}.delta.parquet` with
/// each cell's batch beside it.
#[derive(Debug, Clone)]
pub struct AppliedDelta {
    /// δ's columns, the batch labels sorted.
    pub batches: Vec<Box<str>>,
}

impl SpatialData {
    /// The cells of each section, in data order.
    pub fn section_cells(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.n_sections];
        for (i, &s) in self.section.iter().enumerate() {
            out[s].push(i);
        }
        out
    }
}

/// An input's size, what its peak memory is estimated from: from its files'
/// metadata ([`SpatialInputArgs::stored_shape`]), before its counts are
/// read, at least what the read gives on every field, so an estimate from
/// it bounds the estimates after the read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputShape {
    /// Genes: from the metadata, the row names every file stores, which
    /// the read keeps.
    pub genes: usize,
    /// Nonzeros of the counts: from the metadata, those the files store
    /// over all their rows.
    pub nonzeros: usize,
    /// Cells in each section: from the metadata, each file taken as a
    /// section, which bounds the smaller sections a batch file splits it
    /// into.
    pub section_sizes: Vec<usize>,
}

/// A data file opened ([`try_open_or_convert`]).
pub(crate) type StoredFile = Box<dyn SparseIo<IndexIter = Vec<usize>>>;

impl SpatialInputArgs {
    /// The input's size from its files' stored shape, or `None` when there
    /// are no files or a file does not store its column or nonzero count.
    /// Each file is opened for its metadata alone ([`Self::metadata`]), no
    /// counts read or preloaded, and opened again by [`Self::read`]: a
    /// check on this shape runs before any count is in memory.
    pub fn stored_shape(&self) -> anyhow::Result<Option<InputShape>> {
        if self.data_files.is_empty() {
            return Ok(None);
        }
        let mut shared: Option<HashSet<Box<str>>> = None;
        let mut nonzeros = 0;
        let mut section_sizes = Vec::with_capacity(self.data_files.len());
        for (file, data) in self.data_files.iter().zip(self.metadata()) {
            let data = data?;
            let (Some(cells), Some(stored)) = (data.num_columns(), data.num_non_zeros()) else {
                return Ok(None);
            };
            let rows: HashSet<Box<str>> = data
                .row_names()
                .with_context(|| format!("{file}: row names"))?
                .into_iter()
                .collect();
            shared = Some(match shared {
                Some(mut kept) => {
                    kept.retain(|r| rows.contains(r));
                    kept
                }
                None => rows,
            });
            section_sizes.push(cells);
            nonzeros += stored;
        }
        Ok(Some(InputShape {
            genes: shared.map_or(0, |s| s.len()),
            nonzeros,
            section_sizes,
        }))
    }

    /// What the δ estimate holds above the process's own on an input of
    /// stored shape `s` ([`delta_estimate_bytes`]); 0 when none is made
    /// (`--ignore-batch`, or one data file and no batch files). With batch
    /// files the batches are not known before the read: every cell may be
    /// a sample.
    pub fn delta_bytes(&self, s: &InputShape) -> usize {
        let files = s.section_sizes.len();
        if self.ignore_batch || (files < 2 && self.batch_files.is_none()) {
            return 0;
        }
        let cells: usize = s.section_sizes.iter().sum();
        let samples = match self.batch_files {
            Some(_) => cells,
            None => s
                .section_sizes
                .iter()
                .map(|&n| n.min(DELTA_SAMPLES_PER_BATCH))
                .sum(),
        };
        delta_estimate_bytes(s.genes, s.nonzeros, samples)
    }

    /// Each data file opened for its metadata, its counts not read or
    /// preloaded.
    pub(crate) fn metadata(&self) -> impl Iterator<Item = anyhow::Result<StoredFile>> + '_ {
        self.data_files.iter().map(|file| try_open_or_convert(file))
    }

    /// Read the counts and coordinates, the files opened as the other
    /// commands open them (preloaded by default, as data-beans' budget
    /// allows): a preloaded read leaves less memory resident than a
    /// streamed one.
    pub fn read(&self) -> anyhow::Result<SpatialData> {
        let mut srt = read_data_with_coordinates(SRTReadArgs {
            data_files: self.data_files.clone(),
            coord_files: self.coord_files.clone(),
            preload_data: self.preload_data,
            coord_columns: self.coord_columns.clone().unwrap_or_default(),
            coord_column_names: self.coord_column_names.clone(),
            batch_files: self.batch_files.clone(),
            header_in_coord: self.coord_header_row,
            feature_kind: FeatureNameKind::Exact,
        })?;
        // a section is a (data file, batch label) pair, as the reader keeps
        // them apart; its appended `batch` offset column is not a coordinate
        let section: Vec<usize> =
            ablation::sections(&srt.data.num_columns_by_data()?, &srt.batches)
                .into_iter()
                .map(|s| s as usize)
                .collect();
        let cells = srt.data.column_names()?;
        let genes = srt.data.row_names()?;
        let delta = if self.ignore_batch {
            None
        } else {
            estimate_delta(&mut srt.data, &srt.batches, &self.out)?
        };
        let mut counts = srt.data.read_columns_csc(0..cells.len())?;
        drop(srt.data);
        let applied = delta.map(|(mean, batches)| {
            divide_by_delta(&mut counts, &mean, &batches, &srt.batches);
            AppliedDelta { batches }
        });
        let plane = ablation::non_batch_columns(&srt.coordinate_names);
        let d = plane.len();
        anyhow::ensure!(d >= 1, "no spatial coordinate columns");
        let xy = srt.coordinates.select_columns(&plane);
        let coord_names: Vec<Box<str>> = plane
            .iter()
            .map(|&j| srt.coordinate_names[j].clone())
            .collect();
        let n_sections = ablation::n_sections(&section);
        log::info!(
            "{} cells, {} genes, {} sections, {d}-D coordinates",
            cells.len(),
            genes.len(),
            n_sections
        );
        Ok(SpatialData {
            counts,
            genes,
            cells,
            xy,
            coord_names,
            batches: srt.batches,
            section,
            n_sections,
            delta: applied,
        })
    }
}

/// δ estimated on the input itself, by cage's estimator at cage's default
/// settings (`-p 200`, `--batch-knn 10`, `--num-levels 5`), and written
/// under `out` as cage writes it: its mean, `[genes × batches]`, and its
/// batch labels. `None` with fewer than two batches.
fn estimate_delta(
    data: &mut data_beans::sparse_io_vector::SparseIoVec,
    batches: &[Box<str>],
    out: &str,
) -> anyhow::Result<Option<(crate::util::common::Mat, Vec<Box<str>>)>> {
    use crate::util::batch_effects::{estimate_and_write_batch_effects, EstimateBatchArgs};
    let labels = sorted_labels(batches);
    if labels.len() < 2 {
        return Ok(None);
    }
    legume_numeric::matrix::common_io::mkdir_parent(out)?;
    let Some(posterior) = estimate_and_write_batch_effects(
        data,
        batches,
        &labels,
        EstimateBatchArgs {
            proj_dim: DELTA_PROJ_DIM,
            sort_dim: DELTA_PROJ_DIM.min(10),
            block_size: None,
            batch_knn: DELTA_BATCH_KNN,
            num_levels: DELTA_NUM_LEVELS,
            strata: None,
        },
        Some(out),
    )?
    else {
        return Ok(None);
    };
    let names = posterior.batches.clone();
    Ok(Some((posterior.into_mean()?, names)))
}

/// cage's defaults for the δ estimate (`-p`, `--batch-knn`, `--num-levels`).
const DELTA_PROJ_DIM: usize = 200;
const DELTA_BATCH_KNN: usize = 10;
const DELTA_NUM_LEVELS: usize = 5;

/// The distinct labels of `batches`, sorted: δ's columns.
fn sorted_labels(batches: &[Box<str>]) -> Vec<Box<str>> {
    batches
        .iter()
        .map(AsRef::as_ref)
        .collect::<std::collections::BTreeSet<&str>>()
        .into_iter()
        .map(Box::from)
        .collect()
}

/// Each count divided by its gene's δ in its cell's batch (`x / δ`, as lc
/// divides): `mean` is `[genes × columns]`, its columns named `columns`,
/// and `batch_of_cell` each cell's label.
pub fn divide_by_delta(
    counts: &mut CscMatrix<f32>,
    mean: &crate::util::common::Mat,
    columns: &[Box<str>],
    batch_of_cell: &[Box<str>],
) {
    let col: std::collections::HashMap<&str, usize> = columns
        .iter()
        .enumerate()
        .map(|(j, b)| (b.as_ref(), j))
        .collect();
    crate::util::batch_effects::divide_columns(counts, mean, |j| {
        col.get(batch_of_cell[j].as_ref()).copied()
    });
}
