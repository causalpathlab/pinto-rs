//! The inputs every spatial baseline reads: pinto's own ([`InputArgs`],
//! with the ablations), read through the preprocessing lc, dsvd and cage
//! read through ([`preprocess_srt`]): the same cell QC, the same
//! `--auto-batch` components (on pinto's spatial graph), δ estimated as
//! they estimate it, the same ablations and the same cells table. Cells are named,
//! ordered and split into sections as those fits name, order and split them.

use crate::baselines::spatial::memory::GRAPH_NEIGHBOUR_BYTES;
use crate::link_community::profiles::read_columns_divided;
use crate::util::ablation::{self, Ablation, AblationArgs};
use crate::util::batch_effects::{
    delta_estimate_bytes, n_strata, sorted_batch_labels, Divisor, DELTA_SAMPLES_PER_BATCH,
};
use crate::util::common::{HashSet, Mat};
use crate::util::input::{
    read_batch_file, InputArgs, SrtInputArgs, DEFAULT_NUM_LEVELS, DEFAULT_PROJ_DIM,
    DEFAULT_SPATIAL_GRAPH,
};
use crate::util::memory::{ensure_fits, Budget};
use crate::util::srt_pipeline::{
    preprocess_srt, BatchEstimate, FeatureAxisMode, SrtPreprocessConfig, SrtPreprocessed,
};
use anyhow::Context;
use data_beans::aux::feature_names::FeatureNameKind;
use data_beans::convert::try_open_or_convert;
use data_beans::sparse_io::SparseIo;
use nalgebra_sparse::CscMatrix;

/// `--ignore-batch` as a baseline takes it: no fitted run is read, so
/// cage's word on predict and impute does not apply.
const IGNORE_BATCH_HELP: &str = "Estimate and remove no batch effect: the counts are used as \
                                 stored.\n\
                                 Data files (and --batch-files labels) still keep their cells \
                                 apart, as sections of their own; only the correction goes.";

/// The input every fit takes ([`InputArgs`]: data files, coordinates,
/// batches, QC, `--out`, seed) and the ablations, as lc, dsvd and cage
/// take them. A baseline needs
/// coordinates.
#[derive(clap::Args, Debug, Clone)]
#[command(mut_arg("ignore_batch", |a| a.long_help(IGNORE_BATCH_HELP)))]
pub struct SpatialInputArgs {
    #[command(flatten)]
    pub input: InputArgs,

    #[command(flatten)]
    pub modules: crate::util::feature_modules::FeatureModulesArgs,

    #[command(flatten)]
    pub ablation: AblationArgs,
}

impl std::ops::Deref for SpatialInputArgs {
    type Target = InputArgs;

    fn deref(&self) -> &InputArgs {
        &self.input
    }
}

impl std::ops::DerefMut for SpatialInputArgs {
    fn deref_mut(&mut self) -> &mut InputArgs {
        &mut self.input
    }
}

/// Counts and coordinates of every cell, sections kept apart.
pub struct SpatialData {
    /// Counts, genes × cells: as stored for one batch or under
    /// `--ignore-batch`, otherwise each divided by its gene's δ in its
    /// cell's batch ([`Self::delta`]).
    pub counts: CscMatrix<f32>,
    pub genes: Vec<Box<str>>,
    pub cells: Vec<Box<str>>,
    /// Spatial coordinates, cells × dimensions (no batch column).
    pub xy: Mat,
    /// The coordinate columns' names.
    pub coord_names: Vec<Box<str>>,
    /// Each cell's batch label, what δ is estimated over: as the reader
    /// gives it, or its `--auto-batch` component.
    pub batches: Vec<Box<str>>,
    /// Each cell's section, 0..n_sections, in order of first appearance.
    pub section: Vec<usize>,
    pub n_sections: usize,
    /// The batch effects the counts were divided by, when they were.
    pub delta: Option<AppliedDelta>,
    /// The cells' random projection (`proj_dim × cells`, batch-corrected
    /// with δ when there is one), as cage coarsens on it; read for
    /// the `-pb` commands only ([`SpatialInputArgs::read`]).
    pub projection: Option<Mat>,
    /// pinto's cell-pair graph, as lc's is built (its undirected edges and
    /// their distances): what the `-pb` commands coarsen along; read with
    /// the projection only.
    pub pair_graph: Option<PairGraph>,
    /// What an ablation did, for the manifest.
    pub ablation: Option<Ablation>,
}

/// A cell graph's undirected edges and their distances.
pub type PairGraph = (Vec<(usize, usize)>, Vec<f32>);

/// The δ a baseline's counts were divided by: estimated on the input
/// itself, as cage estimates it, and written to `{out}.delta.parquet` with
/// each cell's batch beside it.
#[derive(Debug, Clone)]
pub struct AppliedDelta {
    /// δ's columns, the batch labels sorted.
    pub batches: Vec<Box<str>>,
}

impl SpatialData {
    /// The shape of the data as read, for the memory checks after the read.
    pub fn shape(&self) -> InputShape {
        InputShape {
            genes: self.genes.len(),
            nonzeros: self.counts.nnz(),
            section_sizes: ablation::section_sizes(&self.section),
            batches: self.n_batches(),
            strata: 1,
        }
    }

    /// Distinct batch labels.
    pub fn n_batches(&self) -> usize {
        sorted_batch_labels(&self.batches).len()
    }

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
/// metadata ([`SpatialInputArgs::stored`]), before its counts are
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
    /// Cells in each section, in the read's order: from the metadata, each
    /// file, split by its batch file's labels when there are batch files.
    pub section_sizes: Vec<usize>,
    /// Distinct batch labels (a δ estimate is made with two or more): the
    /// batch files' labels over all files, or one a file without them.
    /// `--auto-batch` may make more after the read, charged as one a
    /// section ([`SpatialInputArgs::read_bytes`]).
    pub batches: usize,
    /// The `--cnv-clones` strata δ's finest groups are crossed with: those
    /// its table lists and stratum 0 (1 without a table, and after the
    /// read, by which δ is estimated).
    pub strata: usize,
}

/// A data file opened ([`try_open_or_convert`]).
pub(crate) type StoredFile = Box<dyn SparseIo<IndexIter = Vec<usize>>>;

impl SpatialInputArgs {
    /// The input's size from its files' stored shape, or no shape when
    /// there are no files or a file does not store its column or nonzero
    /// count. Each file is opened for its metadata alone
    /// ([`Self::metadata`]), no counts read or preloaded, and opened again
    /// by the read: a check on this shape runs before any count is in
    /// memory. Batch files, a label per cell, are read for the sections
    /// they split files into, and a clone table for its strata.
    pub fn stored(&self) -> anyhow::Result<Option<InputShape>> {
        self.collapse.check()?;
        if self.data_files.is_empty() {
            return Ok(None);
        }
        let batch_files = &self.batch.batch_files;
        if let Some(b) = batch_files {
            anyhow::ensure!(
                b.len() == self.data_files.len(),
                "# batch files != # of data files"
            );
        }
        let mut shared: Option<HashSet<Box<str>>> = None;
        let mut nonzeros = 0;
        let mut section_sizes = Vec::with_capacity(self.data_files.len());
        let mut labels: Vec<Box<str>> = Vec::new();
        for (f, (file, data)) in self.data_files.iter().zip(self.metadata()).enumerate() {
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
            match batch_files {
                Some(b) => {
                    let file_labels = read_batch_file(&b[f], cells)?;
                    let sections = ablation::sections(&[cells], &file_labels);
                    section_sizes.extend(ablation::section_sizes(&sections));
                    labels.extend(file_labels);
                }
                None => section_sizes.push(cells),
            }
            nonzeros += stored;
        }
        Ok(Some(InputShape {
            genes: shared.map_or(0, |s| s.len()),
            nonzeros,
            section_sizes,
            batches: match batch_files {
                Some(_) => sorted_batch_labels(&labels).len(),
                None => self.data_files.len(),
            },
            strata: match &self.collapse.cnv_clones {
                Some(path) => n_strata(path)?,
                None => 1,
            },
        }))
    }

    /// What the read holds above the process's own on an input of stored
    /// shape `s`, checked before it: pinto's spatial graph, built on every
    /// read (for `--auto-batch` and the cells table), and the δ estimate
    /// ([`Self::collapse_bytes`]) unless none is made (`--ignore-batch`, or
    /// fewer than two batch labels and no `--auto-batch`).
    /// Under `--auto-batch` the batches are the graph's components, not
    /// known before the read: each section is charged as one batch.
    pub fn read_bytes(&self, s: &InputShape) -> usize {
        let b = &self.batch;
        let cells: usize = s.section_sizes.iter().sum();
        let graph = GRAPH_NEIGHBOUR_BYTES * DEFAULT_SPATIAL_GRAPH.knn * cells;
        let no_delta = b.ignore_batch || (s.batches < 2 && !b.auto_relabels());
        graph + if no_delta { 0 } else { self.collapse_bytes(s) }
    }

    /// The δ estimate's bytes on stored shape `s` ([`delta_estimate_bytes`]),
    /// each section a batch of its own. A pseudobulk sample is a (group,
    /// stratum, batch) triple of a level, at most
    /// [`DELTA_SAMPLES_PER_BATCH`] groups in each of `s.strata` strata and
    /// no more than the batch's cells at the finest: summed over the
    /// sections, which split each batch (a label shared by files is a
    /// section in each), it bounds a level's samples. The estimate is
    /// calibrated at the default `--num-levels`; more levels hold more
    /// samples, none more than the finest, and a cell's group index at
    /// each. A `-p` above the default widens the projection the estimate
    /// collapses on (its cells' copies, as
    /// [`super::memory::pseudobulk_bytes`] counts them, and its basis).
    fn collapse_bytes(&self, s: &InputShape) -> usize {
        let c = &self.collapse;
        let per_section = DELTA_SAMPLES_PER_BATCH * s.strata;
        let finest: usize = s.section_sizes.iter().map(|&n| n.min(per_section)).sum();
        let levels = c.num_levels.max(DEFAULT_NUM_LEVELS);
        let samples = (finest * levels).div_ceil(DEFAULT_NUM_LEVELS);
        let cells: usize = s.section_sizes.iter().sum();
        let wider = c.proj_dim.saturating_sub(DEFAULT_PROJ_DIM);
        let extra_levels = levels - DEFAULT_NUM_LEVELS;
        delta_estimate_bytes(s.genes, s.nonzeros, samples)
            + 4 * wider * (2 * cells + s.genes)
            + std::mem::size_of::<usize>() * extra_levels * cells
    }

    /// A run's `peak` on an input of stored shape `s`, or the read's
    /// ([`Self::read_bytes`] above the `process`'s own bytes) when that is
    /// larger.
    pub fn peak_with_read(&self, peak: usize, process: usize, s: &InputShape) -> usize {
        peak.max(process + self.read_bytes(s))
    }

    /// Each data file opened for its metadata, its counts not read or
    /// preloaded.
    pub(crate) fn metadata(&self) -> impl Iterator<Item = anyhow::Result<StoredFile>> + '_ {
        self.data_files.iter().map(|file| try_open_or_convert(file))
    }

    /// What the δ estimate holds once `--auto-batch` has relabelled the
    /// cells, on an input of stored shape `s` whose (section, batch) groups
    /// hold `groups` cells each, over `n_batches` batches; 0 when none is
    /// made. An upper bound: a batch's samples are at most each of its
    /// sections' ([`Self::collapse_bytes`]).
    pub fn auto_batch_delta_bytes(
        &self,
        s: &InputShape,
        groups: &[usize],
        n_batches: usize,
    ) -> usize {
        if self.batch.ignore_batch || n_batches < 2 {
            return 0;
        }
        self.collapse_bytes(&InputShape {
            section_sizes: groups.to_vec(),
            batches: n_batches,
            ..s.clone()
        })
    }

    /// Read the counts and coordinates through pinto's preprocessing
    /// ([`preprocess_srt`]), with the cells' projection the super-cells are
    /// coarsened on when `with_projection` ([`SpatialData::projection`]).
    /// The counts of the cells QC kept are divided by δ when one was
    /// estimated. Under
    /// `--auto-batch`, once the components are found and before δ is
    /// estimated over them, the estimate above the `process`'s own bytes is
    /// checked against `budget` ([`Self::auto_batch_delta_bytes`]; unchecked
    /// when the files do not store their shape).
    pub fn read_within(
        &self,
        with_projection: bool,
        budget: &Budget,
        process: usize,
    ) -> anyhow::Result<SpatialData> {
        anyhow::ensure!(
            !self.coord_files.is_empty(),
            "a spatial baseline needs coordinate files (-c), one per data file"
        );
        let srt = SrtInputArgs::with_input(self.input.clone());
        let stored = if self.batch.auto_relabels() {
            self.stored()?
        } else {
            None
        };
        let check = |groups: &[usize], n_batches: usize| -> anyhow::Result<()> {
            let Some(s) = &stored else {
                return Ok(());
            };
            ensure_fits(
                "the δ estimate over the --auto-batch components",
                process + self.auto_batch_delta_bytes(s, groups, n_batches),
                budget,
                "drop --auto-batch, or pass --batch-files with fewer labels",
            )
        };
        let SrtPreprocessed {
            data_vec,
            coordinates,
            coordinate_names,
            batch_membership,
            section_ids,
            batch_effects,
            cell_fold,
            cell_proj,
            permuted_coordinates,
            ablation,
            graph,
            ..
        } = preprocess_srt(SrtPreprocessConfig {
            common: &srt,
            ablation: Some(&self.ablation),
            batch_effects: BatchEstimate::Fit,
            feature_axis: FeatureAxisMode::Rows,
            feature_kind: Some(FeatureNameKind::Exact),
            cell_projection: with_projection,
            auto_batch_memory_check: Some(&check),
            batch_fold: crate::util::srt_pipeline::BatchFoldKind::Cells,
        })?;
        // the -pb commands coarsen along the graph; the rest drop it now
        let pair_graph = with_projection.then_some((graph.edges, graph.distances));
        let cells = data_vec.column_names()?;
        let genes = data_vec.row_names()?;
        let delta = batch_effects.as_ref().map(|p| AppliedDelta {
            batches: p.batches.clone(),
        });
        let mean = batch_effects.map(|p| p.into_mean()).transpose()?;
        // each count divided by its cell's own fold (its gene's δ in its
        // batch, refined by the cell's state), as lc divides; the -pb forms
        // pool these divided cells, so their pseudobulks are unbiased too
        let divisor = match cell_fold.as_ref() {
            Some(f) => Some(Divisor::from(f.table.divisor())),
            None => mean.as_ref().map(Divisor::Registered),
        };
        let counts = read_columns_divided(&data_vec, 0..cells.len(), divisor)?;
        drop(data_vec);
        let plane = ablation::non_batch_columns(&coordinate_names);
        let d = plane.len();
        anyhow::ensure!(d >= 1, "no spatial coordinate columns");
        // the method fits on the positions the graph was built on: the
        // scrambled ones under --permute-coords (the cells table, written by
        // the read, keeps the cells' own)
        let xy = permuted_coordinates
            .as_ref()
            .unwrap_or(&coordinates)
            .select_columns(&plane);
        let coord_names: Vec<Box<str>> =
            plane.iter().map(|&j| coordinate_names[j].clone()).collect();
        // sections numbered 0.. in order of first appearance: QC may have
        // emptied one
        let section: Vec<usize> = ablation::dense_ids(section_ids.iter())
            .into_iter()
            .map(|s| s as usize)
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
            batches: batch_membership,
            section,
            n_sections,
            delta,
            projection: cell_proj.map(|p| p.proj),
            pair_graph,
            ablation,
        })
    }
}

#[cfg(test)]
#[path = "tests/input.rs"]
mod tests;
