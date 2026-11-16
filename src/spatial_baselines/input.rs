//! The inputs every spatial baseline reads, through pinto's data-beans
//! reader ([`crate::util::input::read_data_with_coordinates`]), so cells are
//! named, ordered and split into sections exactly as `lc`, `dsvd` and `cage`
//! name, order and split them.

use crate::link_community::profiles::read_columns_divided;
use crate::spatial_baselines::memory::GRAPH_NEIGHBOUR_BYTES;
use crate::util::ablation;
use crate::util::batch_effects::{
    delta_estimate_bytes, n_strata, sorted_batch_labels, BatchEffects, DELTA_SAMPLES_PER_BATCH,
};
use crate::util::common::{HashSet, Mat};
use crate::util::input::{
    auto_batch_from_coordinates, read_batch_file, read_data_with_coordinates, BatchArgs,
    CollapseArgs, SRTReadArgs, COORD_COLUMN_NAMES, DEFAULT_NUM_LEVELS, DEFAULT_PROJ_DIM,
    DEFAULT_SPATIAL_GRAPH,
};
use crate::util::memory::{ensure_fits, Budget};
use crate::util::srt_pipeline::estimate_batches;
use anyhow::Context;
use data_beans::alg::random_projection::RandProjOps;
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

/// Data files, their coordinates, cage's batch options, the output prefix
/// and the seed.
#[derive(clap::Args, Debug, Clone)]
#[command(mut_arg("ignore_batch", |a| a.long_help(IGNORE_BATCH_HELP)))]
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

    /// cage's batch options, under its names, defaults and help. Each
    /// (data file, `--batch-files` label) is a section of its own; a
    /// multi-batch input's counts are divided by its own δ estimate (as
    /// cage estimates it) before the method's own preprocessing, and
    /// `--ignore-batch` runs the original method.
    #[command(flatten)]
    pub batch: BatchArgs,

    /// How the δ estimate collapses the cells, as cage's.
    #[command(flatten)]
    pub collapse: CollapseArgs,

    /// Header row index in coordinate files (default: detected)
    #[arg(long, hide = true)]
    pub coord_header_row: Option<usize>,

    /// Output prefix
    #[arg(long, short, required = true)]
    pub out: Box<str>,

    #[arg(
        long,
        default_value_t = crate::util::common::DEFAULT_SEED,
        help = crate::util::common::SEED_HELP
    )]
    pub seed: u64,
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
    /// Each cell's batch label as the reader gives it (its file's, or its
    /// `--batch-files` label), before `--auto-batch` relabels it: what
    /// cage's cells table records.
    pub read_batches: Vec<Box<str>>,
    /// Each cell's section, 0..n_sections, in order of first appearance.
    pub section: Vec<usize>,
    pub n_sections: usize,
    /// The batch effects the counts were divided by, when they were.
    pub delta: Option<AppliedDelta>,
    /// The cells' random projection (`proj_dim × cells`, batch-corrected
    /// with δ when there is one), as cage coarsens on it; read for
    /// the `-pb` commands only ([`SpatialInputArgs::read`]).
    pub projection: Option<Mat>,
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
    /// `--auto-batch` may make more after the read; [`SpatialInputArgs::
    /// read_within`] checks them once they are known.
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

    /// What the δ estimate holds above the process's own on an input of
    /// stored shape `s` ([`Self::collapse_bytes`]), checked before the read;
    /// 0 when none is made (`--ignore-batch`, or fewer than two batch
    /// labels and no `--auto-batch`). Under `--auto-batch` the batches are
    /// the spatial graph's components, not known before the read: each
    /// section is charged as one batch here, and the graph they are found
    /// on is counted too; [`Self::read_within`] checks the components
    /// once the cheap relabel has found them ([`Self::relabelled_delta_bytes`]).
    pub fn delta_bytes(&self, s: &InputShape) -> usize {
        let b = &self.batch;
        let auto = b.auto_relabels();
        if b.ignore_batch || (s.batches < 2 && !auto) {
            return 0;
        }
        let graph = if auto {
            let cells: usize = s.section_sizes.iter().sum();
            GRAPH_NEIGHBOUR_BYTES * DEFAULT_SPATIAL_GRAPH.knn * cells
        } else {
            0
        };
        self.collapse_bytes(s) + graph
    }

    /// What the δ estimate holds once `--auto-batch` has relabelled the
    /// cells, on an input of stored shape `s` whose (section, batch) pairs
    /// hold `groups` cells each, over `n_batches` batches; 0 when none is
    /// made. An upper bound: a batch's samples are at most each of its
    /// sections' ([`Self::collapse_bytes`]).
    pub fn relabelled_delta_bytes(
        &self,
        s: &InputShape,
        groups: Vec<usize>,
        n_batches: usize,
    ) -> usize {
        if self.batch.ignore_batch || n_batches < 2 {
            return 0;
        }
        self.collapse_bytes(&InputShape {
            section_sizes: groups,
            batches: n_batches,
            ..s.clone()
        })
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

    /// A run's `peak` on an input of stored shape `s`, or the δ estimate's
    /// during the read ([`Self::delta_bytes`] above the `process`'s own
    /// bytes) when that is larger.
    pub fn peak_with_delta(&self, peak: usize, process: usize, s: &InputShape) -> usize {
        peak.max(process + self.delta_bytes(s))
    }

    /// Each data file opened for its metadata, its counts not read or
    /// preloaded.
    pub(crate) fn metadata(&self) -> impl Iterator<Item = anyhow::Result<StoredFile>> + '_ {
        self.data_files.iter().map(|file| try_open_or_convert(file))
    }

    /// [`Self::read_within`] with no memory check after the relabel.
    #[cfg(test)]
    pub fn read(&self, with_projection: bool) -> anyhow::Result<SpatialData> {
        self.read_checked(with_projection, None)
    }

    /// Read the counts and coordinates, with the cells' projection the
    /// super-cells are coarsened on when `with_projection`
    /// ([`SpatialData::projection`]). The files are opened as the other
    /// commands open them (preloaded by default, as data-beans' budget
    /// allows): a preloaded read leaves less memory resident than a
    /// streamed one. Under `--auto-batch`, once the relabel has found the
    /// batches and before δ is estimated over them, the estimate above
    /// the `process`'s own bytes is checked against `budget`
    /// ([`Self::relabelled_delta_bytes`]).
    pub fn read_within(
        &self,
        with_projection: bool,
        budget: &Budget,
        process: usize,
    ) -> anyhow::Result<SpatialData> {
        self.read_checked(with_projection, Some((budget, process)))
    }

    fn read_checked(
        &self,
        with_projection: bool,
        check: Option<(&Budget, usize)>,
    ) -> anyhow::Result<SpatialData> {
        self.collapse.check()?;
        let mut srt = read_data_with_coordinates(SRTReadArgs {
            data_files: self.data_files.clone(),
            coord_files: self.coord_files.clone(),
            coord_columns: self.coord_columns.clone().unwrap_or_default(),
            coord_column_names: self.coord_column_names.clone(),
            batch_files: self.batch.batch_files.clone(),
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
        let read_batches = srt.batches.clone();
        // `--auto-batch` relabels the cells by the spatial graph's
        // components, as cage's preprocessing does, on cage's graph at its
        // default `-k` and matching (a baseline takes neither flag); the
        // sections stay the files' (and batch files') own
        if self.batch.auto_relabels() {
            auto_batch_from_coordinates(&srt.coordinates, DEFAULT_SPATIAL_GRAPH, &mut srt.batches)?;
            if let Some((budget, process)) = check {
                self.check_components(&srt, budget, process)?;
            }
        }
        let cells = srt.data.column_names()?;
        let genes = srt.data.row_names()?;
        let mut estimate = (!self.batch.ignore_batch)
            .then(|| self.estimate_delta(&mut srt.data, &srt.batches, with_projection))
            .transpose()?
            .flatten()
            // an estimate without a δ is none
            .filter(|e| e.posterior.is_some());
        // the projection cage coarsens on, taken as cage takes it: with a
        // δ, the batch-corrected one its estimate collapsed on
        let projection = match estimate.as_mut().and_then(|e| e.projection.take()) {
            Some(proj) => Some(proj.proj),
            None if with_projection => Some(
                srt.data
                    .project_columns_with_batch_correction::<Box<str>>(
                        self.collapse.proj_dim,
                        None,
                        None,
                    )?
                    .proj,
            ),
            None => None,
        };
        let (mean, applied) = match estimate.and_then(|e| e.posterior) {
            Some(posterior) => {
                let batches = posterior.batches.clone();
                (Some(posterior.into_mean()?), Some(AppliedDelta { batches }))
            }
            None => (None, None),
        };
        // each count divided by its gene's δ in its cell's batch, as lc
        // divides
        let counts = read_columns_divided(&srt.data, 0..cells.len(), mean.as_ref())?;
        drop(srt.data);
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
            read_batches,
            section,
            n_sections,
            delta: applied,
            projection,
        })
    }

    /// Refuse a δ estimate over the `--auto-batch` components `srt`'s
    /// batches now hold that would not fit in `budget` above the
    /// `process`'s own bytes ([`Self::relabelled_delta_bytes`] on the
    /// stored shape; unchecked when the files do not store it, as the read
    /// is). Components nest in files, which the batch offset coordinate
    /// keeps apart, so each (file, component) pair is a group.
    fn check_components(
        &self,
        srt: &crate::util::input::SRTData,
        budget: &Budget,
        process: usize,
    ) -> anyhow::Result<()> {
        let Some(s) = self.stored()? else {
            return Ok(());
        };
        let groups = ablation::section_sizes(&ablation::sections(
            &srt.data.num_columns_by_data()?,
            &srt.batches,
        ));
        let n_batches = sorted_batch_labels(&srt.batches).len();
        ensure_fits(
            "the δ estimate over the --auto-batch components",
            process + self.relabelled_delta_bytes(&s, groups, n_batches),
            budget,
            "drop --auto-batch, or pass --batch-files with fewer labels",
        )
    }

    /// δ estimated on the input itself over `batches`, by cage's estimator
    /// at the batch options' settings, and written under `--out` as cage
    /// writes it, with the projection it collapsed on when
    /// `keep_projection`. `None` with fewer than two batches.
    fn estimate_delta(
        &self,
        data: &mut data_beans::sparse_io_vector::SparseIoVec,
        batches: &[Box<str>],
        keep_projection: bool,
    ) -> anyhow::Result<Option<BatchEffects>> {
        let labels = sorted_batch_labels(batches);
        if labels.len() < 2 {
            return Ok(None);
        }
        legume_numeric::matrix::common_io::mkdir_parent(&self.out)?;
        estimate_batches(
            &self.collapse,
            Some(&self.out),
            data,
            batches,
            &labels,
            keep_projection,
        )
    }
}
