//! pinto's wiring of the shared per-cell covariates
//! ([`genomic_data::covariate`], one implementation for pinto, senna and
//! chickpea): each cell's gene-set scores binned, and its batch label made
//! `batch|cvK`, so δ is estimated per batch and bin.
//!
//! In pinto a batch label also says which section a cell is in (the pair
//! table, the viewer, lc's section × batch centring, cage's per-batch
//! negatives), so the covariate labels drive δ alone: they are the labels δ
//! is estimated over and divided by, and the cells' projection is corrected
//! by; every other use keeps the plain label. A pair's two cells are each
//! divided by their own δ, so each takes its own bin.
//!
//! A fit writes the model ([`files::model`]) and each cell's scores
//! ([`files::cells`]) beside its run, and each gene's covariate effect
//! ([`files::effect`]) beside δ. A query (`pinto apply predict`, `impute`) is
//! binned by the model's recorded model, never fitted afresh.

pub use genomic_data::covariate::{files, plain_batch, CovariateArgs, CovariateModel};

use crate::util::common::*;
use data_beans::alg::random_projection::RandProjOps;
use genomic_data::covariate::ColumnBlocks;
use nalgebra_sparse::CscMatrix;
use std::path::Path;

/// Columns read at a time when cells are scored.
const BLOCK: usize = 1024;

/// A loaded `SparseIoVec` as the shared module reads it: its columns, or
/// the columns `cols` in that order (the fit's cells in the gene files).
///
/// The same as senna's adapter, with the column map. It stays here until
/// the user decides whether data-beans takes one `impl ColumnBlocks for
/// SparseIoVec` for every tool.
struct Blocks<'a> {
    data: &'a SparseIoVec,
    cols: Option<Vec<usize>>,
}

impl<'a> Blocks<'a> {
    fn all(data: &'a SparseIoVec) -> Self {
        Self { data, cols: None }
    }

    /// `genes`' columns of the cells `cells` names, in their order.
    fn of_cells(genes: &'a SparseIoVec, cells: &[Box<str>]) -> anyhow::Result<Self> {
        let at = crate::util::batch_effects::index_by_name(&genes.column_names()?, cells);
        let cols = at
            .iter()
            .zip(cells)
            .map(|(&i, c)| {
                i.ok_or_else(|| {
                    anyhow::anyhow!("--covariate: cell {c} is not in the feature files scored")
                })
            })
            .collect::<anyhow::Result<Vec<usize>>>()?;
        Ok(Self {
            data: genes,
            cols: Some(cols),
        })
    }
}

impl ColumnBlocks for Blocks<'_> {
    fn n_cols(&self) -> usize {
        self.cols
            .as_ref()
            .map_or_else(|| self.data.num_columns(), Vec::len)
    }
    fn row_names(&self) -> anyhow::Result<Vec<Box<str>>> {
        self.data.row_names()
    }
    fn block(&self, lb: usize, ub: usize) -> anyhow::Result<CscMatrix<f32>> {
        match &self.cols {
            Some(cols) => self.data.read_columns_csc(cols[lb..ub].iter().copied()),
            None => self.data.read_columns_csc(lb..ub),
        }
    }
}

/// Every flag of [`CovariateArgs`], by id.
const COVARIATE_FLAGS: [&str; 5] = [
    "covariates",
    "covariate_genes",
    "covariate_bins",
    "covariate_bins_max",
    "covariate_criterion",
];

/// The covariate flags hidden where a model decides the bins (`predict`,
/// `impute`): every flag of [`CovariateArgs`].
pub fn hide_on_queries(arg: clap::Arg) -> clap::Arg {
    if COVARIATE_FLAGS.contains(&arg.get_id().as_str()) {
        arg.hide(true)
    } else {
        arg
    }
}

/// How preprocessing comes by the cells' covariate bins.
#[derive(Clone, Copy)]
pub enum CovariateBins<'a> {
    /// A fit's: modelled on its cells under `--covariate`.
    Fit,
    /// A query's: binned by its model's recorded covariates.
    Query(&'a CovariateModel),
    /// None: a read that estimates no δ.
    Off,
}

/// Each cell's `batch|cvK` label, or `None` when nothing is binned.
///
/// A fit models the covariates (`c`'s `--covariate`) on `data`'s cells
/// under their `plain` labels, on a random projection of `c`'s `-p`
/// corrected by them, and, under `out`, writes the model and the cells'
/// scores. Their genes are `data`'s rows, or, for a fit that reads rows
/// derived from them, the same cells in `c`'s scored data files. Not
/// modelled under `--ignore-batch`, which estimates no δ for them to drive.
/// A query takes its model's bins.
///
/// # Errors
/// As the shared module, or a file that cannot be written.
pub fn bin_cells(
    how: CovariateBins<'_>,
    c: &crate::util::input::SrtInputArgs,
    data: &SparseIoVec,
    plain: &[Box<str>],
    out: Option<&str>,
) -> anyhow::Result<Option<Vec<Box<str>>>> {
    let ignore_batch = c.batch.ignore_batch;
    let args = &c.batch.covariates;
    match how {
        CovariateBins::Query(model) => bin_query(Some(model), ignore_batch, data, plain),
        CovariateBins::Off => Ok(None),
        CovariateBins::Fit => {
            if !args.enabled(&[(ignore_batch, "--ignore-batch, which estimates no δ")])? {
                return Ok(None);
            }
            let several = plain.iter().any(|b| b != &plain[0]);
            let proj = data.project_columns_with_batch_correction(
                c.collapse.proj_dim,
                None,
                several.then_some(plain),
            )?;
            // the genes, when the fit reads rows derived from them
            let genes = c
                .scored_data_files
                .as_deref()
                .map(|f| {
                    crate::util::input::open_fit_data(
                        f,
                        &data_beans::aux::feature_names::FeatureNameKind::Exact,
                    )
                })
                .transpose()?;
            let cells = data.column_names()?;
            let blocks = match &genes {
                Some(g) => Blocks::of_cells(g, &cells)?,
                None => Blocks::all(data),
            };
            let Some(m) = genomic_data::covariate::model_covariates(
                &blocks, plain, &proj.proj, args, None, BLOCK,
            )?
            else {
                return Ok(None);
            };
            if let Some(out) = out {
                m.model.write(Path::new(&files::model(out)))?;
                genomic_data::covariate::write_cells(
                    Path::new(&files::cells(out)),
                    &m.model,
                    &cells,
                    &m.scores,
                    &m.labels,
                )?;
                info!(
                    "--covariate: wrote {} and {}",
                    files::model(out),
                    files::cells(out)
                );
            }
            Ok(Some(m.labels))
        }
    }
}

/// A query's cells binned by its model's recorded covariates (`model`),
/// their `plain` labels made `batch|cvK`; `None` without a model, or under
/// `--ignore-batch`, which divides by no δ.
///
/// # Errors
/// Cells that cannot be scored.
pub fn bin_query(
    model: Option<&CovariateModel>,
    ignore_batch: bool,
    data: &SparseIoVec,
    plain: &[Box<str>],
) -> anyhow::Result<Option<Vec<Box<str>>>> {
    let Some(model) = model.filter(|_| !ignore_batch) else {
        return Ok(None);
    };
    info!(
        "Query cells binned by the model's covariates ({} bins)",
        model.bins.mixture.k()
    );
    Ok(Some(model.labels(
        &Blocks::all(data),
        plain,
        None,
        BLOCK,
    )?))
}

/// Each gene's covariate effect from δ's mean (`rows` × `levels`), under
/// `out`, when δ has covariate levels.
///
/// # Errors
/// A table that cannot be written.
pub fn write_effect(
    out: &str,
    rows: &[Box<str>],
    levels: &[Box<str>],
    delta: &crate::util::common::Mat,
) -> anyhow::Result<()> {
    let log_delta = delta.map(|d| d.max(f32::MIN_POSITIVE).ln());
    if genomic_data::covariate::write_covariate_effect(
        Path::new(&files::effect(out)),
        rows,
        levels,
        &log_delta,
    )? {
        info!("--covariate: wrote {}", files::effect(out));
    }
    Ok(())
}

/// The covariate model `run` recorded, if its manifest records one (a
/// file left under its prefix by an earlier fit is not it).
///
/// # Errors
/// A recorded model that is not there or cannot be read.
pub fn recorded(run: &crate::util::metadata::Run) -> anyhow::Result<Option<CovariateModel>> {
    let recorded = run
        .meta
        .outputs
        .covariates
        .as_ref()
        .map(|f| f.model.as_str());
    match run.recorded_output(recorded, "covariate_model.json")? {
        Some(path) => CovariateModel::read(&path),
        None => Ok(None),
    }
}
