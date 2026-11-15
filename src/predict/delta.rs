//! The batch effect `pinto predict` divides a query by.
//!
//! A cage encoder was trained on counts divided by the run's δ, pinned to
//! geometric mean 1 over the training batches. A query cell from one of
//! those batches is divided by the model's own δ, which is the transform
//! the encoder saw. Only a batch the model never had needs an estimate:
//! the query's own (estimated during preprocessing, and only when such a
//! batch exists), carried into the model's frame through the batches the
//! two share. A model without a δ trained on undivided counts, so its
//! query is not divided either.

use crate::util::batch_effects::RunDelta;
use crate::util::common::*;
use crate::util::input::SrtInputArgs;
use crate::util::lupin_rounds::same_file;
use crate::util::metadata::BatchLabels;
use legume_numeric::matrix::common_io::open_buf_reader;
use std::path::Path;

/// How a query's batch labels name batches, and so how they can be matched
/// to the model's.
pub(crate) enum QueryLabels<'a> {
    /// `--batch-files` labels: a query batch is the model batch of that name.
    Named,
    /// One batch per data file, labelled by the file's position. Positions
    /// mean nothing across runs, so a query file is the model's batch only
    /// when it is one of the model's own data files, and the model labelled
    /// its batches by file too.
    Files {
        query_files: &'a [Box<str>],
        model_files: &'a [String],
    },
    /// `--auto-batch` components: labels a query cannot share with a model.
    Components,
}

impl<'a> QueryLabels<'a> {
    /// The labels a query with label source `source` has.
    pub(crate) fn of(
        source: BatchLabels,
        query_files: &'a [Box<str>],
        model_files: &'a [String],
    ) -> Self {
        match source {
            BatchLabels::BatchFiles => Self::Named,
            BatchLabels::AutoBatch => Self::Components,
            BatchLabels::Files => Self::Files {
                query_files,
                model_files,
            },
        }
    }
}

/// Whether a model labelled its batches by file: as its manifest records
/// (`model_source`), else read from its labels, which are then exactly the
/// positions of its data files.
fn model_labels_files(
    model: &[Box<str>],
    model_source: Option<BatchLabels>,
    model_files: &[String],
) -> bool {
    match model_source {
        Some(source) => source == BatchLabels::Files,
        None => {
            model.len() == model_files.len()
                && (0..model_files.len()).all(|i| model.iter().any(|m| m.as_ref() == i.to_string()))
        }
    }
}

/// For each query batch label, the model's batch it is, if any. `model_source`
/// is how the model labelled its batches, from its manifest; a manifest
/// that does not record it (an older fit) is read from the labels: by file
/// when they are exactly the positions of its data files.
pub(crate) fn match_batches(
    query: &[Box<str>],
    model: &[Box<str>],
    model_source: Option<BatchLabels>,
    how: &QueryLabels<'_>,
) -> Vec<Option<usize>> {
    let col_of = |label: &str| model.iter().position(|m| m.as_ref() == label);
    match how {
        QueryLabels::Named => match model_source {
            Some(BatchLabels::BatchFiles) | None => query.iter().map(|q| col_of(q)).collect(),
            Some(_) => vec![None; query.len()],
        },
        QueryLabels::Components => vec![None; query.len()],
        QueryLabels::Files {
            query_files,
            model_files,
        } => {
            let by_file = model_labels_files(model, model_source, model_files);
            query
                .iter()
                .map(|q| {
                    let file = query_files.get(q.parse::<usize>().ok()?)?;
                    let i = model_files
                        .iter()
                        .position(|m| same_file(Path::new(m), Path::new(file.as_ref())))?;
                    by_file.then(|| col_of(&i.to_string())).flatten()
                })
                .collect()
        }
    }
}

/// Whether every batch a query with inputs `c` (labels `how`) can have is
/// one of the model's, decided before any data is read: per-file labels
/// are the files' positions, `--batch-files` labels are those files' lines,
/// and `--auto-batch` components are never the model's. Read from the
/// inputs, so a label only cells QC drops would carry counts too: it can
/// ask for an estimate the kept cells do not need, never skip one they
/// do. Warns, once, of a query file named as a model file but not that
/// file.
pub(crate) fn every_batch_known(
    c: &SrtInputArgs,
    how: &QueryLabels<'_>,
    model: &[Box<str>],
    model_source: Option<BatchLabels>,
) -> anyhow::Result<bool> {
    let labels: Vec<Box<str>> = match how {
        QueryLabels::Components => return Ok(false),
        QueryLabels::Files {
            query_files,
            model_files,
        } => {
            if model_labels_files(model, model_source, model_files) {
                for file in query_files.iter() {
                    warn_if_renamed(file, model_files);
                }
            }
            (0..query_files.len())
                .map(|i| i.to_string().into())
                .collect()
        }
        // a model labelled otherwise shares no named batch: no need to read
        QueryLabels::Named if model_source.is_some_and(|s| s != BatchLabels::BatchFiles) => {
            return Ok(false)
        }
        QueryLabels::Named => {
            // one entry per distinct label, read a line at a time: no
            // per-cell list
            use std::io::BufRead;
            let mut labels = std::collections::BTreeSet::new();
            for file in c.batch.batch_files.iter().flatten() {
                for line in open_buf_reader(file)?.lines() {
                    let line = line?;
                    if !labels.contains(line.as_str()) {
                        labels.insert(Box::<str>::from(line));
                    }
                }
            }
            labels.into_iter().collect()
        }
    };
    Ok(match_batches(&labels, model, model_source, how)
        .iter()
        .all(Option::is_some))
}

/// A query file not among the model's, but named as one of them is: most
/// likely the model's file moved, or the model's path no longer resolves.
fn warn_if_renamed(file: &str, model_files: &[String]) {
    if model_files
        .iter()
        .any(|m| same_file(Path::new(m), Path::new(file)))
    {
        return;
    }
    let name = Path::new(file).file_name();
    if let Some(m) = model_files
        .iter()
        .find(|m| name.is_some() && Path::new(m).file_name() == name)
    {
        log::warn!(
            "{file} is named as the model's data file {m} but is not that file; its batch \
             is taken as new to the model"
        );
    }
}

/// The δ for the query, `[query rows × query batches]`, from the model's δ
/// on the query's rows (`model_has_row`: the rows the model read) and the
/// query's own estimate on the same rows and batches.
///
/// A query batch that is a model batch (`matched`) takes the model's
/// column. Any other takes the query's estimate, moved into the model's
/// frame: on each row, times the geometric mean, over the shared batches,
/// of model over estimate (the spread scaled alike). A row the model did
/// not read takes the estimate as it is (no shift without one).
pub(crate) fn combine(
    model: &RunDelta,
    model_has_row: &[bool],
    estimate: Option<&RunDelta>,
    matched: &[Option<usize>],
    query_batches: &[Box<str>],
) -> RunDelta {
    let n_rows = model.rows.len();
    let shared: Vec<(usize, usize)> = matched
        .iter()
        .enumerate()
        .filter_map(|(q, m)| m.map(|m| (q, m)))
        .collect();
    let log_anchor: Vec<f32> = (0..n_rows)
        .map(|g| match estimate {
            Some(e) if model_has_row[g] && !shared.is_empty() => {
                let log_ratio: f32 = shared
                    .iter()
                    .map(|&(q, m)| model.mean[(g, m)].ln() - e.mean[(g, q)].ln())
                    .sum();
                log_ratio / shared.len() as f32
            }
            _ => 0.0,
        })
        .collect();
    // how each plane of the estimate moves into the model's frame: the
    // mean and its spread scale, the log mean shifts, the log spread stays
    let moved = |p: usize, v: f32, g: usize| match p {
        0 | 1 => v * log_anchor[g].exp(),
        2 => v + log_anchor[g],
        _ => v,
    };
    let est = estimate.map(RunDelta::planes);
    model.map_planes(
        model.rows.clone(),
        query_batches.to_vec(),
        |p, of_model, fill| {
            Mat::from_fn(n_rows, matched.len(), |g, q| match matched[q] {
                Some(m) if model_has_row[g] => of_model[(g, m)],
                _ => moved(p, est.map_or(fill, |e| e[p][(g, q)]), g),
            })
        },
    )
}

/// The δ predict divides the query by, given the model's δ (read from
/// `path`), each query batch's model batch (`matched`), and the query's
/// own estimate on its rows and batches (`None` when every batch is the
/// model's, or with one batch). `query_batches` are the query's sorted,
/// distinct labels; `rows` its matrix rows. `None`: nothing to divide by.
pub(crate) fn query_delta(
    (model, path): (&RunDelta, &str),
    matched: &[Option<usize>],
    rows: &[Box<str>],
    query_batches: &[Box<str>],
    estimate: Option<&RunDelta>,
) -> Option<RunDelta> {
    let n_known = matched.iter().flatten().count();
    if n_known == 0 {
        log::warn!(
            "No batch of this sample is one of the model's {}; batch effects are the \
             sample's own estimate, not in the model's frame",
            model.batches.len()
        );
        return estimate.cloned();
    }
    let n_new = matched.len() - n_known;
    if n_new > 0 && estimate.is_none() {
        log::warn!(
            "{n_new} batch(es) of this sample are new to the model and could not be \
             estimated; their counts are not divided"
        );
    }
    let (on_rows, has_row) = model.on_rows(rows);
    let n_missing = has_row.iter().filter(|&&h| !h).count();
    if n_missing > 0 {
        info!(
            "{n_missing} of this sample's {} rows have no batch effect in the model",
            rows.len()
        );
    }
    info!(
        "Batch effects: {n_known} of this sample's {} batches take the model's δ ({path}); \
         {n_new} new",
        matched.len()
    );
    Some(combine(
        &on_rows,
        &has_row,
        estimate,
        matched,
        query_batches,
    ))
}

/// What reads a query of a model, for the way out `--ignore-batch` is
/// refused with ([`batch_adjusted`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QueryReader {
    /// `pinto predict` of a cage model, also under `pinto impute`.
    Cage,
    /// `pinto predict` of an lc / dsvd model: held-out scoring.
    Holdout,
    /// `pinto impute` of an lc / dsvd model.
    Impute,
}

impl QueryReader {
    /// What a run without `--ignore-batch` does instead.
    fn without_ignore_batch(self) -> &'static str {
        match self {
            Self::Cage => {
                "each batch the model knows is then divided by the model's δ, a new one by \
                 an estimate on this sample"
            }
            Self::Holdout => {
                "each batch the model knows then takes the model's δ; with \
                 --holdout-delta-iters 0 every other batch is scored unshifted"
            }
            Self::Impute => {
                "the reference is then divided by the run's δ, the query by the model's δ \
                 for a batch it knows and by an estimate for a new one"
            }
        }
    }
}

/// Whether the model at prefix `model` (its manifest `run`) wrote a
/// batch-adjusted feature_community, fitted on counts divided by its δ, as
/// its manifest says. A query of such a model goes onto that scale: each
/// batch the model knows divided by the model's δ, a new one by an
/// estimate. `--ignore-batch` (no δ) is refused for it, since an undivided
/// query is not on the scale of its dictionary or its encoder, with what
/// `reader` does without it; a model without the flag (one batch,
/// `--ignore-batch`, an older fit) takes it.
pub(crate) fn batch_adjusted(
    model: &str,
    run: Option<&crate::util::metadata::Run>,
    ignore_batch: bool,
    reader: QueryReader,
) -> anyhow::Result<bool> {
    let adjusted = run.is_some_and(|r| r.meta.outputs.feature_community_batch_adjusted);
    anyhow::ensure!(
        !(adjusted && ignore_batch),
        "{model} was fitted on counts divided by its batch effect, so an undivided query \
         is not on its scale: drop --ignore-batch ({})",
        reader.without_ignore_batch()
    );
    Ok(adjusted)
}

/// What divides a query of the batch-adjusted model at prefix `model` (its
/// manifest `run`), read with inputs `c` into `data`, its cells' batch
/// labels `membership` as read: the δ on `data`'s rows over the sorted
/// labels, and each cell's column of it. Each batch the model knows takes
/// the model's δ; when one is new and there are two or more, the query's
/// own estimate, moved into the model's frame ([`query_delta`]), as
/// `pinto predict` divides a cage query. `None`: nothing to divide by.
/// `--auto-batch` labels are a graph's components, never the model's
/// batches; no graph is built here, so the query is one component, one
/// batch new to the model, which nothing estimates: undivided, as
/// `pinto predict` leaves a cage query whose graph is one component.
pub(crate) fn adjusted_query_delta(
    model: &str,
    run: &crate::util::metadata::Run,
    c: &SrtInputArgs,
    data: &mut SparseIoVec,
    membership: &[Box<str>],
) -> anyhow::Result<Option<(Mat, Vec<u32>)>> {
    use crate::util::batch_effects::{batch_ids, sorted_batch_labels};
    if BatchLabels::of(c) == BatchLabels::AutoBatch {
        info!(
            "--auto-batch takes batches from a cell graph, which impute does not build: \
             the query is one batch, new to the model, and is not divided"
        );
        return Ok(None);
    }
    let (model_delta, path) = adjusted_model_delta(model, run)?;
    let source = run.meta.batch_labels;
    let model_files = crate::predict::model_data_files(Some(run));
    let how = QueryLabels::of(BatchLabels::of(c), &c.data_files, &model_files);
    let labels = sorted_batch_labels(membership);
    let all_known = every_batch_known(c, &how, &model_delta.batches, source)?;
    let estimate = if all_known {
        None
    } else {
        crate::util::srt_pipeline::estimate_batches(c, None, data, membership, &labels, false)?
            .and_then(|e| e.posterior)
            .map(|p| p.into_delta().map(|d| d.on_batches(&labels)))
            .transpose()?
    };
    let matched = match_batches(&labels, &model_delta.batches, source, &how);
    let rows = data.row_names()?;
    Ok(query_delta(
        (&model_delta, &path),
        &matched,
        &rows,
        &labels,
        estimate.as_ref(),
    )
    .map(|d| (d.mean, batch_ids(membership, &labels))))
}

/// The δ of the batch-adjusted model at prefix `model` (its manifest
/// `run`) and its path: an error when the manifest records none.
pub(crate) fn adjusted_model_delta(
    model: &str,
    run: &crate::util::metadata::Run,
) -> anyhow::Result<(RunDelta, String)> {
    use crate::util::batch_effects::{read_delta, run_delta_path};
    let path = run_delta_path(model, Some(run))?.ok_or_else(|| {
        anyhow::anyhow!(
            "{model}: its manifest says it was fitted on counts divided by its batch \
             effect, but it records no delta.parquet to divide a query by"
        )
    })?;
    Ok((read_delta(&path)?, path))
}
