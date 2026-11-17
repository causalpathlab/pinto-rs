//! The batch effect `pinto apply predict` divides a query by.
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
use crate::util::input::{data_file_names, SrtInputArgs};
use crate::util::lupin_rounds::same_file;
use crate::util::metadata::BatchLabels;
use legume_numeric::matrix::common_io::open_buf_reader;
use std::path::Path;

/// How a model named its batches, from its manifest, for matching a
/// query's batches to them.
#[derive(Debug, Default)]
pub(crate) struct ModelNaming {
    /// What named them; `None` on a manifest that does not record it.
    pub source: Option<BatchLabels>,
    /// Its data files as its manifest records them (canonical), found from
    /// here.
    pub files: Vec<String>,
    /// Each file batch's name, in `files`' order, when recorded.
    pub names: Vec<Box<str>>,
}

impl ModelNaming {
    /// The naming of the model whose run is `run`; none without a run.
    pub(crate) fn of(run: Option<&crate::util::metadata::Run>) -> Self {
        let Some(run) = run else {
            return Self::default();
        };
        Self {
            source: run.meta.batch_labels,
            files: crate::predict::model_data_files(Some(run)),
            names: (run.meta.file_batches.iter().flatten())
                .map(|b| b.name.as_str().into())
                .collect(),
        }
    }

    /// Whether a query labelled by `source` can share a batch with the
    /// model at all: decided without reading the query's labels.
    fn may_share(&self, source: BatchLabels) -> bool {
        use BatchLabels::*;
        match source {
            AutoBatch => false,
            BatchFiles => matches!(self.source, Some(BatchFiles) | None),
            Files => self.source == Some(Files),
        }
    }
}

/// For each query batch label, the model's batch it is, if any, for a
/// query with inputs `c` and a model with batches `model`, named as
/// `naming` says. `--batch-files` labels are the model batch of that name
/// when the model is labelled so too, or records nothing. A file's batch
/// is the model's of the same name when the model named its batches by
/// file too. `--auto-batch` components are never the model's.
pub(crate) fn match_batches(
    query: &[Box<str>],
    c: &SrtInputArgs,
    model: &[Box<str>],
    naming: &ModelNaming,
) -> Vec<Option<usize>> {
    let col_of = |label: &str| model.iter().position(|m| m.as_ref() == label);
    let by_name = || query.iter().map(|q| col_of(q)).collect();
    match BatchLabels::of(&c.batch) {
        BatchLabels::BatchFiles if naming.may_share(BatchLabels::BatchFiles) => by_name(),
        BatchLabels::Files if naming.source == Some(BatchLabels::Files) => by_name(),
        _ => vec![None; query.len()],
    }
}

/// Each of the query's data files `files` (named `names`) that has the
/// name of one of the model's file batches but is not that file (through
/// links), with the model's file: it takes that batch's δ, though it may
/// be another sample.
pub(crate) fn renamed_files<'a>(
    files: &'a [Box<str>],
    names: &[Box<str>],
    naming: &'a ModelNaming,
) -> Vec<(&'a str, &'a str)> {
    files
        .iter()
        .zip(names)
        .filter_map(|(file, name)| {
            let k = naming.names.iter().position(|n| n == name)?;
            let model_file = naming.files.get(k)?;
            (!same_file(Path::new(model_file), Path::new(file.as_ref())))
                .then_some((file.as_ref(), model_file.as_str()))
        })
        .collect()
}

/// Whether every batch a query with inputs `c` can have is one of the
/// model's ([`match_batches`]), decided before any data is read: per-file
/// labels are the files' names, `--batch-files` labels are those files'
/// lines, and `--auto-batch` components are never the model's. Read from
/// the inputs, so a label only cells QC drops would carry counts too: it
/// can ask for an estimate the kept cells do not need, never skip one they
/// do. Warns of a query file named as a model file but not that file.
pub(crate) fn every_batch_known(
    c: &SrtInputArgs,
    model: &[Box<str>],
    naming: &ModelNaming,
) -> anyhow::Result<bool> {
    let source = BatchLabels::of(&c.batch);
    // a model labelled otherwise shares no batch: no need to read
    if !naming.may_share(source) {
        return Ok(false);
    }
    let labels: Vec<Box<str>> = match source {
        BatchLabels::AutoBatch => return Ok(false),
        BatchLabels::BatchFiles => {
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
        BatchLabels::Files => {
            let names = data_file_names(&c.data_files)?;
            if naming.source == Some(BatchLabels::Files) {
                for (file, model_file) in renamed_files(&c.data_files, &names, naming) {
                    log::warn!(
                        "{file} has the name of the model's data file {model_file} but is not \
                         that file; it takes that batch's δ"
                    );
                }
            }
            names
        }
    };
    Ok(match_batches(&labels, c, model, naming)
        .iter()
        .all(Option::is_some))
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
    /// `pinto apply predict` of a cage model, also under `pinto apply impute`.
    Cage,
    /// `pinto apply predict` of an lc / dsvd model: held-out scoring.
    Holdout,
    /// `pinto apply impute` of an lc / dsvd model.
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
/// `pinto apply predict` divides a cage query. `None`: nothing to divide by.
/// `--auto-batch` labels are a graph's components, never the model's
/// batches; no graph is built here, so the query is one component, one
/// batch new to the model, which nothing estimates: undivided, as
/// `pinto apply predict` leaves a cage query whose graph is one component.
pub(crate) fn adjusted_query_delta(
    model: &str,
    run: &crate::util::metadata::Run,
    c: &SrtInputArgs,
    data: &mut SparseIoVec,
    membership: &[Box<str>],
) -> anyhow::Result<Option<(Mat, Vec<u32>)>> {
    use crate::util::batch_effects::{batch_ids, sorted_batch_labels};
    if BatchLabels::of(&c.batch) == BatchLabels::AutoBatch {
        info!(
            "--auto-batch takes batches from a cell graph, which impute does not build: \
             the query is one batch, new to the model, and is not divided"
        );
        return Ok(None);
    }
    let (model_delta, path) = adjusted_model_delta(model, run)?;
    let naming = ModelNaming::of(Some(run));
    let labels = sorted_batch_labels(membership);
    let all_known = every_batch_known(c, &model_delta.batches, &naming)?;
    let estimate = if all_known {
        None
    } else {
        crate::util::srt_pipeline::estimate_batches(
            &c.collapse,
            None,
            data,
            membership,
            &labels,
            false,
        )?
        .and_then(|e| e.posterior)
        .map(|p| p.into_delta().map(|d| d.on_batches(&labels)))
        .transpose()?
    };
    let matched = match_batches(&labels, c, &model_delta.batches, &naming);
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
