//! The batch effect `pinto predict` divides a query by.
//!
//! A cage encoder was trained on counts divided by the run's δ, pinned to
//! geometric mean 1 over the training batches. A query cell from one of
//! those batches is divided by the model's own δ, which is the transform
//! the encoder saw. Only a batch the model never had needs an estimate:
//! the query's own (estimated during preprocessing, over the query's
//! batches), carried into the model's frame through the batches the two
//! share.

use crate::util::batch_effects::{read_delta, run_delta_path, RunDelta};
use crate::util::common::*;

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

/// For each query batch label, the model's batch it is, if any.
pub(crate) fn match_batches(
    query: &[Box<str>],
    model: &[Box<str>],
    how: &QueryLabels<'_>,
) -> Vec<Option<usize>> {
    let col_of = |label: &str| model.iter().position(|m| m.as_ref() == label);
    match how {
        QueryLabels::Named => query.iter().map(|q| col_of(q)).collect(),
        QueryLabels::Components => vec![None; query.len()],
        QueryLabels::Files {
            query_files,
            model_files,
        } => {
            let by_file = model.len() == model_files.len()
                && (0..model_files.len()).all(|i| col_of(&i.to_string()).is_some());
            query
                .iter()
                .map(|q| {
                    let file = query_files.get(q.parse::<usize>().ok()?)?;
                    let i = model_files.iter().position(|m| same_file(m, file))?;
                    by_file.then(|| col_of(&i.to_string())).flatten()
                })
                .collect()
        }
    }
}

/// Whether two paths name one file: by their canonical paths when both
/// resolve, else as written.
fn same_file(a: &str, b: &str) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
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
    let est = |plane: fn(&RunDelta) -> &Mat, g: usize, q: usize, none: f32| {
        estimate.map_or(none, |e| plane(e)[(g, q)])
    };
    let log_anchor: Vec<f32> = (0..n_rows)
        .map(|g| {
            if !model_has_row[g] || shared.is_empty() || estimate.is_none() {
                return 0.0;
            }
            let log_ratio: f32 = shared
                .iter()
                .map(|&(q, m)| model.mean[(g, m)].ln() - est(|e| &e.mean, g, q, 1.0).ln())
                .sum();
            log_ratio / shared.len() as f32
        })
        .collect();
    let plane = |of: fn(&RunDelta) -> &Mat, none: f32, moved: &dyn Fn(f32, usize) -> f32| {
        Mat::from_fn(n_rows, matched.len(), |g, q| match matched[q] {
            Some(m) if model_has_row[g] => of(model)[(g, m)],
            _ => moved(est(of, g, q, none), g),
        })
    };
    let scale = |v: f32, g: usize| v * log_anchor[g].exp();
    RunDelta {
        rows: model.rows.clone(),
        batches: query_batches.to_vec(),
        mean: plane(|d| &d.mean, 1.0, &scale),
        sd: plane(|d| &d.sd, 0.0, &scale),
        log_mean: plane(|d| &d.log_mean, 0.0, &|v, g| v + log_anchor[g]),
        log_sd: plane(|d| &d.log_sd, 0.0, &|v, _| v),
    }
}

/// The δ predict divides the query by, given the query's own estimate
/// (`None` under `--ignore-batch` or with one batch), and whether it is
/// not the query's estimate alone. `query_batches` are the query's sorted,
/// distinct labels; `rows` its matrix rows.
pub(crate) fn query_delta(
    model_prefix: &str,
    how: &QueryLabels<'_>,
    rows: &[Box<str>],
    query_batches: &[Box<str>],
    estimate: Option<RunDelta>,
    ignore_batch: bool,
) -> anyhow::Result<(Option<RunDelta>, bool)> {
    if ignore_batch {
        return Ok((None, false));
    }
    let Some(path) = run_delta_path(model_prefix) else {
        return Ok((estimate, false));
    };
    let model: RunDelta = read_delta(&path)?;
    let matched = match_batches(query_batches, &model.batches, how);
    let n_known = matched.iter().flatten().count();
    if n_known == 0 {
        info!(
            "No batch of this sample is one of the model's {}; batch effects are the \
             sample's own estimate",
            model.batches.len()
        );
        return Ok((estimate, false));
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
    Ok((
        Some(combine(
            &on_rows,
            &has_row,
            estimate.as_ref(),
            &matched,
            query_batches,
        )),
        true,
    ))
}
