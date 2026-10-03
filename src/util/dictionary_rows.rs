//! Which rows of a pre-trained feature dictionary may match this run's
//! features. A mixed-type table (a `senna fne` run's genes beside ontology
//! terms, words and cell types) writes `{prefix}.feature_types.parquet`
//! beside it; only its gene and genomic-window rows name a data feature, and
//! a term or cell type may share a gene's name. They are marked BY POSITION
//! (see [`data_beans::aux::frozen_features::load_frozen_feature_host_matching`]).

use data_beans::aux::feature_types::{feature_rows, read_feature_types};
use log::warn;

/// The run prefix of a dictionary at `path`: `path` less its
/// `.{table}.parquet` suffix (`run.feature_embedding.parquet` → `run`).
fn stem(path: &str) -> &str {
    path.strip_suffix(".parquet")
        .and_then(|p| p.rsplit_once('.').map(|(s, _)| s))
        .unwrap_or(path)
}

/// One flag per row of the dictionary at `path` (`names`, its rows in
/// order): whether it may match a data feature, by the types table beside
/// it. Every row may when there is none, or it lists other rows (an older
/// run's, left under the same prefix), which is warned about.
pub fn matchable_rows(path: &str, names: &[Box<str>]) -> anyhow::Result<Vec<bool>> {
    let Some(types) = read_feature_types(stem(path))? else {
        return Ok(vec![true; names.len()]);
    };
    Ok(feature_rows(&types, names).unwrap_or_else(|| {
        warn!(
            "the types table beside {path} lists other rows (left by another run?); \
             every row of {path} may match"
        );
        vec![true; names.len()]
    }))
}
