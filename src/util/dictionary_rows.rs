//! Which rows of a pre-trained feature dictionary may match this run's
//! features. A mixed-type table (a `senna fne` run's genes beside ontology
//! terms, words and cell types) writes `{prefix}.feature_types.parquet`
//! beside it; only its gene and genomic-window rows name a data feature, and
//! a term or cell type may share a gene's name. They are marked BY POSITION
//! (see [`data_beans::aux::frozen_features::load_frozen_feature_host_matching`]).

use data_beans::aux::feature_types::{feature_rows, read_feature_types};
use graph_embedding_util::transfer::{module_table_paths, MODULE_MEMBERSHIP_SUFFIX};
use log::warn;

/// The run prefix of a dictionary at `path`, by the rule its module tables
/// are found by ([`module_table_paths`]): less `.feature_embedding.parquet`
/// (or another dictionary slot), else less `.parquet`.
#[must_use]
pub fn stem(path: &str) -> String {
    let (pi, _) = module_table_paths(path);
    pi.strip_suffix(&format!(".{MODULE_MEMBERSHIP_SUFFIX}"))
        .unwrap_or(path)
        .to_string()
}

/// One flag per row of the dictionary at `path` (`names`, its rows in
/// order): whether it may match a data feature, by the types table beside
/// it. Every row may when there is none, when it cannot be read, or when it
/// lists other rows (an older run's, left under the same prefix); the last
/// two are warned about.
#[must_use]
pub fn matchable_rows(path: &str, names: &[Box<str>]) -> Vec<bool> {
    let every = || vec![true; names.len()];
    match read_feature_types(&stem(path)) {
        Ok(None) => every(),
        Ok(Some(types)) => feature_rows(&types, names).unwrap_or_else(|| {
            warn!(
                "the types table beside {path} lists other rows (left by another run?); \
                 every row of {path} may match"
            );
            every()
        }),
        Err(e) => {
            warn!("cannot read the types table beside {path} ({e}); every row of it may match");
            every()
        }
    }
}
