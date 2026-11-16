//! cage on an input too small for its defaults to be comfortable. Nothing
//! is fitted to the counts to start the pseudobulk table, so there is no
//! minimum of super-cells or features against the embedding width; what
//! can still stop such a run is its chain levels, in words.

use super::cell_level::try_fit_at;
use crate::test_support::two_sections;

/// cage at its default `-d`, `--num-levels` and `--embedding-dim` on two
/// small sections (levels of the same number of super-cells, fewer features
/// than embedding dimensions), then `extra`; the files stay in `dir`.
fn fit_small(dir: &std::path::Path, extra: &[&str]) -> anyhow::Result<String> {
    let (data, coords) = two_sections(dir, 8);
    try_fit_at(dir, &data, &coords, "small", &[], extra)
}

/// At the default chain levels so small an input keeps no positive pair,
/// and cage says so; it does not panic.
#[test]
fn a_small_input_at_the_defaults_is_refused_in_words() {
    let dir = tempfile::tempdir().unwrap();
    let err = fit_small(dir.path(), &[]).expect_err("no positive pair is left");
    assert!(
        err.to_string().contains("within-batch super edges"),
        "an unclear refusal: {err}"
    );
}

/// With a chain level that keeps its pairs the same input trains and
/// writes its table, though the embedding is wider than the features.
#[test]
fn a_small_input_trains_at_the_default_width() {
    let dir = tempfile::tempdir().unwrap();
    let out = fit_small(dir.path(), &["--chain-levels", "0"]).unwrap();
    assert!(std::path::Path::new(&format!("{out}.pb_embedding.parquet")).exists());
}
