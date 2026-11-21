//! cage on an input too small for its defaults to be comfortable. Nothing
//! is fitted to the counts to start the super-edge table, so there is no
//! minimum of super-cells or features against the embedding width.

use super::cell_level::try_fit_at;
use crate::test_support::two_sections;

/// cage at its default `-d`, `--num-levels` and `--embedding-dim` on two
/// small sections (levels of the same number of super-cells, fewer features
/// than embedding dimensions), then `extra`; the files stay in `dir`.
fn fit_small(dir: &std::path::Path, extra: &[&str]) -> anyhow::Result<String> {
    let (data, coords) = two_sections(dir, 8);
    try_fit_at(dir, &data, &coords, "small", &[], extra)
}

/// No chain level drops a super edge, so the same input trains at the
/// defaults and writes its table, though the embedding is wider than the
/// features.
#[test]
fn a_small_input_trains_at_the_default_width() {
    let dir = tempfile::tempdir().unwrap();
    let out = fit_small(dir.path(), &[]).unwrap();
    assert!(std::path::Path::new(&crate::util::parquet_io::pb_pair_embedding_path(&out)).exists());
}
