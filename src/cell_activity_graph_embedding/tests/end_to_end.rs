//! cage's two remedies for noisy features, end to end on two small
//! sections: `--feature-modules` and `--feature-inclusion moran` (whose
//! fit is in `cell_level.rs`, for cage and cage-cell alike), and the
//! refusal to take both.

use super::cell_level::{fit, try_fit};
use crate::test_support::two_sections;
use crate::util::common::Mat;
use legume_numeric::matrix::traits::IoOps;

/// `--feature-modules M` trains the module layer and writes its tables:
/// each feature's membership a mixture over the M modules.
#[test]
fn feature_modules_train_and_write_their_tables() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    let out = fit(
        dir.path(),
        &data,
        &coords,
        "modules",
        &["--feature-modules", "2"],
    );
    let m = Mat::from_parquet(&format!("{out}.module_membership.parquet")).unwrap();
    assert_eq!(
        (m.mat.nrows(), m.mat.ncols()),
        (12, 2),
        "features x modules"
    );
    for (g, row) in m.mat.row_iter().enumerate() {
        assert!(row.iter().all(|&p| p >= 0.0), "feature {g}: {row:?}");
        let total: f32 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "feature {g} sums to {total}");
    }
    for table in ["module_dictionary", "module_residual", "module_bias"] {
        assert!(
            std::path::Path::new(&format!("{out}.{table}.parquet")).exists(),
            "{table} not written"
        );
    }
}

/// Modules and inclusion are alternatives: a run given both is refused,
/// naming both, before it writes anything.
#[test]
fn feature_modules_with_inclusion_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    let both = ["--feature-modules", "2", "--feature-inclusion", "moran"];
    let e = try_fit(dir.path(), &data, &coords, "both", &both)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("--feature-inclusion") && e.contains("--feature-modules"),
        "{e}"
    );
    assert!(!dir.path().join("both.pinto.json").exists());
}
