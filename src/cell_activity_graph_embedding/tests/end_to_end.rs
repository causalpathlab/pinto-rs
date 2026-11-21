//! cage's two remedies for noisy features, end to end on two small
//! sections: `--feature-modules` and `--feature-inclusion moran` (whose
//! fit is in `cell_level.rs`), for cage and cage-cell alike, and the
//! refusal to take both.

use super::cell_level::{fit, try_fit};
use crate::test_support::two_sections;
use crate::util::common::Mat;
use legume_numeric::matrix::traits::IoOps;

/// `--feature-modules M` trains the module layer and writes its tables,
/// on pseudobulks (cage) and on cells (cage-cell): each feature's
/// membership a mixture over the M modules. The module bias is zero at
/// the start and only the exact unit-module term moves it, so a bias off
/// zero says that term trained on the mode's own units.
#[test]
fn feature_modules_train_and_write_their_tables() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    for (name, extra) in [
        ("modules", &["--feature-modules", "2"][..]),
        ("cell_modules", &["--feature-modules", "2", "cell"][..]),
    ] {
        let out = fit(dir.path(), &data, &coords, name, extra);
        let m = Mat::from_parquet(&format!("{out}.module_membership.parquet")).unwrap();
        assert_eq!(
            (m.mat.nrows(), m.mat.ncols()),
            (12, 2),
            "{name}: features x modules"
        );
        for (g, row) in m.mat.row_iter().enumerate() {
            assert!(row.iter().all(|&p| p >= 0.0), "{name} feature {g}: {row:?}");
            let total: f32 = row.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-4,
                "{name} feature {g} sums to {total}"
            );
        }
        for table in ["module_dictionary", "module_residual", "module_bias"] {
            assert!(
                std::path::Path::new(&format!("{out}.{table}.parquet")).exists(),
                "{name}: {table} not written"
            );
        }
        let bias = Mat::from_parquet(&format!("{out}.module_bias.parquet")).unwrap();
        assert!(
            bias.mat.iter().all(|b| b.is_finite()) && bias.mat.iter().any(|&b| b != 0.0),
            "{name}: the exact module term left the module bias at {:?}",
            bias.mat.as_slice()
        );
        // both train a row per super edge, whatever seeded the modules;
        // cage writes its pseudobulk pairs' table, cage-cell's rows are the latent
        let table =
            std::path::Path::new(&crate::util::parquet_io::pb_pair_embedding_path(&out)).exists();
        assert_eq!(table, name == "modules", "{name}");
    }
}

/// Left at its default, the flag resolves its count from the data and every
/// module table is written at that one count; `--feature-modules 0` writes
/// none, and neither does a run whose `--feature-inclusion moran` the
/// default yielded to. For cage and cage-cell alike.
#[test]
fn default_modules_size_every_table_and_zero_writes_none() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    let rows_cols = |out: &str, table: &str| {
        let m = Mat::from_parquet(&format!("{out}.{table}.parquet")).unwrap();
        (m.mat.nrows(), m.mat.ncols())
    };
    for mode in [&[][..], &["cell"][..]] {
        let tag = mode.first().copied().unwrap_or("pb");
        let out = fit(dir.path(), &data, &coords, &format!("{tag}_auto"), mode);
        let (features, m) = rows_cols(&out, "module_membership");
        assert_eq!(features, 12, "{tag}");
        assert!(m >= 2, "{tag}: auto resolved {m} modules");
        assert_eq!(rows_cols(&out, "module_dictionary"), (m, 4), "{tag}");
        assert_eq!(rows_cols(&out, "module_bias").0, m, "{tag}");
        assert_eq!(rows_cols(&out, "module_residual"), (12, 4), "{tag}");
        for (name, off) in [
            ("zero", &["--feature-modules", "0"][..]),
            ("moran", &["--feature-inclusion", "moran"][..]),
        ] {
            let name = format!("{tag}_{name}");
            let out = fit(dir.path(), &data, &coords, &name, &[off, mode].concat());
            for table in [
                "module_membership",
                "module_dictionary",
                "module_residual",
                "module_bias",
            ] {
                assert!(
                    !std::path::Path::new(&format!("{out}.{table}.parquet")).exists(),
                    "{name}: {table} written"
                );
            }
        }
    }
}

/// Modules and inclusion are alternatives: a run given both is refused,
/// naming both, before it writes anything, in cage and in cage-cell.
#[test]
fn feature_modules_with_inclusion_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    let both = ["--feature-modules", "2", "--feature-inclusion", "moran"];
    for (name, mode) in [("both", &[][..]), ("cell_both", &["cell"][..])] {
        let e = try_fit(
            dir.path(),
            &data,
            &coords,
            name,
            &[&both[..], mode].concat(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("--feature-inclusion") && e.contains("--feature-modules"),
            "{name}: {e}"
        );
        assert!(!dir.path().join(format!("{name}.pinto.json")).exists());
    }
}
