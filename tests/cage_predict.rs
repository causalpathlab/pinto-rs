//! A cage model applied to a new sample, end to end through the binary:
//! train a tiny cage run, predict on cells it never saw, and impute them
//! with the model named by its manifest.

use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use legume_numeric::matrix::parquet::peek_parquet_field_names;
use std::path::Path;
use std::process::Command;

const FEATURES: usize = 12;

/// A section of `side` x `side` cells named `{tag}_{i}`: the left half
/// expresses the first half of the features, the right half the rest. Its
/// zarr file and coordinates CSV, in `dir`.
fn write_section(dir: &Path, tag: &str, side: usize, seed: u64) -> (String, String) {
    let n = side * side;
    let rows: Vec<Box<str>> = (0..FEATURES).map(|g| format!("GENE{g}").into()).collect();
    let cols: Vec<Box<str>> = (0..n).map(|i| format!("{tag}_{i}").into()).collect();
    let mut triplets = Vec::new();
    let mut state = seed;
    for i in 0..n {
        let left = i % side < side / 2;
        for g in 0..FEATURES {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let jitter = (state >> 62) as f32;
            let base = if (g < FEATURES / 2) == left { 6.0 } else { 0.0 };
            let x = base + jitter;
            if x > 0.0 {
                triplets.push((g as u64, i as u64, x));
            }
        }
    }
    let zarr = dir.join(format!("{tag}.zarr"));
    let mut backend = create_sparse_from_triplets(
        &triplets,
        (FEATURES, n, triplets.len()),
        Some(zarr.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )
    .unwrap();
    backend.register_row_names_vec(&rows);
    backend.register_column_names_vec(&cols);

    let coords = dir.join(format!("{tag}.csv"));
    let mut csv = String::from("barcode,x,y\n");
    for (i, name) in cols.iter().enumerate() {
        csv.push_str(&format!("{name},{},{}\n", i % side, i / side));
    }
    std::fs::write(&coords, csv).unwrap();
    (
        zarr.to_string_lossy().into_owned(),
        coords.to_string_lossy().into_owned(),
    )
}

fn pinto(dir: &Path, args: &[&str]) {
    let out = Command::new(env!("CARGO_BIN_EXE_pinto"))
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "pinto {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn manifest(prefix: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(format!("{prefix}.pinto.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn a_cage_model_predicts_and_imputes_a_new_sample() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let (train, train_xy) = write_section(d, "train", 16, 1);
    let (query, query_xy) = write_section(d, "query", 10, 2);
    let model = d.join("model").to_string_lossy().into_owned();
    let pred = d.join("pred").to_string_lossy().into_owned();
    let imputed = d.join("imp").to_string_lossy().into_owned();
    // What every step reads the data with.
    let small = ["--coord-column-names", "x,y", "-k", "4", "--no-qc"];

    let mut fit = vec!["cage", &train, "-c", &train_xy, "-o", &model];
    fit.extend(small);
    fit.extend([
        "--epochs",
        "1",
        "-d",
        "32",
        "--embedding-dim",
        "4",
        "--edge-cluster-method",
        "kmeans",
        "--n-communities",
        "2",
        "--seed",
        "7",
    ]);
    pinto(d, &fit);
    assert_eq!(manifest(&model)["command"], "cage");

    let mut predict = vec![
        "predict", &query, "-c", &query_xy, "--model", &model, "-o", &pred,
    ];
    predict.extend(small);
    pinto(d, &predict);
    let made = manifest(&pred);
    assert_eq!(made["command"], "predict");
    for table in [
        "coord_pairs",
        "latent",
        "link_community",
        "propensity",
        "feature_community",
        "cell_embedding",
    ] {
        let path = format!("{pred}.{table}.parquet");
        assert!(Path::new(&path).is_file(), "{path} not written");
    }
    // One propensity row per query cell, over the model's communities.
    let is_community = |f: &str| {
        f.strip_prefix('C')
            .is_some_and(|k| k.parse::<usize>().is_ok())
    };
    let (cells, prop) = read_rows(&format!("{pred}.propensity.parquet"), is_community);
    assert_eq!(cells.len(), 100);
    assert!(cells.iter().all(|c| c.starts_with("query_")), "{cells:?}");
    for row in &prop {
        let total: f32 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-3, "row sums to {total}");
    }
    assert_eq!(prop[0].len(), 2, "the model's two communities");

    // Imputed by the model's prefix, and by its manifest: the same model.
    let impute = |named: &str, out: &str| {
        let mut argv = vec![
            "impute", &query, "-c", &query_xy, "--model", named, "-o", out,
        ];
        argv.extend(small);
        pinto(d, &argv);
        read_rows(&format!("{out}.imputed.parquet"), |_| true)
    };
    let (cells, counts) = impute(&model, &imputed);
    assert_eq!(cells.len(), 100);
    assert_eq!(counts[0].len(), FEATURES);
    let by_manifest = d.join("imp2").to_string_lossy().into_owned();
    assert!(impute(&format!("{model}.pinto.json"), &by_manifest) == (cells, counts));
}

/// A parquet table's row names and the columns `keep` picks by name, row
/// by row.
fn read_rows(path: &str, keep: impl Fn(&str) -> bool) -> (Vec<String>, Vec<Vec<f32>>) {
    use legume_numeric::matrix::traits::IoOps;
    let fields = peek_parquet_field_names(path).unwrap();
    let wanted: Vec<Box<str>> = fields.into_iter().skip(1).filter(|f| keep(f)).collect();
    assert!(!wanted.is_empty(), "{path}: no columns picked");
    let m =
        nalgebra::DMatrix::<f32>::from_parquet_with_names(path, Some(0), Some(&wanted)).unwrap();
    let rows = (0..m.mat.nrows())
        .map(|i| m.mat.row(i).iter().copied().collect())
        .collect();
    (m.rows.iter().map(|r| r.to_string()).collect(), rows)
}
