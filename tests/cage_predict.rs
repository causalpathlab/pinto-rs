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
    // under --atomic-outputs no message names a staged path
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stdout.lines().chain(stderr.lines()) {
        assert!(!line.contains(".partial-"), "pinto {args:?} said:\n{line}");
    }
}

/// Every name in `dir` that is a staging folder, and every JSON file there
/// that names one.
fn staged(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let json = p.extension().is_some_and(|x| x == "json");
            p.to_string_lossy().contains(".partial-")
                || (json && std::fs::read_to_string(p).unwrap().contains(".partial-"))
        })
        .map(|p| p.display().to_string())
        .collect()
}

fn manifest(prefix: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(format!("{prefix}.pinto.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// What every step reads the data with.
const SMALL: [&str; 5] = ["--coord-column-names", "x,y", "-k", "4", "--no-qc"];

/// A cage run trained in `d` at `{d}/model`, each command given `extra`.
/// The query section's data and coordinates, and the model's prefix.
fn train(d: &Path, extra: &[&str]) -> (String, String, String) {
    let (train, train_xy) = write_section(d, "train", 16, 1);
    let (query, query_xy) = write_section(d, "query", 10, 2);
    let model = d.join("model").to_string_lossy().into_owned();
    let mut fit = vec!["cage", &train, "-c", &train_xy, "-o", &model];
    fit.extend(SMALL);
    fit.extend([
        "--pair-encoder-steps",
        "8",
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
    fit.extend(extra);
    pinto(d, &fit);
    assert_eq!(manifest(&model)["command"], "cage");
    (query, query_xy, model)
}

/// `command` (predict or impute) on the query section with the model
/// `named`, writing `out`, given `extra`.
fn apply(d: &Path, command: &str, query: (&str, &str), named: &str, out: &str, extra: &[&str]) {
    let mut argv = vec![command, query.0, "-c", query.1, "--model", named, "-o", out];
    argv.extend(SMALL);
    argv.extend(extra);
    pinto(d, &argv);
}

/// Trained, predicted and imputed under `--atomic-outputs`: each step lands
/// its outputs whole, records their final paths and leaves no staging
/// folder. The prediction has a propensity row per query cell; the model
/// imputes the same whether named by its prefix (under `--atomic-outputs`)
/// or by its manifest (without).
#[test]
fn a_cage_model_predicts_and_imputes_a_new_sample() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let atomic = ["--atomic-outputs", "--verbose"];
    let (query, query_xy, model) = train(d, &atomic);
    let query = (query.as_str(), query_xy.as_str());
    let pred = d.join("pred").to_string_lossy().into_owned();
    let imputed = d.join("imp").to_string_lossy().into_owned();

    apply(d, "predict", query, &model, &pred, &atomic);
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
    let impute = |named: &str, out: &str, extra: &[&str]| {
        apply(d, "impute", query, named, out, extra);
        read_rows(&format!("{out}.imputed.parquet"), |_| true)
    };
    let (cells, counts) = impute(&model, &imputed, &atomic);
    assert_eq!(cells.len(), 100);
    assert_eq!(counts[0].len(), FEATURES);
    let by_manifest = d.join("imp2").to_string_lossy().into_owned();
    assert!(impute(&format!("{model}.pinto.json"), &by_manifest, &[]) == (cells, counts));

    for (prefix, table) in [
        (&model, "propensity"),
        (&pred, "propensity"),
        (&imputed, "imputed"),
    ] {
        assert_eq!(manifest(prefix)["prefix"], prefix.as_str());
        let path = format!("{prefix}.{table}.parquet");
        assert!(Path::new(&path).is_file(), "{path} not written");
    }
    assert_eq!(staged(d), Vec::<String>::new());
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
