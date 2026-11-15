//! The baselines take cage's batch options: given the same flags, a
//! baseline's read and cage's preprocessing estimate the same δ, and at
//! the defaults the estimate is cage's at its defaults.

use crate::util::batch_effects::{read_cell_batches, read_delta, RunDelta};
use crate::util::input::SrtInputArgs;
use crate::util::srt_pipeline::{
    preprocess_srt, BatchEstimate, FeatureAxisMode, SrtPreprocessConfig,
};
use clap::Parser;
use legume_numeric::matrix::parquet::{write_named_table, Column};

/// One data file holding two `side × side` grids a thousand units apart
/// (two components of the spatial graph), the second's first six features
/// three times the first's; its data and coordinate arguments.
fn two_pieces(dir: &std::path::Path, side: usize) -> Vec<String> {
    let genes: Vec<String> = (0..12).map(|g| format!("GENE{g}")).collect();
    let genes: Vec<&str> = genes.iter().map(String::as_str).collect();
    let n = side * side;
    let mut triplets = Vec::new();
    let mut csv = String::from("cell,x,y\n");
    for piece in 0..2 {
        for c in 0..n {
            let (r, col) = (c / side, c % side);
            let cell = piece * n + c;
            for g in 0..12usize {
                let marker = if col < side / 2 { g < 6 } else { g >= 6 };
                let v = if marker {
                    4.0 + ((c + g) % 3) as f32
                } else {
                    ((c * 7 + g) % 2) as f32
                };
                let fold = if piece == 1 && g < 6 { 3.0 } else { 1.0 };
                if v > 0.0 {
                    triplets.push((g as u64, cell as u64, fold * v));
                }
            }
            let x = col + 1000 * piece;
            csv.push_str(&format!("one_{cell},{x},{r}\n"));
        }
    }
    let data = crate::test_support::write_zarr(dir, "one", &genes, 2 * n, &triplets);
    let coords = dir.join("one.coords.csv");
    std::fs::write(&coords, csv).unwrap();
    let coords = coords.to_str().unwrap().to_string();
    vec![
        data.to_string(),
        "-c".into(),
        coords,
        "--coord-column-names".into(),
        "x,y".into(),
    ]
}

/// [`crate::test_support::two_sections`]' data and coordinate arguments.
fn two_files(dir: &std::path::Path, side: usize) -> Vec<String> {
    let (data, coords) = crate::test_support::two_sections(dir, side);
    vec![
        data,
        "-c".into(),
        coords,
        "--coord-column-names".into(),
        "x,y".into(),
    ]
}

/// A clone table over the cells of `data` (a data argument of two files),
/// as a fit names them: the first sixteen of each section in a stratum of
/// its own, the rest mixable.
fn clone_table(dir: &std::path::Path, data: &str, side: usize) -> String {
    use data_beans::aux::feature_names::FeatureNameKind;
    let path = dir.join("x.clones.parquet");
    let path = path.to_str().unwrap().to_string();
    let files: Vec<Box<str>> = data.split(',').map(Box::from).collect();
    let cell = crate::util::input::open_fit_data(&files, &FeatureNameKind::Exact)
        .unwrap()
        .column_names()
        .unwrap();
    let n = side * side;
    let stratum: Vec<i32> = (0..cell.len())
        .map(|i| if i % n < 16 { (i / n) as i32 + 1 } else { 0 })
        .collect();
    write_named_table(
        &path,
        "cell",
        &cell,
        &[("stratum".into(), Column::I32(&stratum))],
    )
    .unwrap();
    path
}

/// The δ a spatialde2 read writes under `out` with `flags`.
fn baseline_delta(input: &[String], out: &str, flags: &[&str]) -> Option<RunDelta> {
    let mut argv: Vec<String> = vec!["pinto".into(), "spatialde2".into()];
    argv.extend(input.iter().cloned());
    argv.extend(["-o".into(), out.into()]);
    argv.extend(flags.iter().map(|s| s.to_string()));
    let crate::Commands::Spatialde2(args) = <crate::Cli as Parser>::try_parse_from(argv)
        .unwrap_or_else(|e| panic!("{e}"))
        .commands
    else {
        unreachable!()
    };
    let data = args.fit.input.read(false).unwrap();
    data.delta
        .map(|_| read_delta(&format!("{out}.delta.parquet")).unwrap())
}

/// The δ cage's preprocessing writes under `out` with `flags`, its QC off
/// as the baselines read.
fn cage_delta(input: &[String], out: &str, flags: &[&str]) -> Option<RunDelta> {
    let mut argv: Vec<String> = vec!["pinto".into()];
    argv.extend(input.iter().cloned());
    argv.extend(["-o".into(), out.into(), "--no-qc".into()]);
    argv.extend(flags.iter().map(|s| s.to_string()));
    let args = SrtInputArgs::try_parse_from(argv).unwrap_or_else(|e| panic!("{e}"));
    let pre = preprocess_srt(SrtPreprocessConfig {
        common: &args,
        ablation: None,
        batch_effects: BatchEstimate::Fit,
        feature_axis: FeatureAxisMode::Rows,
        feature_kind: None,
        cell_projection: false,
    })
    .unwrap();
    pre.batch_effects
        .map(|_| read_delta(&format!("{out}.delta.parquet")).unwrap())
}

/// Two δ alike: rows, batches and every plane.
fn assert_same(a: &RunDelta, b: &RunDelta, what: &str) {
    assert_eq!(a.rows, b.rows, "{what}: rows");
    assert_eq!(a.batches, b.batches, "{what}: batches");
    for (k, (pa, pb)) in a.planes().iter().zip(b.planes()).enumerate() {
        assert_eq!(*pa, pb, "{what}: plane {k}");
    }
}

/// The two runs' δ under `flags`, each written beside its run.
fn both(dir: &std::path::Path, input: &[String], tag: &str, flags: &[&str]) -> [RunDelta; 2] {
    let out = |who: &str| {
        dir.join(format!("{who}_{tag}"))
            .to_str()
            .unwrap()
            .to_string()
    };
    let base = baseline_delta(input, &out("baseline"), flags).expect("baseline δ");
    let cage = cage_delta(input, &out("cage"), flags).expect("cage δ");
    [base, cage]
}

/// At the defaults, two files are two batches, and the baseline's δ is
/// cage's at its defaults.
#[test]
fn at_the_defaults_a_baseline_s_delta_is_cage_s() {
    let dir = tempfile::tempdir().unwrap();
    let input = two_files(dir.path(), 8);
    let [base, cage] = both(dir.path(), &input, "default", &[]);
    assert_same(&base, &cage, "defaults");
}

/// `--auto-batch` makes one file of two spatial components two batches,
/// as cage makes them, each cell labelled as cage labels it; without it
/// the file is one batch and no δ is estimated.
#[test]
fn auto_batch_estimates_cage_s_delta_over_the_graph_s_components() {
    let dir = tempfile::tempdir().unwrap();
    let input = two_pieces(dir.path(), 8);
    let plain = dir.path().join("plain").to_str().unwrap().to_string();
    assert!(baseline_delta(&input, &plain, &[]).is_none(), "one file");

    let [base, cage] = both(dir.path(), &input, "auto", &["--auto-batch"]);
    assert_eq!(base.batches.len(), 2, "two components");
    assert_same(&base, &cage, "--auto-batch");
    let cells = |who: &str| {
        let path = dir.path().join(format!("{who}_auto.delta_cells.parquet"));
        read_cell_batches(path.to_str().unwrap()).unwrap()
    };
    assert_eq!(cells("baseline"), cells("cage"));

    let ignored = dir.path().join("ignored").to_str().unwrap().to_string();
    let mut argv: Vec<String> = vec!["pinto".into(), "spatialde2".into()];
    argv.extend(input.iter().cloned());
    argv.extend(["-o", &ignored, "--auto-batch", "--ignore-batch"].map(String::from));
    assert!(<crate::Cli as Parser>::try_parse_from(argv).is_err());
}

/// `--cnv-clones`, `-p` and `--num-levels` reach the estimate as they
/// reach cage's: the same δ under each, and each moves it from the
/// defaults'.
#[test]
fn cnv_clones_and_the_collapse_settings_estimate_cage_s_delta() {
    let dir = tempfile::tempdir().unwrap();
    let side = 24;
    let input = super::delta::two_sections(dir.path(), side, 60);
    let clones = clone_table(dir.path(), &input[0], side);
    let [default, _] = both(dir.path(), &input, "default", &[]);
    for (tag, flags) in [
        ("clones", vec!["--cnv-clones", clones.as_str()]),
        ("proj", vec!["-p", "16"]),
        ("levels", vec!["--num-levels", "1"]),
    ] {
        let [base, cage] = both(dir.path(), &input, tag, &flags);
        assert_same(&base, &cage, tag);
        assert_ne!(base.mean, default.mean, "{tag} leaves δ as the defaults'");
    }
}
