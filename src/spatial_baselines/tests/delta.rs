//! δ for the baselines: a multi-batch input's counts are divided by its
//! own δ estimate before the method's preprocessing; `--ignore-batch` reads
//! the counts exactly as the original method gets them.

use crate::util::batch_effects::read_delta;
use crate::util::metadata::PintoMetadata;

/// Two srt-simulate sections, of two donors with donor-wide nuisance (so
/// δ is not 1), `side × side` spots and `genes` genes each, under `dir`:
/// the data and coordinate arguments.
pub(crate) fn two_sections(dir: &std::path::Path, side: usize, genes: usize) -> Vec<String> {
    let out = dir.join("sim").to_string_lossy().to_string();
    crate::simulate::run::run_simulate(&crate::simulate::run::SimulateArgs {
        scenario: "L1".into(),
        set: [
            format!("grid_side={side}"),
            format!("n_genes={genes}"),
            "n_donors=2".into(),
            "sections_per_donor=1".into(),
            "seed=3".into(),
        ]
        .into(),
        coords: Vec::new(),
        coord_column_names: vec!["row_um".into(), "col_um".into()],
        out: out.as_str().into(),
    })
    .unwrap();
    let file = |s: &str, ext: &str| format!("{out}/sections/{s}.{ext}");
    vec![
        format!("{},{}", file("S1", "zarr"), file("S2", "zarr")),
        "-c".into(),
        format!(
            "{},{}",
            file("S1", "coords.parquet"),
            file("S2", "coords.parquet")
        ),
        "--coord-column-names".into(),
        "row_um,col_um".into(),
    ]
}

fn parse(argv: &[String]) -> crate::Commands {
    <crate::Cli as clap::Parser>::try_parse_from(argv)
        .unwrap_or_else(|e| panic!("{e}"))
        .commands
}

fn spatialde2_args(
    files: &[String],
    out: &str,
    extra: &[&str],
) -> Box<crate::spatial_baselines::spatialde2::Args> {
    let mut argv: Vec<String> = vec!["pinto".into(), "spatialde2".into()];
    argv.extend(files.iter().cloned());
    argv.extend(["-o".into(), out.into()]);
    argv.extend(extra.iter().map(|s| s.to_string()));
    let crate::Commands::Spatialde2(args) = parse(&argv) else {
        unreachable!()
    };
    args
}

/// Without `--ignore-batch` the counts read are the counts as stored, each
/// divided by its gene's δ in its cell's batch, the δ written beside the
/// run; with it they are the stored counts, bit for bit, and no δ is
/// estimated.
#[test]
fn counts_are_divided_by_the_input_s_own_delta_unless_ignore_batch() {
    let dir = tempfile::tempdir().unwrap();
    let files = two_sections(dir.path(), 10, 60);
    let plain = dir.path().join("plain").to_string_lossy().to_string();
    let original = spatialde2_args(&files, &plain, &["--ignore-batch"])
        .fit
        .input
        .read(false)
        .unwrap();
    assert!(original.delta.is_none());
    assert!(!std::path::Path::new(&format!("{plain}.delta.parquet")).exists());

    let out = dir.path().join("corrected").to_string_lossy().to_string();
    let corrected = spatialde2_args(&files, &out, &[])
        .fit
        .input
        .read(false)
        .unwrap();
    let applied = corrected.delta.as_ref().expect("two batches give a δ");
    let delta = read_delta(&format!("{out}.delta.parquet")).unwrap();
    assert_eq!(applied.batches, delta.batches);
    let mean = delta.mean_on_rows(&original.genes);
    assert!(mean.iter().any(|&d| (d - 1.0).abs() > 1e-3), "δ is all 1");

    assert_eq!(original.cells, corrected.cells);
    let col: Vec<usize> = original
        .batches
        .iter()
        .map(|b| delta.batches.iter().position(|d| d == b).unwrap())
        .collect();
    let (a, b) = (&original.counts, &corrected.counts);
    assert_eq!(a.col_offsets(), b.col_offsets());
    assert_eq!(a.row_indices(), b.row_indices());
    for (j, (ca, cb)) in a.col_iter().zip(b.col_iter()).enumerate() {
        for ((&g, &x), &y) in ca.row_indices().iter().zip(ca.values()).zip(cb.values()) {
            assert_eq!(y, x / mean[(g, col[j])], "gene {g}, cell {j}");
        }
    }
}

/// A run records what it did about batch effects: the δ files and their
/// labels in the manifest, `batch_correction` in `baseline.json`; under
/// `--ignore-batch`, neither δ file and `delta: false`.
#[test]
fn a_run_records_its_batch_correction() {
    let dir = tempfile::tempdir().unwrap();
    let files = two_sections(dir.path(), 8, 60);
    for (tag, extra) in [("c", &[][..]), ("i", &["--ignore-batch"][..])] {
        let out = dir.path().join(tag).to_string_lossy().to_string();
        let mut flags = vec!["--neighbors", "4", "--max-iter", "5"];
        flags.extend(extra);
        crate::spatial_baselines::spatialde2::run(&spatialde2_args(&files, &out, &flags)).unwrap();
        let meta: PintoMetadata =
            serde_json::from_str(&std::fs::read_to_string(format!("{out}.pinto.json")).unwrap())
                .unwrap();
        let settings: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{out}.baseline.json")).unwrap())
                .unwrap();
        let corrected = extra.is_empty();
        assert_eq!(meta.outputs.batch_effects.is_some(), corrected, "{tag}");
        assert_eq!(meta.outputs.delta_cells.is_some(), corrected, "{tag}");
        assert_eq!(meta.batch_labels.is_some(), corrected, "{tag}");
        assert_eq!(settings["batch_correction"]["delta"], corrected, "{tag}");
    }
}

/// `--reference` is recorded in the manifest and `baseline.json` (the
/// sparse default otherwise), and spatialde2 refuses it with
/// `--neighbors`, which only the sparse graph takes.
#[test]
fn the_mode_is_recorded() {
    use crate::util::metadata::{BaselineInput, BaselineMode};
    let dir = tempfile::tempdir().unwrap();
    let files = two_sections(dir.path(), 6, 60);
    for (tag, flag, mode) in [
        ("s", None, BaselineMode::Sparse),
        ("r", Some("--reference"), BaselineMode::Reference),
    ] {
        let out = dir.path().join(tag).to_string_lossy().to_string();
        let mut flags = vec!["--max-iter", "3", "--ignore-batch"];
        flags.extend(flag);
        crate::spatial_baselines::spatialde2::run(&spatialde2_args(&files, &out, &flags)).unwrap();
        let meta: PintoMetadata =
            serde_json::from_str(&std::fs::read_to_string(format!("{out}.pinto.json")).unwrap())
                .unwrap();
        let run = meta.baseline.unwrap();
        assert_eq!((run.mode, run.input), (mode, BaselineInput::Cells), "{tag}");
        let settings: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{out}.baseline.json")).unwrap())
                .unwrap();
        assert_eq!(settings["run"]["mode"], serde_json::to_value(mode).unwrap());
    }
    let mut argv: Vec<String> = vec!["pinto".into(), "spatialde2".into()];
    argv.extend(files.iter().cloned());
    argv.extend(["-o", "x", "--reference", "--neighbors", "4"].map(String::from));
    assert!(<crate::Cli as clap::Parser>::try_parse_from(argv).is_err());
}

/// A multi-batch `-pb` read takes the projection the δ estimate collapsed
/// on: the batch-corrected projection of the counts as stored, as a fresh
/// projection gives it.
#[test]
fn a_pb_read_reuses_the_delta_estimate_s_projection() {
    use data_beans::alg::random_projection::RandProjOps;
    let dir = tempfile::tempdir().unwrap();
    let files = two_sections(dir.path(), 6, 60);
    let out = dir.path().join("pb").to_string_lossy().to_string();
    let args = spatialde2_args(&files, &out, &[]);
    let data = args.fit.input.read(true).unwrap();
    assert!(data.delta.is_some());
    let i = &args.fit.input;
    let srt = crate::util::input::read_data_with_coordinates(crate::util::input::SRTReadArgs {
        data_files: i.data_files.clone(),
        coord_files: i.coord_files.clone(),
        coord_columns: Vec::new(),
        coord_column_names: i.coord_column_names.clone(),
        batch_files: None,
        header_in_coord: None,
        feature_kind: data_beans::aux::feature_names::FeatureNameKind::Exact,
    })
    .unwrap();
    let fresh = srt
        .data
        .project_columns_with_batch_correction(
            crate::util::input::DEFAULT_PROJ_DIM,
            None,
            Some(srt.batches.as_slice()),
        )
        .unwrap()
        .proj;
    assert_eq!(data.projection.unwrap(), fresh);
}

/// The projection a δ estimate keeps is the one a fresh projection with
/// the same `-p` and batch membership gives after it, basis
/// and all: what the srt pipeline reuses in place of projecting again.
#[test]
fn the_estimate_s_projection_is_a_fresh_one_taken_after_it() {
    use crate::util::batch_effects::{
        estimate_and_write_batch_effects, sorted_batch_labels, EstimateBatchArgs,
    };
    use data_beans::alg::random_projection::RandProjOps;
    let dir = tempfile::tempdir().unwrap();
    let files = two_sections(dir.path(), 6, 60);
    let args = spatialde2_args(&files, "unused", &[]);
    let i = &args.fit.input;
    let mut srt = crate::util::input::read_data_with_coordinates(crate::util::input::SRTReadArgs {
        data_files: i.data_files.clone(),
        coord_files: i.coord_files.clone(),
        coord_columns: Vec::new(),
        coord_column_names: i.coord_column_names.clone(),
        batch_files: None,
        header_in_coord: None,
        feature_kind: data_beans::aux::feature_names::FeatureNameKind::Exact,
    })
    .unwrap();
    let proj_dim = 16;
    let labels = sorted_batch_labels(&srt.batches);
    let kept = estimate_and_write_batch_effects(
        &mut srt.data,
        &srt.batches,
        &labels,
        EstimateBatchArgs::at(proj_dim, 10, 5, None),
        None,
        true,
    )
    .unwrap()
    .expect("two batches give a δ")
    .projection
    .expect("kept");
    let fresh = srt
        .data
        .project_columns_with_batch_correction(proj_dim, None, Some(srt.batches.as_slice()))
        .unwrap();
    assert_eq!(kept.proj, fresh.proj);
    assert_eq!(kept.basis, fresh.basis);
}
