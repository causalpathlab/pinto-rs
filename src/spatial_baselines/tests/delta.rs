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
        scenario: "donor-confounded".into(),
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

/// A run records what it was fitted on and no mode, there being one
/// formulation; a manifest written with a mode (by an earlier pinto) still
/// reads.
#[test]
fn a_run_records_its_input_and_no_mode() {
    use crate::util::metadata::BaselineInput;
    let dir = tempfile::tempdir().unwrap();
    let files = two_sections(dir.path(), 6, 60);
    let out = dir.path().join("s").to_string_lossy().to_string();
    let flags = ["--max-iter", "3", "--ignore-batch"];
    crate::spatial_baselines::spatialde2::run(&spatialde2_args(&files, &out, &flags)).unwrap();
    let text = std::fs::read_to_string(format!("{out}.pinto.json")).unwrap();
    let meta: PintoMetadata = serde_json::from_str(&text).unwrap();
    assert_eq!(meta.baseline.unwrap().input, BaselineInput::Cells);
    let settings: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out}.baseline.json")).unwrap())
            .unwrap();
    assert!(settings["run"].get("mode").is_none(), "{}", settings["run"]);
    // an earlier manifest's mode is read past
    let mut old: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(old["baseline"].get("mode").is_none(), "{}", old["baseline"]);
    for mode in ["sparse", "reference"] {
        old["baseline"]["mode"] = mode.into();
        let read: PintoMetadata = serde_json::from_value(old.clone()).unwrap();
        assert_eq!(read.baseline.unwrap().input, BaselineInput::Cells, "{mode}");
    }
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

/// A planted batch shift shrinks: two sections alike but for every third
/// gene of the second, three times the first's. Read with δ, each shifted
/// gene's total in the second section over its total in the first falls
/// within 1.5 of 1 either way; under `--ignore-batch` it stays near three.
#[test]
fn a_planted_batch_shift_shrinks() {
    use crate::test_support::section;
    const SHIFT: f32 = 3.0;
    let dir = tempfile::tempdir().unwrap();
    let (s1, c1) = section(dir.path(), "s1", 10, &|_| 1.0);
    let (s2, c2) = section(dir.path(), "s2", 10, &|g| {
        if g % 3 == 0 {
            SHIFT
        } else {
            1.0
        }
    });
    let files = vec![
        format!("{s1},{s2}"),
        "-c".into(),
        format!("{c1},{c2}"),
        "--coord-column-names".into(),
        "x,y".into(),
    ];
    let ratios = |tag: &str, extra: &[&str]| -> Vec<(usize, f32)> {
        let out = dir.path().join(tag).to_string_lossy().to_string();
        let data = spatialde2_args(&files, &out, extra)
            .fit
            .input
            .read(false)
            .unwrap();
        let mut total = vec![[0f32; 2]; data.genes.len()];
        for (j, col) in data.counts.col_iter().enumerate() {
            for (&g, &x) in col.row_indices().iter().zip(col.values()) {
                total[g][data.section[j]] += x;
            }
        }
        data.genes
            .iter()
            .zip(&total)
            .map(|(name, t)| {
                let g: usize = name.trim_start_matches("GENE").parse().unwrap();
                (g, t[1] / t[0])
            })
            .collect()
    };
    let corrected = ratios("corrected", &[]);
    let original = ratios("original", &["--ignore-batch"]);
    let shifted = |r: &[(usize, f32)]| -> Vec<f32> {
        r.iter()
            .filter(|(g, _)| g % 3 == 0)
            .map(|&(_, x)| x)
            .collect()
    };
    let (before, after) = (shifted(&original), shifted(&corrected));
    assert_eq!(before.len(), 4);
    for x in before {
        assert!((x - SHIFT).abs() < 0.05 * SHIFT, "uncorrected ratio {x}");
    }
    for x in after {
        assert!((1.0 / 1.5..=1.5).contains(&x), "corrected ratio {x}");
    }
}
