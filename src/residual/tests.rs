use super::*;
use legume_numeric::matrix::parquet::read_parquet_string_columns_by_name;

/// Two communities over three rows, `[K, G]`: community 0 = (0.5, 0.5, 0),
/// community 1 = (0, 0.5, 0.5).
fn total(col: &[(usize, f64)]) -> f64 {
    col.iter().map(|e| e.1).sum()
}

fn profiles() -> nalgebra::DMatrix<f64> {
    nalgebra::DMatrix::from_row_slice(2, 3, &[0.5, 0.5, 0.0, 0.0, 0.5, 0.5])
}

#[test]
fn a_cell_at_its_expectation_leaves_no_excess() {
    // θ = (1, 0), n = 20: μ = (10, 10, 0).
    let col = [(0, 10.0), (1, 10.0)];
    assert!(excess_counts(&col, total(&col), &profiles(), &[1.0, 0.0], Excess::Round).is_empty());
}

#[test]
fn counts_above_the_expectation_are_the_excess() {
    // θ = (0.5, 0.5), y = (8, 20, 12), n = 40: μ = (10, 20, 10).
    let col = [(0, 8.0), (1, 20.0), (2, 12.0)];
    assert_eq!(
        excess_counts(&col, total(&col), &profiles(), &[0.5, 0.5], Excess::Round),
        vec![(2, 2.0)]
    );
}

#[test]
fn excess_is_rounded_and_never_negative() {
    // θ = (1, 0), y = (1, 2), n = 3: μ = (1.5, 1.5). Row 0 is below its
    // mean; row 1 is 0.5 above, which rounds away from zero.
    let col = [(0, 1.0), (1, 2.0)];
    assert_eq!(
        excess_counts(&col, total(&col), &profiles(), &[1.0, 0.0], Excess::Round),
        vec![(1, 1.0)]
    );
}

#[test]
fn floor_drops_an_excess_below_one() {
    // μ = (1.5, 1.5): row 1 is 0.5 above, which floors to nothing.
    let col = [(0, 1.0), (1, 2.0)];
    assert!(excess_counts(&col, total(&col), &profiles(), &[1.0, 0.0], Excess::Floor).is_empty());
    let col = [(0, 1.0), (1, 4.0)]; // n = 5, μ = 2.5: row 1 is 1.5 above
    assert_eq!(
        excess_counts(&col, total(&col), &profiles(), &[1.0, 0.0], Excess::Floor),
        vec![(1, 1.0)]
    );
}

#[test]
fn em_theta_explains_a_pure_cell() {
    let m = profiles();
    let col = [(0, 30.0), (1, 30.0)];
    let pi = cell_theta(&col, &m, 200);
    assert!((pi[0] - 1.0).abs() < 1e-3, "{pi:?}");
    assert!(excess_counts(&col, total(&col), &m, &pi, Excess::Round).is_empty());
}

#[test]
fn a_cell_with_no_counts_has_no_theta_and_no_excess() {
    let m = profiles();
    let pi = cell_theta(&[], &m, 50);
    assert_eq!(pi, vec![0.0, 0.0]);
    assert!(excess_counts(&[], total(&[]), &m, &pi, Excess::Round).is_empty());
}

/// Files of cells over `g0`, `g1` (`(name, cell i's counts)`; cells `a`,
/// `b`, ... in order) and a run on them as a fit leaves it: propensity
/// `prop` over the cells `{barcode}@{file}`, each file its own batch,
/// coordinates and a manifest. Returns the run prefix and the data files.
fn write_run(d: &str, sections: &[(&str, Vec<(f32, f32)>)], prop: Mat) -> (String, Vec<String>) {
    use crate::link_community::outputs::write_propensity_matrix;
    use crate::simulate::write::write_backend;
    use crate::util::metadata::PintoMetadata;
    use crate::util::parquet_io::write_cells_table;
    let genes: Vec<Box<str>> = vec!["g0".into(), "g1".into()];
    let (mut files, mut names, mut batches) = (Vec::new(), Vec::new(), Vec::new());
    for (s, y) in sections {
        let bc: Vec<Box<str>> = (0..y.len())
            .map(|i| char::from(b'a' + i as u8).to_string().into())
            .collect();
        let trip = y
            .iter()
            .enumerate()
            .flat_map(|(i, &(a, b))| [(0, i as u64, a), (1, i as u64, b)])
            .collect();
        let f = format!("{d}/{s}.zarr");
        write_backend(&f, &genes, &bc, trip).unwrap();
        files.push(f);
        for b in &bc {
            names.push(Box::<str>::from(format!("{b}@{s}")));
            batches.push(Box::<str>::from(*s));
        }
    }
    let n = names.len();
    let prefix = format!("{d}/fit");
    write_propensity_matrix(&prefix, &prop, &names).unwrap();
    let coords = Mat::from_fn(n, 2, |i, j| (i * 2 + j) as f32);
    let cn: Vec<Box<str>> = vec!["row_um".into(), "col_um".into()];
    write_cells_table(&prefix, &names, &coords, &cn, &batches, &vec![true; n]).unwrap();
    let mut meta = PintoMetadata {
        command: "lc".into(),
        prefix: prefix.clone(),
        data_files: Some(files.clone()),
        ..Default::default()
    };
    meta.outputs.coord_columns = Some(vec!["row_um".into(), "col_um".into()]);
    meta.outputs.propensity = Some(format!("{prefix}.propensity.parquet"));
    meta.outputs.cells = Some(crate::util::parquet_io::cells_table_path(&prefix));
    meta.write(Path::new(&format!("{prefix}.pinto.json")))
        .unwrap();
    (prefix, files)
}

/// Two files `S1`, `S2` of cells `a`, `b` over `g0`, `g1`, all (10, 10)
/// but S2/b at (10, 30), and a one-community run on them in which S2/b
/// carries no weight, so the profile is the other cells' (1/2, 1/2).
/// Returns the run prefix and the data files.
fn fixture(d: &str) -> (String, Vec<String>) {
    let sections = [
        ("S1", vec![(10.0, 10.0); 2]),
        ("S2", vec![(10.0, 10.0), (10.0, 30.0)]),
    ];
    write_run(
        d,
        &sections,
        Mat::from_column_slice(4, 1, &[1.0, 1.0, 1.0, 0.0]),
    )
}

fn args(run: &str, out: &str, data_files: Option<Vec<Box<str>>>) -> ResidualArgs {
    ResidualArgs {
        run: run.into(),
        out: out.into(),
        data_files,
        operation: Operation::Subtract,
        excess: Excess::Floor,
        em_iters: 50,
        block_size: Some(1),
        feature_names: None,
        adapt: Adapt::Batch,
        adapt_cells: 50.0,
    }
}

/// The fixture's run, rewritten: its propensity over `names`, `prop`
/// (cells x communities), and its command.
fn rewrite_run(prefix: &str, names: &[&str], prop: Mat, command: &str) {
    use crate::link_community::outputs::write_propensity_matrix;
    use crate::util::metadata::PintoMetadata;
    let names: Vec<Box<str>> = names.iter().map(|&n| n.into()).collect();
    write_propensity_matrix(prefix, &prop, &names).unwrap();
    let manifest = format!("{prefix}.pinto.json");
    PintoMetadata::update(Path::new(&manifest), |m| m.command = command.into()).unwrap();
}

const CELLS: [&str; 4] = ["a@S1", "b@S1", "a@S2", "b@S2"];

/// `g1` count of column `j`.
fn g1(c: &nalgebra_sparse::CscMatrix<f32>, j: usize) -> f32 {
    column_entries(c, j)
        .iter()
        .find(|e| e.0 == 1)
        .map_or(0.0, |e| e.1 as f32)
}

fn read(path: &str) -> nalgebra_sparse::CscMatrix<f32> {
    let m = data_beans::sparse_io::open_sparse_matrix_by_path(path).unwrap();
    let n = m.num_columns().unwrap();
    m.read_columns_csc((0..n).collect()).unwrap()
}

/// S2/b, 20 over the others on `g1`, keeps half of it as excess (μ_g1 = 20
/// of its 40); the (10, 10) cells sit at their mean. Each file gets its own
/// cells' coordinates.
#[test]
fn a_run_writes_each_files_excess_and_coordinates() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, _) = fixture(d);
    let out = format!("{d}/resid");
    run_residual(&args(&prefix, &out, None)).unwrap();
    let (s1, s2) = (
        read(&format!("{out}/S1.zarr.zip")),
        read(&format!("{out}/S2.zarr.zip")),
    );
    assert_eq!(g1(&s2, 1), 10.0, "S2/b keeps its excess on g1");
    assert_eq!(g1(&s1, 0), 0.0);
    assert_eq!(g1(&s1, 1), 0.0);
    let spots = read_parquet_string_columns_by_name(&format!("{out}/S2.coords.parquet"), &["spot"])
        .unwrap();
    assert_eq!(spots[0], vec![Box::<str>::from("a"), "b".into()]);
    assert!(Path::new(&format!("{out}/residual.json")).exists());
}

/// The plan names what a run writes, all of it: each file's zarr and
/// coordinates, in the run's order, then residual.json, and nothing else
/// in the output folder.
#[test]
fn the_plan_names_every_file_a_run_writes() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, _) = fixture(d);
    let out = format!("{d}/resid");
    let a = args(&prefix, &out, None);
    let plan = Plan::new(&Run::open(&prefix).unwrap(), &a).unwrap();
    assert_eq!(
        plan.made(),
        [
            "S1.zarr.zip",
            "S1.coords.parquet",
            "S2.zarr.zip",
            "S2.coords.parquet",
            "residual.json"
        ]
        .map(|f| std::path::PathBuf::from(format!("{out}/{f}")))
    );
    assert_eq!(plan.coord_names, ["row_um".into(), "col_um".into()]);
    run_residual(&a).unwrap();
    let mut written: Vec<std::path::PathBuf> = std::fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    written.sort();
    let mut made = plan.made();
    made.sort();
    assert_eq!(written, made);
}

/// An `out` given as a folder, ending in `/`, names each file once under
/// it: no doubled slash in the paths the plan and its reply show.
#[test]
fn a_folder_out_names_its_files_without_a_doubled_slash() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, _) = fixture(d);
    let plan = Plan::new(
        &Run::open(&prefix).unwrap(),
        &args(&prefix, &format!("{d}/resid/"), None),
    )
    .unwrap();
    let under = |f: &str| format!("{d}/resid/{f}");
    assert_eq!(plan.zarrs, [under("S1.zarr.zip"), under("S2.zarr.zip")]);
    assert_eq!(
        plan.coords,
        [under("S1.coords.parquet"), under("S2.coords.parquet")]
    );
    assert_eq!(plan.summary, under("residual.json"));
}

/// A subset of the run's files is refused: a fit names its cells by its
/// files, so the subset could not find the run's cells.
#[test]
fn a_subset_of_the_runs_files_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, files) = fixture(d);
    let out = format!("{d}/resid");
    let err = run_residual(&args(&prefix, &out, Some(vec![files[1].clone().into()])))
        .unwrap_err()
        .to_string();
    assert!(err.contains("fitted on 2"), "{err}");
}

/// `-o` on the data's own directory is refused before any input is touched.
#[test]
fn an_output_over_the_input_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, files) = fixture(d);
    assert!(run_residual(&args(&prefix, d, None)).is_err());
    assert_eq!(g1(&read(&files[1]), 1), 30.0, "the input is intact");
}

/// Profiles are each community's share of every data row, so a feature
/// reads its own profile entry whatever its name.
#[test]
fn profiles_cover_every_data_row_and_sum_to_one() {
    // [G, K]: three rows, two communities
    let gamma = Mat::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 2.0, 2.0, 2.0]).map(f64::from);
    let p_t = normalized_profiles(&gamma);
    assert_eq!(p_t.shape(), (2, 3));
    for k in 0..2 {
        assert!((p_t.row(k).sum() - 1.0).abs() < 1e-12, "{p_t}");
    }
    assert!(
        (p_t[(0, 0)] - 0.25).abs() < 1e-12 && p_t[(1, 0)] == 0.0,
        "{p_t}"
    );
}

/// A run cell missing from the data is refused rather than given no
/// weight, which would leave all its counts as excess.
#[test]
fn a_run_cell_missing_from_the_data_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, _) = fixture(d);
    let mut names = CELLS.to_vec();
    names[0] = "zz@S1";
    rewrite_run(
        &prefix,
        &names,
        Mat::from_column_slice(4, 1, &[1.0, 1.0, 1.0, 0.0]),
        "lc",
    );
    let err = run_residual(&args(&prefix, &format!("{d}/resid"), None))
        .unwrap_err()
        .to_string();
    assert!(err.contains("not in"), "{err}");
}

/// A community with no weight in the data changes nothing: its profile is
/// left out, so it cannot soak up a cell's excess.
#[test]
fn a_community_with_no_weight_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, _) = fixture(d);
    let prop = Mat::from_row_slice(4, 2, &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    rewrite_run(&prefix, &CELLS, prop, "lc");
    let out = format!("{d}/resid");
    run_residual(&args(&prefix, &out, None)).unwrap();
    assert_eq!(g1(&read(&format!("{out}/S2.zarr.zip")), 1), 10.0);
}

/// Outputs never write over anything: an existing output is refused and
/// left as it was, and two data files with one name are refused.
#[test]
fn outputs_never_write_over_anything() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, files) = fixture(d);
    let out = format!("{d}/resid");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(format!("{out}/S1.zarr.zip"), "keep").unwrap();
    let err = run_residual(&args(&prefix, &out, None))
        .unwrap_err()
        .to_string();
    assert!(err.contains("exists"), "{err}");
    assert_eq!(
        std::fs::read_to_string(format!("{out}/S1.zarr.zip")).unwrap(),
        "keep"
    );
    assert!(
        !Path::new(&format!("{out}/S2.zarr.zip")).exists(),
        "nothing written"
    );

    let other = format!("{d}/other");
    std::fs::create_dir_all(&other).unwrap();
    let twin = format!("{other}/S1.zarr");
    copy_dir(&files[0], &twin);
    let err = run_residual(&args(
        &prefix,
        &format!("{d}/resid2"),
        Some(vec![files[0].clone().into(), twin.into()]),
    ))
    .unwrap_err()
    .to_string();
    assert!(err.contains("same name"), "{err}");
}

/// A run is taken by what its manifest records, not by its command: one
/// that records no propensity output, or no cells table (the coordinates
/// and the batches), is refused, the missing record named, before anything
/// is written, even with the file itself under its prefix.
#[test]
fn a_run_that_lacks_a_record_is_refused_by_its_name() {
    use crate::util::metadata::{OutputFiles, PintoMetadata};
    type Unset = fn(&mut OutputFiles);
    let cases: [(Unset, &str); 2] = [
        (
            |o| o.propensity = None,
            "records no propensity output (outputs.propensity)",
        ),
        (|o| o.cells = None, "records no cells table (outputs.cells)"),
    ];
    for (unset, message) in cases {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path().to_str().unwrap();
        let (prefix, _) = fixture(d);
        let manifest = format!("{prefix}.pinto.json");
        PintoMetadata::update(Path::new(&manifest), |m| unset(&mut m.outputs)).unwrap();
        let out = format!("{d}/resid");
        let err = run_residual(&args(&prefix, &out, None))
            .unwrap_err()
            .to_string();
        assert!(err.contains(message), "{err}");
        assert!(!Path::new(&out).exists(), "nothing written");
    }
}

/// A recorded table that is gone, the propensity or the cells table, is
/// refused while planning, by name.
#[test]
fn a_recorded_table_that_is_gone_is_refused_while_planning() {
    type Table = fn(&str) -> String;
    let cases: [Table; 2] = [
        |prefix| format!("{prefix}.propensity.parquet"),
        |prefix| crate::util::parquet_io::cells_table_path(prefix),
    ];
    for table in cases {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path().to_str().unwrap();
        let (prefix, _) = fixture(d);
        let gone = table(&prefix);
        std::fs::remove_file(&gone).unwrap();
        let a = args(&prefix, &format!("{d}/resid"), None);
        let e = Plan::new(&Run::open(&prefix).unwrap(), &a)
            .expect_err("refused")
            .to_string();
        let name = Path::new(&gone).file_name().unwrap().to_str().unwrap();
        assert!(e.contains(name) && e.contains("not there"), "{e}");
    }
}

/// A run whose manifest records no data files is refused while planning,
/// the missing record named, and is taken with its files given as
/// `--data-files`.
#[test]
fn a_run_that_records_no_data_files_takes_them_as_given() {
    use crate::util::metadata::PintoMetadata;
    type Unset = fn(&mut PintoMetadata);
    let cases: [Unset; 2] = [|m| m.data_files = None, |m| m.data_files = Some(vec![])];
    for unset in cases {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path().to_str().unwrap();
        let (prefix, files) = fixture(d);
        PintoMetadata::update(Path::new(&format!("{prefix}.pinto.json")), unset).unwrap();
        let run = Run::open(&prefix).unwrap();
        let out = format!("{d}/resid");
        let e = Plan::new(&run, &args(&prefix, &out, None))
            .expect_err("refused")
            .to_string();
        assert!(
            e.contains("records no data files (data_files)") && e.contains("--data-files"),
            "{e}"
        );
        assert!(!Path::new(&out).exists(), "nothing written");

        let given: Vec<Box<str>> = files.iter().map(|f| f.as_str().into()).collect();
        let plan = Plan::new(&run, &args(&prefix, &out, Some(given.clone()))).unwrap();
        assert_eq!(plan.files, given);
    }
}

/// The command that wrote the run does not matter, and one-hot memberships
/// (hard labels) are read as propensities: with S2/b in a community of its
/// own, each community's profile is its own cells', and nothing is left.
/// With every cell in one community, S2/b keeps its excess as in the
/// soft-weight fixture.
#[test]
fn a_run_of_any_command_with_one_hot_memberships_is_taken() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, _) = fixture(d);
    let apart = Mat::from_row_slice(4, 2, &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    rewrite_run(&prefix, &CELLS, apart, "CT1-labels");
    let out = format!("{d}/apart");
    run_residual(&ResidualArgs {
        adapt: Adapt::None,
        ..args(&prefix, &out, None)
    })
    .unwrap();
    let s2 = read(&format!("{out}/S2.zarr.zip"));
    assert_eq!(s2.nnz(), 0, "each cell sits at its own community's mean");

    rewrite_run(
        &prefix,
        &CELLS,
        Mat::from_column_slice(4, 1, &[1.0; 4]),
        "CT1-labels",
    );
    let out = format!("{d}/together");
    run_residual(&ResidualArgs {
        adapt: Adapt::None,
        ..args(&prefix, &out, None)
    })
    .unwrap();
    assert!(g1(&read(&format!("{out}/S2.zarr.zip")), 1) > 0.0);
}

/// A spatial baseline's run is taken as a fit's is: it records its data files (as a fit does, so that
/// they are found from any working directory), propensity, cells table
/// (with the batch of each cell) and coordinate columns. Each data file
/// gets its excess zarr and coordinates, and a fit reads them as it reads
/// data: the rows every file shares, each file's own cells, a coordinate
/// per cell.
#[test]
fn a_residual_is_taken_from_a_spatial_baseline_run() {
    residual_from_baseline("spatialde2");
}

fn residual_from_baseline(command: &str) {
    use data_beans::aux::feature_names::FeatureNameKind;
    let dir = tempfile::tempdir().unwrap();
    let side = 6;
    let (data, coords) = crate::test_support::two_sections(dir.path(), side);
    let fit = dir.path().join("base").to_string_lossy().into_owned();
    // the data files as typed are not their canonical paths
    std::fs::create_dir(dir.path().join("aside")).unwrap();
    let typed: Vec<String> = ["s1", "s2"]
        .iter()
        .map(|s| format!("{}/aside/../{s}.zarr", dir.path().display()))
        .collect();
    let argv = [
        "pinto",
        "fit",
        command,
        &typed.join(","),
        "-c",
        &coords,
        "--coord-column-names",
        "x,y",
        "-o",
        &fit,
        "--max-iter",
        "5",
    ];
    match <crate::Cli as clap::Parser>::try_parse_from(argv)
        .unwrap_or_else(|e| panic!("{e}"))
        .commands
    {
        crate::Commands::Fit(crate::Fit::Spatialde2(base)) => {
            crate::baselines::spatial::spatialde2::run(&base)
        }
        _ => unreachable!(),
    }
    .unwrap();

    let run = Run::open(&fit).unwrap();
    let canonical: Vec<String> = typed
        .iter()
        .map(|f| {
            let path = std::fs::canonicalize(f).unwrap();
            path.to_string_lossy().into_owned()
        })
        .collect();
    assert_eq!(
        run.meta.data_files.as_deref(),
        Some(&canonical[..]),
        "{command}: data files are recorded as a fit records them"
    );

    // the default --adapt batch groups by the run's own batches
    let recorded = run.meta.outputs.cells.as_deref().unwrap();
    let cells = crate::util::parquet_io::read_cells_table(Path::new(recorded), None).unwrap();
    assert!(cells.batches.is_some(), "the run's cells carry their batch");

    let out = dir.path().join("resid").to_string_lossy().into_owned();
    run_residual(&args(&fit, &out, None)).unwrap();
    let marker: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out}/residual.json")).unwrap())
            .unwrap();
    assert_eq!(marker["counts"], "excess", "{marker}");
    assert_eq!(marker["files"].as_array().unwrap().len(), 2, "{marker}");

    let raw: Vec<Box<str>> = data.split(',').map(Box::from).collect();
    let zarrs: Vec<Box<str>> = ["s1", "s2"]
        .iter()
        .map(|s| format!("{out}/{s}.zarr.zip").into())
        .collect();
    let open = |f: &[Box<str>]| crate::util::input::open_fit_data(f, &FeatureNameKind::Exact);
    let (before, after) = (open(&raw).unwrap(), open(&zarrs).unwrap());
    assert_eq!(after.num_rows(), before.num_rows());
    assert_eq!(after.row_names().unwrap(), before.row_names().unwrap());
    assert_eq!(after.num_columns(), 2 * side * side);
    for (j, s) in ["s1", "s2"].iter().enumerate() {
        let barcodes = after[j].column_names().unwrap();
        assert_eq!(barcodes, before[j].column_names().unwrap(), "{s}");
        let path = format!("{out}/{s}.coords.parquet");
        let spots = read_parquet_string_columns_by_name(&path, &["spot"]).unwrap();
        assert_eq!(spots[0], barcodes, "{s}: a coordinate per cell");
        let table = crate::util::parquet_io::read_cells_table(
            Path::new(&path),
            Some(&["x".to_string(), "y".to_string()]),
        )
        .unwrap();
        assert_eq!(table.coords.len(), side * side, "{s}");
    }
    // some excess is left, and never more than the counts in
    let kept: f64 = marker["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            let (i, o) = (
                f["counts_in"].as_f64().unwrap(),
                f["excess_out"].as_f64().unwrap(),
            );
            assert!(o <= i, "{f}");
            o
        })
        .sum();
    assert!(kept > 0.0, "{marker}");
}

fn copy_dir(from: &str, to: &str) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dst = format!("{to}/{}", e.file_name().to_string_lossy());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path().to_string_lossy(), &dst);
        } else {
            std::fs::copy(e.path(), dst).unwrap();
        }
    }
}

/// Only communities with weight in the data are kept, in order.
#[test]
fn communities_without_weight_are_left_out() {
    let prop = Mat::from_row_slice(2, 3, &[0.5, 0.0, 0.2, 0.5, 0.0, 0.8]);
    let kept = weighted_communities(prop);
    assert_eq!(kept, Mat::from_row_slice(2, 2, &[0.5, 0.2, 0.5, 0.8]));
}

/// residual.json carries the block lra reads (counts: excess, the stage-1
/// run, the raw data files index for index), so lra reads a fit on the
/// residual's files on their raw counts.
#[test]
fn lra_reads_a_fit_on_the_residual_as_a_residual_fit() {
    use crate::util::metadata::PintoMetadata;
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, files) = fixture(d);
    let out = format!("{d}/resid");
    run_residual(&args(&prefix, &out, None)).unwrap();
    let marker: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out}/residual.json")).unwrap())
            .unwrap();
    assert_eq!(marker["counts"], "excess", "{marker}");
    assert_eq!(marker["adapt"], "batch", "{marker}");
    assert_eq!(marker["adapt_cells"], 50.0, "{marker}");
    assert!(marker["formula"].is_string(), "{marker}");
    // paths that resolve from anywhere: canonical, absolute, existing
    let stage1 = std::fs::canonicalize(format!("{prefix}.pinto.json")).unwrap();
    assert_eq!(
        marker["stage1"],
        stage1.to_string_lossy().as_ref(),
        "{marker}"
    );
    let raw: Vec<String> = files
        .iter()
        .map(|f| {
            std::fs::canonicalize(f)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(marker["raw_data_files"], serde_json::json!(raw), "{marker}");

    // a stage-2 fit on the residual's files
    let stage2 = format!("{d}/fit2");
    let meta = PintoMetadata {
        command: "lc".into(),
        prefix: stage2.clone(),
        data_files: Some(vec![
            format!("{out}/S1.zarr.zip"),
            format!("{out}/S2.zarr.zip"),
        ]),
        ..Default::default()
    };
    meta.write(Path::new(&format!("{stage2}.pinto.json")))
        .unwrap();
    let run = Run::open(&stage2).unwrap();
    let found = crate::lr_activity::input::residual_of(&run)
        .unwrap()
        .expect("read as a residual fit");
    let raw: Vec<std::path::PathBuf> = raw.iter().map(std::path::PathBuf::from).collect();
    assert_eq!(found.raw_data_files, raw);
    assert_eq!(found.stage1.unwrap().manifest, stage1);
}

#[test]
fn pooled_profiles_carry_the_gamma_prior() {
    // One community, sums (4, 2) as [G, K]: (4 + 1, 2 + 1) / 8.
    let p = pooled_profiles(&nalgebra::DMatrix::from_column_slice(2, 1, &[4.0, 2.0]));
    assert!((p[(0, 0)] - 5.0 / 8.0).abs() < 1e-12);
    assert!((p[(0, 1)] - 3.0 / 8.0).abs() < 1e-12);
}

#[test]
fn an_adapted_profile_is_the_batchs_sums_shrunk_toward_the_pool() {
    let pooled = nalgebra::DMatrix::from_row_slice(1, 2, &[0.5, 0.5]);
    // Batch sums (30, 10) as [G, K], alpha = 40: (20 + 30, 20 + 10) / 80.
    let sums = nalgebra::DMatrix::from_column_slice(2, 1, &[30.0, 10.0]);
    let p = adapt_profiles(&pooled, &sums, &[40.0]);
    assert!((p[(0, 0)] - 50.0 / 80.0).abs() < 1e-12);
    assert!((p[(0, 1)] - 30.0 / 80.0).abs() < 1e-12);
    // No counts of the community in the batch: the pool.
    let p = adapt_profiles(&pooled, &nalgebra::DMatrix::zeros(2, 1), &[40.0]);
    assert_eq!(p, pooled);
}

/// The run's batch labels set the groups, across files; a cell the cells
/// table lacks (not placed) groups apart; no batch labels is one batch.
#[test]
fn batch_groups_follow_the_runs_batches_across_files() {
    let boxed = |v: &[&str]| -> Vec<Box<str>> { v.iter().map(|&s| s.into()).collect() };
    let table = boxed(&["a@S1", "b@S1", "a@S2", "b@S2"]);
    let names = boxed(&["a@S1", "b@S1", "a@S2", "b@S2", "a@S3"]);
    let batches = boxed(&["D1", "D1", "D1", "D2"]);
    assert_eq!(
        batch_groups(&table, Some(&batches), &names),
        vec![0, 0, 0, 1, 2]
    );
    assert_eq!(batch_groups(&table, None, &names), vec![0; 5]);
}

/// `--adapt` defaults to batch, and `--adapt-cells` refuses a negative
/// weight.
#[test]
fn adapt_defaults_to_batch_and_refuses_a_negative_weight() {
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        r: ResidualArgs,
    }
    let parse =
        |extra: &[&str]| Cli::try_parse_from(["x", "--run", "r", "-o", "o"].iter().chain(extra));
    assert_eq!(parse(&[]).unwrap().r.adapt, Adapt::Batch);
    let err = parse(&["--adapt-cells=-1"])
        .err()
        .expect("refused")
        .to_string();
    assert!(err.contains("must be finite and not negative"), "{err}");
}

/// Two files of 20 cells, one community, each file its own batch. S1: all
/// (20, 20). S2: every cell (20, 40), a shift of the whole community, and
/// cell 0 (20, 120) on top. Returns the run prefix.
fn shifted_fixture(d: &str) -> String {
    let n = 20;
    let mut y2 = vec![(20.0, 40.0); n];
    y2[0] = (20.0, 120.0);
    let sections = [("S1", vec![(20.0, 20.0); n]), ("S2", y2)];
    write_run(d, &sections, Mat::from_element(2 * n, 1, 1.0)).0
}

/// Total `g1` excess over a residual file's cells, and cell 0's.
fn g1_excess(out: &str, s: &str) -> (f32, f32) {
    let m = read(&format!("{out}/{s}.zarr.zip"));
    ((0..m.ncols()).map(|j| g1(&m, j)).sum(), g1(&m, 0))
}

/// Pooled profiles leave S2's whole-community shift as excess in every S2
/// cell; batch-adapted profiles absorb it and keep the one shifted cell.
#[test]
fn batch_adaptation_absorbs_a_community_wide_shift_and_keeps_a_local_one() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let prefix = shifted_fixture(d);

    let pooled = format!("{d}/pooled");
    run_residual(&ResidualArgs {
        adapt: Adapt::None,
        ..args(&prefix, &pooled, None)
    })
    .unwrap();
    let (all, one) = g1_excess(&pooled, "S2");
    assert!(
        all - one > 40.0,
        "pooled: S2's ordinary cells carry the shift ({all}, {one})"
    );

    let adapted = format!("{d}/adapted");
    run_residual(&ResidualArgs {
        adapt: Adapt::Batch,
        adapt_cells: 1.0,
        ..args(&prefix, &adapted, None)
    })
    .unwrap();
    let (all, one) = g1_excess(&adapted, "S2");
    assert!(one > 20.0, "the one shifted cell keeps its excess ({one})");
    assert!(
        all - one < 19.0,
        "the community-wide shift is absorbed ({all}, {one})"
    );
}

/// A data file given that does not exist is refused while planning, before
/// any count is read or a job started, and nothing is written.
#[test]
fn a_missing_data_file_is_refused_while_planning() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, files) = fixture(d);
    let out = format!("{d}/resid");
    let given = Some(vec![
        files[0].clone().into(),
        format!("{d}/nowhere/S2.zarr").into(),
    ]);
    let a = args(&prefix, &out, given);
    let e = Plan::new(&Run::open(&prefix).unwrap(), &a)
        .expect_err("refused")
        .to_string();
    assert!(
        e.contains("nowhere/S2.zarr") && e.contains("not found"),
        "{e}"
    );
    assert!(!Path::new(&out).exists());
}

/// An `-o` that is a file, or under one, is refused while planning: every
/// path under it would read as free, and the run would fail only when it
/// writes.
#[test]
fn an_output_that_is_a_file_is_refused_while_planning() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, _) = fixture(d);
    let out = format!("{d}/resid");
    std::fs::write(&out, "").unwrap();
    for o in [out.clone(), format!("{out}/sub")] {
        let e = Plan::new(&Run::open(&prefix).unwrap(), &args(&prefix, &o, None))
            .expect_err("refused")
            .to_string();
        assert!(e.contains(&format!("{out} is a file")), "{e}");
    }
}

/// A residual of a fit on a residual is refused while planning, whether the
/// fit's manifest names the residual's files or `--data-files` does: its
/// residual.json would name the excess files as raw, and lra on a fit of it
/// would read excess counts as raw counts.
#[test]
fn a_residual_of_a_fit_on_a_residual_is_refused() {
    use crate::util::metadata::PintoMetadata;
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, raw) = fixture(d);
    let out = format!("{d}/resid");
    run_residual(&args(&prefix, &out, None)).unwrap();
    let excess: Vec<String> = ["S1", "S2"]
        .iter()
        .map(|s| format!("{out}/{s}.zarr.zip"))
        .collect();
    let stage2 = format!("{d}/fit2");
    let write = |files: Vec<String>| {
        let mut meta = PintoMetadata {
            command: "lc".into(),
            prefix: stage2.clone(),
            data_files: Some(files),
            ..Default::default()
        };
        meta.outputs.coord_columns = Some(vec!["row_um".into(), "col_um".into()]);
        meta.write(Path::new(&format!("{stage2}.pinto.json")))
            .unwrap();
    };
    let refused = |data_files: Option<Vec<Box<str>>>| {
        let a = args(&stage2, &format!("{d}/resid2"), data_files);
        Plan::new(&Run::open(&stage2).unwrap(), &a)
            .expect_err("refused")
            .to_string()
    };
    write(excess.clone());
    let e = refused(None);
    assert!(e.contains("a fit on a residual"), "{e}");
    // the files moved: the manifest's are gone, --data-files names the excess
    write(vec![
        format!("{d}/moved/S1.zarr"),
        format!("{d}/moved/S2.zarr"),
    ]);
    let e = refused(Some(excess.iter().map(|f| f.as_str().into()).collect()));
    assert!(e.contains("a fit on a residual"), "{e}");
    // a fit whose own manifest carries the residual block, on files with no
    // marker beside them
    let manifest = format!("{stage2}.pinto.json");
    let mut m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest).unwrap()).unwrap();
    m["residual"] = serde_json::json!({"counts": "excess", "raw_data_files": raw});
    std::fs::write(&manifest, m.to_string()).unwrap();
    let e = refused(Some(raw.iter().map(|f| f.as_str().into()).collect()));
    assert!(e.contains("a fit on a residual"), "{e}");
}

/// `--residual divide` writes the tables `μ` is rebuilt from, and the `μ`
/// rebuilt from them is the one subtract takes its excess from, at every
/// nonzero; `p̄`, `φ` and `σ²` are the dense sums over every cell and row
/// (the zeros included) that the closed forms stand for. It writes no
/// count files, and `residual.json` says divide.
#[test]
fn divide_writes_the_tables_mu_is_rebuilt_from() {
    use crate::util::stage1_divisor::{Stage1Divisor, Stage1Tables};
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    // two communities over two batches of 20 cells
    let prefix = shifted_fixture(d);
    let prop = Mat::from_fn(40, 2, |i, k| {
        let w = (i % 5) as f32 / 4.0;
        if k == 0 {
            w
        } else {
            1.0 - w
        }
    });
    let names: Vec<String> = ["S1", "S2"]
        .iter()
        .flat_map(|s| (0..20).map(move |i| format!("{}@{s}", char::from(b'a' + i as u8))))
        .collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    rewrite_run(&prefix, &refs, prop, "lc");
    let out = format!("{d}/divide");
    let a = ResidualArgs {
        operation: Operation::Divide,
        adapt_cells: 5.0,
        ..args(&prefix, &out, None)
    };
    run_residual(&a).unwrap();
    let plan = Plan::new(&Run::open(&prefix).unwrap(), &a).unwrap();
    assert!(plan.zarrs.is_empty());
    let mut written: Vec<std::path::PathBuf> = std::fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    written.sort();
    let mut made = plan.made();
    made.sort();
    assert_eq!(written, made);
    let marker: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out}/residual.json")).unwrap())
            .unwrap();
    assert_eq!(marker["operation"], "divide", "{marker}");
    assert_eq!(marker["counts"], "raw", "{marker}");

    // today's μ, as subtract computes it
    let inp = Inputs::load(&a, &plan).unwrap();
    let profiles = group_profiles(&inp, a.adapt_cells, a.block_size).unwrap();
    let tables = Stage1Tables::read(&out, &crate::util::memory::Budget::new(None, None)).unwrap();
    let rows = inp.data.row_names().unwrap();
    let s1 = Stage1Divisor::align(&tables, &inp.names, &rows).unwrap();
    let n = inp.data.num_columns();
    let csc = inp.data.read_columns_csc(0..n).unwrap();
    let g = rows.len();
    let (mut mu_sum, mut mu2_sum, mut dev) = (vec![0f64; g], vec![0f64; g], vec![0f64; g]);
    let mut n_total = 0.0;
    for j in 0..n {
        let p_t = &profiles[inp.group_of[j]];
        let col = column_entries(&csc, j);
        let pi = cell_theta(&col, p_t, a.em_iters);
        let nn: f64 = col.iter().map(|e| e.1).sum();
        n_total += nn;
        for &(r, _) in &col {
            let want = mu_at(nn, p_t, &pi, r);
            let got = f64::from(s1.mu(j, r));
            assert!(
                (got - want).abs() <= 1e-4 * want.max(1.0),
                "cell {j} row {r}: {got} vs {want}"
            );
        }
        for r in 0..g {
            let mu = mu_at(nn, p_t, &pi, r);
            let y = col.iter().find(|e| e.0 == r).map_or(0.0, |e| e.1);
            mu_sum[r] += mu;
            mu2_sum[r] += mu * mu;
            dev[r] += (y - mu).powi(2) - mu;
        }
    }
    for r in 0..g {
        let pbar = mu_sum[r] / n_total;
        assert!((f64::from(tables.pbar[r]) - pbar).abs() <= 1e-5 * pbar.max(1e-6));
        let phi = (dev[r] / mu2_sum[r]).max(0.0);
        assert!(
            (f64::from(tables.phi[r]) - phi).abs() <= 1e-4 * phi.max(1e-3),
            "row {r}: φ {} vs {phi}",
            tables.phi[r]
        );
    }
}

/// `--residual` takes divide and subtract and their aliases, divide by
/// default.
#[test]
fn the_operation_defaults_to_divide_and_takes_its_aliases() {
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        r: ResidualArgs,
    }
    let parse = |extra: &[&str]| {
        Cli::try_parse_from(["x", "--run", "r", "-o", "o"].iter().chain(extra))
            .unwrap()
            .r
            .operation
    };
    assert_eq!(parse(&[]), Operation::Divide);
    assert_eq!(parse(&["--residual", "multiplicative"]), Operation::Divide);
    assert_eq!(parse(&["--residual", "subtract"]), Operation::Subtract);
    assert_eq!(parse(&["--residual", "additive"]), Operation::Subtract);
}

/// The two-stage fit end to end under divide: an lc run, its residual, and
/// lc, dsvd and cage on the run's own files with `--offset`, each of which
/// runs, estimates no batch effect, and records the folder in its manifest.
#[test]
fn stage_two_fits_run_on_the_divisor() {
    use crate::util::metadata::PintoMetadata;
    use clap::Parser;
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = crate::test_support::two_sections(dir.path(), 8);
    let d = dir.path().to_str().unwrap();
    let stage1 = format!("{d}/lc1");
    let common = |out: &str| -> Vec<String> {
        [
            &data,
            "-c",
            &coords,
            "--coord-column-names",
            "x,y",
            "-o",
            out,
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    };
    let lc = |out: &str, extra: &[&str]| {
        let mut argv = vec!["lc".to_string()];
        argv.extend(common(out));
        argv.extend(
            [
                "--n-communities",
                "2",
                "--num-gibbs",
                "2",
                "--num-greedy",
                "1",
                "--num-levels",
                "2",
            ]
            .iter()
            .chain(extra)
            .map(|s| s.to_string()),
        );
        crate::link_community::fit::fit_srt_link_community(
            &crate::link_community::fit::SrtLinkCommunityArgs::try_parse_from(argv).unwrap(),
        )
        .unwrap();
    };
    lc(&stage1, &[]);
    let out = format!("{d}/resid");
    run_residual(&ResidualArgs {
        operation: Operation::Divide,
        block_size: None,
        ..args(&stage1, &out, None)
    })
    .unwrap();
    let offset = ["--offset", out.as_str()];

    let lc2 = format!("{d}/lc2");
    lc(&lc2, &offset);

    let dsvd2 = format!("{d}/dsvd2");
    let mut argv = vec!["dsvd".to_string()];
    argv.extend(common(&dsvd2));
    argv.extend(["-t", "3", "--num-levels", "2", "--offset", out.as_str()].map(String::from));
    crate::svd::fit::fit_srt_delta_svd(
        &crate::svd::fit::SrtDeltaSvdArgs::try_parse_from(argv).unwrap(),
    )
    .unwrap();

    let cage2 = format!("{d}/cage2");
    {
        use crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            cage: CellActivityGraphEmbeddingArgs,
        }
        let mut argv = vec!["x".to_string()];
        argv.extend(common(&cage2));
        argv.extend(
            [
                "--pair-encoder-steps",
                "8",
                "--epochs",
                "2",
                "--embedding-dim",
                "4",
                "-d",
                "24",
                "--num-levels",
                "3",
                "--n-edge-clusters",
                "2",
                "--edge-cluster-method",
                "kmeans",
                "--feature-inclusion",
                "moran",
                "--offset",
                out.as_str(),
            ]
            .map(String::from),
        );
        crate::cell_activity_graph_embedding::fit::fit_cell_activity_graph_embedding(
            &Cli::try_parse_from(argv).unwrap().cage,
        )
        .unwrap();
    }

    let canonical = std::fs::canonicalize(&out).unwrap();
    for fit in [&lc2, &dsvd2, &cage2] {
        let meta =
            PintoMetadata::read(Path::new(&crate::util::metadata::manifest_path(fit))).unwrap();
        assert_eq!(
            meta.offset.as_deref(),
            Some(canonical.to_string_lossy().as_ref()),
            "{fit}"
        );
        assert!(
            meta.outputs.batch_effects.is_none(),
            "{fit}: a batch effect"
        );
        assert!(
            !Path::new(&format!("{fit}.delta.parquet")).exists(),
            "{fit}"
        );
    }
}

/// A cage run read its gene-named rows (`ENSG…_SYMBOL`) by symbol (its
/// default `--feature-name-mode auto`), two rows of one symbol as one, and
/// its manifest, written before the mode was recorded, says only `cage`.
/// Its stage-1 tables are on the rows a stage-2 cage fit of the same data
/// reads (`--offset` finds every one), and `residual.json` says how they
/// were read; `--feature-name-mode exact` still reads the names as written.
#[test]
fn divide_reads_a_cage_runs_rows_as_cage_read_them() {
    use crate::cell_activity_graph_embedding::args::FeatureNameMode;
    use crate::simulate::write::write_backend;
    use crate::util::metadata::PintoMetadata;
    use crate::util::stage1_divisor::Stage1Divisor;
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let prefix = shifted_fixture(d);
    let genes: Vec<Box<str>> = vec![
        "ENSG00000243485_MIR1302-2HG".into(),
        "ENSG00000237613_FAM138A".into(),
        "ENSG00000284733_OR4F29".into(),
        "ENSG00000284662_OR4F29".into(),
    ];
    std::fs::create_dir(format!("{d}/genes")).unwrap();
    let files: Vec<String> = ["S1", "S2"]
        .iter()
        .map(|s| {
            let bc: Vec<Box<str>> = (0..20)
                .map(|i| char::from(b'a' + i as u8).to_string().into())
                .collect();
            let trip = (0..20u64)
                .flat_map(|i| (0..4u64).map(move |g| (g, i, (1 + g + i % 3) as f32)))
                .collect();
            let f = format!("{d}/genes/{s}.zarr");
            write_backend(&f, &genes, &bc, trip).unwrap();
            f
        })
        .collect();
    let manifest = format!("{prefix}.pinto.json");
    let recorded = files.clone();
    PintoMetadata::update(Path::new(&manifest), move |m| {
        m.command = "cage".into();
        m.data_files = Some(recorded);
    })
    .unwrap();

    // stage 2: a cage fit of the run's data, as it reads it
    let peek = crate::util::input::open_sparse(&files[0])
        .unwrap()
        .row_names()
        .unwrap();
    let kind = FeatureNameMode::Auto.resolve_kind(&peek);
    let boxed: Vec<Box<str>> = files.iter().map(|f| f.as_str().into()).collect();
    let data = crate::util::input::open_fit_data(&boxed, &kind).unwrap();
    let (cells, rows) = (data.column_names().unwrap(), data.row_names().unwrap());
    assert_eq!(rows.len(), 3, "cage reads the rows by symbol: {rows:?}");

    let out = format!("{d}/divide");
    run_residual(&ResidualArgs {
        operation: Operation::Divide,
        ..args(&prefix, &out, None)
    })
    .unwrap();
    let s1 = Stage1Divisor::load(&out, &cells, &rows).unwrap();
    assert_eq!(s1.n_rows(), rows.len());
    let marker: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out}/residual.json")).unwrap())
            .unwrap();
    assert_eq!(marker["feature_names"], "gene", "{marker}");

    // a run that records its mode is read by it, whatever its command
    PintoMetadata::update(Path::new(&manifest), |m| {
        m.command = "lc".into();
        m.feature_names = Some(FeatureNameMode::Gene);
    })
    .unwrap();
    let recorded_out = format!("{d}/recorded");
    run_residual(&ResidualArgs {
        operation: Operation::Divide,
        ..args(&prefix, &recorded_out, None)
    })
    .unwrap();
    Stage1Divisor::load(&recorded_out, &cells, &rows).unwrap();

    // the flag overrides the run: exact names, which the symbols are not
    let exact = format!("{d}/exact");
    run_residual(&ResidualArgs {
        operation: Operation::Divide,
        feature_names: Some(FeatureNameMode::Exact),
        ..args(&prefix, &exact, None)
    })
    .unwrap();
    let e = Stage1Divisor::load(&exact, &cells, &rows).unwrap_err();
    assert!(
        format!("{e:#}").contains("not in the stage-1 tables"),
        "{e:#}"
    );
}
