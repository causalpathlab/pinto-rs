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

/// Two files `S1`, `S2` of cells `a`, `b` over `g0`, `g1`, all (10, 10)
/// but S2/b at (10, 30), and a one-community run on them in which S2/b
/// carries no weight, so the profile is the other cells' (1/2, 1/2).
/// Returns the run prefix and the data files.
fn fixture(d: &str) -> (String, Vec<String>) {
    use crate::link_community::outputs::write_propensity_matrix;
    use crate::simulate::write::write_backend;
    use crate::util::metadata::PintoMetadata;
    use crate::util::parquet_io::write_cells_table;

    let genes: Vec<Box<str>> = vec!["g0".into(), "g1".into()];
    let bc: Vec<Box<str>> = vec!["a".into(), "b".into()];
    let mut names: Vec<Box<str>> = Vec::new();
    let mut files = Vec::new();
    for s in ["S1", "S2"] {
        let extra = if s == "S2" { 20.0 } else { 0.0 };
        let trip = vec![
            (0, 0, 10.0),
            (1, 0, 10.0),
            (0, 1, 10.0),
            (1, 1, 10.0 + extra),
        ];
        let f = format!("{d}/{s}.zarr");
        write_backend(&f, &genes, &bc, trip).unwrap();
        files.push(f);
        names.extend(bc.iter().map(|b| Box::<str>::from(format!("{b}@{s}"))));
    }
    let prefix = format!("{d}/fit");
    let prop = Mat::from_column_slice(4, 1, &[1.0, 1.0, 1.0, 0.0]);
    write_propensity_matrix(&prefix, &prop, &names).unwrap();
    let coords = Mat::from_row_slice(4, 2, &[0., 0., 1., 1., 2., 2., 3., 3.]);
    let cn: Vec<Box<str>> = vec!["row_um".into(), "col_um".into()];
    let batches: Vec<Box<str>> = vec!["S1".into(), "S1".into(), "S2".into(), "S2".into()];
    write_cells_table(&prefix, &names, &coords, &cn, &batches, &[true; 4]).unwrap();
    let mut meta = PintoMetadata {
        command: "lc".into(),
        prefix: prefix.clone(),
        data_files: Some(files.clone()),
        ..Default::default()
    };
    meta.outputs.coord_columns = Some(vec!["row_um".into(), "col_um".into()]);
    meta.write(Path::new(&format!("{prefix}.pinto.json")))
        .unwrap();
    (prefix, files)
}

fn args(run: &str, out: &str, data_files: Option<Vec<Box<str>>>) -> ResidualArgs {
    ResidualArgs {
        run: run.into(),
        out: out.into(),
        data_files,
        excess: Excess::Floor,
        em_iters: 50,
        block_size: Some(1),
        feature_names: crate::cell_activity_graph_embedding::args::FeatureNameMode::Exact,
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
    let mut meta = PintoMetadata::read(Path::new(&manifest)).unwrap();
    meta.command = command.into();
    meta.write(Path::new(&manifest)).unwrap();
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
    data_beans::sparse_io::open_sparse_matrix_by_path(path)
        .unwrap()
        .read_columns_csc(vec![0, 1])
        .unwrap()
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
        read(&format!("{out}/S1.zarr")),
        read(&format!("{out}/S2.zarr")),
    );
    assert_eq!(g1(&s2, 1), 10.0, "S2/b keeps its excess on g1");
    assert_eq!(g1(&s1, 0), 0.0);
    assert_eq!(g1(&s1, 1), 0.0);
    let spots = read_parquet_string_columns_by_name(&format!("{out}/S2.coords.parquet"), &["spot"])
        .unwrap();
    assert_eq!(spots[0], vec![Box::<str>::from("a"), "b".into()]);
    assert!(Path::new(&format!("{out}/residual.json")).exists());
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
    let p_t = normalised_profiles(&gamma);
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
    assert_eq!(g1(&read(&format!("{out}/S2.zarr")), 1), 10.0);
}

/// Outputs never write over anything: an existing output is refused and
/// left as it was, and two data files with one name are refused.
#[test]
fn outputs_never_write_over_anything() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, files) = fixture(d);
    let out = format!("{d}/resid");
    std::fs::create_dir_all(format!("{out}/S1.zarr")).unwrap();
    std::fs::write(format!("{out}/S1.zarr/keep"), "").unwrap();
    let err = run_residual(&args(&prefix, &out, None))
        .unwrap_err()
        .to_string();
    assert!(err.contains("exists"), "{err}");
    assert!(Path::new(&format!("{out}/S1.zarr/keep")).exists());
    assert!(
        !Path::new(&format!("{out}/S2.zarr")).exists(),
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

/// Only an lc, cage or dsvd run is taken, and a run is checked whole before
/// anything is written.
#[test]
fn the_run_is_checked_before_anything_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let (prefix, _) = fixture(d);
    rewrite_run(
        &prefix,
        &CELLS,
        Mat::from_column_slice(4, 1, &[1.0; 4]),
        "lra",
    );
    let out = format!("{d}/resid");
    let err = run_residual(&args(&prefix, &out, None))
        .unwrap_err()
        .to_string();
    assert!(err.contains("lc, cage or dsvd"), "{err}");
    assert!(!Path::new(&out).exists(), "nothing written");
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
