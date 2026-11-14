use super::*;
use legume_numeric::matrix::parquet::read_parquet_string_columns_by_name;

/// Two communities over three rows, `[K, G]`: community 0 = (0.5, 0.5, 0),
/// community 1 = (0, 0.5, 0.5).
fn profiles() -> nalgebra::DMatrix<f64> {
    nalgebra::DMatrix::from_row_slice(2, 3, &[0.5, 0.5, 0.0, 0.0, 0.5, 0.5])
}

#[test]
fn a_cell_at_its_expectation_leaves_no_excess() {
    // θ = (1, 0), n = 20: μ = (10, 10, 0).
    let col = [(0, 10.0), (1, 10.0)];
    assert!(excess_counts(&col, &profiles(), &[1.0, 0.0], Excess::Round).is_empty());
}

#[test]
fn counts_above_the_expectation_are_the_excess() {
    // θ = (0.5, 0.5), y = (8, 20, 12), n = 40: μ = (10, 20, 10).
    let col = [(0, 8.0), (1, 20.0), (2, 12.0)];
    assert_eq!(
        excess_counts(&col, &profiles(), &[0.5, 0.5], Excess::Round),
        vec![(2, 2.0)]
    );
}

#[test]
fn excess_is_rounded_and_never_negative() {
    // θ = (1, 0), y = (1, 2), n = 3: μ = (1.5, 1.5). Row 0 is below its
    // mean; row 1 is 0.5 above, which rounds away from zero.
    let col = [(0, 1.0), (1, 2.0)];
    assert_eq!(
        excess_counts(&col, &profiles(), &[1.0, 0.0], Excess::Round),
        vec![(1, 1.0)]
    );
}

#[test]
fn floor_drops_an_excess_below_one() {
    // μ = (1.5, 1.5): row 1 is 0.5 above, which floors to nothing.
    let col = [(0, 1.0), (1, 2.0)];
    assert!(excess_counts(&col, &profiles(), &[1.0, 0.0], Excess::Floor).is_empty());
    let col = [(0, 1.0), (1, 4.0)]; // n = 5, μ = 2.5: row 1 is 1.5 above
    assert_eq!(
        excess_counts(&col, &profiles(), &[1.0, 0.0], Excess::Floor),
        vec![(1, 1.0)]
    );
}

#[test]
fn em_theta_explains_a_pure_cell() {
    let m = profiles();
    let col = [(0, 30.0), (1, 30.0)];
    let pi = cell_theta(&col, &m, 200);
    assert!((pi[0] - 1.0).abs() < 1e-3, "{pi:?}");
    assert!(excess_counts(&col, &m, &pi, Excess::Round).is_empty());
}

#[test]
fn a_cell_with_no_counts_has_no_theta_and_no_excess() {
    let m = profiles();
    let pi = cell_theta(&[], &m, 50);
    assert_eq!(pi, vec![0.0, 0.0]);
    assert!(excess_counts(&[], &m, &pi, Excess::Round).is_empty());
}

/// A run on `files` as a fit would leave it: propensity `prop` over the
/// cells `names` (`{barcode}@{file base}`), their batches and coordinates,
/// and a manifest. Returns its prefix.
fn write_run(
    d: &str,
    files: &[String],
    names: &[Box<str>],
    prop: &Mat,
    batches: &[Box<str>],
) -> String {
    use crate::link_community::outputs::write_propensity_matrix;
    use crate::util::metadata::PintoMetadata;
    use crate::util::parquet_io::write_cells_table;
    let n = names.len();
    let prefix = format!("{d}/fit");
    write_propensity_matrix(&prefix, prop, names).unwrap();
    let coords = Mat::from_fn(n, 2, |i, j| (i * 2 + j) as f32);
    let cn: Vec<Box<str>> = vec!["row_um".into(), "col_um".into()];
    write_cells_table(&prefix, names, &coords, &cn, batches, &vec![true; n]).unwrap();
    let mut meta = PintoMetadata {
        command: "lc".into(),
        prefix: prefix.clone(),
        data_files: Some(files.to_vec()),
        ..Default::default()
    };
    meta.outputs.coord_columns = Some(vec!["row_um".into(), "col_um".into()]);
    meta.write(Path::new(&format!("{prefix}.pinto.json")))
        .unwrap();
    prefix
}

/// Each file's cells as the run names them, and their batch: the file base.
fn run_names(files: &[(&str, usize)]) -> (Vec<Box<str>>, Vec<Box<str>>) {
    files
        .iter()
        .flat_map(|&(s, n)| (0..n).map(move |i| (format!("c{i}@{s}").into(), s.into())))
        .unzip()
}

/// One file of cells over `g0`, `g1`, cell `i` at `y[i]`.
fn write_section(d: &str, s: &str, y: &[(f32, f32)]) -> String {
    let genes: Vec<Box<str>> = vec!["g0".into(), "g1".into()];
    let bc: Vec<Box<str>> = (0..y.len()).map(|i| format!("c{i}").into()).collect();
    let trip = y
        .iter()
        .enumerate()
        .flat_map(|(i, &(a, b))| [(0, i as u64, a), (1, i as u64, b)])
        .collect();
    let f = format!("{d}/{s}.zarr");
    crate::simulate::write::write_backend(&f, &genes, &bc, trip).unwrap();
    f
}

/// Two files `S1`, `S2` of cells `c0`, `c1` over `g0`, `g1`, all (10, 10)
/// but S2/c1 at (10, 30), and a one-community run on them in which S2/c1
/// carries no weight, so the profile is the other cells' (1/2, 1/2).
/// Returns the run prefix and the data files.
fn fixture(d: &str) -> (String, Vec<String>) {
    let files = vec![
        write_section(d, "S1", &[(10.0, 10.0); 2]),
        write_section(d, "S2", &[(10.0, 10.0), (10.0, 30.0)]),
    ];
    let (names, batches) = run_names(&[("S1", 2), ("S2", 2)]);
    let prop = Mat::from_column_slice(4, 1, &[1.0, 1.0, 1.0, 0.0]);
    (write_run(d, &files, &names, &prop, &batches), files)
}

fn args(run: &str, out: &str, data_files: Option<Vec<Box<str>>>) -> ResidualArgs {
    ResidualArgs {
        run: run.into(),
        out: out.into(),
        data_files,
        excess: Excess::Floor,
        em_iters: 50,
        block_size: Some(1),
        adapt: Adapt::None,
        adapt_cells: 50.0,
    }
}

/// `g1` count of column `j`.
fn g1(c: &nalgebra_sparse::CscMatrix<f32>, j: usize) -> f32 {
    column_entries(c, j)
        .iter()
        .find(|e| e.0 == 1)
        .map_or(0.0, |e| e.1 as f32)
}

fn read(path: &str, n: usize) -> nalgebra_sparse::CscMatrix<f32> {
    data_beans::sparse_io::open_sparse_matrix_by_path(path)
        .unwrap()
        .read_columns_csc((0..n).collect())
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
        read(&format!("{out}/S1.zarr"), 2),
        read(&format!("{out}/S2.zarr"), 2),
    );
    assert_eq!(g1(&s2, 1), 10.0, "S2/b keeps its excess on g1");
    assert_eq!(g1(&s1, 0), 0.0);
    assert_eq!(g1(&s1, 1), 0.0);
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out}/residual.json")).unwrap())
            .unwrap();
    assert_eq!(meta["counts"], "excess");
    assert!(meta["stage1"].as_str().unwrap().ends_with("fit.pinto.json"));
    let raw = meta["raw_data_files"].as_array().unwrap();
    assert!(raw[1].as_str().unwrap().ends_with("S2.zarr"));
    assert!(Path::new(raw[1].as_str().unwrap()).is_absolute());
    let spots = read_parquet_string_columns_by_name(&format!("{out}/S2.coords.parquet"), &["spot"])
        .unwrap();
    assert_eq!(spots[0], vec![Box::<str>::from("c0"), "c1".into()]);
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
    assert_eq!(g1(&read(&files[1], 2), 1), 30.0, "the input is intact");
}

/// Rows of a `[K, G]` matrix.
fn kg(rows: &[&[f64]]) -> nalgebra::DMatrix<f64> {
    let g = rows[0].len();
    nalgebra::DMatrix::from_row_iterator(rows.len(), g, rows.iter().flat_map(|r| r.iter().copied()))
}

#[test]
fn a_negative_prior_weight_is_refused() {
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        r: ResidualArgs,
    }
    let parse = |v: &str| Cli::try_parse_from(["x", "--run", "r", "-o", "o", "--adapt-cells", v]);
    assert!(parse("-1").is_err());
    assert!(parse("NaN").is_err());
    assert_eq!(parse("0").unwrap().r.adapt_cells, 0.0);
}

#[test]
fn pooled_profiles_add_one_to_the_summed_groups() {
    // One community over two genes, summed sums (4, 2): (5, 3) / 8.
    let p = pooled_profiles(&kg(&[&[4.0, 2.0]]));
    assert!((p[(0, 0)] - 5.0 / 8.0).abs() < 1e-12);
    assert!((p[(0, 1)] - 3.0 / 8.0).abs() < 1e-12);
}

#[test]
fn an_adapted_profile_is_the_groups_sums_shrunk_toward_the_pool() {
    let pooled = kg(&[&[0.5, 0.5]]);
    // Group sums (30, 10), alpha = 40: (20 + 30, 20 + 10) / 80.
    let mut p = kg(&[&[30.0, 10.0]]);
    adapt_profiles(&pooled, &mut p, &[40.0]);
    assert!((p[(0, 0)] - 50.0 / 80.0).abs() < 1e-12);
    assert!((p[(0, 1)] - 30.0 / 80.0).abs() < 1e-12);
    // No counts in the group: the pool.
    let mut p = kg(&[&[0.0, 0.0]]);
    adapt_profiles(&pooled, &mut p, &[40.0]);
    assert_eq!(p, pooled);
}

/// The run's batch labels set the groups, across files; a cell the cells
/// table lacks groups by its file.
#[test]
fn batch_groups_follow_the_runs_batches_across_files() {
    use crate::util::parquet_io::write_cells_table;
    let dir = tempfile::tempdir().unwrap();
    let prefix = format!("{}/fit", dir.path().to_str().unwrap());
    let names: Vec<Box<str>> = ["a@S1", "b@S1", "a@S2", "b@S2", "a@S3"]
        .iter()
        .map(|&s| s.into())
        .collect();
    let batches: Vec<Box<str>> = ["D1", "D1", "D1", "D2", "D2"]
        .iter()
        .map(|&s| s.into())
        .collect();
    let cn: Vec<Box<str>> = vec!["x".into(), "y".into()];
    write_cells_table(
        &prefix,
        &names[..4],
        &Mat::zeros(4, 2),
        &cn,
        &batches[..4],
        &[true; 4],
    )
    .unwrap();
    let path = crate::util::parquet_io::cells_table_path(&prefix);
    let (group, n) = batch_groups(&path, &names, &[2, 2, 1]).unwrap();
    assert_eq!(group, vec![0, 0, 0, 1, 2]);
    assert_eq!(n, 3);
}

/// Two sections of 20 cells, one community. S1: all (20, 20). S2: every
/// cell (20, 40), a shift of the whole community, and cell 0 (20, 120) on
/// top. Returns the run prefix.
fn shifted_fixture(d: &str) -> String {
    let n = 20;
    let mut y2 = vec![(20.0, 40.0); n];
    y2[0] = (20.0, 120.0);
    let files = vec![
        write_section(d, "S1", &vec![(20.0, 20.0); n]),
        write_section(d, "S2", &y2),
    ];
    let (names, batches) = run_names(&[("S1", n), ("S2", n)]);
    write_run(
        d,
        &files,
        &names,
        &Mat::from_element(2 * n, 1, 1.0),
        &batches,
    )
}

/// Total `g1` excess over a file's cells, and cell 0's.
fn g1_excess(out: &str, s: &str, n: usize) -> (f32, f32) {
    let m = read(&format!("{out}/{s}.zarr"), n);
    let all = (0..n).map(|j| g1(&m, j)).sum();
    (all, g1(&m, 0))
}

/// Pooled profiles leave S2's whole-community shift as excess in every S2
/// cell; batch-adapted profiles absorb it and keep the one shifted cell.
#[test]
fn batch_adaptation_absorbs_a_community_wide_shift_and_keeps_a_local_one() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().to_str().unwrap();
    let prefix = shifted_fixture(d);

    let pooled = format!("{d}/pooled");
    run_residual(&args(&prefix, &pooled, None)).unwrap();
    let (all, one) = g1_excess(&pooled, "S2", 20);
    assert!(
        all - one > 40.0,
        "pooled: S2's ordinary cells carry the shift ({all}, {one})"
    );

    let adapted = format!("{d}/adapted");
    let a = ResidualArgs {
        adapt: Adapt::Batch,
        adapt_cells: 1.0,
        ..args(&prefix, &adapted, None)
    };
    run_residual(&a).unwrap();
    let (all, one) = g1_excess(&adapted, "S2", 20);
    assert!(one > 20.0, "the one shifted cell keeps its excess ({one})");
    assert!(
        all - one < 19.0,
        "the community-wide shift is absorbed ({all}, {one})"
    );
}
