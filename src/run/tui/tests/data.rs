use super::*;

fn pair(name: &str) -> Pair {
    Pair::pending(PathBuf::from(name))
}

fn paths(v: &[&str]) -> Vec<PathBuf> {
    v.iter().map(PathBuf::from).collect()
}

fn sides(pairs: &[Pair], side: Side) -> Vec<Option<PathBuf>> {
    pairs.iter().map(|p| p.side(side).cloned()).collect()
}

fn touch(dir: &Path, name: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, "").unwrap();
    p
}

#[test]
fn stems_drop_data_coordinate_and_label_endings() {
    assert_eq!(stem(Path::new("/d/s1.zarr.zip")), "s1");
    assert_eq!(stem(Path::new("s1_batch.tsv.gz")), "s1_batch");
    assert_eq!(stem(Path::new("s1.h5")), "s1");
    assert_eq!(stem(Path::new("s1_coords.parquet")), "s1_coords");
}

#[test]
fn files_pair_by_name_whatever_their_order() {
    let mut pairs = vec![pair("d/s1.zarr.zip"), pair("d/s2.zarr.zip")];
    let got = assign(
        &mut pairs,
        &paths(&["d/s2_batch.tsv", "d/s1_batch.tsv"]),
        Side::Batch,
    );
    assert_eq!(got, Paired::ByName(2));
    assert_eq!(
        sides(&pairs, Side::Batch),
        [Some("d/s1_batch.tsv".into()), Some("d/s2_batch.tsv".into())]
    );
    let got = assign(
        &mut pairs,
        &paths(&["d/S2.positions.csv", "d/s1-coord.csv"]),
        Side::Coord,
    );
    assert_eq!(got, Paired::ByName(2));
    assert_eq!(
        sides(&pairs, Side::Coord),
        [
            Some("d/s1-coord.csv".into()),
            Some("d/S2.positions.csv".into())
        ]
    );
    // The batch files stay as they were.
    assert_eq!(pairs[0].batch, Some("d/s1_batch.tsv".into()));
}

#[test]
fn names_alike_only_in_their_start_are_not_crossed() {
    // Data listed rep2 first: a shared `rep` start must not decide.
    let mut pairs = vec![pair("rep2.zarr"), pair("rep1.zarr")];
    let got = assign(
        &mut pairs,
        &paths(&["rep_batch_1.txt", "rep_batch_2.txt"]),
        Side::Batch,
    );
    assert_eq!(got, Paired::ByName(2));
    assert_eq!(
        sides(&pairs, Side::Batch),
        [
            Some("rep_batch_2.txt".into()),
            Some("rep_batch_1.txt".into())
        ]
    );
}

#[test]
fn a_number_that_runs_on_is_another_sample() {
    let mut pairs = vec![pair("s1.zarr"), pair("s2.zarr")];
    let got = assign(
        &mut pairs,
        &paths(&["s10_coords.csv", "s2_coords.csv"]),
        Side::Coord,
    );
    assert_eq!(got, Paired::Partly(1));
    assert_eq!(
        sides(&pairs, Side::Coord),
        [None, Some("s2_coords.csv".into())]
    );
}

#[test]
fn unrelated_names_go_in_order_only_when_nothing_matches() {
    let mut pairs = vec![pair("a.zarr"), pair("b.zarr")];
    assert_eq!(
        assign(&mut pairs, &paths(&["x.tsv", "y.tsv"]), Side::Batch),
        Paired::InOrder
    );
    assert_eq!(
        sides(&pairs, Side::Batch),
        [Some("x.tsv".into()), Some("y.tsv".into())]
    );
    let mut pairs = vec![pair("a.zarr"), pair("b.zarr")];
    assert_eq!(
        assign(&mut pairs, &paths(&["x.tsv"]), Side::Batch),
        Paired::Partly(0)
    );
    assert_eq!(sides(&pairs, Side::Batch), [None, None]);
}

#[test]
fn a_label_file_beside_the_data_is_found() {
    let dir = tempfile::tempdir().unwrap();
    let d = touch(dir.path(), "s1.zarr.zip");
    let near = || side_files_in(dir.path(), Side::Batch);
    assert_eq!(beside(&[d.as_path()], &near(), true)[0], None);
    touch(dir.path(), "s10_batch.txt");
    assert_eq!(
        beside(&[d.as_path()], &near(), true)[0],
        None,
        "s10's labels are not s1's"
    );
    touch(dir.path(), "s1.batch.tsv");
    touch(dir.path(), "s2.batch.tsv");
    assert_eq!(
        beside(&[d.as_path()], &near(), true)[0],
        Some(dir.path().join("s1.batch.tsv"))
    );
    touch(dir.path(), "s1_batch.txt");
    assert_eq!(
        beside(&[d.as_path()], &near(), true)[0],
        None,
        "two candidates: none is guessed"
    );
}

#[test]
fn coordinates_and_labels_are_told_apart_by_their_words() {
    let dir = tempfile::tempdir().unwrap();
    touch(dir.path(), "s1_coords.csv");
    touch(dir.path(), "s1_batch.csv");
    touch(dir.path(), "s1_spatial_labels.csv");
    touch(dir.path(), "notes.csv");
    assert_eq!(
        side_files_in(dir.path(), Side::Coord),
        [dir.path().join("s1_coords.csv")]
    );
    assert_eq!(
        side_files_in(dir.path(), Side::Batch),
        [dir.path().join("s1_batch.csv")]
    );
}

#[test]
fn a_generic_positions_file_goes_to_the_only_data_file_beside_it() {
    let dir = tempfile::tempdir().unwrap();
    let d = touch(dir.path(), "filtered.h5");
    std::fs::create_dir(dir.path().join("spatial")).unwrap();
    let pos = touch(&dir.path().join("spatial"), "tissue_positions.csv");
    let near = side_files_in(dir.path(), Side::Coord);
    assert_eq!(near, std::slice::from_ref(&pos));
    assert_eq!(beside(&[d.as_path()], &near, true)[0], Some(pos));
    // With other data beside it, whose it is cannot be told.
    assert_eq!(beside(&[d.as_path()], &near, false)[0], None);
    touch(dir.path(), "other.h5");
    assert_eq!(data_in(dir.path()), 2);
}

#[test]
fn coordinate_files_are_all_or_none() {
    let mut pairs = vec![pair("a.zarr"), pair("b.zarr")];
    assert_eq!(coord_problem(&pairs), None);
    pairs[0].coord = Some("a.csv".into());
    assert!(coord_problem(&pairs).is_some_and(|w| w.contains("coordinates")));
    pairs[1].coord = Some("b.csv".into());
    assert_eq!(coord_problem(&pairs), None);
    // Batch files may be mixed: the run writes the rest.
    pairs[0].batch = Some("a.tsv".into());
    assert_eq!(coord_problem(&pairs), None);
}

#[test]
fn an_exact_name_wins_and_a_file_goes_to_one_data_file() {
    let d = |n: &str| PathBuf::from(n);
    let (s1, rep) = (d("s1.zarr"), d("s1_rep.zarr"));
    let data = [s1.as_path(), rep.as_path()];
    // `s1_coords` is named for s1 exactly: s1_rep, which it only
    // extends, does not take it too.
    assert_eq!(
        beside(&data, &paths(&["s1_coords.csv"]), false),
        [Some(d("s1_coords.csv")), None]
    );
    // With both, each takes its own; the exact match beats the longer one.
    assert_eq!(
        beside(
            &data,
            &paths(&["s1_coords.csv", "s1_rep_coords.csv"]),
            false
        ),
        [Some(d("s1_coords.csv")), Some(d("s1_rep_coords.csv"))]
    );
    // Words are whole: `s1b` is not `s1` extended.
    let s1b = d("s1b.zarr");
    assert_eq!(
        beside(&[s1b.as_path()], &paths(&["s1_coords.csv"]), false),
        [None]
    );
    // A file two data files would take only by extending them goes to neither.
    let (a, b) = (d("s1_a.zarr"), d("s1_b.zarr"));
    assert_eq!(
        beside(
            &[a.as_path(), b.as_path()],
            &paths(&["s1_coords.csv"]),
            false
        ),
        [None, None]
    );
}
