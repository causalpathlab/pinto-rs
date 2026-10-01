use super::*;

fn pair(name: &str, cells: Option<usize>) -> Pair {
    let mut p = Pair::pending(PathBuf::from(name));
    p.cells = cells;
    p
}

#[test]
fn files_left_to_pinto_pass_no_batch_files_and_are_numbered() {
    let pairs = [pair("d/s1.zarr", Some(3)), pair("d/s2.zarr", None)];
    assert_eq!(args(&pairs), Ok(None));
    let (batches, notes) = summary(&pairs);
    assert!(notes.is_empty());
    assert_eq!(
        batches,
        [
            Batch {
                name: "0".into(),
                files: 1,
                cells: Some(3)
            },
            Batch {
                name: "1".into(),
                files: 1,
                cells: None
            },
        ]
    );
}

#[test]
fn a_name_given_to_two_files_makes_one_batch_and_the_rest_are_their_stems() {
    let mut pairs = [
        pair("d/s1.zarr", Some(3)),
        pair("d/s2.zarr", Some(2)),
        pair("e/s3.zarr.zip", Some(4)),
    ];
    pairs[0].name = Some("b1".into());
    pairs[1].name = Some("b1".into());
    assert_eq!(
        args(&pairs),
        Ok(Some(vec![
            Arg::Made("s1.txt".into(), Made::Repeat("b1".into(), 3)),
            Arg::Made("s2.txt".into(), Made::Repeat("b1".into(), 2)),
            Arg::Made("s3.txt".into(), Made::Repeat("s3".into(), 4)),
        ]))
    );
    let (batches, _) = summary(&pairs);
    let named: Vec<(&str, usize, Option<usize>)> = batches
        .iter()
        .map(|b| (b.name.as_str(), b.files, b.cells))
        .collect();
    assert_eq!(named, [("b1", 2, Some(5)), ("s3", 1, Some(4))]);
}

#[test]
fn a_file_of_unknown_size_cannot_be_named_yet() {
    let mut pairs = [pair("d/s1.zarr", None)];
    pairs[0].name = Some("b1".into());
    assert!(args(&pairs).unwrap_err().contains("s1.zarr"));
}

#[test]
fn label_files_pass_as_they_are_unless_renamed_and_same_stems_do_not_collide() {
    let mut pairs = [pair("d/s1.zarr", Some(3)), pair("e/s1.zarr", Some(3))];
    pairs[0].batch = Some("d/s1_batch.txt".into());
    pairs[1].batch = Some("e/s1_batch.txt".into());
    pairs[1].renames.insert("A".into(), "b2".into());
    assert_eq!(
        args(&pairs),
        Ok(Some(vec![
            Arg::Given("d/s1_batch.txt".into()),
            Arg::Made(
                "s1.txt".into(),
                Made::Renamed("e/s1_batch.txt".into(), pairs[1].renames.clone())
            ),
        ]))
    );
    pairs[0].renames.insert("A".into(), "b1".into());
    let Ok(Some(a)) = args(&pairs) else {
        panic!("both renamed")
    };
    let names: Vec<&str> = a
        .iter()
        .map(|a| match a {
            Arg::Made(n, _) => n.as_str(),
            Arg::Given(_) => "",
        })
        .collect();
    assert_eq!(names, ["s1.txt", "s1-2.txt"]);
}

#[test]
fn renamed_labels_are_written_line_for_line_and_never_over_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let labels = dir.path().join("s1_batch.txt");
    std::fs::write(&labels, "A\nB\nA\n").unwrap();
    let counts = label_counts(&labels).unwrap();
    assert_eq!(counts, BTreeMap::from([("A".into(), 2), ("B".into(), 1)]));

    let out = dir.path().join("s1.txt");
    let renames = BTreeMap::from([("A".to_string(), "b1".to_string())]);
    write(&out, &Made::Renamed(labels.clone(), renames)).unwrap();
    assert_eq!(std::fs::read_to_string(&out).unwrap(), "b1\nB\nb1\n");
    assert!(write(&out, &Made::Repeat("x".into(), 1)).is_err());

    let rep = dir.path().join("s2.txt");
    write(&rep, &Made::Repeat("b2".into(), 2)).unwrap();
    assert_eq!(std::fs::read_to_string(&rep).unwrap(), "b2\nb2\n");
}

#[test]
fn renamed_labels_merge_in_the_summary_and_unread_ones_are_noted() {
    let mut pairs = [pair("d/s1.zarr", Some(3)), pair("d/s2.zarr", Some(2))];
    pairs[0].batch = Some("d/s1_batch.txt".into());
    pairs[0].labels = Some(Ok(std::sync::Arc::new(BTreeMap::from([
        ("A".into(), 2),
        ("B".into(), 1),
    ]))));
    pairs[0].renames.insert("B".into(), "A".into());
    pairs[1].batch = Some("d/s2_batch.txt".into());
    let (batches, notes) = summary(&pairs);
    assert_eq!(
        batches,
        [Batch {
            name: "A".into(),
            files: 1,
            cells: Some(3)
        }]
    );
    assert_eq!(notes.len(), 1);
    assert!(notes[0].contains("s2_batch.txt"));
}

#[test]
fn a_new_batch_file_forgets_the_old_names_and_a_cleared_row_is_its_own_again() {
    let mut p = pair("d/s1.zarr", Some(3));
    p.name = Some("b1".into());
    p.set(
        super::super::data::Side::Batch,
        Some("d/s1_batch.txt".into()),
    );
    assert_eq!(kind(&p), Kind::Labels(Path::new("d/s1_batch.txt")));
    p.renames.insert("A".into(), "b1".into());
    p.set(super::super::data::Side::Batch, Some("d/other.txt".into()));
    assert!(p.renames.is_empty() && p.labels.is_none());
    p.clear_batch();
    assert_eq!(kind(&p), Kind::File);
}
