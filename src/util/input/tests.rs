//! Tests of reading inputs and coordinates.

mod reading {
    use super::super::*;

    #[test]
    fn test_append_batch_coordinate_single_batch() {
        let coords = Mat::from_vec(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let batches = [0u32, 0, 0];
        let result = append_batch_coordinate(&coords, &batches).unwrap();
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 3); // original 2 + 1 batch
                                       // Single batch → all batch coordinates should be 0.0
        for i in 0..3 {
            assert_eq!(result[(i, 2)], 0.0);
        }
    }

    #[test]
    fn test_append_batch_coordinate_two_batches() {
        let coords = Mat::from_vec(4, 2, vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]);
        let batches = [0u32, 0, 1, 1];
        let result = append_batch_coordinate(&coords, &batches).unwrap();
        assert_eq!(result.ncols(), 3);
        // Batch coordinates should differ between A and B
        let batch_a = result[(0, 2)];
        let batch_b = result[(2, 2)];
        assert_ne!(
            batch_a, batch_b,
            "different batches should have different coordinates"
        );
        // Same batch should have same coordinate
        assert_eq!(result[(0, 2)], result[(1, 2)]);
        assert_eq!(result[(2, 2)], result[(3, 2)]);
    }

    #[test]
    fn test_append_batch_coordinate_scaling() {
        // Width = max - min of all coords
        let coords = Mat::from_vec(2, 1, vec![0.0, 10.0]);
        let batches = [0u32, 1];
        let result = append_batch_coordinate(&coords, &batches).unwrap();
        // Width = 10.0, one batch at 0*width, other at 1*width
        let vals: Vec<f32> = (0..2).map(|i| result[(i, 1)]).collect();
        let (lo, hi) = if vals[0] < vals[1] {
            (vals[0], vals[1])
        } else {
            (vals[1], vals[0])
        };
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 10.0); // 1 * width
    }

    #[test]
    fn test_append_batch_coordinate_mismatch() {
        let coords = Mat::from_vec(2, 1, vec![0.0, 1.0]);
        let batches = [0u32]; // wrong length
        assert!(append_batch_coordinate(&coords, &batches).is_err());
    }

    use std::io::Write as _;

    fn write_csv(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_read_one_coord_file_text_match_no_fallback() {
        let f = write_csv(
            "barcode,cell_centroid_x,cell_centroid_y\n\
             cell_a,1.0,2.0\n\
             cell_b,3.0,4.0\n",
        );
        let names: Vec<Box<str>> = vec!["cell_centroid_x".into(), "cell_centroid_y".into()];
        let r = read_one_coord_file(f.path().to_str().unwrap(), &[], &names, None).unwrap();
        assert_eq!(r.mat.ncols(), 2);
        assert_eq!(r.mat.nrows(), 2);
        assert_eq!(r.cols[0].as_ref(), "cell_centroid_x");
        assert_eq!(r.cols[1].as_ref(), "cell_centroid_y");
        assert_eq!(r.rows[0].as_ref(), "cell_a");
    }

    #[test]
    fn unresolved_coordinate_columns_are_reported_with_the_file_s_own_fields() {
        // Reading by position is a guess about the file, and a wrong guess does
        // not fail: any two numeric columns pass as coordinates. So nothing is
        // guessed. The error quotes the first line back, because the caller
        // cannot fix this without knowing what the file actually holds.
        let f = write_csv(
            "barcode,in_tissue,array_row,array_col,my_x,my_y\n\
             cell_a,1,0,0,100.5,200.5\n\
             cell_b,1,1,1,300.5,400.5\n",
        );
        let names: Vec<Box<str>> = vec!["pxl_row_in_fullres".into(), "pxl_col_in_fullres".into()];
        let err = match read_one_coord_file(f.path().to_str().unwrap(), &[], &names, None) {
            Ok(_) => panic!("columns that resolve to nothing must not be guessed at"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("my_x"),
            "the file's own fields guide the fix: {err}"
        );
        assert!(
            err.contains("--coord-column-indices"),
            "and say how to say it: {err}"
        );
    }

    #[test]
    fn a_headerless_positional_file_is_read_when_the_caller_says_which_columns() {
        // The classic spot layout has no header at all, so there are no names
        // to match. Naming the positions is the whole answer.
        let f = write_csv(
            "cell_a,1,0,0,100.5,200.5\n\
             cell_b,1,1,1,300.5,400.5\n",
        );
        let names: Vec<Box<str>> = vec!["pxl_row_in_fullres".into()];
        let r = read_one_coord_file(f.path().to_str().unwrap(), &[4, 5], &names, None).unwrap();
        assert_eq!(r.mat.ncols(), 2);
        assert_eq!(r.mat[(0, 0)], 100.5);
        assert_eq!(r.mat[(0, 1)], 200.5);
        assert_eq!(r.mat[(1, 0)], 300.5);
        assert_eq!(r.rows[0].as_ref(), "cell_a");
    }

    #[test]
    fn test_read_one_coord_file_user_indices_skip_fallback() {
        let f = write_csv(
            "barcode,foo,bar\n\
             cell_a,10.0,20.0\n",
        );
        let names: Vec<Box<str>> = vec!["nonexistent".into()];
        let r = read_one_coord_file(f.path().to_str().unwrap(), &[1, 2], &names, None).unwrap();
        assert_eq!(r.mat.ncols(), 2);
        // Columns came from indices, so names are "foo","bar" (from header), not "x","y".
        assert_eq!(r.cols[0].as_ref(), "foo");
        assert_eq!(r.cols[1].as_ref(), "bar");
    }

    /// A parquet of two cells, a row-name column then `cols`.
    fn write_coord_parquet(dir: &tempfile::TempDir, cols: &[&str]) -> String {
        use legume_numeric::matrix::traits::IoOps;
        let path = dir.path().join("c.parquet").to_string_lossy().into_owned();
        let rows: Vec<Box<str>> = vec!["cell_a".into(), "cell_b".into()];
        let cols: Vec<Box<str>> = cols.iter().map(|&c| c.into()).collect();
        let m = Mat::from_fn(2, cols.len(), |i, j| (i * 10 + j) as f32);
        m.to_parquet_with_names(&path, (Some(&rows), Some("cell")), Some(&cols))
            .unwrap();
        path
    }

    #[test]
    fn parquet_coordinates_named_but_not_there_are_refused() {
        // Names that match none of the file's columns once read every
        // column back, and the first two passed as coordinates unseen.
        let dir = tempfile::tempdir().unwrap();
        let path = write_coord_parquet(&dir, &["a", "b"]);
        let names: Vec<Box<str>> = vec!["nope_x".into(), "nope_y".into()];
        let err = match read_one_coord_file(&path, &[], &names, None) {
            Ok(r) => panic!("read {:?} for {names:?}", r.cols),
            Err(e) => format!("{e:#}"),
        };
        assert!(
            err.contains("nope_x") && err.contains("coordinate columns"),
            "{err}"
        );
        // Of several conventions named, the one the file has is read.
        let names: Vec<Box<str>> = vec!["x".into(), "y".into(), "a".into(), "b".into()];
        let r = read_one_coord_file(&path, &[], &names, None).unwrap();
        assert_eq!(r.cols, vec![Box::from("a"), Box::from("b")]);
    }

    #[test]
    fn a_quoted_header_is_read_by_its_names() {
        let f = write_csv(
            "\"barcode\",\"x_centroid\",\"y_centroid\"\n\
             \"cell_a\",1.0,2.0\n",
        );
        let names: Vec<Box<str>> = vec!["x_centroid".into(), "y_centroid".into()];
        let r = read_one_coord_file(f.path().to_str().unwrap(), &[], &names, None).unwrap();
        assert_eq!(r.mat.ncols(), 2);
        assert_eq!(r.mat[(0, 1)], 2.0);
    }

    #[test]
    fn a_missing_coordinate_file_is_called_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("none.parquet");
        let path = path.to_str().unwrap();
        let names: Vec<Box<str>> = vec!["x".into(), "y".into()];
        let Err(err) = read_one_coord_file(path, &[], &names, None) else {
            panic!("read a file that is not there");
        };
        let err = format!("{err:#}");
        assert!(
            err.contains(path) && err.contains("does not exist"),
            "{err}"
        );
        assert!(!err.contains("--coord-column-indices"), "{err}");
    }

    fn xenium_zarr_path() -> Option<std::path::PathBuf> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("docs/temp/cells.zarr.zip");
        p.exists().then_some(p)
    }

    #[test]
    fn zarr_coordinates_that_match_no_requested_name_are_reported() {
        // This used to assert a fallback to indices [0, 1]. There is no
        // fallback now, so it asserts the replacement contract. It is skipped
        // without the fixture, which is why the old version stayed green for
        // as long as it did after the behaviour under it changed.
        let Some(p) = xenium_zarr_path() else { return };
        let names: Vec<Box<str>> = vec!["pxl_row_in_fullres".into(), "pxl_col_in_fullres".into()];
        match read_one_coord_file(p.to_str().unwrap(), &[], &names, None) {
            Ok(_) => panic!("names that match nothing must not be guessed at"),
            Err(e) => {
                let m = e.to_string();
                assert!(m.contains("coordinate columns"), "{m}");
            }
        }
    }

    #[test]
    fn test_read_one_coord_file_zarr_match_no_fallback() {
        let Some(p) = xenium_zarr_path() else { return };
        let names: Vec<Box<str>> = vec!["cell_centroid_x".into(), "cell_centroid_y".into()];
        let r = read_one_coord_file(p.to_str().unwrap(), &[], &names, None).unwrap();
        assert_eq!(r.mat.ncols(), 2);
        assert_eq!(r.cols[0].as_ref(), "cell_centroid_x");
        assert_eq!(r.cols[1].as_ref(), "cell_centroid_y");
    }
}

mod row_alignment {
    use super::super::*;
    use std::io::Write as _;

    /// Each row's value in cell 1 of every file, so a value names its row.
    fn value(row: &str) -> f32 {
        ["GENE1", "GENE2", "GENE3", "GENE4"]
            .iter()
            .position(|&r| r == row)
            .unwrap() as f32
            + 1.
    }

    /// A file of `rows` by 2 cells, cell 1 holding each row's [`value`].
    fn write_data(dir: &tempfile::TempDir, tag: &str, rows: &[&str]) -> Box<str> {
        let triplets: Vec<(u64, u64, f32)> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| (i as u64, 1, value(r)))
            .collect();
        crate::test_support::write_zarr(dir.path(), tag, rows, 2, &triplets)
    }

    fn write_coord(dir: &tempfile::TempDir, tag: &str) -> Box<str> {
        let path = dir.path().join(format!("{tag}.csv"));
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "barcode,x,y\n{tag}_0,0,0\n{tag}_1,1,1").unwrap();
        path.to_str().unwrap().into()
    }

    fn args(data_files: Vec<Box<str>>, coord_files: Vec<Box<str>>) -> SRTReadArgs {
        SRTReadArgs {
            data_files,
            coord_files,
            preload_data: false,
            coord_columns: vec![],
            coord_column_names: vec!["x".into(), "y".into()],
            batch_files: None,
            header_in_coord: None,
            feature_kind: FeatureNameKind::Exact,
        }
    }

    /// Samples from different chemistries list different genes, in different
    /// orders. They are read on the genes all of them measured, each value
    /// under its own name; a gene some sample never measured is not a zero.
    fn assert_shared_rows(data: &SparseIoVec) {
        let rows = data.row_names().unwrap();
        let mut sorted = rows.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            vec!["GENE1".into(), "GENE3".into()] as Vec<Box<str>>
        );
        let x = data.read_columns_csc([1usize, 3].into_iter()).unwrap();
        for (r, name) in rows.iter().enumerate() {
            let want = value(name);
            for c in 0..2 {
                let col = x.col(c);
                let got = col
                    .row_indices()
                    .iter()
                    .position(|&i| i == r)
                    .map(|k| col.values()[k]);
                assert_eq!(got, Some(want), "{name} in column {c}");
            }
        }
    }

    #[test]
    fn files_with_different_rows_are_read_on_the_rows_they_share() {
        let dir = tempfile::tempdir().unwrap();
        let data = vec![
            write_data(&dir, "a", &["GENE1", "GENE2", "GENE3"]),
            write_data(&dir, "b", &["GENE3", "GENE1", "GENE4"]),
        ];
        let coord = vec![write_coord(&dir, "a"), write_coord(&dir, "b")];

        let with = read_data_with_coordinates(args(data.clone(), coord)).unwrap();
        assert_shared_rows(&with.data);

        let without = read_data_without_coordinates(args(data, vec![])).unwrap();
        assert_shared_rows(&without.data);
    }

    #[test]
    fn files_sharing_no_row_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let data = vec![
            write_data(&dir, "a", &["GENE1"]),
            write_data(&dir, "b", &["GENE2"]),
        ];
        let coord = vec![write_coord(&dir, "a"), write_coord(&dir, "b")];
        assert!(read_data_with_coordinates(args(data.clone(), coord)).is_err());
        assert!(read_data_without_coordinates(args(data, vec![])).is_err());
    }
}

mod separation {
    use super::super::*;

    #[test]
    fn files_under_one_label_get_their_own_offsets() {
        let one: Vec<Box<str>> = vec!["all".into(); 4];
        let keys = crate::util::ablation::sections(&[2, 2], &one);
        let out = append_batch_coordinate(&Mat::from_vec(4, 2, vec![0.0; 8]), &keys).unwrap();
        assert_eq!(out[(0, 2)], out[(1, 2)]);
        assert_ne!(out[(0, 2)], out[(2, 2)]);
    }
}

/// Two `zarr` backends of one gene, cells `a` and `b`, at `names` under
/// `dir`.
fn backends(dir: &std::path::Path, names: &[&str]) -> Vec<Box<str>> {
    let genes: Vec<Box<str>> = vec!["g0".into()];
    let bc: Vec<Box<str>> = vec!["a".into(), "b".into()];
    names
        .iter()
        .map(|n| {
            let f = dir.join(n);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            let f = f.to_string_lossy().into_owned();
            crate::simulate::write::write_backend(&f, &genes, &bc, vec![(0, 0, 1.0), (0, 1, 2.0)])
                .unwrap();
            f.into()
        })
        .collect()
}

/// A fit names its cells as data-beans' own shared-row reader does (the
/// one impute reads a reference with), with one file and with several, so
/// training and reference agree on every name.
#[test]
fn fits_name_cells_as_data_beans_reads_them() {
    use data_beans::aux::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
    use data_beans::aux::feature_names::FeatureNameKind;
    let dir = tempfile::tempdir().unwrap();
    let files = backends(dir.path(), &["S1.zarr", "S2.zarr"]);
    for n in [1, 2] {
        let fit = super::open_fit_data(&files[..n], &FeatureNameKind::Exact, false).unwrap();
        let reference = read_data_on_shared_rows(ReadSharedRowsArgs {
            data_files: files[..n].to_vec(),
            feature_kind: Some(FeatureNameKind::Exact),
            ..Default::default()
        })
        .unwrap()
        .data;
        assert_eq!(
            fit.column_names().unwrap(),
            reference.column_names().unwrap(),
            "{n} files"
        );
    }
}

/// Data files that would give their cells the same name are refused: two
/// files with one stem would name two cells `a@S1`.
#[test]
fn files_with_one_name_are_refused() {
    use data_beans::aux::feature_names::FeatureNameKind;
    let dir = tempfile::tempdir().unwrap();
    let files = backends(dir.path(), &["x/S1.zarr", "y/S1.zarr"]);
    let e = super::open_fit_data(&files, &FeatureNameKind::Exact, false)
        .err()
        .expect("refused")
        .to_string();
    assert!(e.contains("same name"), "{e}");
    let e = super::read_expr_data(&files)
        .err()
        .expect("refused")
        .to_string();
    assert!(e.contains("same name"), "{e}");
}
