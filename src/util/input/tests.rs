//! Tests of reading inputs and coordinates.

mod reading {
    use super::super::*;

    #[test]
    fn test_append_batch_coordinate_single_batch() {
        let coords = Mat::from_vec(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let batches = vec!["A", "A", "A"];
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
        let batches = vec!["A", "A", "B", "B"];
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
        let batches = vec!["X", "Y"];
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
        let batches = vec!["A"]; // wrong length
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
