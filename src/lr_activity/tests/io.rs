//! Reader tolerance for the two coord_pairs batch schemas: the current
//! writer emits string labels, but tables from older runs carry the numeric
//! batch pseudo-coordinate under the same column names. Those files do not
//! vanish, so the reader must load either, stringifying numerics.

use crate::lr_activity::io::{attach_batch_from_coord_pairs, EdgeRecord};
use legume_numeric::matrix::parquet::{write_named_table, Column};

fn records(n: usize) -> Vec<EdgeRecord> {
    (0..n)
        .map(|i| EdgeRecord {
            left_cell: format!("cell{i}").into(),
            right_cell: format!("cell{}", i + 1).into(),
            community: 0,
            batch: None,
            side_batches: None,
            is_spatial: true,
        })
        .collect()
}

/// A legacy table's `left_batch` / `right_batch` are FLOAT columns holding
/// the batch pseudo-coordinate. They still read: integral floats become
/// their integer string, and a straddling edge still resolves to None.
#[test]
fn a_legacy_numeric_batch_table_still_reads() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("legacy.coord_pairs.parquet");
    let path = path.to_str().unwrap();

    let left_cells: Vec<Box<str>> = vec!["cell0".into(), "cell1".into(), "cell2".into()];
    let right_cells: Vec<Box<str>> = vec!["cell1".into(), "cell2".into(), "cell3".into()];
    let left_b = [0.0f32, 0.0, 1000.0];
    let right_b = [0.0f32, 1000.0, 1000.0];
    let row_names: Vec<Box<str>> = (0..3).map(|i| i.to_string().into()).collect();
    write_named_table(
        path,
        "cell_pair",
        &row_names,
        &[
            ("left_cell".into(), Column::Str(&left_cells)),
            ("right_cell".into(), Column::Str(&right_cells)),
            ("left_batch".into(), Column::F32(&left_b)),
            ("right_batch".into(), Column::F32(&right_b)),
        ],
    )?;

    let mut edges = records(3);
    attach_batch_from_coord_pairs(&mut edges, path)?;
    assert_eq!(edges[0].batch.as_deref(), Some("0"));
    assert_eq!(edges[1].batch, None, "straddling edge belongs to no batch");
    assert_eq!(edges[2].batch.as_deref(), Some("1000"));
    Ok(())
}

/// The current schema's string labels pass through verbatim.
#[test]
fn string_batch_labels_pass_through_verbatim() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("current.coord_pairs.parquet");
    let path = path.to_str().unwrap();

    let left_cells: Vec<Box<str>> = vec!["cell0".into(), "cell1".into()];
    let right_cells: Vec<Box<str>> = vec!["cell1".into(), "cell2".into()];
    let left_b: Vec<Box<str>> = vec!["coreA".into(), "coreA".into()];
    let right_b: Vec<Box<str>> = vec!["coreA".into(), "coreB".into()];
    let row_names: Vec<Box<str>> = (0..2).map(|i| i.to_string().into()).collect();
    write_named_table(
        path,
        "cell_pair",
        &row_names,
        &[
            ("left_cell".into(), Column::Str(&left_cells)),
            ("right_cell".into(), Column::Str(&right_cells)),
            ("left_batch".into(), Column::Str(&left_b)),
            ("right_batch".into(), Column::Str(&right_b)),
        ],
    )?;

    let mut edges = records(2);
    attach_batch_from_coord_pairs(&mut edges, path)?;
    assert_eq!(edges[0].batch.as_deref(), Some("coreA"));
    assert_eq!(edges[1].batch, None);
    Ok(())
}

/// A pair table reads whatever width its writer chose for each column: an
/// integer `community`, an `edge_kind` that marks an expression pair, and
/// per-end batch labels, the shared batch only where the two ends agree.
#[test]
fn a_pair_table_reads_its_columns_at_any_width() -> anyhow::Result<()> {
    use crate::lr_activity::io::read_pair_table;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("p.link_community.parquet");
    let path = path.to_str().unwrap();
    let left: Vec<Box<str>> = vec!["cell0".into(), "cell1".into(), "cell2".into()];
    let right: Vec<Box<str>> = vec!["cell1".into(), "cell2".into(), "cell3".into()];
    let community = [2i64, 0, 1];
    let kind = [
        crate::util::cell_pairs::EDGE_KIND_SPATIAL,
        crate::util::cell_pairs::EDGE_KIND_EXPRESSION,
        crate::util::cell_pairs::EDGE_KIND_SPATIAL,
    ];
    let left_b: Vec<Box<str>> = vec!["B1".into(), "B1".into(), "B2".into()];
    let right_b: Vec<Box<str>> = vec!["B1".into(), "B2".into(), "B2".into()];
    let row_names: Vec<Box<str>> = (0..3).map(|i| i.to_string().into()).collect();
    write_named_table(
        path,
        "cell_pair",
        &row_names,
        &[
            ("left_cell".into(), Column::Str(&left)),
            ("right_cell".into(), Column::Str(&right)),
            ("community".into(), Column::I64(&community)),
            ("edge_kind".into(), Column::I32(&kind)),
            ("left_batch".into(), Column::Str(&left_b)),
            ("right_batch".into(), Column::Str(&right_b)),
        ],
    )?;

    let edges = read_pair_table(path, true)?;
    let got: Vec<_> = edges
        .iter()
        .map(|e| {
            (
                e.left_cell.as_ref(),
                e.right_cell.as_ref(),
                e.community,
                e.is_spatial,
                e.batch.as_deref(),
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("cell0", "cell1", 2, true, Some("B1")),
            ("cell1", "cell2", 0, false, None),
            ("cell2", "cell3", 1, true, Some("B2")),
        ]
    );
    assert_eq!(edges[1].side_batches, Some(("B1".into(), "B2".into())));
    // Read as coord_pairs, there is no community: every pair is in 0.
    assert!(read_pair_table(path, false)?
        .iter()
        .all(|e| e.community == 0));
    Ok(())
}

/// A batch named as lra's own rows are (cells of no batch, the pooled
/// rows) is refused as its table is read, by either reader, saying which.
#[test]
fn a_batch_named_as_lras_own_rows_is_refused() -> anyhow::Result<()> {
    use crate::lr_activity::fit::{BATCH_LABEL_ALL, BATCH_LABEL_META};
    let dir = tempfile::tempdir()?;
    for reserved in [BATCH_LABEL_ALL, BATCH_LABEL_META] {
        let path = dir.path().join(format!("{reserved}.coord_pairs.parquet"));
        let path = path.to_str().unwrap();
        let left_cells: Vec<Box<str>> = vec!["cell0".into()];
        let right_cells: Vec<Box<str>> = vec!["cell1".into()];
        let left_b: Vec<Box<str>> = vec!["s1".into()];
        let right_b: Vec<Box<str>> = vec![reserved.into()];
        write_named_table(
            path,
            "cell_pair",
            &["0".into()],
            &[
                ("left_cell".into(), Column::Str(&left_cells)),
                ("right_cell".into(), Column::Str(&right_cells)),
                ("left_batch".into(), Column::Str(&left_b)),
                ("right_batch".into(), Column::Str(&right_b)),
            ],
        )?;
        let e = attach_batch_from_coord_pairs(&mut records(1), path).unwrap_err();
        assert!(e.to_string().contains(&format!("batch {reserved}")), "{e}");
        let e = crate::lr_activity::io::read_pair_table(path, false)
            .err()
            .expect("refused");
        assert!(e.to_string().contains(&format!("batch {reserved}")), "{e}");
    }
    Ok(())
}
