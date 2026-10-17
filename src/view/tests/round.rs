//! A lupin round's per-cell labels, from its parquet or its TSV.

use super::fixture::synth_cells;
use crate::view::data::{Geometry, NO_CLUSTER};
use crate::view::round::cell_types;

#[test]
fn a_rounds_labels_read_alike_from_parquet_and_tsv() {
    use legume_numeric::matrix::parquet::{write_named_table, Column};
    let dir = tempfile::tempdir().unwrap();
    let geom = Geometry::from_cells(synth_cells(3, 1));
    let tsv = dir.path().join("r.argmax.tsv");
    std::fs::write(
        &tsv,
        "cell\tcell_type\tprobability\nc0\tT\t0.9\nc1\tunassigned\tNaN\nc2\tB\t0.8\n",
    )
    .unwrap();
    let parquet = dir.path().join("r.argmax.parquet");
    let types: Vec<Box<str>> = vec!["T".into(), "unassigned".into(), "B".into()];
    write_named_table(
        parquet.to_str().unwrap(),
        "cell",
        &["c0".into(), "c1".into(), "c2".into()],
        &[
            ("cell_type".into(), Column::Str(&types)),
            ("probability".into(), Column::F32(&[0.9, f32::NAN, 0.8])),
        ],
    )
    .unwrap();
    let from_tsv = cell_types(&tsv, &geom).unwrap();
    let from_parquet = cell_types(&parquet, &geom).unwrap();
    assert_eq!(from_parquet, from_tsv);
    let (group, names) = from_parquet;
    assert_eq!(
        names,
        ["B".into(), "T".into()] as [Box<str>; 2],
        "alphabetical"
    );
    assert_eq!(group, [1, NO_CLUSTER, 0], "unassigned is ungrouped");
}
