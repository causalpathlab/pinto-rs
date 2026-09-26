use super::fixture::synth_cells;
use crate::util::common::*;
use crate::view::data::{Communities, Edges, Geometry, Run, NO_CLUSTER};

#[test]
fn single_batch_keeps_coordinates() {
    let cells = synth_cells(100, 1);
    let before = cells.coords.clone();
    let geom = Geometry::from_cells(cells);
    assert_eq!(geom.tiles.len(), 1);
    for (i, &(x, y)) in before.iter().enumerate() {
        assert_eq!((geom.x[i], geom.y[i]), (x, y));
    }
}

#[test]
fn batches_are_tiled_without_overlap() {
    let geom = Geometry::from_cells(synth_cells(1000, 4));
    assert_eq!(geom.tiles.len(), 4);
    for (a, ta) in geom.tiles.iter().enumerate() {
        assert_eq!(ta.n_cells, 250);
        for tb in &geom.tiles[a + 1..] {
            let disjoint = ta.bounds.x1 < tb.bounds.x0
                || tb.bounds.x1 < ta.bounds.x0
                || ta.bounds.y1 < tb.bounds.y0
                || tb.bounds.y1 < ta.bounds.y0;
            assert!(disjoint, "{} overlaps {}", ta.name, tb.name);
        }
    }
    for i in 0..geom.n() {
        let t = &geom.tiles[geom.batch[i] as usize].bounds;
        assert!(t.x0 <= geom.x[i] && geom.x[i] <= t.x1);
        assert!(t.y0 <= geom.y[i] && geom.y[i] <= t.y1);
    }
}

#[test]
fn propensity_joins_by_name_and_reports_gaps() {
    let geom = Geometry::from_cells(synth_cells(10, 1));
    let k = 3;
    // Rows in reverse order, one unknown cell, cell c0 missing.
    let names: Vec<Box<str>> = (1..10)
        .rev()
        .map(|i| format!("c{i}").into_boxed_str())
        .chain(std::iter::once("ghost".into()))
        .collect();
    let mut prop = Mat::zeros(names.len(), k);
    let mut cluster = vec![];
    for (r, name) in names.iter().enumerate() {
        let c = name[1..].parse::<usize>().map_or(0, |i| i % k);
        prop[(r, c)] = 1.;
        cluster.push(c as i64);
    }
    let comm = Communities::join(&geom, "L1", (prop, cluster, None, names));

    assert_eq!(comm.n_missing, 1);
    assert_eq!(comm.n_unmatched, 1);
    assert_eq!(comm.cluster[0], NO_CLUSTER);
    for i in 1..10 {
        let c = i % k;
        assert_eq!(comm.cluster[i], c as u16);
        assert_eq!(comm.prop[i * k + c], 255);
    }
    assert!(comm.entropy.is_none());
}

#[test]
fn edges_map_names_to_rows() {
    let geom = Geometry::from_cells(synth_cells(4, 1));
    let pairs: Vec<(Box<str>, Box<str>)> = vec![
        ("c0".into(), "c1".into()),
        ("c2".into(), "ghost".into()),
        ("c3".into(), "c2".into()),
    ];
    let edges = Edges::join(&geom, &pairs, &[4, 5, -1]);
    assert_eq!(edges.len(), 2);
    assert_eq!(edges.n_unmatched, 1);
    assert_eq!((edges.a[0], edges.b[0], edges.community[0]), (0, 1, 4));
    assert_eq!(
        (edges.a[1], edges.b[1], edges.community[1]),
        (3, 2, NO_CLUSTER)
    );
}

#[test]
fn a_run_without_a_manifest_is_found_by_file_name() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    for name in [
        "coord_pairs",
        "L2.propensity",
        "L2.link_community",
        "L2.gene_topic",
        "L10.propensity",
        "propensity",
        "link_community",
        "feature_community",
        // Neither a level nor the tail: ignored.
        "draft.propensity",
    ] {
        std::fs::write(format!("{prefix}.{name}.parquet"), b"").unwrap();
    }

    let run = Run::open(&prefix).unwrap();
    assert!(run.manifest.is_none());
    let tags: Vec<&str> = run.levels.iter().map(|l| l.tag.as_str()).collect();
    assert_eq!(tags, ["L2", "L10", "final"]);
    let l2 = &run.levels[0];
    assert!(l2
        .feature_community
        .as_deref()
        .unwrap()
        .ends_with("run.L2.gene_topic.parquet"));
    assert!(run.levels[1].link_community.is_none());
    assert!(run.meta.outputs.coord_pairs.is_some());
}

#[test]
fn a_prefix_with_neither_manifest_nor_coord_pairs_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("missing").to_string_lossy().to_string();
    let err = Run::open(&prefix).err().unwrap().to_string();
    assert!(err.contains("coord_pairs"), "{err}");
}
