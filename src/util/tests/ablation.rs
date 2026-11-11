use crate::util::ablation::*;
use crate::util::common::*;
use crate::util::knn_graph::KnnGraph;
use legume_numeric::matrix::knn_graph::symmetric_adjacency;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|&s| s.into()).collect()
}

fn args() -> AblationArgs {
    AblationArgs {
        permute_block: 8,
        ..Default::default()
    }
}

/// The flags as a fit command parses them: its input args and its ablation.
fn parse(extra: &[&str]) -> Result<AblationArgs, clap::Error> {
    use clap::Parser;
    #[derive(Parser)]
    struct Fit {
        #[command(flatten)]
        common: crate::util::input::SrtInputArgs,
        #[command(flatten)]
        ablation: AblationArgs,
    }
    let mut argv = vec!["fit", "d.zarr", "-o", "o"];
    argv.extend_from_slice(extra);
    Fit::try_parse_from(argv).map(|f| f.ablation)
}

#[test]
fn one_ablation_per_run_and_a_parent_only_with_one() {
    assert!(parse(&["--drop-features", "^MT-", "--mask-cells", "c.txt"]).is_err());
    assert!(parse(&["--ablation-parent", "base"]).is_err());
    let a = parse(&["--drop-features", "^MT-", "--ablation-parent", "base"]).unwrap();
    assert_eq!(a.kind(), Some(AblationKind::FeatureDrop));
    assert_eq!(parse(&[]).unwrap().kind(), None);
}

#[test]
fn permuting_positions_needs_coordinates() {
    assert!(parse(&["--permute-coords", "block"]).is_err());
    let a = parse(&["--permute-coords", "block", "--coord", "c.csv"]).unwrap();
    assert_eq!(a.kind(), Some(AblationKind::CoordPermutation));
    a.validate(true).unwrap();
    let mut a = a;
    a.permute_block = 1;
    assert!(a.validate(true).is_err());
}

#[test]
fn a_parent_needs_a_manifest() {
    let mut a = args();
    a.drop_features = Some("^MT-".into());
    a.ablation_parent = Some("nowhere".into());
    let e = a.validate(true).unwrap_err().to_string();
    assert!(e.contains("no manifest nowhere.pinto.json"), "{e}");

    let dir = tempfile::tempdir().unwrap();
    let run = dir.path().join("base");
    std::fs::write(dir.path().join("base.pinto.json"), "{}").unwrap();
    a.ablation_parent = Some(run.to_str().unwrap().into());
    a.validate(true).unwrap();
    assert!(parent_manifest(run.to_str().unwrap())
        .unwrap()
        .ends_with("base.pinto.json"));
}

#[test]
fn features_drop_by_regex_on_name_parts() {
    let rows = names(&[
        "ENSG1_MT-CO1",
        "ENSG2_ACTB",
        "MT-ND1",
        "ENSG3_GAPDH/count/spliced",
    ]);
    let (keep, spec) = feature_keep(&rows, "^MT-").unwrap();
    assert_eq!(keep, vec![false, true, false, true]);
    assert_eq!(spec["n_dropped"], 2);
    let (keep, _) = feature_keep(&rows, "^GAPDH$").unwrap();
    assert_eq!(keep, vec![true, true, true, false]);
    assert!(feature_keep(&rows, "^NOPE$").is_err());
    assert!(feature_keep(&rows, ".").is_err());
}

#[test]
fn features_drop_by_listed_name() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("drop.txt");
    std::fs::write(&f, "# housekeeping\nACTB\n\nENSG3\n").unwrap();
    let rows = names(&["ENSG1_MT-CO1", "ENSG2_ACTB", "ENSG3_GAPDH"]);
    let (keep, spec) = feature_keep(&rows, f.to_str().unwrap()).unwrap();
    assert_eq!(keep, vec![true, false, false]);
    assert_eq!(spec["n_listed"], 2);
}

#[test]
fn a_cell_list_takes_the_first_field_and_skips_a_header() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("cells.tsv");
    std::fs::write(
        &f,
        "cell\tregion\nA-1@s1\tnecrosis\nB-1@s1\tedge\n\n# note\n",
    )
    .unwrap();
    let listed = read_cell_list(f.to_str().unwrap()).unwrap();
    assert_eq!(listed.len(), 2);
    let keep = cell_keep(&names(&["A-1@s1", "C-1@s1", "B-1@s1"]), &listed).unwrap();
    assert_eq!(keep, vec![false, true, false]);
    let none: HashSet<Box<str>> = ["Z".into()].into_iter().collect();
    assert!(cell_keep(&names(&["A-1@s1"]), &none).is_err());
}

#[test]
fn sections_split_files_and_batches() {
    let b = names(&["x", "x", "y", "x", "x"]);
    assert_eq!(sections(&[3, 2], &b), vec![0, 0, 1, 2, 2]);
}

#[test]
fn a_within_section_shuffle_keeps_each_sections_positions() {
    let coords = Mat::from_fn(10, 2, |i, j| (i * 10 + j) as f32);
    let sec: Vec<u32> = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let moved = permute_within(&coords, &sec, 7);
    let rows = |m: &Mat, r: std::ops::Range<usize>| {
        let mut v: Vec<(i64, i64)> = r.map(|i| (m[(i, 0)] as i64, m[(i, 1)] as i64)).collect();
        v.sort_unstable();
        v
    };
    assert_eq!(rows(&moved, 0..5), rows(&coords, 0..5));
    assert_eq!(rows(&moved, 5..10), rows(&coords, 5..10));
    assert_ne!(moved, coords);
    assert_eq!(moved, permute_within(&coords, &sec, 7), "seeded");
    assert_ne!(moved, permute_within(&coords, &sec, 8));
}

#[test]
fn block_moves_keep_offsets_inside_a_tile() {
    // A 4×4 grid of tiles of side 1, one cell per tile at (tx + 0.25, ty + 0.5),
    // with corners at 0 and 4 so the section spans 4 by 4.
    let mut pts: Vec<(f32, f32)> = (0..16)
        .map(|t| ((t % 4) as f32 + 0.25, (t / 4) as f32 + 0.5))
        .collect();
    pts.push((0.0, 0.0));
    pts.push((4.0, 4.0));
    let n = pts.len();
    let coords = Mat::from_fn(n, 3, |i, j| match j {
        0 => pts[i].0,
        1 => pts[i].1,
        _ => 99.0,
    });
    let sec = vec![0u32; n];
    let (moved, n_tiles) = permute_blocks(&coords, (0, 1), &sec, 4, 3);
    assert!((15..=16).contains(&n_tiles));
    for i in 0..16 {
        let (x, y) = (moved[(i, 0)], moved[(i, 1)]);
        assert!((x - x.floor() - 0.25).abs() < 1e-5, "{x}");
        assert!((y - y.floor() - 0.5).abs() < 1e-5, "{y}");
        assert_eq!(moved[(i, 2)], 99.0, "the batch column stays");
    }
    assert_ne!(moved, coords);
}

#[test]
fn plane_columns_skip_the_batch_offset() {
    assert_eq!(plane_columns(&names(&["batch", "x", "y"])).unwrap(), (1, 2));
    assert!(plane_columns(&names(&["x", "batch"])).is_err());
}

#[test]
fn the_induced_subgraph_drops_edges_touching_a_masked_node() {
    let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3)];
    let distances = vec![1.0, 2.0, 3.0, 4.0];
    let g = KnnGraph {
        adjacency: symmetric_adjacency(4, &edges, &distances),
        edges,
        distances,
        n_nodes: 4,
    };
    let sub = induced_subgraph(&g, &[true, false, true, true]);
    assert_eq!(sub.n_nodes, 3);
    assert_eq!(sub.edges, vec![(0, 1), (1, 2)]);
    assert_eq!(sub.distances, vec![2.0, 4.0]);
    assert_eq!(sub.neighbors(1), &[0, 2]);
}

#[test]
fn the_record_carries_parent_and_seed() {
    let mut a = args();
    a.ablation_parent = Some("res/base".into());
    let r = a.record(
        AblationKind::CoordPermutation,
        serde_json::json!({"mode": "block"}),
        Some(a.seed(42)),
    );
    assert_eq!(
        r.parent.as_deref(),
        Some("res/base.pinto.json"),
        "named by its manifest"
    );
    assert_eq!(r.seed, Some(42));
    let v = serde_json::to_value(&r).unwrap();
    assert_eq!(v["kind"], "coord_permutation");
    let r = a.record(AblationKind::FeatureDrop, serde_json::json!({}), None);
    assert!(serde_json::to_value(&r).unwrap().get("seed").is_none());
}

#[test]
fn geojson_features_carry_their_section() {
    let v: serde_json::Value = serde_json::from_str(
        r#"{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"section":"S1"},
             "geometry":{"type":"Polygon","coordinates":[[[0,0],[2,0],[2,2],[0,2],[0,0]]]}},
            {"type":"Feature","properties":{"sample":"s2","name":"Tumor"},
             "geometry":{"type":"MultiPolygon","coordinates":[[[[5,5],[6,5],[6,6]]],[[[7,7],[8,7],[8,8]]]]}}
        ]}"#,
    )
    .unwrap();
    let r = parse_regions(&v).unwrap();
    assert_eq!(r.len(), 3);
    assert_eq!(r[0].section.as_deref(), Some("S1"));
    assert_eq!(
        r[2].section.as_deref(),
        Some("s2"),
        "a QuPath label in `name` is not a section"
    );
}

#[test]
fn a_hole_takes_its_area_back() {
    let outer = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
    let hole = vec![(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)];
    let rings = vec![outer, hole];
    assert!(inside(&rings, 1.0, 1.0));
    assert!(!inside(&rings, 5.0, 5.0));
    assert!(!inside(&rings, 11.0, 5.0));
}

#[test]
fn polygons_pick_cells_of_their_own_section() {
    let names = names(&["a@GSM1_s1", "b@GSM1_s1", "c@GSM2_s2"]);
    // All three sit at the same point in their own frames.
    let (x, y) = (vec![1.0f32; 3], vec![1.0f32; 3]);
    let square = vec![vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]];
    let region = |s: Option<&str>| Region {
        section: s.map(str::to_string),
        rings: square.clone(),
    };
    let (picked, counts) = cells_in_regions(&names, &x, &y, &[region(Some("s1"))]).unwrap();
    assert_eq!(counts, vec![2]);
    assert!(picked.contains("a@GSM1_s1") && !picked.contains("c@GSM2_s2"));
    let e = cells_in_regions(&names, &x, &y, &[region(None)]).unwrap_err();
    assert!(e.to_string().contains("section"), "{e}");
    assert!(cells_in_regions(&names, &x, &y, &[region(Some("s9"))]).is_err());
    // One section: a polygon without one applies to it.
    let one = names[..2].to_vec();
    assert_eq!(
        cells_in_regions(&one, &x[..2], &y[..2], &[region(None)])
            .unwrap()
            .1,
        vec![2]
    );
}

#[test]
fn seeding_sees_the_plane_per_section_and_batch() {
    // Columns x, the batch offset and y; two sections, the second split
    // into two batches (as `--auto-batch` components would be).
    let coords = Mat::from_fn(4, 3, |i, j| (10 * i + j) as f32);
    let batches: Vec<Box<str>> = ["a", "a", "b", "c"].iter().map(|&s| s.into()).collect();
    let (plane, groups) = seeding_inputs(
        &coords,
        &names(&["x", "batch", "y"]),
        &[0, 0, 1, 1],
        &batches,
    );
    assert_eq!(plane.ncols(), 2);
    for i in 0..4 {
        assert_eq!(
            (plane[(i, 0)], plane[(i, 1)]),
            (coords[(i, 0)], coords[(i, 2)])
        );
    }
    assert_eq!(groups, [0, 0, 1, 2]);
}
