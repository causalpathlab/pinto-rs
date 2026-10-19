use super::fixture::synth_cells;
use crate::util::common::*;
use crate::view::data::{Communities, Geometry};
use crate::view::gene::*;
use crate::view::index::Grid;
use crate::view::markers::FeatureRates;

#[test]
fn features_are_found_by_name_or_symbol() {
    let names: Vec<Box<str>> = ["ENSG00000198851_CD3E", "CD45RA", "ENSG00000116824_CD2"]
        .map(Box::from)
        .to_vec();
    assert_eq!(find(&names, "ENSG00000198851_CD3E"), Some(0));
    assert_eq!(find(&names, "CD3E"), Some(0));
    assert_eq!(find(&names, "CD45RA"), Some(1));
    assert_eq!(find(&names, "CD2"), Some(2));
    assert_eq!(find(&names, "CD8A"), None);
}

#[test]
fn expected_levels_follow_the_propensity_weighted_rates() {
    let geom = Geometry::from_cells(synth_cells(4, 1));
    // Cells 0, 1 wholly in community 0, cells 2, 3 wholly in community 1.
    let names = geom.names.clone();
    let mut prop = Mat::zeros(4, 2);
    for i in 0..4 {
        prop[(i, i / 2)] = 1.;
    }
    let comm = Communities::join(&geom, "final", (prop, vec![0, 0, 1, 1], None, names));
    // The gene runs at rate 1 in community 0 and 4 in community 1.
    let rates = FeatureRates {
        names: vec!["G".into()],
        rates: Mat::from_row_slice(1, 2, &[1., 4.]),
    };
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);

    let g = GeneMap::expected("G", &rates, &comm, 99., &grid).unwrap();
    assert_eq!(g.source, Source::Expected);
    assert_eq!(g.top, 4.);
    // Scaled to the top of the positive values: 1/4 and 4/4.
    assert_eq!(g.comm.prop, vec![64, 64, 255, 255]);
    assert_eq!(g.comm.k, 1);
    assert!(GeneMap::expected("nope", &rates, &comm, 99., &grid).is_err());
}

#[test]
fn a_few_extreme_cells_clip_instead_of_darkening_the_rest() {
    // 100 cells at level 1, 100 at level 2, one outlier at 100.
    let n = 201;
    let geom = Geometry::from_cells(synth_cells(n, 1));
    let names = geom.names.clone();
    let group = |i: usize| {
        if i < 100 {
            0
        } else if i < 200 {
            1
        } else {
            2
        }
    };
    let mut prop = Mat::zeros(n, 3);
    for i in 0..n {
        prop[(i, group(i))] = 1.;
    }
    let cluster = (0..n).map(|i| group(i) as i64).collect();
    let comm = Communities::join(&geom, "final", (prop, cluster, None, names));
    let rates = FeatureRates {
        names: vec!["G".into()],
        rates: Mat::from_row_slice(1, 3, &[1., 2., 100.]),
    };
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);

    let g = GeneMap::expected("G", &rates, &comm, 99., &grid).unwrap();
    // The top is the 99th percentile, 2, not the outlier's 100.
    assert_eq!(g.top, 2.);
    assert_eq!(g.comm.prop[0], 128);
    assert_eq!(g.comm.prop[150], 255);
    assert_eq!(g.comm.prop[200], 255, "the outlier clips");
}

#[test]
fn the_clip_percentile_moves_the_top() {
    // Positive values 1..=100: p95 sits at 95, p100 at the maximum.
    let n = 100;
    let geom = Geometry::from_cells(synth_cells(n, 1));
    let names = geom.names.clone();
    let mut prop = Mat::zeros(n, n);
    for i in 0..n {
        prop[(i, i)] = 1.;
    }
    let cluster = (0..n as i64).collect();
    let comm = Communities::join(&geom, "final", (prop, cluster, None, names));
    let levels: Vec<f32> = (1..=n).map(|v| v as f32).collect();
    let rates = FeatureRates {
        names: vec!["G".into()],
        rates: Mat::from_row_slice(1, n, &levels),
    };
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);
    let map = |clip| GeneMap::expected("G", &rates, &comm, clip, &grid).unwrap();
    assert_eq!(map(100.).top, 100.);
    assert_eq!(map(95.).top, 95.);
    assert!(map(95.).top_label().ends_with("(p95)"));
    assert!(!map(100.).top_label().contains('+'));
}

#[test]
fn ramp_tops_keep_three_significant_digits() {
    assert_eq!(significant(1.6094, 3), "1.61");
    assert_eq!(significant(0.012345, 3), "0.0123");
    assert_eq!(significant(123.4, 3), "123");
    assert_eq!(significant(0., 3), "0");
}

/// Two cells' counts of `rows` in a zarr file `tag`.
fn write_data(dir: &tempfile::TempDir, tag: &str, rows: &[&str], counts: &[[f32; 2]]) -> Box<str> {
    let triplets: Vec<(u64, u64, f32)> = counts
        .iter()
        .enumerate()
        .flat_map(|(r, c)| (0..2).map(move |j| (r as u64, j as u64, c[j])))
        .filter(|t| t.2 > 0.)
        .collect();
    crate::test_support::write_zarr(dir.path(), tag, rows, 2, &triplets)
}

#[test]
fn a_gene_one_file_never_measured_is_not_a_zero_there() {
    use crate::util::parquet_io::CellTable;
    use crate::view::data::NO_CLUSTER;
    let dir = tempfile::tempdir().unwrap();
    // GENE2 is on the first file's panel only; its second cell has none.
    let files = vec![
        write_data(&dir, "a", &["GENE1", "GENE2"], &[[1., 1.], [3., 0.]]),
        write_data(&dir, "b", &["GENE1"], &[[1., 1.]]),
    ];
    let mut names = crate::util::input::read_expr_data(&files)
        .unwrap()
        .column_names()
        .unwrap();
    // A map cell no data file holds: no level either.
    names.push("elsewhere".into());
    let cells = CellTable {
        index: names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect(),
        coords: (0..names.len())
            .map(|i| ((i % 2) as f32, (i / 2) as f32))
            .collect(),
        names,
        batches: None,
        in_graph: None,
        coord_col_names: vec!["x".into(), "y".into()],
    };
    let geom = Geometry::from_cells(cells);
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);
    let expr = Expression::open(&files, &geom).unwrap();

    let g = expr.observed("GENE2", &geom, 100., &grid).unwrap();
    // Measured: a count, or a measured zero. Not measured: no value.
    assert_eq!(&g.comm.cluster[..2], &[0, 0]);
    assert_eq!(&g.comm.cluster[2..], &[NO_CLUSTER; 3]);
    assert_eq!(&g.comm.prop[..2], &[255, 0]);
    // Zoomed out, a bin averages the cells that were measured.
    let top = g.pyramid.levels.last().unwrap();
    assert_eq!((top.count[0], top.absent[0]), (5, 3));

    // A group's mean is over the cells whose file measured the gene.
    let means = expr.group_means(&["GENE2", "GENE1"], &[0; 5], 1).unwrap();
    assert_eq!(means[0].as_deref(), Some(&[3f32.ln_1p() / 2.][..]));
    assert_eq!(means[1].as_deref(), Some(&[1f32.ln_1p()][..]));
}
