use super::fixture::synth_cells;
use crate::util::common::*;
use crate::view::data::{Communities, Geometry};
use crate::view::gene::*;
use crate::view::index::Grid;
use crate::view::markers::FeatureRates;

#[test]
fn features_are_found_by_name_or_symbol() {
    let names: Vec<Box<str>> = ["ENSG99999999901_GENE1", "PROT1", "ENSG99999999902_GENE2"]
        .map(Box::from)
        .to_vec();
    assert_eq!(find(&names, "ENSG99999999901_GENE1"), Some(0));
    assert_eq!(find(&names, "GENE1"), Some(0));
    assert_eq!(find(&names, "PROT1"), Some(1));
    assert_eq!(find(&names, "GENE2"), Some(2));
    assert_eq!(find(&names, "GENE9"), None);
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

/// The map of the data files' cells, two to a row, and of `extra` cells
/// no file holds. Each row slants, so two cells still span both axes.
fn geom_of(files: &[Box<str>], extra: &[&str]) -> Geometry {
    use crate::util::parquet_io::CellTable;
    let mut names = crate::util::input::read_expr_data(files)
        .unwrap()
        .column_names()
        .unwrap();
    names.extend(extra.iter().map(|&s| Box::from(s)));
    let cells = CellTable {
        index: names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect(),
        coords: (0..names.len())
            .map(|i| ((i % 2) as f32, (i / 2) as f32 + 0.5 * (i % 2) as f32))
            .collect(),
        names,
        batches: None,
        in_graph: None,
        coord_col_names: vec!["x".into(), "y".into()],
    };
    Geometry::from_cells(cells)
}

/// ln(1 + 1e4 · count / total), spelled out apart from the code's helper.
fn cp10k(count: f32, total: f32) -> f32 {
    (1e4 * count / total).ln_1p()
}

#[test]
fn a_gene_one_file_never_measured_is_not_a_zero_there() {
    use crate::view::data::NO_CLUSTER;
    let dir = tempfile::tempdir().unwrap();
    // GENE2 is on the first file's panel only; its second cell has none.
    let files = vec![
        write_data(&dir, "a", &["GENE1", "GENE2"], &[[1., 1.], [3., 0.]]),
        write_data(&dir, "b", &["GENE1"], &[[1., 1.]]),
    ];
    // A map cell no data file holds: no level either.
    let geom = geom_of(&files, &["elsewhere"]);
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

    // A group's mean is over the cells whose file measured the gene. The
    // cells total 4, 1, 1 and 1.
    let means = expr.group_means(&["GENE2", "GENE1"], &[0; 5], 1).unwrap();
    assert_eq!(means[0].as_deref(), Some(&[cp10k(3., 4.) / 2.][..]));
    let gene1 = means[1].as_deref().unwrap()[0];
    let want = (cp10k(1., 4.) + 3. * cp10k(1., 1.)) / 4.;
    assert!((gene1 - want).abs() < 1e-5 * want, "{gene1} vs {want}");
}

/// Two cells of one composition, the second five times as deep.
fn one_composition(dir: &tempfile::TempDir) -> Vec<Box<str>> {
    vec![write_data(
        dir,
        "a",
        &["GENE1", "GENE2"],
        &[[1., 5.], [3., 15.]],
    )]
}

#[test]
fn cells_of_one_composition_show_the_same_level_at_any_depth() {
    let dir = tempfile::tempdir().unwrap();
    let files = one_composition(&dir);
    let geom = geom_of(&files, &[]);
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);
    let expr = Expression::open(&files, &geom).unwrap();

    let levels = expr.levels(0, geom.n()).unwrap();
    assert_eq!(levels[0], levels[1]);
    assert_eq!(levels[0], cp10k(1., 4.));
    // So both sit at the top of the ramp.
    let g = expr.observed("GENE1", &geom, 100., &grid).unwrap();
    assert_eq!(g.comm.prop, [255, 255]);
}

#[test]
fn a_single_count_shows_its_share_of_the_cells_total() {
    let dir = tempfile::tempdir().unwrap();
    // The first cell: one count of GENE1 among a total of 7 over all rows.
    let rows = ["GENE1", "GENE2", "GENE3"];
    let files = vec![write_data(
        &dir,
        "a",
        &rows,
        &[[1., 0.], [4., 2.], [2., 1.]],
    )];
    let geom = geom_of(&files, &[]);
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);
    let expr = Expression::open(&files, &geom).unwrap();

    assert_eq!(
        expr.levels(0, geom.n()).unwrap(),
        [(1e4f32 / 7.).ln_1p(), 0.]
    );
    let g = expr.observed("GENE1", &geom, 100., &grid).unwrap();
    assert_eq!(g.top, (1e4f32 / 7.).ln_1p());
}

#[test]
fn group_means_are_of_the_same_depth_normalized_level() {
    let dir = tempfile::tempdir().unwrap();
    let files = one_composition(&dir);
    let geom = geom_of(&files, &[]);
    let expr = Expression::open(&files, &geom).unwrap();

    // A group per cell: one composition, one mean, whatever the depth.
    let apart = expr.group_means(&["GENE2"], &[0, 1], 2).unwrap();
    assert_eq!(
        apart[0].as_deref(),
        Some(&[cp10k(3., 4.), cp10k(3., 4.)][..])
    );
    // And each mean is of the levels the gene's map shows.
    let together = expr.group_means(&["GENE1"], &[0, 0], 1).unwrap();
    let levels = expr.levels(0, geom.n()).unwrap();
    assert_eq!(
        together[0].as_deref(),
        Some(&[(levels[0] + levels[1]) / 2.][..])
    );
}

#[test]
fn a_cell_with_no_counts_shows_zero() {
    let dir = tempfile::tempdir().unwrap();
    // The second cell's only entry is a stored zero: its total is 0.
    let triplets = [(0, 0, 2.), (1, 0, 2.), (0, 1, 0.)];
    let files = vec![crate::test_support::write_zarr(
        dir.path(),
        "a",
        &["GENE1", "GENE2"],
        2,
        &triplets,
    )];
    let geom = geom_of(&files, &[]);
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);
    let expr = Expression::open(&files, &geom).unwrap();

    assert_eq!(expr.levels(0, geom.n()).unwrap(), [cp10k(2., 4.), 0.]);
    // A measured zero, at the bottom of the ramp.
    let g = expr.observed("GENE1", &geom, 100., &grid).unwrap();
    assert_eq!(g.comm.cluster, [0, 0]);
    assert_eq!(g.comm.prop, [255, 0]);
    let means = expr.group_means(&["GENE1"], &[0, 1], 2).unwrap();
    assert_eq!(means[0].as_deref(), Some(&[cp10k(2., 4.), 0.][..]));
}

#[test]
fn the_legend_names_the_scale_shown() {
    let dir = tempfile::tempdir().unwrap();
    let files = one_composition(&dir);
    let geom = geom_of(&files, &[]);
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);
    let expr = Expression::open(&files, &geom).unwrap();
    let g = expr.observed("GENE1", &geom, 100., &grid).unwrap();
    assert_eq!(g.title(), "GENE1 log1p CP10k");
}
