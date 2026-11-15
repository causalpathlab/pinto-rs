//! `spatialde2-pb`: super-cells as cage builds them, the collapsed
//! graph, and the unravelling back to cells.
//!
//! Success bar, set before measuring: on an srt-simulate section (a few
//! thousand cells at most), fitted on about one super-cell per 8 cells,
//! the unravelled labels recover the planted domains within 0.1 ARI of
//! the cell-level fit, and beat the super-cells' labels broadcast to their
//! cells on the cells at a domain boundary. A method that fails it has no
//! `-pb` command.

use super::graph::grid;
use crate::spatial_baselines::graph::knn_within_sections;
use crate::spatial_baselines::pseudobulk::*;
use crate::util::common::Mat;
use legume_numeric::matrix::parquet::read_table_columns;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use std::collections::HashMap;

/// Super-cells sum their cells' counts, sit at their centroid, keep their
/// section, and never span two sections.
#[test]
fn super_cells_sum_counts_within_a_section() {
    let mut d = grid(8, 2);
    let n = d.cells.len();
    let mut coo = CooMatrix::<f32>::new(3, n);
    for i in 0..n {
        coo.push(i % 3, i, (i % 5 + 1) as f32);
        coo.push((i + 1) % 3, i, 1.0);
    }
    d.counts = CscMatrix::from(&coo);
    let proj = Mat::from_fn(4, n, |r, c| ((r + 1) * (c % 8)) as f32 + 0.5);
    let sc = super_cells(&d, proj, 10, knn_within_sections(&d, COARSEN_K)).unwrap();
    let m = sc.data.cells.len();
    assert!(m >= 2 && m < n, "{m} super-cells");
    for (c, &s) in sc.of_cell.iter().enumerate() {
        assert_eq!(sc.data.section[s], d.section[c], "cell {c}");
    }
    let total = |x: &CscMatrix<f32>, g: usize| -> f32 {
        x.triplet_iter().filter(|t| t.0 == g).map(|t| *t.2).sum()
    };
    for g in 0..3 {
        assert_eq!(total(&sc.data.counts, g), total(&d.counts, g));
    }
    let s0 = sc.of_cell[0];
    let members: Vec<usize> = (0..n).filter(|&c| sc.of_cell[c] == s0).collect();
    let mean_x: f32 = members.iter().map(|&c| d.xy[(c, 0)]).sum::<f32>() / members.len() as f32;
    assert!((sc.data.xy[(s0, 0)] - mean_x).abs() < 1e-5);
}

/// A super-edge sums the weights of the cell-cell edges it collapses;
/// edges inside a super-cell fold into it.
#[test]
fn fold_sums_the_collapsed_edges() {
    let of_cell = [0, 0, 1, 1, 2];
    let edges = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 4)];
    let w: Vec<f64> = (1..=edges.len()).map(|e| e as f64).collect();
    let (super_edges, w) = fold_weighted(&of_cell, &edges, &w);
    let got: HashMap<(usize, usize), f64> = super_edges.into_iter().zip(w).collect();
    assert_eq!(got.len(), 2);
    assert_eq!(got[&(0, 1)], 2.0 + 3.0 + 4.0);
    assert_eq!(got[&(1, 2)], 5.0);
}

/// `undirected` lists each edge of symmetric neighbour lists once.
#[test]
fn undirected_lists_each_edge_once() {
    let d = grid(4, 1);
    let nb = knn_within_sections(&d, 4).symmetrised();
    let (edges, dist) = nb.undirected();
    assert_eq!(edges.len(), nb.n_edges() / 2);
    assert!(edges.iter().all(|&(i, j)| i < j));
    assert_eq!(edges.len(), dist.len());
}

// ------------------------------------------------- the success bar

/// Adjusted Rand index of two labellings.
pub(crate) fn ari(a: &[usize], b: &[usize]) -> f64 {
    let mut table: HashMap<(usize, usize), f64> = HashMap::new();
    let (mut ra, mut rb): (HashMap<usize, f64>, HashMap<usize, f64>) = Default::default();
    for (&x, &y) in a.iter().zip(b) {
        *table.entry((x, y)).or_default() += 1.0;
        *ra.entry(x).or_default() += 1.0;
        *rb.entry(y).or_default() += 1.0;
    }
    let c2 = |v: f64| v * (v - 1.0) / 2.0;
    let index: f64 = table.values().map(|&v| c2(v)).sum();
    let (sa, sb): (f64, f64) = (
        ra.values().map(|&v| c2(v)).sum(),
        rb.values().map(|&v| c2(v)).sum(),
    );
    let expected = sa * sb / c2(a.len() as f64);
    (index - expected) / ((sa + sb) / 2.0 - expected)
}

/// Labels as integers, in the order of `cells`, from a `cell`/label table.
pub(crate) fn labels_of(cells: &[Box<str>], table: &HashMap<Box<str>, Box<str>>) -> Vec<usize> {
    let mut ids: HashMap<&str, usize> = HashMap::new();
    cells
        .iter()
        .map(|c| {
            let l = table[c].as_ref();
            let next = ids.len();
            *ids.entry(l).or_insert(next)
        })
        .collect()
}

pub(crate) fn read_labels(prefix: &str) -> HashMap<Box<str>, Box<str>> {
    std::fs::read_to_string(format!("{prefix}.labels.tsv"))
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let (c, v) = l.split_once('\t').unwrap();
            (c.into(), v.into())
        })
        .collect()
}

pub(crate) fn read_column(path: &str, key: &str, col: &str) -> HashMap<Box<str>, Box<str>> {
    let (strs, _) = read_table_columns(path, &[key, col], &[]).unwrap();
    strs[0]
        .iter()
        .cloned()
        .zip(strs[1].iter().cloned())
        .collect()
}

/// spatialde2-pb against the bar: within 0.1 ARI of spatialde2 on cells,
/// and above the broadcast labels on the boundary cells.
#[test]
fn spatialde2_pb_meets_the_bar_on_a_simulated_section() {
    let dir = tempfile::tempdir().unwrap();
    let sim = dir.path().join("sim").to_string_lossy().to_string();
    crate::simulate::run::run_simulate(&crate::simulate::run::SimulateArgs {
        scenario: "L0".into(),
        set: [
            "grid_side=40".into(),
            "n_genes=500".into(),
            "n_donors=1".into(),
            "sections_per_donor=1".into(),
            "seed=2".into(),
        ]
        .into(),
        coords: Vec::new(),
        coord_column_names: vec!["row_um".into(), "col_um".into()],
        out: sim.as_str().into(),
    })
    .unwrap();
    let run = |cmd: &str, out: &str, extra: &[&str]| {
        let mut argv = vec![
            "pinto".to_string(),
            cmd.into(),
            format!("{sim}/sections/S1.zarr"),
            "-c".into(),
            format!("{sim}/sections/S1.coords.parquet"),
            "--coord-column-names".into(),
            "row_um,col_um".into(),
            "-o".into(),
            out.into(),
        ];
        argv.extend(extra.iter().map(|s| s.to_string()));
        match <crate::Cli as clap::Parser>::try_parse_from(argv)
            .unwrap()
            .commands
        {
            crate::Commands::Spatialde2(args) => {
                crate::spatial_baselines::spatialde2::run(&args).unwrap();
                args.fit.input.read(false).unwrap()
            }
            crate::Commands::Spatialde2Pb(args) => {
                crate::spatial_baselines::spatialde2::run_pb(&args).unwrap();
                args.fit.input.read(false).unwrap()
            }
            _ => unreachable!(),
        }
    };
    let cell_out = dir.path().join("cells").to_string_lossy().to_string();
    let pb_out = dir.path().join("pb").to_string_lossy().to_string();
    let data = run("spatialde2", &cell_out, &[]);
    let cells = &data.cells;
    run("spatialde2-pb", &pb_out, &["-d", "200"]);

    let truth = read_column(&format!("{sim}/truth/spots.parquet"), "barcode", "domain");
    let domain = labels_of(cells, &truth);
    let cell_fit = labels_of(cells, &read_labels(&cell_out));
    let unravelled = labels_of(cells, &read_labels(&pb_out));
    let broadcast = labels_of(
        cells,
        &read_column(&format!("{pb_out}.cell_pb.parquet"), "cell", "pb_label"),
    );
    // cells with a planted domain other than their own among their 6 nearest
    let nb = knn_within_sections(&data, 6);
    let boundary: Vec<usize> = (0..cells.len())
        .filter(|&i| nb.index[i].iter().any(|&j| domain[j] != domain[i]))
        .collect();
    assert!(boundary.len() > 50, "{} boundary cells", boundary.len());
    let on = |l: &[usize]| boundary.iter().map(|&i| l[i]).collect::<Vec<_>>();
    let (a_cell, a_pb) = (ari(&domain, &cell_fit), ari(&domain, &unravelled));
    let (b_pb, b_bc) = (
        ari(&on(&domain), &on(&unravelled)),
        ari(&on(&domain), &on(&broadcast)),
    );
    eprintln!(
        "ARI cells {a_cell:.3}, unravelled {a_pb:.3}; boundary {b_pb:.3} vs broadcast {b_bc:.3}"
    );
    assert!(a_pb >= a_cell - 0.1, "unravelled {a_pb} vs cells {a_cell}");
    assert!(
        b_pb > b_bc,
        "boundary: unravelled {b_pb} vs broadcast {b_bc}"
    );
}

/// A factor baseline's `-pb` command against the bar: within 0.1 ARI of
/// the cell-level command, and above the broadcast labels on the boundary
/// cells; both at the planted number of domains.
fn factor_pb_meets_the_bar(cmd: &str) {
    use super::spatialpca::{parse, planted, simulated_section};
    let dir = tempfile::tempdir().unwrap();
    let (sim, files) = simulated_section(dir.path(), 40, 500, 2);
    let cell_out = dir.path().join("cells").to_string_lossy().to_string();
    let pb_out = dir.path().join("pb").to_string_lossy().to_string();
    let probe = parse(cmd, &files, &cell_out, &[]);
    let data = match &probe {
        crate::Commands::Spatialpca(a) => a.fit.core.input.read(false).unwrap(),
        crate::Commands::Mefisto(a) => a.fit.core.input.read(false).unwrap(),
        _ => unreachable!(),
    };
    let cells = &data.cells;
    let (domain, k) = planted(&sim, cells);
    let k = k.to_string();
    let pb_cmd = format!("{cmd}-pb");
    // quadrature on the cell graph, as on any section above 400 cells
    let quad = ["--exact-eig-max", "400"];
    match parse(
        cmd,
        &files,
        &cell_out,
        &["--n-clusters", &k, quad[0], quad[1]],
    ) {
        crate::Commands::Spatialpca(a) => crate::spatial_baselines::spatialpca::run(&a).unwrap(),
        crate::Commands::Mefisto(a) => crate::spatial_baselines::mefisto::run(&a).unwrap(),
        _ => unreachable!(),
    }
    match parse(
        &pb_cmd,
        &files,
        &pb_out,
        &["--n-clusters", &k, "-d", "200", quad[0], quad[1]],
    ) {
        crate::Commands::SpatialpcaPb(a) => {
            crate::spatial_baselines::spatialpca::run_pb(&a).unwrap()
        }
        crate::Commands::MefistoPb(a) => crate::spatial_baselines::mefisto::run_pb(&a).unwrap(),
        _ => unreachable!(),
    }
    let cell_fit = labels_of(cells, &read_labels(&cell_out));
    let unravelled = labels_of(cells, &read_labels(&pb_out));
    let broadcast = labels_of(
        cells,
        &read_column(&format!("{pb_out}.cell_pb.parquet"), "cell", "pb_label"),
    );
    let nb = knn_within_sections(&data, 6);
    let boundary: Vec<usize> = (0..cells.len())
        .filter(|&i| nb.index[i].iter().any(|&j| domain[j] != domain[i]))
        .collect();
    assert!(boundary.len() > 50, "{} boundary cells", boundary.len());
    let on = |l: &[usize]| boundary.iter().map(|&i| l[i]).collect::<Vec<_>>();
    let (a_cell, a_pb) = (ari(&domain, &cell_fit), ari(&domain, &unravelled));
    let a_bc = ari(&domain, &broadcast);
    let (b_cell, b_pb, b_bc) = (
        ari(&on(&domain), &on(&cell_fit)),
        ari(&on(&domain), &on(&unravelled)),
        ari(&on(&domain), &on(&broadcast)),
    );
    eprintln!(
        "{cmd}: ARI cells {a_cell:.3}, unravelled {a_pb:.3}, broadcast {a_bc:.3}; \
         boundary cells {b_cell:.3}, unravelled {b_pb:.3}, broadcast {b_bc:.3}"
    );
    assert!(
        a_pb >= a_cell - 0.1,
        "{cmd}: unravelled {a_pb} vs cells {a_cell}"
    );
    assert!(
        b_pb > b_bc,
        "{cmd} boundary: unravelled {b_pb} vs broadcast {b_bc}"
    );
}

#[test]
fn spatialpca_pb_meets_the_bar_on_a_simulated_section() {
    factor_pb_meets_the_bar("spatialpca");
}

#[test]
fn mefisto_pb_meets_the_bar_on_a_simulated_section() {
    factor_pb_meets_the_bar("mefisto");
}
