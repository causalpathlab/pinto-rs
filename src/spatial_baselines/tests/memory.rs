//! The peak memory estimates, against peaks measured on full runs, and the
//! refusal of a run they say would not fit.

use super::graph::toy;
use crate::spatial_baselines::memory::*;
use crate::spatial_baselines::{spatialde2, stagate};
use nalgebra_sparse::{CooMatrix, CscMatrix};

const MIB: f64 = (1u64 << 20) as f64;
/// Threads of the machine the peaks were measured on.
const THREADS: usize = 10;

/// Maximum resident memory (MiB) of full CPU runs on square grids at the
/// defaults (2000 genes, all trained on; for stagate hidden width 512 and
/// 6 neighbours, for spatialde2 100 neighbours): (cells per section,
/// sections, MiB).
const STAGATE_MEASURED: [(usize, usize, f64); 5] = [
    (1024, 1, 426.19),
    (4096, 1, 1394.23),
    (8100, 1, 2699.27),
    (16384, 1, 4917.27),
    (4096, 2, 2627.53),
];
const SPATIALDE2_MEASURED: [(usize, usize, f64); 8] = [
    (1024, 1, 89.84),
    (4096, 1, 310.94),
    (8100, 1, 698.50),
    (16384, 1, 1490.67),
    (32761, 1, 2397.05),
    (65536, 1, 4202.14),
    (99856, 1, 5664.56),
    (4096, 2, 598.58),
];
/// Nonzeros per cell of those runs' counts: what their slope, 12 + 40
/// bytes a nonzero, fits.
const NONZEROS_PER_CELL: usize = 1000;

fn stagate_shape(per_section: usize, sections: usize) -> StagateShape {
    let sizes = vec![per_section; sections];
    let n = per_section * sections;
    StagateShape {
        cells: n,
        edges: n + knn_edges(&sizes, 6),
        genes: 2000,
        hidden: 512,
        nonzeros: NONZEROS_PER_CELL * n,
        gram: knn_gram_bytes(&sizes, THREADS),
        searched: knn_edges(&sizes, 6),
        train_on_host: true,
    }
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / MIB
}

#[test]
fn stagate_estimate_covers_every_measured_peak_and_is_not_far_above() {
    for (per, sections, measured) in STAGATE_MEASURED {
        let est = mib(stagate_peak_bytes(&stagate_shape(per, sections)));
        assert!(
            est >= measured,
            "{per}x{sections}: {est:.0} < {measured} MiB"
        );
        if per * sections >= 4096 {
            assert!(est <= 1.25 * measured, "{per}x{sections}: {est:.0} MiB");
        }
    }
}

/// The safety factor's calibration: per cell, never below the 0.285 MiB
/// the measured runs grew by.
#[test]
fn stagate_estimate_grows_by_at_least_the_measured_slope() {
    let at = |n| stagate_training_bytes(&stagate_shape(n, 1));
    let slope = mib(at(200_000) - at(100_000)) / 100_000.0;
    assert!((0.285..0.35).contains(&slope), "{slope} MiB per cell");
}

/// A radius graph's edges, the input's genes and the hidden width all
/// count; on CUDA the host holds the counts and the input only.
#[test]
fn stagate_estimate_follows_edges_genes_and_device() {
    let base = stagate_shape(10_000, 1);
    let peak = stagate_peak_bytes(&base);
    let radius = StagateShape {
        edges: 257 * base.cells,
        ..base.clone()
    };
    // 4 bytes, 10 E×H floats per edge, times the safety factor
    let added = stagate_peak_bytes(&radius) - peak;
    let want = 1.5 * 4.0 * 10.0 * 512.0 * (radius.edges - base.edges) as f64;
    assert!(
        (added as f64 - want).abs() < 1e-6 * want,
        "{added} vs {want}"
    );
    let genes = StagateShape {
        genes: 3000,
        ..base.clone()
    };
    assert!(stagate_peak_bytes(&genes) > peak);
    let cuda = StagateShape {
        train_on_host: false,
        ..base.clone()
    };
    let counts = 12 * base.nonzeros;
    let dense = 4 * base.cells * base.genes;
    assert_eq!(
        stagate_peak_bytes(&cuda),
        STAGATE_PROCESS_BYTES + (counts + base.gram + 12 * base.searched).max(counts + dense)
    );
}

fn spatialde2_shape(per_section: usize, sections: usize) -> Spatialde2Shape {
    let sizes = vec![per_section; sections];
    let n = per_section * sections;
    Spatialde2Shape {
        cells: n,
        nonzeros: NONZEROS_PER_CELL * n,
        selected_nonzeros: NONZEROS_PER_CELL * n,
        edges: 2 * knn_edges(&sizes, 100),
        regions: spatialde2::initial_regions(n, None),
        gram: knn_gram_bytes(&sizes, THREADS),
    }
}

#[test]
fn spatialde2_estimate_covers_every_measured_peak_and_is_not_far_above() {
    for (per, sections, measured) in SPATIALDE2_MEASURED {
        let est = mib(spatialde2_peak_bytes(&spatialde2_shape(per, sections)));
        assert!(
            est >= measured,
            "{per}x{sections}: {est:.0} < {measured} MiB"
        );
        if per * sections >= 16384 {
            assert!(est <= 1.5 * measured, "{per}x{sections}: {est:.0} MiB");
        }
    }
}

/// The task holding a section's Gram block holds `cells × block` floats,
/// the block 64 MiB over the cells' bytes (16 to 1024 columns); at most
/// one task per thread is in flight.
#[test]
fn gram_blocks_in_flight_are_bounded_by_the_threads_and_the_blocks() {
    let block = 64 << 20;
    // 65536 cells: 256-column blocks of 64 MiB, 256 of them
    assert_eq!(knn_gram_bytes(&[65536], 8), 8 * block);
    assert_eq!(knn_gram_bytes(&[65536], 1000), 256 * block);
    // 100 cells: one block of every column
    assert_eq!(knn_gram_bytes(&[100], 8), 4 * 100 * 100);
    // the largest blocks first, across sections
    assert_eq!(
        knn_gram_bytes(&[65536, 100], 257),
        256 * block + 4 * 100 * 100
    );
    assert_eq!(knn_gram_bytes(&[1, 0], 8), 0);
    assert_eq!(knn_edges(&[10, 3, 1], 6), 10 * 6 + 3 * 2);
}

// ---------------------------------------------------------------- runs

/// `side × side` cells in each of two sections, 12 genes of counts that
/// vary along the grid.
fn tissue(side: usize) -> crate::spatial_baselines::input::SpatialData {
    let mut xy = Vec::new();
    let mut sec = Vec::new();
    for s in 0..2 {
        for r in 0..side {
            for c in 0..side {
                xy.push((r as f32, c as f32));
                sec.push(s);
            }
        }
    }
    let mut d = toy(&xy, &sec);
    let genes = 12;
    let mut coo = CooMatrix::<f32>::new(genes, xy.len());
    for (i, &(r, c)) in xy.iter().enumerate() {
        for g in 0..genes {
            let v = ((r as usize * (g + 1) + c as usize * (g % 3) + i) % 7) as f32;
            if v > 0.0 {
                coo.push(g, i, v);
            }
        }
    }
    d.counts = CscMatrix::from(&coo);
    d.genes = crate::test_support::numbered("GENE", genes);
    d
}

fn parse(argv: &[&str]) -> crate::Commands {
    <crate::Cli as clap::Parser>::try_parse_from(argv)
        .unwrap_or_else(|e| panic!("{e}"))
        .commands
}

fn stagate_args(out: &std::path::Path, extra: &[&str]) -> stagate::Args {
    let mut argv = vec![
        "pinto",
        "stagate",
        "d.zarr",
        "-c",
        "c.csv",
        "-o",
        out.to_str().unwrap(),
        "--n-top-genes",
        "8",
        "--hidden-dims",
        "8,2",
        "--n-clusters",
        "2",
    ];
    argv.extend(extra);
    let crate::Commands::Stagate(args) = parse(&argv) else {
        unreachable!()
    };
    *args
}

/// A budget of a few bytes refuses the run before the graph and training
/// (a run that trained first would not finish its billion epochs), with
/// the estimate, the budget and what to change; a large one runs it.
#[test]
fn stagate_refuses_a_run_over_the_budget_before_training() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("st");
    let run = |epochs: &str, budget: &str| {
        let args = stagate_args(&out, &["--n-epochs", epochs]);
        stagate::run_on(&args, tissue(6), Some(budget))
    };
    let e = run("1000000000", "1000").unwrap_err().to_string();
    for want in [
        "estimated",
        "budget",
        "--n-top-genes",
        "--device cuda",
        "PINTO_MEMORY_BUDGET",
    ] {
        assert!(e.contains(want), "{want} missing from: {e}");
    }
    assert!(!dir.path().join("st.labels.tsv").exists());
    run("3", &usize::MAX.to_string()).unwrap();
    assert!(dir.path().join("st.labels.tsv").exists());
}

/// A radius graph's edges are counted once it is built: a budget that
/// holds its self loops alone refuses it then.
#[test]
fn stagate_counts_a_radius_graph_once_built() {
    let dir = tempfile::tempdir().unwrap();
    let data = tissue(6);
    let loops_only = StagateShape {
        cells: data.cells.len(),
        edges: data.cells.len(),
        genes: 8,
        hidden: 8,
        nonzeros: data.counts.nnz(),
        gram: knn_gram_bytes(&[36, 36], rayon::current_num_threads()),
        searched: knn_edges(&[36, 36], 256),
        train_on_host: true,
    };
    let budget = (stagate_peak_bytes(&loops_only) + 1).to_string();
    let args = stagate_args(
        &dir.path().join("st"),
        &["--rad-cutoff", "1.5", "--n-epochs", "1"],
    );
    let e = stagate::run_on(&args, data, Some(&budget)).unwrap_err();
    assert!(e.to_string().contains("estimated"), "{e}");
}

#[test]
fn spatialde2_refuses_a_run_over_the_budget_before_fitting() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sd");
    let run = |budget: &str| {
        let crate::Commands::Spatialde2(args) = parse(&[
            "pinto",
            "spatialde2",
            "d.zarr",
            "-c",
            "c.csv",
            "-o",
            out.to_str().unwrap(),
            "--neighbors",
            "4",
        ]) else {
            unreachable!()
        };
        spatialde2::run_on(&args, tissue(6), Some(budget))
    };
    let e = run("1000").unwrap_err().to_string();
    for want in ["estimated", "budget", "--n-clusters", "PINTO_MEMORY_BUDGET"] {
        assert!(e.contains(want), "{want} missing from: {e}");
    }
    assert!(!dir.path().join("sd.labels.tsv").exists());
    run(&usize::MAX.to_string()).unwrap();
    assert!(dir.path().join("sd.labels.tsv").exists());
}

/// Each command's help names the budget variable.
#[test]
fn help_names_the_memory_budget() {
    for cmd in ["stagate", "spatialde2"] {
        let help = crate::built_cli()
            .find_subcommand_mut(cmd)
            .unwrap()
            .render_long_help()
            .to_string();
        assert!(help.contains("PINTO_MEMORY_BUDGET"), "{cmd} --help");
    }
}
