//! The peak memory estimates, against peaks measured on full runs, and the
//! refusal of a run they say would not fit.

use super::graph::toy;
use crate::spatial_baselines::memory::*;
use crate::spatial_baselines::{spatialde2, stagate};
use crate::util::memory::{ensure_fits, Budget, MEMORY_BUDGET_VAR};
use nalgebra_sparse::{CooMatrix, CscMatrix};

const MIB: f64 = (1u64 << 20) as f64;

/// Maximum resident memory (MiB) of full CPU runs on srt-simulate square
/// grids at the defaults (2000 genes, all trained on; for stagate hidden
/// width 512 and 6 neighbours, for spatialde2 100 neighbours): (cells per
/// section, sections, threads, MiB).
const STAGATE_MEASURED: [(usize, usize, usize, f64); 9] = [
    (1024, 1, 10, 426.19),
    (4096, 1, 10, 1394.23),
    (8100, 1, 10, 2699.27),
    (10000, 1, 10, 3492.0),
    (10000, 1, 4, 3280.91),
    (14400, 1, 4, 4691.03),
    (16384, 1, 4, 4901.41),
    (16384, 1, 10, 4917.27),
    (4096, 2, 10, 2627.53),
];
const SPATIALDE2_MEASURED: [(usize, usize, usize, f64); 13] = [
    (1024, 1, 10, 89.84),
    (4096, 1, 4, 299.66),
    (4096, 1, 10, 310.94),
    (8100, 1, 10, 698.50),
    (10000, 1, 4, 688.91),
    (10000, 1, 10, 926.0),
    (14400, 1, 4, 968.00),
    (16384, 1, 4, 1090.64),
    (16384, 1, 10, 1490.67),
    (32761, 1, 10, 2397.05),
    (65536, 1, 10, 4202.14),
    (99856, 1, 10, 5664.56),
    (4096, 2, 10, 598.58),
];
/// Nonzeros per cell of those runs' counts, as srt-simulate draws them at
/// those settings ([`srt_simulate_reads_as_the_measured_runs_assume`]).
const NONZEROS_PER_CELL: usize = 765;

fn stagate_shape(per_section: usize, sections: usize, threads: usize) -> StagateShape {
    let sizes = vec![per_section; sections];
    let n = per_section * sections;
    StagateShape {
        cells: n,
        edges: n + knn_edges(&sizes, 6),
        genes: 2000,
        hidden: 512,
        nonzeros: NONZEROS_PER_CELL * n,
        gram: knn_gram_bytes(&sizes, threads),
        searched: knn_edges(&sizes, 6),
        train_on_host: true,
    }
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / MIB
}

#[test]
fn stagate_estimate_covers_every_measured_peak_and_is_not_far_above() {
    for (per, sections, threads, measured) in STAGATE_MEASURED {
        let est = mib(stagate_peak_bytes(&stagate_shape(per, sections, threads)));
        assert!(
            est >= measured,
            "{per}x{sections}: {est:.0} < {measured} MiB"
        );
        if per * sections >= 4096 {
            assert!(est <= 1.25 * measured, "{per}x{sections}: {est:.0} MiB");
        }
    }
}

/// The safety factor's calibration: the largest ratio of a measured peak,
/// above the process's own, to the counted training memory, with a small
/// margin.
#[test]
fn stagate_safety_is_the_largest_measured_residual() {
    let largest = STAGATE_MEASURED
        .iter()
        .map(|&(per, sections, threads, measured)| {
            let shape = stagate_shape(per, sections, threads);
            let counted = stagate_training_bytes(&shape) as f64 / STAGATE_SAFETY;
            (measured * MIB - STAGATE_PROCESS_BYTES as f64) / counted
        })
        .fold(0.0, f64::max);
    assert!(
        (largest..largest + 0.1).contains(&STAGATE_SAFETY),
        "{STAGATE_SAFETY} against {largest}"
    );
}

/// A radius graph's edges, the input's genes and the hidden width all
/// count; on CUDA the host holds the counts, the input and the graph's
/// search only.
#[test]
fn stagate_estimate_follows_edges_genes_and_device() {
    let base = stagate_shape(10_000, 1, 10);
    let peak = stagate_peak_bytes(&base);
    let radius = StagateShape {
        edges: 257 * base.cells,
        ..base.clone()
    };
    // 4 bytes, 10 E×H floats per edge, times the safety factor
    let added = stagate_peak_bytes(&radius) - peak;
    let want = STAGATE_SAFETY * 4.0 * 10.0 * 512.0 * (radius.edges - base.edges) as f64;
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
    // the counts are freed once the dense input is built, before the graph
    let counts = 12 * base.nonzeros;
    let dense = 4 * base.cells * base.genes;
    let graph = dense + base.gram + 28 * base.searched;
    assert!(graph > read_bytes(base.nonzeros) && graph > counts + dense);
    assert_eq!(stagate_peak_bytes(&cuda), STAGATE_PROCESS_BYTES + graph);
}

fn spatialde2_shape(per_section: usize, sections: usize, threads: usize) -> Spatialde2Shape {
    let sizes = vec![per_section; sections];
    let n = per_section * sections;
    Spatialde2Shape {
        cells: n,
        nonzeros: NONZEROS_PER_CELL * n,
        selected_nonzeros: NONZEROS_PER_CELL * n,
        genes: 2000,
        edges: 2 * knn_edges(&sizes, 100),
        regions: spatialde2::initial_regions(n, None),
        gram: knn_gram_bytes(&sizes, threads),
    }
}

/// Building the symmetrised graph holds 76 bytes a kNN edge (38 a
/// symmetrised edge, of which there are at most twice as many): the kNN
/// lists (12), their sorted copy (8), the reverse edges (24) and the
/// merged lists (16 each, the reverse ones among them).
#[test]
fn spatialde2_estimate_charges_the_graph_as_it_is_built() {
    let base = spatialde2_shape(16384, 1, 4);
    let more = Spatialde2Shape {
        edges: base.edges + 1000,
        ..base.clone()
    };
    assert_eq!(
        spatialde2_peak_bytes(&more) - spatialde2_peak_bytes(&base),
        38 * 1000
    );
}

/// Each update holds six genes × regions arrays of f64: E[λ], E[log λ] in
/// both layouts, π̂ Y, and the old and new gammahat_1.
#[test]
fn spatialde2_estimate_counts_the_genes_by_regions_state() {
    let base = spatialde2_shape(16384, 1, 4);
    let more = Spatialde2Shape {
        genes: base.genes + 1000,
        ..base.clone()
    };
    assert_eq!(
        spatialde2_peak_bytes(&more) - spatialde2_peak_bytes(&base),
        6 * 8 * 1000 * base.regions
    );
}

#[test]
fn spatialde2_estimate_covers_every_measured_peak_and_is_not_far_above() {
    for (per, sections, threads, measured) in SPATIALDE2_MEASURED {
        let est = mib(spatialde2_peak_bytes(&spatialde2_shape(
            per, sections, threads,
        )));
        assert!(
            est >= measured,
            "{per}x{sections}: {est:.0} < {measured} MiB"
        );
        if per * sections >= 16384 {
            assert!(est <= 1.5 * measured, "{per}x{sections}: {est:.0} MiB");
        }
    }
}

/// Reading the counts holds more per nonzero than they keep once read,
/// so a run whose counts dwarf the rest peaks in the read.
#[test]
fn the_read_of_the_counts_is_charged() {
    let nonzeros = 1 << 30;
    let stagate = StagateShape {
        cells: 10,
        edges: 10,
        genes: 1,
        hidden: 1,
        nonzeros,
        gram: 0,
        searched: 0,
        train_on_host: true,
    };
    assert_eq!(
        stagate_peak_bytes(&stagate),
        STAGATE_PROCESS_BYTES + read_bytes(nonzeros)
    );
    let spatialde2 = Spatialde2Shape {
        cells: 10,
        nonzeros,
        selected_nonzeros: 0,
        genes: 1,
        edges: 0,
        regions: 1,
        gram: 0,
    };
    assert_eq!(
        spatialde2_peak_bytes(&spatialde2),
        SPATIALDE2_PROCESS_BYTES + read_bytes(nonzeros)
    );
    assert!(read_bytes(nonzeros) > 12 * nonzeros);
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

/// The budget `bytes` sets ([`MEMORY_BUDGET_VAR`]).
fn set(bytes: usize) -> Budget {
    Budget::new(Some(bytes), None)
}

/// Unset, the budget is half of the memory available when it was taken
/// (before the data were read), not of what is left when a check runs.
#[test]
fn an_unset_budget_is_half_the_memory_available_when_taken() {
    let gib = 1usize << 30;
    let budget = Budget::new(None, Some(8 * gib));
    let e = ensure_fits("run", 4 * gib + 1, &budget, "lower it")
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("4.00 GiB") && e.contains("half the available"),
        "{e}"
    );
    ensure_fits("run", 4 * gib, &budget, "lower it").unwrap();
    assert!(set(1000).to_string().contains(MEMORY_BUDGET_VAR));
}

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
    let run = |epochs: &str, budget: usize| {
        let args = stagate_args(&out, &["--n-epochs", epochs]);
        stagate::run_on(&args, tissue(6), &set(budget))
    };
    let e = run("1000000000", 1000).unwrap_err().to_string();
    for want in [
        "estimated",
        "budget",
        "--n-top-genes",
        "PINTO_MEMORY_BUDGET",
    ] {
        assert!(e.contains(want), "{want} missing from: {e}");
    }
    // a GPU is suggested only where this build can train on one
    assert_eq!(e.contains("--device cuda"), cfg!(feature = "cuda"), "{e}");
    assert!(!dir.path().join("st.labels.tsv").exists());
    run("3", usize::MAX).unwrap();
    assert!(dir.path().join("st.labels.tsv").exists());
}

/// A run already on a GPU is not told to train on one.
#[test]
fn stagate_on_a_gpu_is_not_told_to_use_cuda() {
    let dir = tempfile::tempdir().unwrap();
    for device in ["cuda", "metal"] {
        let args = stagate_args(&dir.path().join("st"), &["--device", device]);
        let e = stagate::run_on(&args, tissue(6), &set(1000))
            .unwrap_err()
            .to_string();
        assert!(e.contains("--n-top-genes"), "{e}");
        assert!(!e.contains("--device cuda"), "{device}: {e}");
    }
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
    let budget = stagate_peak_bytes(&loops_only) + 1;
    let args = stagate_args(
        &dir.path().join("st"),
        &["--rad-cutoff", "1.5", "--n-epochs", "1"],
    );
    let e = stagate::run_on(&args, data, &set(budget)).unwrap_err();
    assert!(e.to_string().contains("estimated"), "{e}");
}

#[test]
fn spatialde2_refuses_a_run_over_the_budget_before_fitting() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sd");
    let run = |budget: usize| {
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
        spatialde2::run_on(&args, tissue(6), &set(budget))
    };
    let e = run(1000).unwrap_err().to_string();
    for want in ["estimated", "budget", "--n-clusters", "PINTO_MEMORY_BUDGET"] {
        assert!(e.contains(want), "{want} missing from: {e}");
    }
    assert!(!dir.path().join("sd.labels.tsv").exists());
    run(usize::MAX).unwrap();
    assert!(dir.path().join("sd.labels.tsv").exists());
}

// ------------------------------------------------- an srt-simulate read

/// A `side × side` srt-simulate section at the measured runs' settings
/// (2000 genes, one section), written under `dir`: its data and
/// coordinate files.
fn simulated(dir: &std::path::Path, side: usize) -> (String, String) {
    let out = dir.join("sim").to_string_lossy().to_string();
    crate::simulate::run::run_simulate(&crate::simulate::run::SimulateArgs {
        scenario: "L0".into(),
        set: [
            format!("grid_side={side}"),
            "n_genes=2000".into(),
            "n_donors=1".into(),
            "sections_per_donor=1".into(),
            "seed=1".into(),
        ]
        .into(),
        coords: Vec::new(),
        coord_column_names: vec!["row_um".into(), "col_um".into()],
        out: out.as_str().into(),
    })
    .unwrap();
    (
        format!("{out}/sections/S1.zarr"),
        format!("{out}/sections/S1.coords.parquet"),
    )
}

fn on_files(
    cmd: &str,
    (data, coord): &(String, String),
    out: &std::path::Path,
    extra: &[&str],
) -> crate::Commands {
    let mut argv = vec![
        "pinto",
        cmd,
        data.as_str(),
        "-c",
        coord.as_str(),
        "--coord-column-names",
        "row_um,col_um",
        "-o",
        out.to_str().unwrap(),
    ];
    argv.extend(extra);
    parse(&argv)
}

/// The check before the read counts every stored nonzero and as many genes
/// as could be kept, so a budget of exactly its estimate passes the checks
/// after the read too.
#[test]
fn the_check_before_the_read_bounds_the_checks_after_it() {
    let dir = tempfile::tempdir().unwrap();
    let files = simulated(dir.path(), 8);
    let out = dir.path().join("run");
    let st = [
        "--n-epochs",
        "1",
        "--hidden-dims",
        "8,2",
        "--n-clusters",
        "2",
    ];
    let crate::Commands::Stagate(args) = on_files("stagate", &files, &out, &st) else {
        unreachable!()
    };
    let input = args.input.open().unwrap();
    let bound = stagate::input_peak_bytes(&args, &input).unwrap();
    stagate::run_on(&args, input.read().unwrap(), &set(bound)).unwrap();

    let crate::Commands::Spatialde2(args) =
        on_files("spatialde2", &files, &out, &["--neighbors", "4"])
    else {
        unreachable!()
    };
    let input = args.input.open().unwrap();
    let bound = spatialde2::input_peak_bytes(&args, &input).unwrap();
    spatialde2::run_on(&args, input.read().unwrap(), &set(bound)).unwrap();
}

/// Opening the input preloads no counts, `--preload-data` or not, so the
/// check from the stored shape runs before the counts are in memory.
#[test]
fn opening_the_input_preloads_no_counts() {
    let dir = tempfile::tempdir().unwrap();
    let files = simulated(dir.path(), 8);
    let out = dir.path().join("run");
    for extra in [&[][..], &["--preload-data"][..]] {
        let crate::Commands::Spatialde2(args) = on_files("spatialde2", &files, &out, extra) else {
            unreachable!()
        };
        let input = args.input.open().unwrap();
        assert!(!input.preloaded(), "{extra:?}");
        assert_eq!(input.read().unwrap().cells.len(), 64);
    }
}

/// The measured runs' shapes, from an srt-simulate read at their settings:
/// the nonzeros per cell [`NONZEROS_PER_CELL`] takes, and every gene kept
/// (spatialde2's selection holds every nonzero; stagate's highly variable
/// genes are all 2000).
#[test]
fn srt_simulate_reads_as_the_measured_runs_assume() {
    let dir = tempfile::tempdir().unwrap();
    let files = simulated(dir.path(), 32);
    let crate::Commands::Spatialde2(args) =
        on_files("spatialde2", &files, &dir.path().join("x"), &[])
    else {
        unreachable!()
    };
    let data = args.input.open().unwrap().read().unwrap();
    let per_cell = data.counts.nnz() as f64 / data.cells.len() as f64;
    let want = NONZEROS_PER_CELL as f64;
    assert!(
        (per_cell - want).abs() < 0.03 * want,
        "{per_cell} nonzeros per cell"
    );
    let top = spatialde2::top_genes(&data, args.n_top_genes);
    assert_eq!(top.nonzeros, data.counts.nnz());
    let (hvg, _) = stagate::seurat_v3_hvg(&data.counts, 3000);
    assert_eq!((top.genes.len(), hvg.len()), (2000, 2000));
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
