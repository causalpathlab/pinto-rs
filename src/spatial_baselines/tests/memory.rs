//! The peak memory estimates, against peaks measured on full runs, and the
//! refusal of a run they say would not fit.

use super::graph::grid;
use crate::spatial_baselines::input::InputShape;
use crate::spatial_baselines::memory::*;
use crate::spatial_baselines::{spatialde2, stagate};
use crate::util::batch_effects::{delta_estimate_bytes, DELTA_SAMPLES_PER_BATCH};
use crate::util::memory::{ensure_fits, Budget, MEMORY_BUDGET_VAR};
use crate::util::metadata::InputLayout;
use nalgebra_sparse::{CooMatrix, CscMatrix};

const MIB: f64 = (1u64 << 20) as f64;

/// Maximum resident memory (MiB) of full CPU runs on srt-simulate square
/// grids at the defaults (2000 genes, all trained on; for stagate hidden
/// width 512 and 6 neighbours, its dense input, now `--reference`; for
/// spatialde2 100 neighbours): (cells per
/// section, sections, threads, MiB).
const STAGATE_MEASURED: [(usize, usize, usize, f64); 9] = [
    (1024, 1, 10, 426.19),
    (4096, 1, 10, 1394.23),
    (8100, 1, 10, 2699.27),
    (10000, 1, 10, 3492.0),
    (10000, 1, 4, 3280.91),
    (14400, 1, 4, 4691.03),
    (16384, 1, 4, 5264.30),
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
/// Maximum resident memory (MiB) of spatialde2 runs whose peak is the δ
/// estimate (two srt-simulate L1 sections, the fit cut to nothing:
/// `--neighbors 0 --max-iter 1 --n-clusters 2`, 4 threads): (cells per
/// section, genes, nonzeros, MiB).
const DELTA_MEASURED: [(usize, usize, usize, f64); 5] = [
    (256, 500, 100_906, 82.69),
    (1024, 2000, 1_584_782, 566.12),
    (4096, 500, 1_615_747, 261.30),
    (4096, 2000, 6_360_230, 783.44),
    (8100, 2000, 12_581_100, 1079.61),
];

/// The input arguments of a spatialde2 run with `extra`.
fn input_args(extra: &[&str]) -> crate::spatial_baselines::input::SpatialInputArgs {
    let mut argv = vec!["pinto", "spatialde2", "d.zarr", "-c", "c.csv", "-o", "x"];
    argv.extend(extra);
    let crate::Commands::Spatialde2(args) = parse(&argv) else {
        unreachable!()
    };
    args.fit.input
}

/// The δ estimate's peak covers every measured run whose peak it is, and
/// is not far above it.
#[test]
fn delta_estimate_covers_every_measured_peak_and_is_not_far_above() {
    for (per, genes, nonzeros, measured) in DELTA_MEASURED {
        let shape = InputShape {
            genes,
            nonzeros,
            section_sizes: vec![per; 2],
            batches: 2,
        };
        let args = input_args(&[]);
        let est = mib(SPATIALDE2_PROCESS_BYTES + args.delta_bytes(&shape));
        assert!(
            est >= measured,
            "{per}x2, {genes} genes: {est:.0} < {measured} MiB"
        );
        assert!(
            est <= 1.25 * measured,
            "{per}x2, {genes} genes: {est:.0} MiB"
        );
        let ignored = input_args(&["--ignore-batch"]);
        assert_eq!(ignored.delta_bytes(&shape), 0);
        let one = InputShape {
            section_sizes: vec![per],
            batches: 1,
            ..shape
        };
        assert_eq!(args.delta_bytes(&one), 0, "one file, no batch files");
    }
}

/// With batch files the δ estimate holds at most
/// [`DELTA_SAMPLES_PER_BATCH`] samples in each section the stored shape
/// lists, not one per cell: a large input split by batch files passes a
/// check its δ fits, and its estimate is that of the same sections without
/// them.
#[test]
fn a_large_input_with_batch_files_is_charged_its_delta_samples() {
    let shape = InputShape {
        genes: 2000,
        nonzeros: 400_000 * 800,
        section_sizes: vec![100_000; 4],
        batches: 4,
    };
    let with = input_args(&["-b", "a.batch"]);
    let without = input_args(&[]);
    assert_eq!(with.delta_bytes(&shape), without.delta_bytes(&shape));
    assert_eq!(
        with.delta_bytes(&shape),
        delta_estimate_bytes(2000, shape.nonzeros, 4 * DELTA_SAMPLES_PER_BATCH)
    );
    let budget = set(2 * (SPATIALDE2_PROCESS_BYTES + without.delta_bytes(&shape)));
    let est = SPATIALDE2_PROCESS_BYTES + with.delta_bytes(&shape);
    check_stored_shape("spatialde2", Some(est), &budget, "lower it").unwrap();
    let one = InputShape {
        section_sizes: vec![400_000],
        batches: 1,
        ..shape
    };
    assert_eq!(with.delta_bytes(&one), 0, "one section has one batch");
}

/// Maximum resident memory (MiB) of sparse (default) stagate runs on one
/// srt-simulate section at the defaults (2000 genes, all of them highly
/// variable; hidden width 512, 6 neighbours; 20 epochs, which peak as 1000
/// do), 4 threads: (cells, nonzeros, MiB).
const STAGATE_SPARSE_MEASURED: [(usize, usize, f64); 4] = [
    (1024, 788_258, 398.84),
    (4096, 3_156_044, 1106.34),
    (8100, 6_238_351, 2067.0),
    (16384, 12_572_357, 4162.81),
];

/// The sparse estimate covers every measured sparse run, and from 4096
/// cells is not far above it; the dense estimate of the same input is
/// larger (the dense input and reconstruction are gone).
#[test]
fn stagate_sparse_estimate_covers_every_measured_peak_and_is_not_far_above() {
    for (cells, nonzeros, measured) in STAGATE_SPARSE_MEASURED {
        let sizes = stagate::StagateSizes {
            layout: InputLayout::Sparse,
            ..STAGATE_MEASURED_SIZES
        };
        let mut shape = sizes.memory_shape(&InputShape {
            genes: 2000,
            nonzeros,
            section_sizes: vec![cells],
            batches: 1,
        });
        shape.gram = knn_gram_bytes(&[cells], 4);
        let est = mib(stagate_peak_bytes(&shape));
        assert!(est >= measured, "{cells}: {est:.0} < {measured} MiB");
        if cells >= 4096 {
            assert!(est <= 1.35 * measured, "{cells}: {est:.0} MiB");
        }
        let dense = StagateShape {
            sparse: false,
            ..shape
        };
        assert!(stagate_peak_bytes(&dense) > stagate_peak_bytes(&shape));
    }
}

/// Maximum resident memory (MiB) of spatialde2-pb on one srt-simulate
/// section (2000 genes, `-d 1024`, at the defaults otherwise, 4 threads):
/// (cells, nonzeros, MiB).
const PSEUDOBULK_MEASURED: [(usize, usize, f64); 4] = [
    (1024, 788_258, 136.20),
    (4096, 3_156_044, 466.98),
    (8100, 6_238_351, 695.89),
    (16384, 12_572_357, 1316.26),
];

/// spatialde2-pb's estimate, its cell-level fit's and the coarsening's,
/// covers every measured run (it fits on super-cells, but unravels on
/// every cell).
#[test]
fn pseudobulk_estimate_covers_every_measured_peak() {
    for (cells, nonzeros, measured) in PSEUDOBULK_MEASURED {
        let input = InputShape {
            genes: 2000,
            nonzeros,
            section_sizes: vec![cells],
            batches: 1,
        };
        let sd = spatialde2::Spatialde2Sizes {
            neighbors: 100,
            n_clusters: None,
            all_pairs: false,
        }
        .memory_shape(&input, nonzeros);
        let est = mib(spatialde2_peak_bytes(&sd) + pseudobulk_bytes(cells, nonzeros, sd.edges));
        assert!(
            est >= measured,
            "spatialde2-pb {cells}: {est:.0} < {measured}"
        );
    }
}

/// Nonzeros per cell of those runs' counts, as srt-simulate draws them at
/// those settings ([`srt_simulate_reads_as_the_measured_runs_assume`]).
const NONZEROS_PER_CELL: usize = 765;

/// A measured run's shape, its kNN's Gram blocks on `threads` threads.
/// The measured STAGATE runs' settings: 6 neighbours, 512 wide, on the CPU.
const STAGATE_MEASURED_SIZES: stagate::StagateSizes = stagate::StagateSizes {
    k_cutoff: 6,
    rad_cutoff: None,
    hidden: 512,
    train_on_host: true,
    layout: InputLayout::Dense,
};

fn stagate_shape(per_section: usize, sections: usize, threads: usize) -> StagateShape {
    let sizes = vec![per_section; sections];
    let nonzeros = NONZEROS_PER_CELL * per_section * sections;
    StagateShape {
        gram: knn_gram_bytes(&sizes, threads),
        ..STAGATE_MEASURED_SIZES.memory_shape(&InputShape {
            genes: 2000,
            nonzeros,
            section_sizes: sizes,
            batches: sections,
        })
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
    let on_cuda = stagate_peak_bytes(&cuda);
    assert!(on_cuda < peak);
    // training's edges and width are on the device, not the host
    let wider = StagateShape {
        edges: 10 * cuda.edges,
        hidden: 4 * cuda.hidden,
        ..cuda.clone()
    };
    assert_eq!(stagate_peak_bytes(&wider), on_cuda);
    // the graph's search is the host's peak: the counts are freed once
    // the dense input is built, before it
    let more_counts = StagateShape {
        nonzeros: cuda.nonzeros + 1000,
        ..cuda.clone()
    };
    assert_eq!(stagate_peak_bytes(&more_counts), on_cuda);
    let searched = StagateShape {
        searched: cuda.searched + 1000,
        ..cuda.clone()
    };
    assert_eq!(stagate_peak_bytes(&searched) - on_cuda, 28 * 1000);
}

/// A measured run's shape, its kNN's Gram blocks on `threads` threads.
fn spatialde2_shape(per_section: usize, sections: usize, threads: usize) -> Spatialde2Shape {
    let sizes = vec![per_section; sections];
    let nonzeros = NONZEROS_PER_CELL * per_section * sections;
    Spatialde2Shape {
        gram: knn_gram_bytes(&sizes, threads),
        ..spatialde2::Spatialde2Sizes {
            neighbors: 100,
            n_clusters: None,
            all_pairs: false,
        }
        .memory_shape(
            &InputShape {
                genes: 2000,
                nonzeros,
                section_sizes: sizes,
                batches: sections,
            },
            nonzeros,
        )
    }
}

/// Maximum resident memory (MiB) of `spatialde2 --reference` runs on one
/// srt-simulate L1 section (2000 genes, every gene segmented on, 4
/// threads): (cells, nonzeros, MiB).
const SPATIALDE2_REFERENCE_MEASURED: [(usize, usize, f64); 3] = [
    (1024, 788_258, 80.22),
    (4096, 3_156_044, 238.45),
    (8100, 6_238_351, 434.44),
];

/// `--reference` holds no edge list and no kNN search: its estimate
/// covers every measured run and is not far above it.
#[test]
fn spatialde2_reference_holds_no_graph_and_its_estimate_covers_the_measured_runs() {
    for (cells, nonzeros, measured) in SPATIALDE2_REFERENCE_MEASURED {
        let shape = spatialde2::Spatialde2Sizes {
            neighbors: 100,
            n_clusters: None,
            all_pairs: true,
        }
        .memory_shape(
            &InputShape {
                genes: 2000,
                nonzeros,
                section_sizes: vec![cells],
                batches: 1,
            },
            nonzeros,
        );
        assert_eq!((shape.edges, shape.gram, shape.pair_cells), (0, 0, cells));
        let est = mib(spatialde2_peak_bytes(&shape));
        assert!(est >= measured, "{cells}: {est:.0} < {measured} MiB");
        assert!(est <= 1.5 * measured, "{cells}: {est:.0} MiB");
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
        sparse: false,
        selected_nonzeros: 0,
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
        pair_cells: 0,
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
    let mut d = grid(side, 2);
    let genes = 12;
    let mut coo = CooMatrix::<f32>::new(genes, d.cells.len());
    for i in 0..d.cells.len() {
        let (r, c) = (d.xy[(i, 0)] as usize, d.xy[(i, 1)] as usize);
        for g in 0..genes {
            let v = ((r * (g + 1) + c * (g % 3) + i) % 7) as f32;
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

/// The sparse formulation is the CPU path: on a GPU the input is held
/// dense, and the estimate charges it so; the layout is recorded apart
/// from the mode, which is `--reference`'s alone.
#[test]
fn stagate_holds_its_input_dense_off_the_cpu() {
    let dir = tempfile::tempdir().unwrap();
    for (extra, dense, reference) in [
        (&["--device", "cpu"][..], false, false),
        (&["--device", "cpu", "--reference"][..], true, true),
        (&["--device", "cuda"][..], true, false),
        (&["--device", "metal"][..], true, false),
    ] {
        let args = stagate_args(&dir.path().join("st"), extra);
        assert_eq!(args.reference, reference, "{extra:?}");
        let layout = match dense {
            true => InputLayout::Dense,
            false => InputLayout::Sparse,
        };
        assert_eq!(args.input_layout(), layout, "{extra:?}");
        assert_eq!(args.sizes().layout, layout, "{extra:?}");
    }
}

/// A CPU run records `baseline.mode` from `--reference` and the layout
/// held as `input_layout`, in the manifest and `baseline.json`'s `run`.
#[test]
fn stagate_records_its_mode_and_layout_apart() {
    use crate::util::metadata::{BaselineMode, PintoMetadata};
    let dir = tempfile::tempdir().unwrap();
    for (tag, extra, mode, layout) in [
        ("s", &[][..], BaselineMode::Sparse, InputLayout::Sparse),
        (
            "r",
            &["--reference"][..],
            BaselineMode::Reference,
            InputLayout::Dense,
        ),
    ] {
        let out = dir.path().join(tag);
        let mut flags = vec!["--n-epochs", "2"];
        flags.extend(extra);
        stagate::run_on(&stagate_args(&out, &flags), tissue(6), &set(usize::MAX)).unwrap();
        let prefix = out.to_string_lossy();
        let meta: PintoMetadata =
            serde_json::from_str(&std::fs::read_to_string(format!("{prefix}.pinto.json")).unwrap())
                .unwrap();
        let run = meta.baseline.unwrap();
        assert_eq!(run.mode, mode, "{tag}");
        assert_eq!(run.input_layout, Some(layout), "{tag}");
        let settings: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{prefix}.baseline.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            settings["run"]["input_layout"],
            serde_json::to_value(layout).unwrap(),
            "{tag}"
        );
    }
}

/// A radius graph's edges are counted once it is built: a budget that
/// holds its self loops alone refuses it then.
#[test]
fn stagate_counts_a_radius_graph_once_built() {
    let dir = tempfile::tempdir().unwrap();
    let data = tissue(6);
    let loops_only = stagate::StagateSizes {
        k_cutoff: 6,
        rad_cutoff: Some(1.5),
        hidden: 8,
        train_on_host: true,
        layout: InputLayout::Dense,
    }
    .memory_shape(&InputShape {
        genes: 8,
        nonzeros: data.counts.nnz(),
        section_sizes: vec![36, 36],
        batches: 2,
    });
    assert_eq!(loops_only.edges, data.cells.len());
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
    let shape = args.input.stored().unwrap().unwrap();
    let bound = stagate::input_peak_bytes(&args, &shape);
    stagate::run_on(&args, args.input.read(false).unwrap(), &set(bound)).unwrap();

    let crate::Commands::Spatialde2(args) =
        on_files("spatialde2", &files, &out, &["--neighbors", "4"])
    else {
        unreachable!()
    };
    let shape = args.fit.input.stored().unwrap().unwrap();
    let bound = spatialde2::input_peak_bytes(&args.fit, args.variant(), &shape);
    spatialde2::run_on(&args, args.fit.input.read(false).unwrap(), &set(bound)).unwrap();
}

/// The stored shape of one file, taken from its metadata before any count
/// is read (`--preload-data` or not), is the shape the read then gives.
#[test]
fn one_file_s_stored_shape_is_its_read_shape() {
    let dir = tempfile::tempdir().unwrap();
    let files = simulated(dir.path(), 8);
    let out = dir.path().join("run");
    for extra in [&[][..], &["--preload-data"][..]] {
        let crate::Commands::Spatialde2(args) = on_files("spatialde2", &files, &out, extra) else {
            unreachable!()
        };
        let shape = args.fit.input.stored().unwrap().unwrap();
        let data = args.fit.input.read(false).unwrap();
        assert_eq!(
            shape,
            InputShape {
                genes: data.genes.len(),
                nonzeros: data.counts.nnz(),
                section_sizes: vec![64],
                batches: 1,
            },
            "{extra:?}"
        );
    }
}

/// The files the stored shape is taken from are opened with no counts in
/// memory, `--preload-data` or not, so the check on it comes before any.
#[test]
fn the_stored_shape_is_taken_with_no_counts_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let files = simulated(dir.path(), 8);
    let out = dir.path().join("run");
    for extra in [&[][..], &["--preload-data"][..]] {
        let crate::Commands::Spatialde2(args) = on_files("spatialde2", &files, &out, extra) else {
            unreachable!()
        };
        for file in args.fit.input.metadata() {
            assert!(file.unwrap().csc_column_arrays().is_none(), "{extra:?}");
        }
    }
}

/// No data files store no shape: the check from it is skipped.
#[test]
fn no_data_files_have_no_stored_shape() {
    let crate::Commands::Spatialde2(mut args) =
        parse(&["pinto", "spatialde2", "d.zarr", "-c", "c.csv", "-o", "x"])
    else {
        unreachable!()
    };
    args.fit.input.data_files.clear();
    assert_eq!(args.fit.input.stored().unwrap(), None);
}

/// Data files `inputs`, each a (tag, rows, cells) with every count 1 and
/// its cells on a line one apart, and `labels` (a batch label per cell of
/// each file; `None` for no batch files): the arguments naming the data,
/// coordinate and batch files.
fn several_files(
    dir: &std::path::Path,
    inputs: &[(&str, &[&str], usize)],
    labels: Option<&[&[&str]]>,
) -> Vec<String> {
    use std::io::Write as _;
    let (mut data, mut coords, mut batches) = (Vec::new(), Vec::new(), Vec::new());
    for (f, &(tag, rows, cells)) in inputs.iter().enumerate() {
        let triplets: Vec<(u64, u64, f32)> = (0..rows.len() as u64)
            .flat_map(|r| (0..cells as u64).map(move |c| (r, c, 1.0)))
            .collect();
        data.push(crate::test_support::write_zarr(dir, tag, rows, cells, &triplets).to_string());
        let path = dir.join(format!("{tag}.csv"));
        let mut out = std::fs::File::create(&path).unwrap();
        writeln!(out, "barcode,x,y").unwrap();
        for c in 0..cells {
            writeln!(out, "{tag}_{c},{c},0").unwrap();
        }
        coords.push(path.to_string_lossy().to_string());
        if let Some(labels) = labels {
            let path = dir.join(format!("{tag}.batch"));
            std::fs::write(&path, labels[f].join("\n") + "\n").unwrap();
            batches.push(path.to_string_lossy().to_string());
        }
    }
    let mut argv = vec![data.join(","), "-c".into(), coords.join(",")];
    if labels.is_some() {
        argv.extend(["-b".into(), batches.join(",")]);
    }
    argv
}

/// On files with different rows, and files a batch file splits into
/// several sections, the stored shape bounds the read's on every field an
/// estimate takes: the genes the files share (exactly those the read
/// keeps), the nonzeros, and the sections' kNN edges and Gram blocks; its
/// sections are the read's, batch files read for them.
#[test]
fn the_stored_shape_of_several_files_bounds_their_read_shape() {
    let dir = tempfile::tempdir().unwrap();
    let inputs: [(&str, &[&str], usize); 2] = [
        ("a", &["GENE1", "GENE2", "GENE3"], 6),
        ("b", &["GENE3", "GENE1", "GENE4"], 4),
    ];
    let split: [&[&str]; 2] = [&["L1", "L1", "L1", "L2", "L2", "L2"], &["L1"; 4]];
    for labels in [None, Some(&split[..])] {
        let files = several_files(dir.path(), &inputs, labels);
        let out = dir.path().join("run").to_string_lossy().to_string();
        let mut argv = vec!["pinto", "spatialde2"];
        argv.extend(files.iter().map(String::as_str));
        argv.extend(["--coord-column-names", "x,y", "-o", &out]);
        let crate::Commands::Spatialde2(args) = parse(&argv) else {
            unreachable!()
        };
        let shape = args.fit.input.stored().unwrap().unwrap();
        let data = args.fit.input.read(false).unwrap();
        let read = crate::util::ablation::section_sizes(&data.section);
        assert_eq!(shape.genes, data.genes.len(), "{labels:?}");
        assert!(shape.nonzeros >= data.counts.nnz(), "{labels:?}");
        assert_eq!(shape.section_sizes.iter().sum::<usize>(), data.cells.len());
        assert_eq!(shape.section_sizes, read, "{labels:?}");
        assert_eq!(shape.batches, data.n_batches(), "{labels:?}");
        for k in 1..=4 {
            let (stored, read) = (knn_edges(&shape.section_sizes, k), knn_edges(&read, k));
            assert!(stored >= read, "{labels:?}, k = {k}");
        }
        for threads in [1, 2, 8] {
            let stored = knn_gram_bytes(&shape.section_sizes, threads);
            assert!(stored >= knn_gram_bytes(&read, threads), "{labels:?}");
        }
    }
}

/// Several files whose batch files give them one shared label are one
/// batch: no δ is estimated, so none is charged, though each file is a
/// section of its own; without batch files each file is a batch, and
/// charged.
#[test]
fn files_sharing_one_batch_label_are_not_charged_a_delta() {
    let dir = tempfile::tempdir().unwrap();
    let inputs: [(&str, &[&str], usize); 3] = [
        ("a", &["GENE1", "GENE2"], 5),
        ("b", &["GENE1", "GENE2"], 4),
        ("c", &["GENE1", "GENE2"], 3),
    ];
    let shared: [&[&str]; 3] = [&["L1"; 5], &["L1"; 4], &["L1"; 3]];
    for (labels, batches) in [(Some(&shared[..]), 1), (None, 3)] {
        let files = several_files(dir.path(), &inputs, labels);
        let mut argv = vec!["pinto", "spatialde2"];
        argv.extend(files.iter().map(String::as_str));
        argv.extend(["--coord-column-names", "x,y", "-o", "x"]);
        let crate::Commands::Spatialde2(args) = parse(&argv) else {
            unreachable!()
        };
        let shape = args.fit.input.stored().unwrap().unwrap();
        assert_eq!(shape.section_sizes, vec![5, 4, 3], "{labels:?}");
        assert_eq!(shape.batches, batches, "{labels:?}");
        assert_eq!(
            args.fit.input.delta_bytes(&shape) == 0,
            batches == 1,
            "{labels:?}"
        );
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
    let data = args.fit.input.read(false).unwrap();
    let per_cell = data.counts.nnz() as f64 / data.cells.len() as f64;
    let want = NONZEROS_PER_CELL as f64;
    assert!(
        (per_cell - want).abs() < 0.03 * want,
        "{per_cell} nonzeros per cell"
    );
    let top = spatialde2::top_genes(&data, args.fit.n_top_genes);
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

/// Maximum resident memory (MiB) of full CPU runs of the factor baselines
/// on one srt-simulate section (2000 genes, all modelled; at their
/// defaults otherwise, `-d 1024` for the `-pb` forms, 4 threads): (cells,
/// MiB) for spatialpca, mefisto, spatialpca-pb and mefisto-pb.
const FACTOR_MEASURED: [(&str, [(usize, f64); 4]); 4] = [
    (
        "spatialpca",
        [
            (1024, 98.77),
            (4096, 265.45),
            (8100, 463.78),
            (16384, 850.14),
        ],
    ),
    (
        "mefisto",
        [
            (1024, 87.45),
            (4096, 254.97),
            (8100, 462.45),
            (16384, 839.08),
        ],
    ),
    (
        "spatialpca-pb",
        [
            (1024, 113.34),
            (4096, 368.91),
            (8100, 569.27),
            (16384, 1166.91),
        ],
    ),
    (
        "mefisto-pb",
        [
            (1024, 107.98),
            (4096, 383.13),
            (8100, 590.23),
            (16384, 1173.61),
        ],
    ),
];

/// The factor baselines' estimates cover every measured run, and from 4096
/// cells are not far above it.
#[test]
fn factor_estimates_cover_every_measured_peak_and_are_not_far_above() {
    use crate::spatial_baselines::{mefisto, spatialpca};
    for (cmd, runs) in FACTOR_MEASURED {
        for (cells, measured) in runs {
            let input = InputShape {
                genes: 2000,
                nonzeros: NONZEROS_PER_CELL * cells,
                section_sizes: vec![cells],
                batches: 1,
            };
            let argv = ["pinto", cmd, "d.zarr", "-c", "c.csv", "-o", "x"];
            let shape = match parse(&argv) {
                crate::Commands::Spatialpca(a) => {
                    a.fit
                        .memory_shape(&input, 2000, spatialpca::Variant::Sparse)
                }
                crate::Commands::SpatialpcaPb(a) => {
                    a.fit
                        .memory_shape(&input, 2000, spatialpca::Variant::Pseudobulk(1024))
                }
                crate::Commands::Mefisto(a) => {
                    a.fit
                        .memory_shape(&input, 2000, mefisto::Variant::Sparse, 0)
                }
                crate::Commands::MefistoPb(a) => {
                    a.fit
                        .memory_shape(&input, 2000, mefisto::Variant::Pseudobulk(1024), 0)
                }
                _ => unreachable!(),
            };
            let est = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap()
                .install(|| mib(factor_peak_bytes(&shape)));
            assert!(est >= measured, "{cmd} {cells}: {est:.0} < {measured} MiB");
            if cells >= 4096 {
                assert!(est <= 1.5 * measured, "{cmd} {cells}: {est:.0} MiB");
            }
        }
    }
}

/// Maximum resident memory (MiB) of a default `mefisto --reference` run
/// on the 1024-cell section below (one group: the eigenbasis search, its
/// bases kept), 4 threads.
const MEFISTO_REFERENCE_MIB: f64 = 508.82;

/// The `--reference` estimates cover the measured runs (one srt-simulate
/// section of 1024 cells, 2000 genes, 4 threads; MEFISTO's on its 1000
/// inducing cells): spatialpca 208.19 MiB, mefisto
/// [`MEFISTO_REFERENCE_MIB`]; and spatialpca on three sections of 400
/// cells (`--ignore-batch`), 249.16 MiB.
#[test]
fn factor_reference_estimates_cover_the_measured_runs() {
    use crate::spatial_baselines::{mefisto, spatialpca};
    let input = InputShape {
        genes: 2000,
        nonzeros: NONZEROS_PER_CELL * 1024,
        section_sizes: vec![1024],
        batches: 1,
    };
    let crate::Commands::Spatialpca(a) =
        parse(&["pinto", "spatialpca", "d.zarr", "-c", "c.csv", "-o", "x"])
    else {
        unreachable!()
    };
    let sp = a
        .fit
        .memory_shape(&input, 2000, spatialpca::Variant::Reference(None));
    let crate::Commands::Mefisto(m) =
        parse(&["pinto", "mefisto", "d.zarr", "-c", "c.csv", "-o", "x"])
    else {
        unreachable!()
    };
    let mf = m
        .fit
        .memory_shape(&input, 2000, mefisto::Variant::Reference, 1000);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let (est_sp, est_mf) =
        pool.install(|| (mib(factor_peak_bytes(&sp)), mib(factor_peak_bytes(&mf))));
    assert!(est_sp >= 208.19, "spatialpca --reference {est_sp:.0}");
    assert!(
        est_mf >= MEFISTO_REFERENCE_MIB,
        "mefisto --reference {est_mf:.0}"
    );
    // three sections of 400 cells, one kernel block each
    let three = InputShape {
        nonzeros: NONZEROS_PER_CELL * 1200,
        section_sizes: vec![400; 3],
        ..input
    };
    let sp3 = a
        .fit
        .memory_shape(&three, 2000, spatialpca::Variant::Reference(None));
    let est3 = pool.install(|| mib(factor_peak_bytes(&sp3)));
    assert!(
        est3 >= 249.16,
        "spatialpca --reference, 3 sections {est3:.0}"
    );
}
