//! A baseline reads its input as lc, dsvd and cage read theirs: masked
//! cells (and QC's) left out of the fit but kept in the cells table, and
//! the ablation recorded in the manifest.

use crate::test_support::two_sections;
use crate::util::metadata::Run;
use std::path::Path;

/// `pinto fit spatialde2` on `data` and `coords` into `out`, with `extra`,
/// as parsed.
fn spatialde2_args(
    data: &str,
    coords: &str,
    out: &str,
    extra: &[&str],
) -> anyhow::Result<crate::baselines::spatial::spatialde2::Args> {
    let mut argv = vec![
        "pinto",
        "fit",
        "spatialde2",
        data,
        "-c",
        coords,
        "--coord-column-names",
        "x,y",
        "-o",
        out,
        "--max-iter",
        "5",
    ];
    argv.extend_from_slice(extra);
    match <crate::Cli as clap::Parser>::try_parse_from(argv)?.commands {
        crate::Commands::Fit(crate::Fit::Spatialde2(a)) => Ok(*a),
        _ => unreachable!(),
    }
}

/// [`spatialde2_args`] run.
fn spatialde2(data: &str, coords: &str, out: &str, extra: &[&str]) -> anyhow::Result<()> {
    crate::baselines::spatial::spatialde2::run(&spatialde2_args(data, coords, out, extra)?)
}

fn labelled(out: &str) -> usize {
    std::fs::read_to_string(format!("{out}.labels.tsv"))
        .unwrap()
        .lines()
        .count()
        - 1
}

#[test]
fn masked_cells_leave_the_fit_and_stay_in_the_cells_table() {
    let dir = tempfile::tempdir().unwrap();
    let side = 6;
    let (data, coords) = two_sections(dir.path(), side);
    let list = dir.path().join("masked.txt");
    std::fs::write(&list, "s1_0@s1\ns2_3@s2\n").unwrap();
    let out = dir.path().join("masked").to_string_lossy().into_owned();
    spatialde2(
        &data,
        &coords,
        &out,
        &["--no-qc", "--mask-cells", list.to_str().unwrap()],
    )
    .unwrap();
    let n = 2 * side * side;
    assert_eq!(labelled(&out), n - 2);
    let run = Run::open(&out).unwrap();
    assert!(run.meta.ablation.is_some(), "the mask is recorded");
    let table = crate::util::parquet_io::read_cells_table(
        Path::new(run.meta.outputs.cells.as_deref().unwrap()),
        None,
    )
    .unwrap();
    assert_eq!(table.names.len(), n, "every cell read is listed");
    #[cfg(feature = "view")]
    assert_eq!(
        table.in_graph.unwrap().iter().filter(|&&k| !k).count(),
        2,
        "the masked cells are not in the graph"
    );
}

#[test]
fn a_permutation_moves_the_positions_the_method_reads() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 6);
    let read = |name: &str, extra: &[&str]| {
        let out = dir.path().join(name).to_string_lossy().into_owned();
        let extra = [&["--no-qc"][..], extra].concat();
        spatialde2_args(&data, &coords, &out, &extra)
            .unwrap()
            .fit
            .input
            .read_within(false, &crate::util::memory::Budget::new(None, None), 0)
            .unwrap()
    };
    let own = read("own", &[]);
    let moved = read("moved", &["--permute-coords", "within-batch"]);
    assert_eq!(own.cells, moved.cells);
    assert_ne!(own.xy, moved.xy, "the method reads the scrambled positions");
    let sorted = |m: &crate::util::common::Mat| {
        let mut v: Vec<(i64, i64)> = (0..m.nrows())
            .map(|i| (m[(i, 0)] as i64, m[(i, 1)] as i64))
            .collect();
        v.sort_unstable();
        v
    };
    assert_eq!(
        sorted(&own.xy),
        sorted(&moved.xy),
        "the same positions, moved"
    );
    assert!(moved.ablation.is_some());
}
