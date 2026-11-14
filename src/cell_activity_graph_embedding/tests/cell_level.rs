//! `--cell-level`: cells are the trained unit and nothing is pseudobulked.

use crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs;
use crate::cell_activity_graph_embedding::fit::{
    cell_level_hierarchy, fit_cell_activity_graph_embedding,
};
use crate::test_support::write_zarr;
use crate::util::common::Mat;
use clap::Parser;
use legume_numeric::matrix::traits::IoOps;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    cage: CellActivityGraphEmbeddingArgs,
}

fn parse(argv: &[&str]) -> Result<CellActivityGraphEmbeddingArgs, clap::Error> {
    Cli::try_parse_from(["x"].into_iter().chain(argv.iter().copied())).map(|c| c.cage)
}

#[test]
fn the_hierarchy_is_the_batch_then_the_cell() {
    let batches: Vec<Box<str>> = ["a", "a", "b", "b", "a"].map(Box::from).to_vec();
    let pairs = [(0, 1), (2, 3), (1, 4)];
    let ml = cell_level_hierarchy(&batches, &pairs);
    assert_eq!(
        ml.all_cell_labels,
        vec![vec![0, 0, 1, 1, 0], vec![0, 1, 2, 3, 4]]
    );
    // coarse: pairs within batch a share a sample; finest: one per pair
    assert_eq!(ml.all_num_samples, vec![2, 3]);
    assert_eq!(ml.all_pair_to_sample[0][0], ml.all_pair_to_sample[0][2]);
    assert_ne!(ml.all_pair_to_sample[0][0], ml.all_pair_to_sample[0][1]);
}

#[test]
fn cell_level_refuses_chain_levels() {
    let base = ["-o", "out", "d.zarr", "--cell-level"];
    assert!(parse(&base).is_ok());
    assert!(parse(&[&base[..], &["--chain-levels", "0"]].concat()).is_err());
}

/// Two sections of a `side × side` grid, the left and right halves
/// expressing different genes; returns the data and coordinate files.
fn two_sections(dir: &std::path::Path, side: usize) -> (String, String) {
    let genes: Vec<String> = (0..12).map(|g| format!("G{g}")).collect();
    let genes: Vec<&str> = genes.iter().map(String::as_str).collect();
    let (mut data, mut coords) = (Vec::new(), Vec::new());
    for tag in ["s1", "s2"] {
        let n = side * side;
        let mut triplets = Vec::new();
        let mut csv = String::from("cell,x,y\n");
        for c in 0..n {
            let (r, col) = (c / side, c % side);
            let left = col < side / 2;
            for g in 0..12u64 {
                let marker = if left { g < 6 } else { g >= 6 };
                let v = if marker {
                    4.0 + ((c + g as usize) % 3) as f32
                } else {
                    ((c * 7 + g as usize) % 2) as f32
                };
                if v > 0.0 {
                    triplets.push((g, c as u64, v));
                }
            }
            csv.push_str(&format!("{tag}_{c},{col},{r}\n"));
        }
        data.push(write_zarr(dir, tag, &genes, n, &triplets).to_string());
        let path = dir.join(format!("{tag}.coords.csv"));
        std::fs::write(&path, csv).unwrap();
        coords.push(path.to_str().unwrap().to_string());
    }
    (data.join(","), coords.join(","))
}

fn fit(dir: &std::path::Path, data: &str, coords: &str, name: &str, extra: &[&str]) -> String {
    let out = dir.join(name).to_str().unwrap().to_string();
    let argv: Vec<&str> = [
        data,
        "-c",
        coords,
        "--coord-column-names",
        "x,y",
        "-o",
        &out,
        "--epochs",
        "2",
        "--embedding-dim",
        "4",
        "-d",
        "24",
        "--num-levels",
        "3",
        "--n-edge-clusters",
        "2",
        "--edge-cluster-method",
        "kmeans",
    ]
    .into_iter()
    .chain(extra.iter().copied())
    .collect();
    fit_cell_activity_graph_embedding(&parse(&argv).unwrap()).unwrap();
    out
}

#[test]
fn cell_level_trains_on_cells_and_builds_no_pseudobulks() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    let n_cells = 2 * 8 * 8;

    let cell = fit(dir.path(), &data, &coords, "cell", &["--cell-level"]);
    let trained = Mat::from_parquet(&format!("{cell}.pb_embedding.parquet")).unwrap();
    assert_eq!(trained.mat.nrows(), n_cells, "one trained row per cell");
    for skipped in ["delta", "pseudobulk_cells"] {
        assert!(
            !std::path::Path::new(&format!("{cell}.{skipped}.parquet")).exists(),
            "--cell-level wrote {skipped}.parquet"
        );
    }
    assert!(std::path::Path::new(&format!("{cell}.pinto.json")).exists());

    // the same data without the flag trains on pseudobulks, and builds both
    let pb = fit(dir.path(), &data, &coords, "pb", &[]);
    let trained = Mat::from_parquet(&format!("{pb}.pb_embedding.parquet")).unwrap();
    assert!(
        trained.mat.nrows() < n_cells,
        "the default trains on super-cells"
    );
    for built in ["delta", "pseudobulk_cells"] {
        assert!(std::path::Path::new(&format!("{pb}.{built}.parquet")).exists());
    }
}
