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
fn every_cell_is_its_own_unit_at_both_levels() {
    let pairs = [(0, 1), (2, 3), (1, 4)];
    let ml = cell_level_hierarchy(5, &pairs);
    // the chain level is the identity too, so a negative is redrawn only
    // when it is the anchor itself (not whenever it shares its batch)
    assert_eq!(ml.all_cell_labels, vec![(0..5).collect::<Vec<_>>(); 2]);
    assert_eq!(ml.all_num_samples, vec![3, 3]);
}

/// `cage-cell` sets the mode after parsing, so fit itself refuses what
/// needs a hierarchy of pseudobulks or a batch-effect estimate.
#[test]
fn cell_level_refuses_hierarchy_flags() {
    for extra in [
        &["--chain-levels", "1"][..],
        &["--feature-embedding", "dict.parquet"][..],
        // an input to the batch-effect estimate cage-cell skips: refused,
        // not left unopened
        &["--cnv-clones", "clones.tsv"][..],
    ] {
        let mut a = parse(&[&["-o", "out", "d.zarr"][..], extra].concat()).unwrap();
        a.cell_level = true;
        let e = fit_cell_activity_graph_embedding(&a)
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("cage-cell:"), "{extra:?}: {e}");
    }
}

/// Two sections of a `side × side` grid, the left and right halves
/// expressing different genes, and the second section's every third gene
/// four times the first's (a batch effect); returns the data and
/// coordinate files.
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
                let v = if tag == "s2" && g % 3 == 0 {
                    4.0 * v
                } else {
                    v
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
    let cell_level = extra.contains(&"cell");
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
    .chain(extra.iter().copied().filter(|&e| e != "cell"))
    .collect();
    let mut args = parse(&argv).unwrap();
    args.cell_level = cell_level;
    fit_cell_activity_graph_embedding(&args).unwrap();
    out
}

fn cell_level_of(prefix: &str) -> bool {
    let path = crate::util::metadata::manifest_path(prefix);
    crate::util::metadata::PintoMetadata::read(std::path::Path::new(&path))
        .unwrap()
        .cell_level
}

/// `pinto predict` of `model` on the files it was fitted on; returns the
/// output prefix.
fn predict(dir: &std::path::Path, data: &str, coords: &str, model: &str) -> String {
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        predict: crate::predict::PredictArgs,
    }
    let out = dir.join("predicted").to_str().unwrap().to_string();
    let argv = [
        "x",
        data,
        "-c",
        coords,
        "--coord-column-names",
        "x,y",
        "--model",
        model,
        "-o",
        &out,
    ];
    crate::predict::predict(&Cli::try_parse_from(argv).unwrap().predict).unwrap();
    out
}

fn cell_embedding(prefix: &str) -> Mat {
    Mat::from_parquet(&format!("{prefix}.cell_embedding.parquet"))
        .unwrap()
        .mat
}

#[test]
fn cell_level_trains_on_cells_and_builds_no_pseudobulks() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    let n_cells = 2 * 8 * 8;

    let cell = fit(dir.path(), &data, &coords, "cell", &["cell"]);
    let trained = Mat::from_parquet(&format!("{cell}.pb_embedding.parquet")).unwrap();
    assert_eq!(trained.mat.nrows(), n_cells, "one trained row per cell");
    for skipped in ["delta", "pseudobulk_cells"] {
        assert!(
            !std::path::Path::new(&format!("{cell}.{skipped}.parquet")).exists(),
            "--cell-level wrote {skipped}.parquet"
        );
    }
    assert!(
        cell_level_of(&cell),
        "the manifest says the run is cell-level"
    );

    // Predicted on its own training files, a cell-level model places the
    // cells where it trained them: the query is not batch-corrected, as
    // the training was not.
    let predicted = predict(dir.path(), &data, &coords, &cell);
    let (fitted, placed) = (cell_embedding(&cell), cell_embedding(&predicted));
    let diff = (&fitted - &placed).abs().max();
    assert!(diff < 1e-4, "predict moved the cells by up to {diff}");

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
    assert!(
        !cell_level_of(&pb),
        "an ordinary cage run is not cell-level"
    );
}

/// On cells the chain group is the cell, so the upstream rejection
/// excludes only the anchor; the sampler also redraws a negative that is
/// the anchor's positive partner, so no pair is scored as both.
#[test]
fn a_cell_level_negative_is_never_the_anchor_or_its_partner() {
    use crate::cell_activity_graph_embedding::feature_chain_sampler::{
        build_feature_exp_batch_cache, FeatureGatedChainSampler,
    };
    use crate::cell_activity_graph_embedding::feature_gating::FeatureActiveEdges;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    let n = 6;
    let pairs: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
    let ml = cell_level_hierarchy(n, &pairs);
    let edges: Vec<(u32, u32)> = pairs.iter().map(|&(i, j)| (i as u32, j as u32)).collect();
    let (samplers, _) = graph_embedding_util::loss::build_per_batch_unit_samplers(
        &edges,
        &vec![0; n],
        1,
        n,
        1.0,
        None,
    );
    let activities = FeatureActiveEdges {
        feature_active_edges: vec![(0..edges.len() as u32).collect()],
        feature_active_edge_weights: vec![vec![1.0; edges.len()]],
    };
    let cache = build_feature_exp_batch_cache(&activities, &samplers, 1.0);
    let groups: Vec<&[usize]> = vec![&ml.all_cell_labels[0]];
    let sampler = FeatureGatedChainSampler {
        super_edges: &edges,
        samplers_per_exp_batch: &samplers,
        cache: &cache,
        unit_to_group_per_level: &groups,
        positives_per_draw: 8,
        n_negatives: 8,
        exclude_partner: true,
    };
    let mut rng = SmallRng::seed_from_u64(0);
    for _ in 0..200 {
        let (batch, _) = sampler.sample(0, 0, &mut rng).unwrap();
        for neg in &batch.per_level_neg {
            for (b, row) in neg.chunks(batch.n_negatives).enumerate() {
                let pair = [batch.left_units[b], batch.right_units[b]];
                assert!(
                    row.iter().all(|c| !pair.contains(c)),
                    "negatives {row:?} for the pair {pair:?}"
                );
            }
        }
    }
}
