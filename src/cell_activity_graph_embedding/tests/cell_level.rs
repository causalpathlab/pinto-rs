//! `--cell-level`: cells are the trained unit and nothing is pseudobulked.

use crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs;
use crate::cell_activity_graph_embedding::fit::{
    cell_level_hierarchy, fit_cell_activity_graph_embedding,
};
use crate::test_support::two_sections;
use crate::util::common::Mat;
use clap::Parser;
use legume_numeric::matrix::parquet::{write_named_table, Column};
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

/// The small widths and the short encoder budget the test fits run at.
const SMALL_WIDTHS: [&str; 8] = [
    "--embedding-dim",
    "4",
    "-d",
    "24",
    "--num-levels",
    "3",
    "--pair-encoder-steps",
    "8",
];

/// A short cage fit of `data` (`cage-cell` when `extra` holds `"cell"`)
/// to `{dir}/{name}`; its prefix, or why it failed.
pub(super) fn try_fit(
    dir: &std::path::Path,
    data: &str,
    coords: &str,
    name: &str,
    extra: &[&str],
) -> anyhow::Result<String> {
    try_fit_at(dir, data, coords, name, &SMALL_WIDTHS, extra)
}

/// [`try_fit`] at the `widths` given (none: cage's defaults).
pub(super) fn try_fit_at(
    dir: &std::path::Path,
    data: &str,
    coords: &str,
    name: &str,
    widths: &[&str],
    extra: &[&str],
) -> anyhow::Result<String> {
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
        "--n-edge-clusters",
        "2",
        "--edge-cluster-method",
        "kmeans",
    ]
    .into_iter()
    .chain(widths.iter().copied())
    .chain(extra.iter().copied().filter(|&e| e != "cell"))
    .collect();
    let mut args = parse(&argv).unwrap();
    args.cell_level = cell_level;
    fit_cell_activity_graph_embedding(&args)?;
    Ok(out)
}

/// [`try_fit`], which must succeed.
pub(super) fn fit(
    dir: &std::path::Path,
    data: &str,
    coords: &str,
    name: &str,
    extra: &[&str],
) -> String {
    try_fit(dir, data, coords, name, extra).unwrap()
}

fn cell_level_of(prefix: &str) -> bool {
    let path = crate::util::metadata::manifest_path(prefix);
    crate::util::metadata::PintoMetadata::read(std::path::Path::new(&path))
        .unwrap()
        .cell_level
}

/// `pinto apply predict` of `model` on the files it was fitted on; the output
/// prefix, or why it failed.
fn try_predict(
    dir: &std::path::Path,
    data: &str,
    coords: &str,
    model: &str,
) -> anyhow::Result<String> {
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
    crate::predict::predict(
        &Cli::try_parse_from(argv).unwrap().predict,
        &Default::default(),
    )?;
    Ok(out)
}

/// A cage-cell run at `prefix` over `n_cells` cells writes its pairs'
/// rows as the latent, one L2-normalized row per listed pair (as many as
/// the manifest's edges), and a cell embedding of one row per cell; no
/// unit table and no pair encoder, and its manifest lists neither.
fn assert_writes_the_pairs_rows(prefix: &str, n_cells: usize) {
    let meta = crate::util::metadata::PintoMetadata::read(std::path::Path::new(
        &crate::util::metadata::manifest_path(prefix),
    ))
    .unwrap();
    let latent = Mat::from_parquet(&format!("{prefix}.latent.parquet"))
        .unwrap()
        .mat;
    assert_eq!(Some(latent.nrows()), meta.n_edges, "a latent row per pair");
    assert_eq!(latent.ncols(), 4, "the embedding width");
    for (k, row) in latent.row_iter().enumerate() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "pair {k}: norm {norm}");
    }
    let cells = cell_embedding(prefix);
    assert_eq!((cells.nrows(), cells.ncols()), (n_cells, 4));
    assert!(
        cells.iter().all(|x| x.is_finite()) && cells.iter().any(|&x| x != 0.0),
        "the cell embedding is empty"
    );
    for file in ["pb_embedding.parquet", "pair_encoder.safetensors"] {
        assert!(
            !std::path::Path::new(&format!("{prefix}.{file}")).exists(),
            "cage-cell wrote {file}"
        );
    }
    assert!(meta.outputs.pair_encoder.is_none() && meta.outputs.pb_embedding.is_none());
}

fn cell_embedding(prefix: &str) -> Mat {
    Mat::from_parquet(&format!("{prefix}.cell_embedding.parquet"))
        .unwrap()
        .mat
}

/// `feature_inclusion.parquet` of the run at `prefix` has a row per
/// feature and every π in [0, 1].
fn assert_scores_every_feature(prefix: &str) {
    let t = Mat::from_parquet(&format!("{prefix}.feature_inclusion.parquet")).unwrap();
    assert_eq!(t.rows.len(), 12, "{prefix}: one row per feature");
    let pi = t
        .mat
        .column(t.cols.iter().position(|c| c.as_ref() == "pi").unwrap());
    assert!(
        pi.iter().all(|&p| (0.0..=1.0).contains(&p)),
        "{prefix}: π outside [0, 1]"
    );
}

fn pb_embedding(prefix: &str) -> Mat {
    Mat::from_parquet(&format!("{prefix}.pb_embedding.parquet"))
        .unwrap()
        .mat
}

/// The short fits on one pair of sections: cage-cell trains on cells and
/// estimates batch effects (stratified by `--cnv-clones` when given);
/// `--positives-per-epoch` changes its fit; `--ignore-batch` skips the
/// batch effects; plain cage trains on pseudobulks; and `--feature-inclusion
/// moran` scores every feature in both modes.
#[test]
fn cell_level_trains_on_cells_with_batch_effects_and_no_pseudobulk_table() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    let n_cells = 2 * 8 * 8;

    let cell = fit(dir.path(), &data, &coords, "cell", &["cell"]);
    assert_writes_the_pairs_rows(&cell, n_cells);
    // batch effects are estimated and removed, as in cage, unless asked not to
    assert!(
        std::path::Path::new(&format!("{cell}.delta.parquet")).exists(),
        "cage-cell estimated no batch effects"
    );
    assert!(std::path::Path::new(&format!("{cell}.pinto.json")).exists());
    assert!(
        cell_level_of(&cell),
        "the manifest says the run is cell-level"
    );

    // A cell-level model has no pair encoder to place a new sample with,
    // so predict (and impute, which predicts) refuses it, naming why.
    let e = try_predict(dir.path(), &data, &coords, &cell).unwrap_err();
    assert!(
        format!("{e:#}").contains("cage-cell models have no pair encoder"),
        "{e:#}"
    );
    #[derive(Parser)]
    struct ImputeCli {
        #[command(flatten)]
        impute: crate::impute::ImputeArgs,
    }
    let imputed = dir.path().join("imputed");
    let argv = [
        "x",
        &data,
        "-c",
        &coords,
        "--coord-column-names",
        "x,y",
        "--model",
        &cell,
        "-o",
        imputed.to_str().unwrap(),
    ];
    let e = crate::impute::run_impute(&ImputeCli::try_parse_from(argv).unwrap().impute, &[])
        .unwrap_err();
    assert!(
        format!("{e:#}").contains("cage-cell models have no pair encoder"),
        "impute: {e:#}"
    );

    // `--positives-per-epoch` (hidden) replaces the per-feature budget
    let one = fit(
        dir.path(),
        &data,
        &coords,
        "one",
        &["cell", "--positives-per-epoch", "1"],
    );
    assert!(
        cell_embedding(&cell) != cell_embedding(&one),
        "the budget left the fit unchanged"
    );

    let raw = fit(
        dir.path(),
        &data,
        &coords,
        "cell_raw",
        &["cell", "--ignore-batch"],
    );
    assert!(
        !std::path::Path::new(&format!("{raw}.delta.parquet")).exists(),
        "--ignore-batch still estimated batch effects"
    );

    // the same data without the flag trains on pseudobulks
    let inclusion = ["--feature-inclusion", "moran", "--inclusion-blocks", "4"];
    let pb = fit(dir.path(), &data, &coords, "pb", &inclusion);
    assert!(
        pb_embedding(&pb).nrows() < n_cells,
        "the default trains on super-cells"
    );
    assert!(std::path::Path::new(&format!("{pb}.delta.parquet")).exists());
    assert!(
        !cell_level_of(&pb),
        "an ordinary cage run is not cell-level"
    );
    assert_scores_every_feature(&pb);

    // cage-cell estimates batch effects as cage does, so `--cnv-clones`
    // stratifies that estimate here too: every other cell in a private
    // clone (stratum 1), the rest mixable
    let clones = dir.path().join("x.clones.parquet");
    let clones = clones.to_str().unwrap();
    let cells: Vec<Box<str>> = ["s1", "s2"]
        .iter()
        .flat_map(|tag| (0..8 * 8).map(move |c| format!("{tag}_{c}@{tag}").into()))
        .collect();
    let stratum: Vec<i32> = (0..cells.len() as i32).map(|c| c % 2).collect();
    write_named_table(
        clones,
        "cell",
        &cells,
        &[("stratum".into(), Column::I32(&stratum))],
    )
    .unwrap();
    let extra = [&inclusion[..], &["cell", "--cnv-clones", clones]].concat();
    let cnv = fit(dir.path(), &data, &coords, "cnv", &extra);
    assert!(std::path::Path::new(&format!("{cnv}.delta.parquet")).exists());
    assert_scores_every_feature(&cnv);
}

/// cage-cell's draw names pairs: every positive is a pair active for the
/// feature in the draw's batch (both its cells in that batch), and every
/// negative another pair of the same batch, never the positive.
#[test]
fn a_cell_level_draw_names_pairs_of_its_batch() {
    use crate::cell_activity_graph_embedding::feature_chain_sampler::{
        build_feature_exp_batch_cache, FeatureGatedChainSampler,
    };
    use crate::cell_activity_graph_embedding::feature_gating::FeatureActiveEdges;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    // two batches of six cells, each a path; one pair crosses them
    let n = 12;
    let mut pairs: Vec<(usize, usize)> =
        (0..5).flat_map(|i| [(i, i + 1), (i + 6, i + 7)]).collect();
    pairs.push((5, 6));
    let batch: Vec<u32> = (0..n).map(|c| (c / 6) as u32).collect();
    let edges: Vec<(u32, u32)> = pairs.iter().map(|&(i, j)| (i as u32, j as u32)).collect();
    let (samplers, _) =
        graph_embedding_util::loss::build_per_batch_unit_samplers(&edges, &batch, 2, n, 1.0, None);
    // the feature is active on every pair but the first of each batch
    let active: Vec<u32> = (2..edges.len() as u32).collect();
    let activities = FeatureActiveEdges {
        feature_active_edge_weights: vec![vec![1.0; active.len()]],
        feature_active_edges: vec![active.clone()],
    };
    let cache = build_feature_exp_batch_cache(&activities, &samplers, 1.0);
    let ml = cell_level_hierarchy(n, &pairs);
    let groups: Vec<&[usize]> = vec![&ml.all_cell_labels[0]];
    let sampler = FeatureGatedChainSampler {
        super_edges: &edges,
        samplers_per_exp_batch: &samplers,
        cache: &cache,
        unit_to_group_per_level: &groups,
        positives_per_draw: 8,
        n_negatives: 6,
    };
    let mut rng = SmallRng::seed_from_u64(0);
    for b in 0..2u32 {
        let in_batch = |e: u32| {
            let (u, v) = edges[e as usize];
            batch[u as usize] == b && batch[v as usize] == b
        };
        for _ in 0..100 {
            let draw = sampler.sample_edges(0, b as usize, &mut rng).unwrap();
            assert_eq!(draw.pos_edges.len(), 8);
            assert_eq!(draw.neg_edges.len(), 8 * 6);
            for (i, &e) in draw.pos_edges.iter().enumerate() {
                assert!(
                    active.contains(&e) && in_batch(e),
                    "positive {e} in batch {b}"
                );
                for &w in &draw.neg_edges[i * 6..(i + 1) * 6] {
                    assert!((w as usize) < edges.len() && in_batch(w), "negative {w}");
                    assert_ne!(w, e, "a negative is its own positive");
                }
            }
        }
    }
}

/// `--positives-per-epoch` (hidden) parses to an epoch total; unset keeps
/// the per-feature default.
#[test]
fn positives_per_epoch_parses() {
    let base = ["-o", "out", "d.zarr"];
    assert_eq!(parse(&base).unwrap().positives_per_epoch, None);
    let set = parse(&[&base[..], &["--positives-per-epoch", "5"]].concat()).unwrap();
    assert_eq!(set.positives_per_epoch, Some(5));
}

/// The per-(feature, batch) draws: the default when unset, else the epoch
/// total split over the expected visits and the batches, floored, at least
/// one, and no division by zero visits or batches.
#[test]
fn positives_per_feature_batch_splits_the_total_over_visits_and_batches() {
    use crate::cell_activity_graph_embedding::args::PER_FEATURE_BATCH;
    use crate::cell_activity_graph_embedding::fit::positives_per_feature_batch as per;
    assert_eq!(per(None, 5, 2), PER_FEATURE_BATCH);
    assert_eq!(per(Some(100), 5, 2), 10);
    assert_eq!(per(Some(29), 4, 3), 2, "floored");
    assert_eq!(per(Some(1), 5, 2), 1, "at least one");
    assert_eq!(per(Some(7), 0, 0), 7, "no visits or batches divides by one");
}
