//! lc end to end on two small sections, with feature modules from an
//! external feature network.

use crate::link_community::args::SrtLinkCommunityArgs;
use crate::link_community::fit::fit_srt_link_community;
use crate::test_support::two_sections;
use crate::util::common::Mat;
use clap::Parser;
use legume_numeric::matrix::traits::IoOps;

/// `--feature-network` is a two-column file of feature pairs. lc groups
/// the features into modules on it (Leiden) and profiles each cell pair
/// by module pairs: here two cliques, one per half of the tissue's
/// features, and lines naming no feature of the data are skipped.
#[test]
fn lc_builds_its_profiles_from_an_external_feature_network() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    let mut tsv = String::from("feature1\tfeature2\nGENE0\tNOT_IN_DATA\n");
    for half in [0..6, 6..12] {
        for a in half.clone() {
            for b in a + 1..half.end {
                tsv.push_str(&format!("GENE{a}\tGENE{b}\n"));
            }
        }
    }
    let network = dir.path().join("network.tsv");
    std::fs::write(&network, tsv).unwrap();
    let out = dir.path().join("lc").to_str().unwrap().to_string();
    let args = SrtLinkCommunityArgs::try_parse_from([
        "lc",
        &data,
        "-c",
        &coords,
        "--coord-column-names",
        "x,y",
        "-o",
        &out,
        "--feature-network",
        network.to_str().unwrap(),
        "--n-communities",
        "2",
        "--num-gibbs",
        "2",
        "--num-greedy",
        "1",
        "--num-levels",
        "2",
    ])
    .unwrap();
    fit_srt_link_community(&args).unwrap();

    // the two cliques, 15 pairs each, and nothing between them
    let graph = format!("{out}.feature_graph.parquet");
    let (names, _) =
        legume_numeric::matrix::parquet::read_table_columns(&graph, &["feature1", "feature2"], &[])
            .unwrap();
    let half = |f: &str| f["GENE".len()..].parse::<usize>().unwrap() / 6;
    assert_eq!(names[0].len(), 30, "{graph}");
    assert!(names[0]
        .iter()
        .zip(&names[1])
        .all(|(a, b)| half(a) == half(b)));
    let propensity = Mat::from_parquet(&format!("{out}.propensity.parquet")).unwrap();
    assert_eq!(propensity.rows.len(), 2 * 8 * 8, "one row per cell");
}

/// A short lc fit of `data` to `{dir}/{name}` with `extra` flags; its prefix.
fn fit_lc(dir: &std::path::Path, data: &str, coords: &str, name: &str, extra: &[&str]) -> String {
    fit_lc_with(dir, data, coords, name, extra, |_| {})
}

/// [`fit_lc`] with its arguments changed by `change` once parsed.
fn fit_lc_with(
    dir: &std::path::Path,
    data: &str,
    coords: &str,
    name: &str,
    extra: &[&str],
    change: impl FnOnce(&mut SrtLinkCommunityArgs),
) -> String {
    let out = dir.join(name).to_str().unwrap().to_string();
    let argv: Vec<&str> = [
        "lc",
        data,
        "-c",
        coords,
        "--coord-column-names",
        "x,y",
        "-o",
        &out,
        "--n-communities",
        "2",
        "--num-gibbs",
        "2",
        "--num-greedy",
        "1",
        "--num-levels",
        "2",
    ]
    .into_iter()
    .chain(extra.iter().copied())
    .collect();
    let mut args = SrtLinkCommunityArgs::try_parse_from(argv).unwrap();
    change(&mut args);
    fit_srt_link_community(&args).unwrap();
    out
}

/// Fits lc once on two sections and runs, in order, the checks that read
/// it; the last ones rewrite and rename its manifest.
#[test]
fn one_lc_model_records_scores_and_predicts() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    let lc = fit_lc(dir.path(), &data, &coords, "lc", &[]);
    lc_records_its_batch_effects_in_the_manifest(dir.path(), (&data, &coords), &lc);
    a_known_batch_is_scored_under_the_models_delta(dir.path(), (&data, &coords), &lc);
    an_lc_model_named_by_a_renamed_manifest_predicts(dir.path(), (&data, &coords), &lc);
}

/// The batch effects a run's manifest names.
fn recorded_delta(prefix: &str) -> Option<String> {
    let path = crate::util::metadata::manifest_path(prefix);
    crate::util::metadata::PintoMetadata::read(std::path::Path::new(&path))
        .unwrap()
        .outputs
        .batch_effects
}

/// lc estimates batch effects on a multi-batch run, as dsvd and cage do, and
/// its manifest names the file; `--ignore-batch` writes and names none. The
/// manifest also records the `--max-threads` that set lc's Gibbs blocks.
fn lc_records_its_batch_effects_in_the_manifest(
    dir: &std::path::Path,
    (data, coords): (&str, &str),
    lc: &str,
) {
    let delta = recorded_delta(lc).expect("the manifest names no delta.parquet");
    assert_eq!(delta, "lc.delta.parquet", "recorded beside the manifest");
    let run = crate::util::metadata::Run::open(lc).unwrap();
    assert_eq!(
        crate::util::batch_effects::run_delta_path(lc, Some(&run)).unwrap(),
        Some(format!("{lc}.delta.parquet"))
    );

    let raw = fit_lc_with(dir, data, coords, "raw", &["--ignore-batch"], |a| {
        a.max_threads = 3
    });
    assert!(recorded_delta(&raw).is_none());
    assert!(!std::path::Path::new(&format!("{raw}.delta.parquet")).exists());
    let meta = crate::util::metadata::PintoMetadata::read(std::path::Path::new(
        &crate::util::metadata::manifest_path(&raw),
    ))
    .unwrap();
    assert_eq!(
        meta.max_threads,
        Some(3),
        "the manifest does not record --max-threads"
    );
}

/// Two sections of the same tissue, a left half and a right half that
/// differ only weakly, the second section's `folded` features four times
/// the first's. Undivided, that batch fold is the strongest signal and lc
/// splits the pairs by section; divided by the estimated batch effect, it
/// does not.
///
/// Projection profiles, with every other feature folded: lc then finds the
/// halves. Module-pair profiles from a feature network (one clique per
/// half), with one module's features folded: those features carry almost
/// no Fisher weight (it is taken on raw counts, where the fold reads as
/// overdispersion), so the halves are too faint to find and only the
/// section split is checked.
#[test]
#[ignore = "benchmark: cargo test -- --ignored"]
fn lc_divides_its_profiles_by_the_batch_effect() {
    let dir = tempfile::tempdir().unwrap();
    let (by_half, by_section) = halves_and_sections(dir.path(), "proj", &|g| g % 2 == 0, &[]);
    assert!(by_half > 0.9, "lc missed the halves ({by_half})");
    assert!(
        by_section < 0.75,
        "lc split the pairs by section ({by_section})"
    );

    let mut tsv = String::from("feature1\tfeature2\n");
    for half in [0..6, 6..12] {
        for a in half.clone() {
            for b in a + 1..half.end {
                tsv.push_str(&format!("GENE{a}\tGENE{b}\n"));
            }
        }
    }
    let network = dir.path().join("network.tsv");
    std::fs::write(&network, tsv).unwrap();
    let extra = ["--feature-network", network.to_str().unwrap()];
    let (_, by_section) = halves_and_sections(dir.path(), "modules", &|g| g < 6, &extra);
    assert!(
        by_section < 0.75,
        "lc on modules split the pairs by section ({by_section})"
    );
}

/// lc (with `extra`) on [`lc_divides_its_profiles_by_the_batch_effect`]'s
/// sections, written under `{dir}/{tag}*`: how far its two communities
/// agree with the halves, and with the sections (1 is exactly, 0.5 not at
/// all).
fn halves_and_sections(
    dir: &std::path::Path,
    tag: &str,
    folded: &dyn Fn(usize) -> bool,
    extra: &[&str],
) -> (f32, f32) {
    use crate::test_support::write_zarr;
    let side = 10usize;
    let genes: Vec<String> = (0..12).map(|g| format!("GENE{g}")).collect();
    let genes: Vec<&str> = genes.iter().map(String::as_str).collect();
    let (mut data, mut coords) = (Vec::new(), Vec::new());
    for section in ["s1", "s2"] {
        let name = format!("{tag}_{section}");
        let mut triplets = Vec::new();
        let mut csv = String::from("cell,x,y\n");
        for c in 0..side * side {
            let (r, col) = (c / side, c % side);
            for g in 0..12usize {
                let marker = (col < side / 2) == (g < 6);
                let v = if marker {
                    3.0 + ((c + g) % 3) as f32
                } else {
                    3.0 + ((c * 7 + g) % 2) as f32
                };
                let v = if section == "s2" && folded(g) {
                    4.0 * v
                } else {
                    v
                };
                triplets.push((g as u64, c as u64, v));
            }
            csv.push_str(&format!("{name}_{c},{col},{r}\n"));
        }
        data.push(write_zarr(dir, &name, &genes, side * side, &triplets).to_string());
        let path = dir.join(format!("{name}.coords.csv"));
        std::fs::write(&path, csv).unwrap();
        coords.push(path.to_str().unwrap().to_string());
    }
    let out = fit_lc(dir, &data.join(","), &coords.join(","), tag, extra);
    let p = Mat::from_parquet(&format!("{out}.propensity.parquet"))
        .unwrap()
        .mat;
    // each cell's top community against a two-way split, up to the labels' order
    let agreement = |side_of: &dyn Fn(usize) -> bool| {
        let n = p.nrows();
        let same = (0..n)
            .filter(|&i| (p[(i, 0)] >= p[(i, 1)]) == side_of(i))
            .count();
        same.max(n - same) as f32 / n as f32
    };
    (
        agreement(&|i| (i % (side * side)) % side < side / 2),
        agreement(&|i| i >= side * side),
    )
}

/// An lc model named by its manifest under another file name is predicted
/// from (held-out scoring), as one named by its prefix: the manifest is
/// the model's, though no `{prefix}.pinto.json` is there. It moves the
/// manifest, so it runs last.
fn an_lc_model_named_by_a_renamed_manifest_predicts(
    dir: &std::path::Path,
    (data, coords): (&str, &str),
    lc: &str,
) {
    let renamed = dir.join("first.json");
    std::fs::rename(crate::util::metadata::manifest_path(lc), &renamed).unwrap();

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        predict: crate::predict::PredictArgs,
    }
    let out = dir.join("pred").to_str().unwrap().to_string();
    let argv = [
        "x",
        data,
        "-c",
        coords,
        "--coord-column-names",
        "x,y",
        "--model",
        renamed.to_str().unwrap(),
        "-o",
        &out,
    ];
    crate::predict::predict(
        &Cli::try_parse_from(argv).unwrap().predict,
        &Default::default(),
    )
    .expect("a model opened by its manifest was refused");
}

/// `pinto predict` (held-out scoring) of the lc model at `model` on `data`
/// (with `coords` when given), with `extra` flags, to `{dir}/{name}`: its
/// output's manifest, or the error.
fn score(
    dir: &std::path::Path,
    model: &str,
    (data, coords): (&str, Option<&str>),
    name: &str,
    extra: &[&str],
) -> anyhow::Result<crate::util::metadata::PintoMetadata> {
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        predict: crate::predict::PredictArgs,
    }
    let out = dir.join(name).to_str().unwrap().to_string();
    let mut argv = vec!["x", data, "--model", model, "-o", &out];
    if let Some(coords) = coords {
        argv.extend_from_slice(&["-c", coords, "--coord-column-names", "x,y"]);
    }
    argv.extend_from_slice(extra);
    crate::predict::predict(
        &Cli::try_parse_from(argv).unwrap().predict,
        &Default::default(),
    )?;
    crate::util::metadata::PintoMetadata::read(std::path::Path::new(
        &crate::util::metadata::manifest_path(&out),
    ))
}

/// The manifest at `prefix`, its batch-adjusted flag set to `adjusted`.
fn set_adjusted(prefix: &str, adjusted: bool) {
    let path = crate::util::metadata::manifest_path(prefix);
    let mut meta = crate::util::metadata::PintoMetadata::read(std::path::Path::new(&path)).unwrap();
    meta.outputs.feature_community_batch_adjusted = adjusted;
    meta.write(std::path::Path::new(&path)).unwrap();
}

/// A multi-batch lc model's profiles are batch-adjusted, so a held-out
/// batch the model knows takes the model's δ: with no δ rounds, its
/// training sections score better than through the same profiles read as
/// an older model's (every batch at δ = 1). `--ignore-batch` is refused
/// for it, and taken for the older one.
///
/// It sets the model's batch-adjusted flag and restores it.
fn a_known_batch_is_scored_under_the_models_delta(
    dir: &std::path::Path,
    (data, coords): (&str, &str),
    lc: &str,
) {
    let input = (data, Some(coords));
    let no_rounds = ["--holdout-delta-iters", "0"];
    let llik =
        |m: &crate::util::metadata::PintoMetadata| m.holdout.as_ref().unwrap().eval_llik_per_count;
    let adjusted = score(dir, lc, input, "adjusted", &no_rounds).unwrap();
    // with no δ rounds the known batches still take the model's δ, and the
    // manifest says so
    let info = adjusted.holdout.as_ref().unwrap();
    assert_eq!((info.delta_iters, info.model_delta_batches), (0, Some(2)));
    assert!(
        info.notes.iter().any(|n| n.contains(
            "2 of this sample's 2 batches are the \
             model's and take its δ; no other batch"
        )),
        "{:?}",
        info.notes
    );
    let err = score(dir, lc, input, "ignored", &["--ignore-batch"]).unwrap_err();
    assert!(err.to_string().contains("--ignore-batch"), "{err}");
    assert!(err.to_string().contains("--holdout-delta-iters 0"), "{err}");

    set_adjusted(lc, false);
    let older = score(dir, lc, input, "older", &no_rounds).unwrap();
    let info = older.holdout.as_ref().unwrap();
    assert_eq!(info.model_delta_batches, None);
    assert!(
        info.notes
            .iter()
            .any(|n| n.starts_with("every batch takes the model's profiles unshifted")),
        "{:?}",
        info.notes
    );
    assert!(
        llik(&adjusted) > llik(&older),
        "known batches under the model's δ {} vs at 1 {}",
        llik(&adjusted),
        llik(&older)
    );
    score(dir, lc, input, "ignored", &["--ignore-batch"]).unwrap();
    set_adjusted(lc, true);
}

/// Without coordinates an adjusted model's graph is built on the
/// batch-corrected projection, and so is the held-out sample's: scored on
/// its own training data, its pairs are the model's. The query's graph
/// follows the manifest's graph flag alone, whatever its profiles'
/// batch-adjusted flag says, so each is set without the other.
#[test]
fn an_expression_model_is_scored_on_a_graph_built_as_its_own() {
    let dir = tempfile::tempdir().unwrap();
    let (data, _) = two_sections(dir.path(), 8);
    let model = dir.path().join("expr").to_str().unwrap().to_string();
    let argv = [
        "lc",
        data.as_str(),
        "-o",
        model.as_str(),
        "--n-communities",
        "2",
        "--num-gibbs",
        "2",
        "--num-greedy",
        "1",
        "--num-levels",
        "2",
    ];
    fit_srt_link_community(&SrtLinkCommunityArgs::try_parse_from(argv).unwrap()).unwrap();
    let pairs_of = |left: &[Box<str>], right: &[Box<str>]| {
        let mut p: Vec<(Box<str>, Box<str>)> = left
            .iter()
            .zip(right)
            .map(|(a, b)| {
                if a <= b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                }
            })
            .collect();
        p.sort();
        p
    };
    let trained =
        crate::lr_activity::io::read_pair_table(&format!("{model}.coord_pairs.parquet"), false)
            .unwrap();
    let (l, r): (Vec<Box<str>>, Vec<Box<str>>) = trained
        .into_iter()
        .map(|e| (e.left_cell, e.right_cell))
        .unzip();
    let trained = pairs_of(&l, &r);

    let scored_pairs = |name: &str, extra: &[&str]| {
        score(dir.path(), &model, (data.as_str(), None), name, extra).unwrap();
        let path = dir.path().join(format!("{name}.predictive.parquet"));
        let (strs, _) = legume_numeric::matrix::parquet::read_table_columns(
            path.to_str().unwrap(),
            &["left", "right"],
            &[],
        )
        .unwrap();
        pairs_of(&strs[0], &strs[1])
    };
    let manifest = crate::util::metadata::manifest_path(&model);
    let meta = crate::util::metadata::PintoMetadata::read(std::path::Path::new(&manifest)).unwrap();
    assert!(
        meta.base_graph_batch_corrected(),
        "the fit records its graph"
    );
    assert_eq!(scored_pairs("pred", &[]), trained);
    let flag = |adjusted: bool, corrected: bool| {
        let mut meta = meta.clone();
        meta.outputs.feature_community_batch_adjusted = adjusted;
        meta.graph.as_mut().unwrap().base_batch_corrected = corrected;
        meta.write(std::path::Path::new(&manifest)).unwrap();
    };

    // the graph flag alone: profiles read as raw, the graph still corrected
    flag(false, true);
    assert_eq!(scored_pairs("graph_only", &[]), trained);

    // an older model's manifest records neither: its query graph is the
    // uncorrected one, as an --ignore-batch query builds it
    flag(false, false);
    let older = scored_pairs("older", &[]);
    assert_eq!(older, scored_pairs("ignored", &["--ignore-batch"]));
    assert_ne!(older, trained, "the corrected graph should differ");

    // the profiles' flag alone: the graph is the uncorrected one
    flag(true, false);
    assert_eq!(scored_pairs("profiles_only", &[]), older);
}
