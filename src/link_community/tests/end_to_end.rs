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
    fit_srt_link_community(&SrtLinkCommunityArgs::try_parse_from(argv).unwrap()).unwrap();
    out
}

/// lc estimates batch effects on a multi-batch run, as dsvd and cage do, and
/// its manifest names the file; `--ignore-batch` writes and names none.
#[test]
fn lc_records_its_batch_effects_in_the_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    let recorded = |prefix: &str| {
        let path = crate::util::metadata::manifest_path(prefix);
        crate::util::metadata::PintoMetadata::read(std::path::Path::new(&path))
            .unwrap()
            .outputs
            .batch_effects
    };
    let lc = fit_lc(dir.path(), &data, &coords, "lc", &[]);
    let delta = recorded(&lc).expect("the manifest names no delta.parquet");
    assert_eq!(delta, "lc.delta.parquet", "recorded beside the manifest");
    let run = crate::util::metadata::Run::open(&lc).unwrap();
    assert_eq!(
        crate::util::batch_effects::run_delta_path(&lc, Some(&run)).unwrap(),
        Some(format!("{lc}.delta.parquet"))
    );

    let raw = fit_lc(dir.path(), &data, &coords, "raw", &["--ignore-batch"]);
    assert!(recorded(&raw).is_none());
    assert!(!std::path::Path::new(&format!("{raw}.delta.parquet")).exists());
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
/// the model's, though no `{prefix}.pinto.json` is there.
#[test]
fn an_lc_model_named_by_a_renamed_manifest_predicts() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    let lc = fit_lc(dir.path(), &data, &coords, "lc", &[]);
    let renamed = dir.path().join("first.json");
    std::fs::rename(crate::util::metadata::manifest_path(&lc), &renamed).unwrap();

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        predict: crate::predict::PredictArgs,
    }
    let out = dir.path().join("pred").to_str().unwrap().to_string();
    let argv = [
        "x",
        data.as_str(),
        "-c",
        coords.as_str(),
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
