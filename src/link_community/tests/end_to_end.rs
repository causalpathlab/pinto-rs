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
