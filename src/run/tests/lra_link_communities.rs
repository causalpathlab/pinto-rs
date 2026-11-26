//! `run_lra_link_communities` lists the communities of the link-community
//! table the run records, as `pinto lra link-communities` does.

use super::*;
use crate::util::metadata::PintoMetadata;
use std::path::Path;

#[test]
fn each_link_community_comes_with_its_links_receivers_and_whether_it_is_fitted() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let cells: Vec<Box<str>> = ["a", "b", "c", "d"].map(Box::from).to_vec();
    // community 0: a-b and b-c (three receivers); community 1: c-d (two)
    let table = format!("{prefix}.link_community.parquet");
    crate::link_community::outputs::write_link_communities(
        &table,
        &[(0, 1), (1, 2), (2, 3)],
        &[0, 0, 1],
        &cells,
        None,
    )
    .unwrap();
    let mut meta = PintoMetadata {
        prefix: prefix.clone(),
        ..Default::default()
    };
    meta.outputs.link_community = Some(table);
    meta.write(Path::new(&format!("{prefix}.pinto.json")))
        .unwrap();

    let reply = link_communities(LinkCommunitiesToolArgs {
        run: prefix,
        min_receivers: Some(3),
    })
    .unwrap();
    let Content::Json(v) = &reply[1] else {
        panic!("no JSON reply")
    };
    let rows: Vec<(i64, u64, u64, bool)> = v["link_communities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r["link_community"].as_i64().unwrap(),
                r["links"].as_u64().unwrap(),
                r["receivers"].as_u64().unwrap(),
                r["fitted"].as_bool().unwrap(),
            )
        })
        .collect();
    assert_eq!(rows, [(0, 2, 3, true), (1, 1, 2, false)]);
    assert_eq!(v["min_receivers"], 3);
}
