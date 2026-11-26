//! `view_lra`'s `start` asks `run_lra` for the command the terminal's
//! ctrl+r builds from the same options.

use super::*;
use crate::lr_activity::args::LraMethod;

#[test]
fn a_subset_run_passes_what_the_terminal_passes() {
    let options = LraOptions {
        // absolute, as run_lra writes paths out in full
        lr_pairs: Some("/data/pairs.tsv".into()),
        targets: vec!["/data/t.tsv".into()],
        seed: 7,
        marginal_baselines: Some("r.lra.celltype.parquet".into()),
        marginal_permutations: Some(50),
        null_permutations: Some(3),
        ..LraOptions::default()
    };
    let only = vec!["L1:R1".to_string(), "L2:R2".to_string()];
    let manifest = Path::new("r.pinto.json");
    let a = crate::mcp::parse(subset_args(manifest, "r", &options, &only)).unwrap();
    let mut mcp =
        super::super::lra::lra_argv(&a, LraMethod::ContactEffect, Some("r.pinto.json"), "r");
    let mut tui = options.argv(
        Some("r.pinto.json"),
        Some("r"),
        LraMethod::ContactEffect,
        &only,
    );
    mcp.sort();
    tui.sort();
    assert_eq!(mcp, tui);
}
