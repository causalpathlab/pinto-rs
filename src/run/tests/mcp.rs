use super::*;

fn flags(v: Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn an_ablation_fit_is_named_for_its_kind_and_parent() {
    let f = flags(json!({"permute-coords": "block", "ablation-parent": "res/base.pinto.json"}));
    assert_eq!(
        ablation_summary(&f).unwrap(),
        "coord_permutation ablation of res/base.pinto.json"
    );
    let f = flags(json!({"--mask-cells": "region.txt"}));
    assert_eq!(
        ablation_summary(&f).unwrap(),
        "cell_mask ablation of an unnamed parent"
    );
    assert!(ablation_summary(&flags(json!({"n-communities": 12}))).is_none());
}

#[test]
fn a_run_given_by_prefix_is_named_by_its_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run");
    let prefix = prefix.to_str().unwrap();
    assert_eq!(as_run(prefix), shown_path(Path::new(prefix)));
    std::fs::write(format!("{prefix}.pinto.json"), "{}").unwrap();
    assert!(as_run(prefix).ends_with("run.pinto.json"));
    let args = CompareArgs {
        a: Runs(vec![format!("{prefix}.pinto.json")]),
        b: Runs(vec!["x/other".into()]),
        out: None,
        level_a: Some("draft".into()),
        level_b: Some("final".into()),
        tau: None,
        paired: false,
        donor_map: None,
        bootstrap: None,
    };
    assert_eq!(default_name(&args), "run.draft_vs_other");
}
