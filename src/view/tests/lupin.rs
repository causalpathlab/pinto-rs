use crate::view::data::{Communities, NO_CLUSTER};
use crate::view::lupin::*;
use std::path::Path;

fn write(path: &Path, body: &str) {
    std::fs::write(path, body).unwrap();
}

#[test]
fn labels_compare_as_lupin_folds_them() {
    assert_eq!(label_key("CD4 T, naive"), "cd4_t_naive");
    assert_eq!(label_key("CD4_T__naive"), label_key("cd4 t naive"));
}

#[test]
fn panels_read_tab_or_comma_lines_and_skip_headers() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("panel.tsv");
    write(
        &p,
        "gene\tcell_type\n# note\n\nCD3E\tT cell\nCD2,T_cell\nMS4A1\tB cell\n",
    );
    let panel = Panel::read(&p).unwrap();
    assert_eq!(panel.types.len(), 2);
    assert_eq!(panel.genes("t cell").unwrap(), ["CD3E", "CD2"]);
    assert_eq!(panel.types_of("ms4a1"), ["B cell"]);
}

#[test]
fn a_first_round_takes_the_first_unused_name() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").display().to_string();
    assert_eq!(annotate_out(&prefix, "L2"), format!("{prefix}.L2.a1"));
    write(Path::new(&format!("{prefix}.L2.a1.lupin.json")), "{}");
    assert_eq!(annotate_out(&prefix, "L2"), format!("{prefix}.L2.a2"));
}

#[test]
fn the_newest_round_of_each_chain_is_found_and_its_level_read() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    write(&d.join("run.pinto.json"), "{}");
    let round = |name: &str, source: &str, level: Option<&str>| {
        let settings = level.map_or(
            serde_json::Value::Null,
            |l| serde_json::json!({"enrichment": {"level": l}}),
        );
        let m = serde_json::json!({"annotate": {"source": source, "settings": settings}});
        write(&d.join(name), &m.to_string());
    };
    round("run.L2.a1.lupin.json", "run.pinto.json", None);
    round("run.L2.a1.r1.lupin.json", "run.L2.a1.lupin.json", None);
    round("run.final.a1.lupin.json", "run.pinto.json", Some("final"));
    round("other.a1.lupin.json", "other.pinto.json", None);

    let mut tips: Vec<String> = latest_rounds(&d.join("run.pinto.json"))
        .iter()
        .map(|p| crate::tui::name(p))
        .collect();
    tips.sort();
    assert_eq!(tips, ["run.L2.a1.r1.lupin.json", "run.final.a1.lupin.json"]);

    let tags = ["L1", "L2", "final"];
    // Recorded by lupin, or read from the name pinto gave the first round.
    let level = |name: &str| round_level(&d.join(name), &tags);
    assert_eq!(level("run.final.a1.lupin.json").as_deref(), Some("final"));
    assert_eq!(level("run.L2.a1.r1.lupin.json").as_deref(), Some("L2"));
}

#[test]
fn a_grouping_draws_as_one_hot_communities() {
    let names = ["b", "a"].map(Box::<str>::from).to_vec();
    let c = Communities::from_groups("types", vec![1, 0, 1, NO_CLUSTER], names);
    assert_eq!(c.k, 2);
    assert_eq!(c.sizes, [1, 2]);
    assert_eq!(c.by_size, [1, 0]);
    assert_eq!(&c.prop[..4], &[0, 255, 255, 0]);
    assert_eq!(&c.prop[6..], &[0, 0]);
    assert_eq!(c.name(1), "a");
}

#[test]
fn a_cluster_goes_by_its_id_and_a_type_by_its_whole_name() {
    use crate::view::focus_name;
    let names = ["K3 T_cell", "K12 –"].map(Box::<str>::from).to_vec();
    let ids = ["K3", "K12"].map(Box::<str>::from).to_vec();
    let clusters = Communities::from_groups("g", vec![0, 1], names).with_ids(ids);
    let got: Vec<String> = (0..2).map(|g| focus_name(&clusters, g)).collect();
    assert_eq!(got, ["K3", "K12"]);
    // A cell type is its whole name, even one that starts like an id.
    let names = ["Kupffer cell", "K2 cell"].map(Box::<str>::from).to_vec();
    let types = Communities::from_groups("g", vec![0, 1], names);
    let got: Vec<String> = (0..2).map(|g| focus_name(&types, g)).collect();
    assert_eq!(got, ["Kupffer cell", "K2 cell"]);
}
