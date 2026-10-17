use crate::view::data::{Communities, NO_CLUSTER};
use crate::view::lupin::*;
use std::path::Path;

fn write(path: &Path, body: &str) {
    std::fs::write(path, body).unwrap();
}

#[test]
fn labels_compare_as_lupin_folds_them() {
    assert_eq!(label_key("CD4 T, naive"), "CD4_T_naive");
    assert_eq!(label_key("CD4_T__naive"), label_key("CD4 T naive"));
    // lupin keeps case: two spellings are two types.
    assert_ne!(label_key("T cell"), label_key("t cell"));
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
    assert_eq!(panel.genes("T cell").unwrap(), ["CD3E", "CD2"]);
    assert_eq!(panel.types_of("ms4a1"), ["B cell"]);
}

/// Warnings logged on this thread, counted by [`count_logs`].
mod quiet {
    use std::cell::Cell;

    thread_local!(static LOGGED: Cell<usize> = const { Cell::new(0) });

    struct Counter;
    impl log::Log for Counter {
        fn enabled(&self, _: &log::Metadata) -> bool {
            true
        }
        fn log(&self, _: &log::Record) {
            LOGGED.with(|n| n.set(n.get() + 1));
        }
        fn flush(&self) {}
    }

    /// Records `f` logs on this thread, at any level. A probe logged first
    /// must be counted, so a 0 means nothing was logged, not that another
    /// logger or level filter swallowed it.
    pub fn count_logs(f: impl FnOnce()) -> usize {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            let _ = log::set_logger(&Counter);
            log::set_max_level(log::LevelFilter::Trace);
        });
        let before = LOGGED.with(Cell::get);
        log::trace!("probe");
        assert_eq!(LOGGED.with(Cell::get), before + 1, "logs are not counted");
        f();
        LOGGED.with(Cell::get) - before - 1
    }
}

#[test]
fn panels_read_by_the_shared_two_column_rule() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("panel.csv");
    write(
        &p,
        "% note, with a comma\n\
         GENE1,\"CT1, sub a\"\n\
         GENE2,\"CT1, sub a\"\n\
         GENE3,CT3, sub b\n\
         GENE4\tCT4, sub c\textra\n",
    );
    let panel = Panel::read(&p).unwrap();
    let names: Vec<&str> = panel.types.iter().map(|(t, _)| t.as_str()).collect();
    // A quoted label keeps its comma; an unquoted comma ends the label; a
    // tab line splits on its tabs only; a `%` line is a comment.
    assert_eq!(names, ["CT1, sub a", "CT3", "CT4, sub c"]);
    assert_eq!(panel.genes("CT1, sub a").unwrap(), ["GENE1", "GENE2"]);
    assert_eq!(panel.genes("CT3").unwrap(), ["GENE3"]);
}

#[test]
fn panels_keep_case_and_drop_labels_that_key_to_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("panel.tsv");
    write(&p, "CD3E\tT cell\nCD2\tt cell\nCD19\t_\n");
    let panel = Panel::read(&p).unwrap();
    assert_eq!(panel.genes("T cell").unwrap(), ["CD3E"]);
    assert_eq!(panel.genes("t cell").unwrap(), ["CD2"]);
    assert!(panel.types_of("CD19").is_empty());
}

#[test]
fn reading_a_panel_logs_nothing() {
    // The panel picker sniffs every file in a folder with this reader.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("panel.csv");
    write(&p, "gene,type\nGENE3,CT3, sub b\nGENE5,\n,CT5\n");
    let logged = quiet::count_logs(|| {
        Panel::read(&p).unwrap();
    });
    assert_eq!(logged, 0);
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
