use crate::view::draft::*;
use std::path::PathBuf;

fn draft() -> Draft {
    Draft {
        round: PathBuf::from("/runs/r.final.a1.lupin.json"),
        ..Draft::default()
    }
}

fn lines(d: &Draft) -> Vec<serde_json::Value> {
    d.decisions()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn decisions_go_marker_edits_then_merges_then_labels() {
    let mut d = draft();
    d.verdicts.insert(
        3,
        Verdict::Label {
            label: "T cell".into(),
            rationale: "CD3E".into(),
        },
    );
    d.merges.push(Merge {
        clusters: vec![2, 6],
        label: "B cell".into(),
        rationale: "one program".into(),
    });
    for gene in ["CD2", "CD5"] {
        d.toggle_mark(Mark {
            label: "T cell".into(),
            feature: gene.into(),
            add: true,
        });
    }
    let got = lines(&d);
    let actions: Vec<&str> = got.iter().map(|v| v["action"].as_str().unwrap()).collect();
    assert_eq!(actions, ["markers_add", "merge", "label"]);
    // One marker line per type and direction.
    assert_eq!(got[0]["features"], serde_json::json!(["CD2", "CD5"]));
    assert_eq!(got[1]["clusters"], serde_json::json!([2, 6]));
    assert_eq!(got[2]["cluster"], 3);
    for v in &got {
        assert_eq!(v["decided_by"], "user");
        assert!(!v["rationale"].as_str().unwrap().is_empty());
        assert!(v["round"]
            .as_str()
            .unwrap()
            .ends_with("r.final.a1.lupin.json"));
    }
}

#[test]
fn a_mark_toggles_and_the_opposite_replaces_it() {
    let mut d = draft();
    let mark = |add| Mark {
        label: "T_cell".into(),
        feature: "CD3E".into(),
        add,
    };
    assert!(d.toggle_mark(mark(true)));
    // The same type under another spelling is the same type.
    assert!(!d.toggle_mark(Mark {
        label: "t cell".into(),
        ..mark(true)
    }));
    assert!(d.marks.is_empty());
    d.toggle_mark(mark(true));
    assert!(d.toggle_mark(mark(false)));
    assert_eq!(d.marks, [mark(false)]);
}

#[test]
fn unstaging_takes_back_a_verdict_and_its_merge() {
    let mut d = draft();
    d.merges.push(Merge {
        clusters: vec![1, 4],
        label: "x".into(),
        rationale: "y".into(),
    });
    assert_eq!(d.merge_of(4), Some(0));
    assert!(d.unstage(4));
    assert!(d.merges.is_empty());
    assert!(!d.unstage(4));
}

#[test]
fn a_draft_is_kept_beside_its_round_and_only_for_it() {
    let dir = tempfile::tempdir().unwrap();
    let round = dir.path().join("r.final.a1.lupin.json");
    assert_eq!(
        Draft::path_for(&round),
        dir.path().join("r.final.a1.relabel_draft.json")
    );
    let mut d = Draft::load(&round);
    d.verdicts.insert(
        0,
        Verdict::Keep {
            label: "x".into(),
            rationale: "y".into(),
        },
    );
    d.save().unwrap();
    assert_eq!(Draft::load(&round).verdicts, d.verdicts);
    // Another round does not pick it up.
    let other = dir.path().join("r.final.a1.r1.lupin.json");
    std::fs::copy(Draft::path_for(&round), Draft::path_for(&other)).unwrap();
    assert!(Draft::load(&other).is_empty());
    d.discard();
    assert!(!Draft::path_for(&round).exists());
}
