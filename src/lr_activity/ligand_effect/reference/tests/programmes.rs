//! Networks, the signed search, targets, translation and number formats.

use super::{ix_tsv, resources_json};
use crate::lr_activity::ligand_effect::reference::programmes::*;
use crate::lr_activity::ligand_effect::reference::sources::{parse_interactions, Licences};
use crate::lr_activity::ligand_effect::reference::LicenceMode;
use rustc_hash::FxHashMap;

fn graph(edges: &[(&str, &str, i8)]) -> Graph {
    let mut g = Graph::default();
    for &(u, v, s) in edges {
        g.entry(u.into()).or_default().insert(v.into(), s);
    }
    g
}

fn regulons(entries: &[(&str, &[(&str, i8)])]) -> Regulons {
    entries
        .iter()
        .map(|(tf, tg)| {
            (
                Box::from(*tf),
                tg.iter().map(|&(g, m)| (Box::from(g), m)).collect(),
            )
        })
        .collect()
}

/// R reaches T at distance 2 along three paths: + · +, − · +, and
/// unsigned (+1) · −, so S_T = −1; X behind T inherits it; the edge B → A
/// is not a shortest path to A.
fn diamond() -> Graph {
    graph(&[
        ("R", "A", 1),
        ("R", "B", -1),
        ("R", "C", 0),
        ("A", "T", 1),
        ("B", "T", 1),
        ("C", "T", -1),
        ("B", "A", -1),
        ("T", "X", 1),
        ("X", "Y", 1),
    ])
}

#[test]
fn bfs_sums_signs_over_shortest_paths() {
    let g = diamond();
    let reach = signed_bfs(&g, &["R"], 3);
    assert_eq!(reach["R"], (0, 1));
    assert_eq!(reach["A"], (1, 1));
    assert_eq!(reach["B"], (1, -1));
    assert_eq!(reach["C"], (1, 1), "unsigned counts +1");
    assert_eq!(reach["T"], (2, -1));
    assert_eq!(reach["X"], (3, -1));
    assert!(!reach.contains_key("Y"), "beyond max_steps");
    let short = signed_bfs(&g, &["R"], 1);
    assert!(!short.contains_key("T"));
}

#[test]
fn complex_receptor_starts_from_every_subunit() {
    let g = graph(&[("R1", "T", 1), ("R2", "T", 1)]);
    let reach = signed_bfs(&g, &split_entity("R1_R2"), 3);
    assert_eq!(reach["T"], (1, 2));
}

#[test]
fn targets_weights_counts_and_top_n() {
    let g = diamond();
    // w_T = −1 / 2; the complex T_U has U unreached: (−0.5 + 0) / 2.
    let reg = regulons(&[
        ("T", &[("G1", 1), ("G2", -1)]),
        ("T_U", &[("G1", 1), ("G3", 1)]),
        ("U", &[("G4", 1)]),
    ]);
    let tfs = tf_parts(&reg);
    let rows = receptor_targets("R", &g, &reg, &tfs, 3, 10);
    let got: Vec<(&str, f64, usize, usize)> = rows
        .iter()
        .map(|r| (&*r.target, r.weight, r.n_tfs, r.min_path))
        .collect();
    assert_eq!(
        got,
        [("G1", -0.75, 2, 3), ("G2", 0.5, 1, 3), ("G3", -0.25, 1, 3)]
    );
    let top2 = receptor_targets("R", &g, &reg, &tfs, 3, 2);
    assert_eq!(top2.len(), 2);
    assert_eq!(&*top2[1].target, "G2");
}

#[test]
fn ties_by_name_and_cancelling_targets_dropped() {
    let g = graph(&[("R", "T1", 1), ("R", "T2", 1)]);
    let reg = regulons(&[
        ("T1", &[("ZZ", 1), ("AA", -1), ("GONE", 1)]),
        ("T2", &[("GONE", -1)]),
    ]);
    let rows = receptor_targets("R", &g, &reg, &tf_parts(&reg), 3, 10);
    let names: Vec<&str> = rows.iter().map(|r| &*r.target).collect();
    assert_eq!(names, ["AA", "ZZ"], "GONE sums to zero");
}

#[test]
fn receptor_that_is_a_tf_and_tf_with_zero_weight() {
    // R is itself a TF (d = 0, w = 1); T is reached by +1 and −1 paths
    // (S_T = 0), so it adds nothing and is not counted.
    let g = graph(&[("R", "A", 1), ("R", "B", -1), ("A", "T", 1), ("B", "T", 1)]);
    let reg = regulons(&[("R", &[("G", 1)]), ("T", &[("G", 1), ("H", 1)])]);
    let rows = receptor_targets("R", &g, &reg, &tf_parts(&reg), 3, 10);
    assert_eq!(
        rows,
        [TargetRow {
            target: "G".into(),
            weight: 1.0,
            n_tfs: 1,
            min_path: 1
        }]
    );
}

#[test]
fn signalling_consensus_licence_and_complexes() {
    let lic = Licences::from_json(resources_json()).unwrap();
    let mut text = ix_tsv(&[
        ("P1", "P2", "A", "B", 1, "Free"),
        ("P1", "P2", "A", "B", 0, "Free"),
        ("P1", "P3", "A", "C", 1, "Free"),
        ("P1", "P3", "A", "C", -1, "Free"),
        ("P4", "P5", "D", "E", 1, "NC"),
        ("C1", "P6", "F_G", "F", -1, "Free"),
    ]);
    text.push_str(
        "P7\tP8\tH\tI\tFalse\tFalse\tFalse\tFalse\tFalse\tFalse\tFree\tpost_translational\tprotein\tprotein\n",
    );
    let rows = parse_interactions(&text).unwrap();
    let (g, stats) = build_signalling(&rows, &lic, LicenceMode::Permissive);
    assert_eq!(g["A"]["B"], 1, "one non-zero sign wins over unsigned");
    assert_eq!(g["A"]["C"], 0, "both signs: unsigned");
    assert!(!g.contains_key("D"), "licence");
    assert_eq!(g["G"]["F"], -1);
    assert!(!g.contains_key("F"), "self-loop F → F dropped");
    assert!(!g.contains_key("H"), "undirected");
    assert_eq!(stats["licence_dropped"], 1);
    assert_eq!(stats["undirected"], 1);
    let (all, _) = build_signalling(&rows, &lic, LicenceMode::All);
    assert_eq!(all["D"]["E"], 1);
}

#[test]
fn regulons_drop_conflicts_and_complex_targets() {
    let lic = Licences::from_json(resources_json()).unwrap();
    let rows = parse_interactions(&ix_tsv(&[
        ("P1", "P2", "TF", "G1", 1, "Free"),
        ("P1", "P3", "TF", "G2", -1, "Free"),
        ("P1", "P4", "TF", "G3", 0, "Free"),
        ("P1", "P5", "TF", "G4", 1, "Free"),
        ("P1", "P5", "TF", "G4", -1, "Free"),
        ("P1", "C1", "TF", "X_Y", 1, "Free"),
        ("C2", "P2", "JUN_FOS", "G1", 1, "Foo_CollecTRI2"),
        ("P9", "P2", "NCTF", "G1", 1, "NC"),
    ]))
    .unwrap();
    let (reg, stats) = build_regulons(&rows, &lic, LicenceMode::Permissive);
    assert_eq!(reg["TF"].len(), 3);
    assert_eq!(reg["TF"]["G2"], -1);
    assert_eq!(reg["TF"]["G3"], 1, "unsigned regulation counts +1");
    assert_eq!(reg["JUN_FOS"]["G1"], 1);
    assert!(!reg.contains_key("NCTF"));
    assert_eq!(stats["conflicting_mor_dropped"], 1);
    assert_eq!(stats["complex_target"], 1);
}

#[test]
fn translation_keeps_mapped_genes() {
    let h2m: FxHashMap<Box<str>, Box<str>> = [("A", "a"), ("B", "b"), ("G", "g")]
        .into_iter()
        .map(|(h, m)| (h.into(), m.into()))
        .collect();
    let g = translate_graph(
        &graph(&[("A", "B", -1), ("A", "C", 1), ("C", "B", 1)]),
        &h2m,
    );
    assert_eq!(g.len(), 1);
    assert_eq!(g["a"].len(), 1);
    assert_eq!(g["a"]["b"], -1);
    let reg = translate_regulons(
        &regulons(&[
            ("A_B", &[("G", 1), ("H", -1)]),
            ("A_C", &[("G", 1)]),
            ("B", &[("H", 1)]),
        ]),
        &h2m,
    );
    assert_eq!(
        reg.len(),
        1,
        "A_C has an unmapped subunit, B no mapped target"
    );
    assert_eq!(reg["a_b"].len(), 1);
    assert_eq!(reg["a_b"]["g"], 1);
}

#[test]
fn fsum_is_exactly_rounded() {
    assert_eq!(fsum([0.1; 10]), 1.0);
    assert_eq!(fsum([1e100, 1.0, -1e100]), 1.0);
    // The exact sum of the doubles, as Python's math.fsum gives it.
    assert_eq!(
        fsum([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, -1.0]),
        -5.551115123125783e-17
    );
    assert_eq!(fsum(std::iter::empty()), 0.0);
}

#[test]
fn percent_g_formatting() {
    let cases = [
        (-14.0 / 3.0, "-4.66667"),
        (1.0 / 3.0, "0.333333"),
        (87.7, "87.7"),
        (2.0, "2"),
        (-0.25, "-0.25"),
        (100000.0, "100000"),
        (999999.5, "1e+06"),
        (1234567.0, "1.23457e+06"),
        (0.0001, "0.0001"),
        (0.00001234, "1.234e-05"),
        (0.0, "0"),
    ];
    for (x, want) in cases {
        assert_eq!(fmt_g(x), want, "{x}");
    }
}
