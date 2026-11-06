//! Licences, OmniPath and LIANA+ rows, provenance and HCOP orthologs.

use super::{ix_tsv, resources_json};
use crate::lr_activity::ligand_effect::reference::sources::*;
use crate::lr_activity::ligand_effect::reference::LicenceMode;

#[test]
fn licence_resolution_and_permissive_rule() {
    let lic = Licences::from_json(resources_json()).unwrap();
    assert!(lic.is_permissive("Free"));
    assert!(lic.is_permissive("Foo_CollecTRI"), "for_profit counts");
    assert!(
        lic.is_permissive("Foo_CollecTRI2"),
        "*_CollecTRI2 → *_CollecTRI"
    );
    assert!(!lic.is_permissive("NC"));
    assert!(!lic.is_permissive("SA"));
    assert!(!lic.is_permissive("hprd_ProtMapper"), "alias to HPRD");
    assert!(lic.resolve("hprd_ProtMapper").is_some());
    assert!(lic.resolve("Bare").is_some() && !lic.is_permissive("Bare"));
    assert!(lic.resolve("Unknown").is_none());
}

#[test]
fn keep_sources_filters_by_mode() {
    let lic = Licences::from_json(resources_json()).unwrap();
    assert_eq!(
        lic.keep_sources("NC;Free;Free;Foo_CollecTRI2;", LicenceMode::Permissive),
        Some(vec!["Foo_CollecTRI2", "Free"])
    );
    assert_eq!(lic.keep_sources("NC;SA", LicenceMode::Permissive), None);
    assert_eq!(
        lic.keep_sources("SA;NC", LicenceMode::All),
        Some(vec!["NC", "SA"])
    );
    assert_eq!(lic.keep_sources("", LicenceMode::All), Some(vec![]));
}

#[test]
fn interactions_parse_signs_and_flags() {
    let rows = parse_interactions(&ix_tsv(&[
        ("P1", "P2", "A", "B", 1, "Free"),
        ("P3", "P4", "C", "D_E", -1, "NC;Free"),
        ("P5", "P6", "F", "G", 0, "SA"),
    ]))
    .unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter().map(Interaction::sign).collect::<Vec<_>>(),
        [1, -1, 0]
    );
    assert!(rows[0].is_directed && rows[0].consensus_direction);
    assert_eq!(&*rows[1].entity_type_target, "complex");
    assert_eq!(&*rows[1].sources, "NC;Free");
}

#[test]
fn lr_pairs_with_provenance() {
    let lic = Licences::from_json(resources_json()).unwrap();
    let omni = parse_interactions(&ix_tsv(&[
        ("L1", "R1", "LIGA", "RECA", 1, "NC;Free"),
        ("L1", "R1", "LIGA", "RECA", 1, "Foo_CollecTRI2"),
        ("L2", "R2", "LIGB", "RECB", 1, "NC"),
    ]))
    .unwrap();
    let liana = parse_liana(
        "\"\",\"source\",\"target\",\"source_genesymbol\",\"target_genesymbol\",\"resource\"\n\
         \"1\",\"L1\",\"R1\",\"LIGA\",\"RECA\",\"consensus\"\n\
         \"2\",\"L2\",\"R2\",\"LIGB\",\"RECB\",\"consensus\"\n\
         \"3\",\"L3\",\"R3\",\"LIGC\",\"RECC\",\"consensus\"\n\
         \"4\",\"L1\",\"R1\",\"Liga\",\"Reca\",\"mouseconsensus\"\n",
    )
    .unwrap();

    let prov = Provenance::new(omni.iter(), &lic, LicenceMode::Permissive);
    let (human, unsupported) = lr_pairs(&liana, &prov, "consensus");
    assert_eq!(unsupported, 2, "L2-R2 rests on NC only, L3-R3 on nothing");
    assert_eq!(
        human,
        [LrPair {
            ligand: "LIGA".into(),
            receptor: "RECA".into(),
            sources: "Foo_CollecTRI2;Free".into(),
        }]
    );
    assert_eq!(
        human[0].row(),
        ["LIGA", "RECA", "Foo_CollecTRI2;Free", "LIGA", "RECA"].map(String::from)
    );
    let (mouse, _) = lr_pairs(&liana, &prov, "mouseconsensus");
    assert_eq!(&*mouse[0].ligand, "Liga", "mouse keeps the human ids");

    let prov = Provenance::new(omni.iter(), &lic, LicenceMode::All);
    let (all, unsupported) = lr_pairs(&liana, &prov, "consensus");
    assert_eq!(unsupported, 1);
    let sources: Vec<&str> = all.iter().map(|p| p.sources.as_str()).collect();
    assert_eq!(sources, ["Foo_CollecTRI2;Free;NC", "NC", "LIANA_consensus"]);
}

#[test]
fn hcop_keeps_one_to_one_with_evidence() {
    let text = "human_symbol\tmouse_symbol\tsupport\n\
                GENE1\tTrp53\ta,b,c\n\
                ONE\tOne1\ta,b,c\n\
                ONE\tOne2\ta,b,c,d\n\
                WEAK\tWeak\ta,b\n\
                -\tGm1\ta,b,c\n\
                SHARE1\tShared\ta,b,c\n\
                SHARE2\tShared\ta,b,c\n\
                SHARE2\tShared\ta,b,c\n";
    let h2m = hcop_one_to_one(text, 3).unwrap();
    assert_eq!(h2m.len(), 1, "{h2m:?}");
    assert_eq!(h2m.get("GENE1").map(|m| &**m), Some("Trp53"));
}

#[test]
fn only_transmitted_ligands_are_kept() {
    let ligands = parse_intercell_ligands(
        "category\tgenesymbol\nligand\tLIGB1\nligand\tCOMPLEX:LIGI1_LIGI2\n",
    )
    .unwrap();
    assert!(ligands.contains("LIGI2") && ligands.contains("LIGB1"));
    let pair = |l: &str| LrPair {
        ligand: l.into(),
        receptor: "R".into(),
        sources: String::new(),
    };
    let (kept, dropped) = transmitted_pairs(
        vec![
            pair("LIGB1"),
            pair("GENE2"),
            pair("LIGI1_LIGI2"),
            pair("LIGI1_RECK1"),
        ],
        &ligands,
        &|s| s.to_string(),
    );
    assert_eq!(dropped, 2);
    let names: Vec<&str> = kept.iter().map(|p| &*p.ligand).collect();
    assert_eq!(names, ["LIGB1", "LIGI1_LIGI2"]);
    // Mouse symbols go through the ortholog map, else compare upper-cased.
    let (kept, _) = transmitted_pairs(vec![pair("Ligb1")], &ligands, &|s| s.to_string());
    assert_eq!(kept.len(), 1);
}
