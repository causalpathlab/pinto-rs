//! The whole build from synthetic cached downloads, offline.

use super::{ix_tsv, resources_json};
use crate::lr_activity::ligand_effect::reference::sources::*;
use crate::lr_activity::ligand_effect::reference::*;
use flate2::write::GzEncoder;
use std::fs;
use std::io::Write;
use std::path::Path;

/// Human: LIGA → RECA (receptor) → K → TF → {G1 +, G2 −}; RECB has no
/// path to a TF. Mouse via HCOP one-to-one orthologs.
fn write_downloads(dir: &Path) {
    let d = dir.join("downloads");
    fs::create_dir_all(&d).unwrap();
    fs::write(d.join(RESOURCES), resources_json()).unwrap();
    fs::write(
        d.join(LIANA),
        "\"\",\"source\",\"target\",\"source_genesymbol\",\"target_genesymbol\",\"resource\"\n\
         \"1\",\"L1\",\"R1\",\"LIGA\",\"RECA\",\"consensus\"\n\
         \"2\",\"L2\",\"R2\",\"LIGB\",\"RECB\",\"consensus\"\n\
         \"3\",\"L1\",\"R1\",\"Liga\",\"Reca\",\"mouseconsensus\"\n",
    )
    .unwrap();
    fs::write(
        d.join(SIGNALLING),
        ix_tsv(&[
            ("R1", "PK", "RECA", "K", 1, "Free"),
            ("PK", "PT", "K", "TF", -1, "Free"),
            ("R2", "PX", "RECB", "X", 1, "NC"),
        ]),
    )
    .unwrap();
    fs::write(
        d.join(LIGREC),
        ix_tsv(&[
            ("L1", "R1", "LIGA", "RECA", 1, "Free;NC"),
            ("L2", "R2", "LIGB", "RECB", 1, "Free"),
        ]),
    )
    .unwrap();
    fs::write(
        d.join(COLLECTRI),
        ix_tsv(&[
            ("PT", "P1", "TF", "G1", 1, "Foo_CollecTRI2"),
            ("PT", "P2", "TF", "G2", -1, "Free"),
        ]),
    )
    .unwrap();
    // LIGB is not annotated as transmitted, so LIGB-RECB is dropped.
    fs::write(
        d.join(INTERCELL),
        "category\tparent\tdatabase\tscope\taspect\tsource\tuniprot\tgenesymbol\n\
         ligand\tligand\tOmniPath\tgeneric\tfunctional\tcomposite\tL1\tLIGA\n",
    )
    .unwrap();
    let hcop = "human_symbol\tmouse_symbol\tsupport\n\
                RECA\tReca\ta,b,c\nK\tKk\ta,b,c\nTF\tTf\ta,b,c\nG1\tGg1\ta,b,c\nG2\tGg2\ta,b\n";
    let mut gz = GzEncoder::new(Vec::new(), flate2::Compression::default());
    gz.write_all(hcop.as_bytes()).unwrap();
    fs::write(d.join(HCOP), gz.finish().unwrap()).unwrap();
}

#[test]
fn builds_both_species_offline_and_reuses() {
    let dir = tempfile::tempdir().unwrap();
    write_downloads(dir.path());
    let opts = ReferenceOpts {
        dir: Some(dir.path().to_path_buf()),
        ..Default::default()
    };
    let human = ensure_with(Species::Human, &opts, true).unwrap();
    assert_eq!(
        human.lr_pairs,
        dir.path().join("built/r2-permissive/lr_pairs.human.tsv")
    );
    assert_eq!(
        fs::read_to_string(&human.lr_pairs).unwrap(),
        "ligand\treceptor\tsource\tligand_subunits\treceptor_subunits\n\
         LIGA\tRECA\tFree\tLIGA\tRECA\n",
        "LIGB is not annotated as transmitted, so its pair is dropped"
    );
    // w_TF = −1 / 2: G1 −0.5, G2 +0.5, ties by name.
    assert_eq!(
        fs::read_to_string(&human.receptor_targets).unwrap(),
        "receptor\ttarget\tweight\tn_tfs\tmin_path\n\
         RECA\tG1\t-0.5\t1\t3\n\
         RECA\tG2\t0.5\t1\t3\n"
    );

    let record: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&human.record).unwrap()).unwrap();
    assert_eq!(record["licence"], "permissive");
    assert_eq!(record["builder_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(record["liana_commit"], LIANA_COMMIT);
    assert_eq!(record["hcop_one_to_one_pairs"], 4);
    assert_eq!(
        record["species"]["human"]["counts"]["receptors_with_targets"],
        1
    );
    assert_eq!(
        record["downloads"][RESOURCES]["sha256"],
        download::sha256_bytes(resources_json().as_bytes())
    );

    // Mouse was built alongside: G2's ortholog lacks evidence.
    let before = fs::read_to_string(&human.record).unwrap();
    let mouse = ensure_with(Species::Mouse, &opts, true).unwrap();
    assert_eq!(fs::read_to_string(&human.record).unwrap(), before, "reused");
    assert_eq!(
        fs::read_to_string(&mouse.receptor_targets).unwrap(),
        "receptor\ttarget\tweight\tn_tfs\tmin_path\nReca\tGg1\t-0.5\t1\t3\n"
    );
    assert_eq!(
        fs::read_to_string(&mouse.lr_pairs).unwrap().lines().nth(1),
        Some("Liga\tReca\tFree\tLiga\tReca")
    );
}

#[test]
fn all_licences_build_apart() {
    let dir = tempfile::tempdir().unwrap();
    write_downloads(dir.path());
    let opts = ReferenceOpts {
        dir: Some(dir.path().to_path_buf()),
        licence: LicenceMode::All,
        ..Default::default()
    };
    let files = ensure_with(Species::Human, &opts, true).unwrap();
    assert!(files.lr_pairs.starts_with(dir.path().join("built/r2-all")));
    let lr = fs::read_to_string(&files.lr_pairs).unwrap();
    assert!(lr.contains("LIGA\tRECA\tFree;NC\t"), "{lr}");
}

#[test]
fn offline_without_downloads_names_the_cache() {
    let dir = tempfile::tempdir().unwrap();
    let opts = ReferenceOpts {
        dir: Some(dir.path().to_path_buf()),
        ..Default::default()
    };
    let err = ensure_with(Species::Human, &opts, true)
        .unwrap_err()
        .to_string();
    assert!(err.contains(LIANA) && err.contains(HCOP), "{err}");
    assert!(err.contains(&dir.path().join("downloads").display().to_string()));
}

#[test]
fn civil_dates() {
    assert_eq!(civil_from_days(0), (1970, 1, 1));
    assert_eq!(civil_from_days(20_760), (2026, 11, 3));
    assert_eq!(civil_from_days(11_016), (2000, 2, 29));
}
