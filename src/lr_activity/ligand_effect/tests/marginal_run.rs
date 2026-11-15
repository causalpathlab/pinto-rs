//! `pinto lra --marginal-baselines` from data to tables: the contact-effect
//! pipeline on a small tissue with named cell types, in full and with
//! `--only`, its tables and its record in the manifest.

use super::super::only;
use super::super::output::{celltype_beside, lra_path};
use super::super::pipeline::run;
use super::super::reference::{with_test_root, write_empty_reference, Species};
use super::super::tissue::ALL_SECTIONS;
use super::super::{RunInputs, Settings};
use super::{lattice_contacts, type_names};
use crate::lr_activity::args::{ReferenceSpecies, RegionJackknife};
use crate::lr_activity::recorded;
use crate::test_support::{numbered, sparse_vec};
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use crate::util::metadata::PintoMetadata;
use legume_numeric::matrix::parquet::read_table_columns;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};
use std::path::Path;

/// Cells per side of each of the two sections.
const SIDE: usize = 20;
const SECTIONS: usize = 2;
/// Permutations asked for: more than one round of the early stop, so a
/// test that stopped early would show fewer.
const PERMUTATIONS: usize = 1000;
const IDENTITY_GENES: usize = 30;
const TARGETS: usize = 8;

fn gene(g: usize) -> Box<str> {
    format!("GENE{}", g + 1).into()
}

/// GENE1 and GENE3 the ligands, GENE2 their receptor, GENE4 to GENE11 the
/// targets, the rest the receivers' identity.
fn n_genes() -> usize {
    3 + TARGETS + IDENTITY_GENES
}

/// A cell's row and column within its section.
fn place(c: usize) -> (usize, usize) {
    let i = c % (SIDE * SIDE);
    (i / SIDE, i % SIDE)
}

/// The identity group (by rows), which the identity genes and the
/// receptor follow, and the cell type (by columns), which nothing follows.
fn group(c: usize) -> usize {
    place(c).0 / 7
}

fn type_code(c: usize) -> u32 {
    (place(c).1 / 7) as u32
}

/// Counts per gene and cell: the identity genes high in their group, the
/// receptor rare in the last group, ligands and targets everywhere.
fn triplets(n: usize) -> Vec<(u64, u64, f32)> {
    let mut rng = SmallRng::seed_from_u64(5);
    let mut out = Vec::new();
    for c in 0..n {
        for g in 0..n_genes() {
            let rate = match g {
                0 | 2 => 0.4,
                1 if group(c) == 2 => 0.02,
                1 => 0.8,
                g if g < 3 + TARGETS => 1.0,
                g => 0.3 + 2.0 * f64::from(u8::from((g - 3 - TARGETS) % 3 == group(c))),
            };
            let x = Poisson::new(rate).unwrap().sample(&mut rng);
            if x > 0.0 {
                out.push((g as u64, c as u64, x as f32));
            }
        }
    }
    out
}

fn data(dir: &Path, n: usize) -> SparseIoVec {
    let rows: Vec<Box<str>> = (0..n_genes()).map(gene).collect();
    sparse_vec(dir, "d", &rows, &numbered("c", n), &triplets(n))
}

/// The pair file and the receptor's programme, beside an empty reference.
fn reference_and_files(dir: &Path) -> (String, String) {
    write_empty_reference(&dir.join("reference"), Species::Human).unwrap();
    let pairs = dir.join("pairs.tsv");
    std::fs::write(&pairs, "ligand\treceptor\nGENE1\tGENE2\nGENE3\tGENE2\n").unwrap();
    let targets = dir.join("targets.tsv");
    let mut t = String::from("receptor\ttarget\tweight\n");
    for g in 3..3 + TARGETS {
        t += &format!("GENE2\t{}\t1\n", gene(g));
    }
    std::fs::write(&targets, t).unwrap();
    let s = |p: &Path| p.to_string_lossy().into_owned();
    (s(&pairs), s(&targets))
}

/// `pinto lra --marginal-baselines --marginal-permutations` [`PERMUTATIONS`]
/// with `only` into `out`, the run's manifest at `manifest`.
fn marginal_run(dir: &Path, out: &str, manifest: &Path, only: &[Box<str>]) {
    let n = SECTIONS * SIDE * SIDE;
    let data = data(dir, n);
    let axis =
        FeatureAxis::resolve_or_identity(&(0..n_genes()).map(gene).collect::<Vec<_>>()).unwrap();
    let (pairs_file, targets_file) = reference_and_files(dir);
    let targets = [targets_file.into_boxed_str()];
    let coords: Vec<Option<(f32, f32)>> = (0..n)
        .map(|c| {
            let (r, col) = place(c);
            let shift = (c / (SIDE * SIDE) * 10 * SIDE) as f32;
            Some((col as f32 * 10.0 + shift, r as f32 * 10.0))
        })
        .collect();
    let section: Vec<Box<str>> = (0..n)
        .map(|c| format!("s{}", c / (SIDE * SIDE)).into())
        .collect();
    let codes: Vec<u32> = (0..n).map(type_code).collect();
    let names = type_names();
    let pairs = lattice_contacts(SIDE, SECTIONS);
    with_test_root(&dir.join("reference"), || {
        run(&RunInputs {
            settings: Settings {
                out,
                manifest,
                lr_pairs: Some(&pairs_file),
                targets: &targets,
                species: ReferenceSpecies::Human,
                seed: 7,
                block_size: None,
                only,
                cell_types: Some("ct.tsv"),
                region: None,
                plant_region: None,
                region_jackknife: RegionJackknife::NoRefit,
                marginal: Some(PERMUTATIONS),
            },
            data: &data,
            axis: &axis,
            pairs: &pairs,
            cell_section: &section,
            coords: &coords,
            cell_type: Some((&codes, &names)),
            region: None,
        })
    })
    .unwrap();
}

/// The per-type-pair table at `path`: its type names and labels, and the
/// permutations of each row.
fn celltype_table(path: &str) -> (Vec<Box<str>>, Vec<Box<str>>, Vec<f64>) {
    let (s, x) = read_table_columns(
        path,
        &["sender_type", "receiver_type", "labels"],
        &["permutations"],
    )
    .unwrap();
    let mut s = s.into_iter();
    let (senders, receivers, labels) = (s.next().unwrap(), s.next().unwrap(), s.next().unwrap());
    let types = senders.into_iter().chain(receivers).collect();
    (types, labels, x.into_iter().next().unwrap())
}

/// Every row of the per-type-pair table is on the run's cell types, never
/// `unassigned`, and has every permutation asked for.
fn assert_full_celltype_table(path: &str) {
    assert!(Path::new(path).is_file(), "{path}");
    let (types, labels, permutations) = celltype_table(path);
    assert!(!labels.is_empty(), "type pairs were tested");
    assert!(labels.iter().all(|l| &**l == "cell-types"), "{labels:?}");
    assert!(
        types.iter().all(|t| ["CT1", "CT2", "CT3"].contains(&&**t)),
        "{types:?}"
    );
    assert!(
        permutations.iter().all(|&p| p == PERMUTATIONS as f64),
        "{permutations:?}"
    );
}

/// From data to tables: real labels, every permutation, the baselines on
/// the pooled rows alone, and the per-type-pair table recorded in the
/// manifest, for the full table and for a `--only` subset.
#[test]
fn a_run_with_marginal_baselines_writes_and_records_its_tables() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("run.pinto.json");
    PintoMetadata {
        command: "lra".into(),
        ..Default::default()
    }
    .write(&manifest)
    .unwrap();
    let out = dir.path().join("o").to_string_lossy().into_owned();

    marginal_run(dir.path(), &out, &manifest, &[]);
    let table = lra_path(&out);
    let celltype = celltype_beside(&table);
    assert_full_celltype_table(&celltype);
    let (s, x) =
        read_table_columns(&table, &["section"], &["celltype_p", "celltype_tested"]).unwrap();
    let pooled: Vec<bool> = s[0].iter().map(|r| &**r == ALL_SECTIONS).collect();
    assert!(pooled.iter().any(|&p| p), "{:?}", s[0]);
    assert!(
        pooled.iter().any(|&p| !p),
        "rows besides the pooled: {:?}",
        s[0]
    );
    for (i, &p) in pooled.iter().enumerate() {
        let (cp, tested) = (x[0][i], x[1][i]);
        if p {
            assert!(cp.is_finite() && tested > 0.0, "{} {cp} {tested}", s[0][i]);
        } else {
            assert!(cp.is_nan() && tested == 0.0, "{} {cp} {tested}", s[0][i]);
        }
    }
    let lra = PintoMetadata::read(&manifest).unwrap().outputs.lra;
    assert_eq!(
        lra.celltype.as_deref(),
        Some(recorded(Path::new(&celltype)).as_str())
    );
    let options = lra.options.unwrap();
    assert_eq!(options.marginal_baselines.as_deref(), Some("cell-types"));
    assert_eq!(options.marginal_permutations, Some(PERMUTATIONS));

    let subset: Vec<Box<str>> = vec!["GENE1:GENE2".into()];
    marginal_run(dir.path(), &out, &manifest, &subset);
    let sub_table = only::table_path(&out, &subset);
    let sub_celltype = celltype_beside(&sub_table);
    assert_full_celltype_table(&sub_celltype);
    let lra = PintoMetadata::read(&manifest).unwrap().outputs.lra;
    let entry = lra
        .subsets
        .iter()
        .find(|e| e.table == recorded(Path::new(&sub_table)))
        .expect("the subset is recorded");
    assert_eq!(
        entry.celltype.as_deref(),
        Some(recorded(Path::new(&sub_celltype)).as_str())
    );
    assert_eq!(
        lra.celltype.as_deref(),
        Some(recorded(Path::new(&celltype)).as_str()),
        "the full table's record stays"
    );
}
