//! `pinto lra --link-community K`: the exposure through one link community's
//! contacts, its receivers alone fitted and reported, and the tables it
//! writes and `pinto lra merge` collects.

use super::lattice;
use super::marginal_run::{
    run_on_the_test_tissue, test_tissue_contacts, test_tissue_row, TestRunChoices,
};
use crate::lr_activity::ligand_effect::crossfit::LEFT_OUT;
use crate::lr_activity::ligand_effect::dose::contact_dose;
use crate::lr_activity::ligand_effect::link_community_scope::{
    link_community_sizes, pairs_of_link_community, q_scope, shard_path, table_and_curve_paths,
    LinkCommunityInputs,
};
use crate::lr_activity::ligand_effect::output::collect::{
    are_tables, by_link_community_path, collect_link_community_tables,
};
use crate::lr_activity::ligand_effect::pipeline::merge;
use crate::lr_activity::ligand_effect::shard::Shard;
use crate::lr_activity::ligand_effect::tissue::{Tissue, TissueInputs, TissueOpts};
use crate::util::common::Mat;
use crate::util::metadata::PintoMetadata;
use legume_numeric::matrix::parquet::{read_table_columns, write_named_table, Column};
use std::path::Path;

const SIDE: usize = 30;

/// The [`SIDE`]-square lattice as a tissue; with `community` (its number,
/// each contact's community, the fewest receivers) a `--link-community` one.
fn tissue_of(community: Option<(i64, &[i64], usize)>) -> anyhow::Result<Tissue> {
    let n = SIDE * SIDE;
    let (coords, pairs) = lattice(SIDE);
    let bin: Vec<usize> = (0..n).map(|i| (i / SIDE) / 10).collect();
    let identity = Mat::from_fn(2, n, |k, i| bin[i] as f32 * (k as f32 + 1.0));
    let section: Vec<Box<str>> = vec!["s0".into(); n];
    let depth = vec![100f32.ln_1p(); n];
    let community_inputs =
        community.map(
            |(link_community, of_pair, min_receivers)| LinkCommunityInputs {
                link_community,
                pairs: pairs_of_link_community(&pairs, of_pair, link_community),
                min_receivers,
            },
        );
    Tissue::build(
        &TissueInputs {
            coords: &coords,
            section: &section,
            pairs: &pairs,
            identity: &identity,
            identity_bin: &bin,
            log1p_depth: &depth,
            cell_type: None,
            region: None,
            link_community: community_inputs.as_ref(),
        },
        &TissueOpts {
            folds: 4,
            tile: 6.0,
            ..TissueOpts::default()
        },
    )
}

/// Each lattice contact's community: 1 when both cells are in the left
/// `left_columns` columns, else 2.
fn left_and_right_communities(left_columns: usize) -> Vec<i64> {
    let (_, pairs) = lattice(SIDE);
    pairs
        .iter()
        .map(|&(i, j)| {
            if i % SIDE < left_columns && j % SIDE < left_columns {
                1
            } else {
                2
            }
        })
        .collect()
}

#[test]
fn a_community_of_every_link_exposes_folds_and_scopes_as_the_marginal_run() {
    let (_, pairs) = lattice(SIDE);
    let marginal = tissue_of(None).unwrap();
    let everything = vec![5i64; pairs.len()];
    let community = tissue_of(Some((5, &everything, 1))).unwrap();
    assert_eq!(community.cf.fold, marginal.cf.fold);
    assert_eq!(community.scope_rows, marginal.scope_rows);
    let ligand: Vec<f64> = (0..marginal.n()).map(|i| (i % 7) as f64).collect();
    let marginal_dose =
        contact_dose(marginal.exposure_contacts(), &marginal.size_factor, &ligand).1;
    let community_dose = contact_dose(
        community.exposure_contacts(),
        &community.size_factor,
        &ligand,
    )
    .1;
    assert_eq!(community_dose, marginal_dose);
}

#[test]
fn receivers_outside_the_community_are_out_of_every_fold_and_scope() {
    let communities = left_and_right_communities(SIDE / 2);
    let tissue = tissue_of(Some((1, &communities, 1))).unwrap();
    let scope = tissue.community.as_ref().unwrap();
    // the left half's cells, and none of the right half's, have a community-1 link
    for row in 0..tissue.n() {
        let in_left_half = tissue.cell[row] % SIDE < SIDE / 2;
        assert_eq!(scope.in_community[row], in_left_half, "row {row}");
        assert_eq!(tissue.cf.fold[row] == LEFT_OUT, !in_left_half, "row {row}");
    }
    assert!(!tissue.scope_rows.is_empty());
    for rows in &tissue.scope_rows {
        assert!(rows.iter().all(|&row| scope.in_community[row as usize]));
    }
    assert_eq!(tissue.scope_rows[0], scope.receiver_rows);
}

#[test]
fn a_receivers_dose_counts_only_its_community_links() {
    let communities = left_and_right_communities(SIDE / 2);
    let marginal = tissue_of(None).unwrap();
    let tissue = tissue_of(Some((1, &communities, 1))).unwrap();
    // ligand on one cell at the community's edge: column SIDE/2 - 1 of row 5
    let sender = 5 * SIDE + SIDE / 2 - 1;
    let row_of = |cell: usize| tissue.cell.iter().position(|&c| c == cell).unwrap();
    let mut ligand = vec![0.0; tissue.n()];
    ligand[row_of(sender)] = 10.0;
    let dose = contact_dose(tissue.exposure_contacts(), &tissue.size_factor, &ligand).1;
    let marginal_dose =
        contact_dose(marginal.exposure_contacts(), &marginal.size_factor, &ligand).1;
    let (left, right) = (row_of(sender - 1), row_of(sender + 1));
    // the left neighbour reaches it through a community-1 link, the right one does not
    assert!(dose[left] > 0.0 && marginal_dose[left] > 0.0);
    assert_eq!(dose[right], 0.0);
    assert!(marginal_dose[right] > 0.0);
}

#[test]
fn a_community_with_too_few_receivers_is_refused() {
    let communities = left_and_right_communities(2);
    let e = tissue_of(Some((1, &communities, 200)))
        .err()
        .expect("60 receivers is fewer than 200")
        .to_string();
    assert!(e.contains("link community 1 has 60 receivers"), "{e}");
}

#[test]
fn communities_are_listed_with_their_links_and_receivers() {
    let pairs = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")];
    let sizes = link_community_sizes(&pairs, &[7, 7, 3, 7]);
    let listed: Vec<(i64, usize, usize)> = sizes
        .iter()
        .map(|s| (s.link_community, s.links, s.receivers))
        .collect();
    assert_eq!(listed, [(3, 1, 2), (7, 3, 4)]);
}

#[test]
fn a_communitys_tables_shards_and_q_scope_are_its_own() {
    let only: [Box<str>; 1] = ["GENE1".into()];
    assert_eq!(
        table_and_curve_paths("o", &[], None),
        ("o.lra.parquet".into(), "o.lra.curve.parquet".into())
    );
    assert_eq!(
        table_and_curve_paths("o", &[], Some(4)).0,
        "o.lra.link-community-4.parquet"
    );
    assert!(table_and_curve_paths("o", &only, Some(4))
        .0
        .starts_with("o.lra.link-community-4.only-"));
    let s = Shard { index: 1, count: 3 };
    assert_eq!(
        shard_path("o", s, Some(4)),
        "o.lra.link-community-4.shard-1-of-3.cbor"
    );
    assert_eq!(
        q_scope(&[], 2, 2, Some(4)),
        "co-expression lfdr fitted within link community 4"
    );
}

#[test]
fn community_shards_merged_write_the_community_runs_tables() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("run.pinto.json");
    PintoMetadata {
        command: "lra".into(),
        ..Default::default()
    }
    .write(&manifest)
    .unwrap();
    // community 1: the contacts within the top 12 rows of each section
    let communities: Vec<i64> = test_tissue_contacts()
        .iter()
        .map(|&(i, j)| {
            if test_tissue_row(i) < 12 && test_tissue_row(j) < 12 {
                1
            } else {
                2
            }
        })
        .collect();
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let choices = |shard| TestRunChoices {
        only: &[],
        shard,
        marginal_baselines: false,
        link_community: Some((1, &communities)),
    };
    let whole = at("whole");
    run_on_the_test_tissue(dir.path(), &whole, &manifest, &choices(None));
    let (whole_table, whole_curve) = table_and_curve_paths(&whole, &[], Some(1));
    assert!(Path::new(&whole_table).exists());
    let files: Vec<String> = (0..2)
        .map(|index| {
            let s = Shard { index, count: 2 };
            let out = at(&format!("shard{index}/o"));
            run_on_the_test_tissue(dir.path(), &out, &manifest, &choices(Some(s)));
            shard_path(&out, s, Some(1))
        })
        .collect();
    let merged = at("merged");
    merge(&files, &merged).unwrap();
    let (merged_table, merged_curve) = table_and_curve_paths(&merged, &[], Some(1));
    for (a, b) in [(&whole_table, &merged_table), (&whole_curve, &merged_curve)] {
        assert_eq!(std::fs::read(a).unwrap(), std::fs::read(b).unwrap(), "{b}");
    }
    // recorded as a subset of the community, never as the full table
    let lra = PintoMetadata::read(&manifest).unwrap().outputs.lra;
    assert!(lra.table.is_none());
    assert!(lra
        .subsets
        .iter()
        .all(|s| s.options.as_ref().and_then(|o| o.link_community) == Some(1)));
    assert!(!lra.subsets.is_empty());
}

#[test]
fn collecting_tables_adds_their_link_community() {
    let dir = tempfile::tempdir().unwrap();
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let write = |path: &str, pairs: &[&str], effects: &[f32]| {
        let names: Vec<Box<str>> = pairs.iter().map(|&p| p.into()).collect();
        write_named_table(
            path,
            "pair",
            &names,
            &[("effect".into(), Column::F32(effects))],
        )
        .unwrap();
    };
    let marginal = at("run.lra.parquet");
    let community = at("run.lra.link-community-3.parquet");
    write(&marginal, &["L1:R1", "L2:R1"], &[0.5, 0.1]);
    write(&community, &["L1:R1"], &[0.9]);
    let tables = vec![marginal, community];
    assert!(are_tables(&tables));
    let path = collect_link_community_tables(&tables, &at("all")).unwrap();
    assert_eq!(path, by_link_community_path(&at("all")));
    let (labels, numbers) =
        read_table_columns(&path, &["link_community", "pair"], &["effect"]).unwrap();
    let community_of_row: Vec<&str> = labels[0].iter().map(AsRef::as_ref).collect();
    assert_eq!(community_of_row, ["all", "all", "3"]);
    assert_eq!(numbers[0], [0.5, 0.1, 0.9].map(|x: f32| x as f64));
    // --only subsets' tables name their community the same way; a curve
    // table is no effect table
    let subset = at("run.lra.link-community-3.only-gene1.parquet");
    write(&subset, &["L1:R1"], &[0.2]);
    let path = collect_link_community_tables(&[subset], &at("x")).unwrap();
    let (labels, _) = read_table_columns(&path, &["link_community"], &[]).unwrap();
    assert_eq!(&*labels[0][0], "3");
    let curve = at("run.lra.curve.parquet");
    write(&curve, &["L1:R1"], &[0.2]);
    assert!(collect_link_community_tables(&[curve], &at("y")).is_err());
}

#[test]
fn collecting_lines_up_columns_a_community_table_lacks() {
    let dir = tempfile::tempdir().unwrap();
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    // the marginal table with a --marginal-baselines column the community lacks
    let marginal = at("run.lra.parquet");
    let names: Vec<Box<str>> = vec!["L1:R1".into()];
    write_named_table(
        &marginal,
        "pair",
        &names,
        &[
            ("effect".into(), Column::F32(&[0.5])),
            ("moran_i".into(), Column::F32(&[0.2])),
        ],
    )
    .unwrap();
    let community = at("run.lra.link-community-2.parquet");
    write_named_table(
        &community,
        "pair",
        &names,
        &[("effect".into(), Column::F32(&[0.9]))],
    )
    .unwrap();
    let path = collect_link_community_tables(&[marginal, community], &at("all")).unwrap();
    let (labels, numbers) = read_table_columns(&path, &["link_community"], &["effect"]).unwrap();
    let community_of_row: Vec<&str> = labels[0].iter().map(AsRef::as_ref).collect();
    assert_eq!(community_of_row, ["all", "2"]);
    assert_eq!(numbers[0], [0.5f32 as f64, 0.9f32 as f64]);
    // the column the community lacks is kept, empty (null) on its row
    use arrow_array::Array;
    let batch = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
        std::fs::File::open(&path).unwrap(),
    )
    .unwrap()
    .build()
    .unwrap()
    .next()
    .unwrap()
    .unwrap();
    let moran_i = batch.column_by_name("moran_i").unwrap();
    assert_eq!((moran_i.len(), moran_i.null_count()), (2, 1));
    assert!(moran_i.is_null(1));
}

/// A table from before local fdrs (its BH `effect_dr_q`) is not collected
/// beside tables with lfdrs, which would leave each half of the column empty.
#[test]
fn collecting_refuses_a_table_with_q_values() {
    let dir = tempfile::tempdir().unwrap();
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let names: Vec<Box<str>> = vec!["L1:R1".into()];
    let old = at("run.lra.parquet");
    write_named_table(
        &old,
        "pair",
        &names,
        &[("effect_dr_q".into(), Column::F32(&[0.01]))],
    )
    .unwrap();
    let e = collect_link_community_tables(&[old], &at("all"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("rerun pinto lra"), "{e}");
}

#[cfg(feature = "view")]
#[test]
fn a_merges_table_is_known_from_its_inputs_names() {
    use crate::lr_activity::ligand_effect::link_community_scope::merge_output_of_inputs;
    let shards = |name: &str| vec![name.to_string()];
    assert_eq!(
        merge_output_of_inputs(&shards("o.lra.shard-0-of-2.cbor"), "m"),
        "m.lra.parquet"
    );
    assert_eq!(
        merge_output_of_inputs(&shards("o.lra.link-community-7.shard-0-of-2.cbor"), "m"),
        "m.lra.link-community-7.parquet"
    );
    assert_eq!(
        merge_output_of_inputs(&shards("o.lra.parquet"), "m"),
        "m.lra.by-link-community.parquet"
    );
}
