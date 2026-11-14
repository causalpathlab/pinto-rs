//! Round-trip tests for the run manifest.
//!
//! The manifest is a published contract: `pinto plot`, `pinto prop` and
//! `pinto lra` all locate their inputs through it, so a field that fails to
//! serialize is a broken pipeline rather than a cosmetic defect.

use crate::util::metadata::*;

/// Without `--merge-cut` the merge tree is still written and the manifest
/// points at it, but there is no cut, so no cut file is named.
#[test]
fn an_lc_run_without_a_cut_names_the_tree_and_no_cut() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let data_files: Vec<Box<str>> = vec!["a.h5".into()];
    let meta = create_lc_metadata(
        &RunInputs {
            prefix: &prefix,
            data_files: &data_files,
            coord_file: None,
            coord_columns: &[],
            n_cells: 10,
            n_features: 20,
            n_edges: 30,
            k: 4,
            graph: GraphParams::default(),
            ablation: None,
        },
        Some(DictMergeSummary {
            min_nnz: 1,
            features_scored: 10,
            cut_cosine: None,
            centred_sections: None,
        }),
        None,
        &[0],
    );
    let path = dir.path().join("run.pinto.json");
    meta.write(&path).unwrap();
    let back = PintoMetadata::read(&path).unwrap();
    let merge = back.outputs.dict_merge.as_ref().expect("dict_merge");
    assert!(merge.merges.ends_with("run.dict_merges.parquet"));
    assert!(merge.cut.is_none());
    assert!(merge.cut_cosine.is_none());
}

#[test]
fn metadata_roundtrip_lc() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let data_files: Vec<Box<str>> = vec!["a.h5".into(), "b.h5".into()];
    let coord_cols: Vec<Box<str>> = vec!["pxl_row_in_fullres".into(), "pxl_col_in_fullres".into()];
    let meta = create_lc_metadata(
        &RunInputs {
            prefix: &prefix,
            data_files: &data_files,
            coord_file: Some("a.tsv,b.tsv"),
            coord_columns: &coord_cols,
            n_cells: 1234,
            n_features: 18000,
            n_edges: 55555,
            k: 12,
            graph: GraphParams::default(),
            ablation: None,
        },
        Some(DictMergeSummary {
            min_nnz: 1,
            features_scored: 10,
            cut_cosine: Some(0.25),
            centred_sections: Some(3),
        }),
        // A channelized `lc` run reports the structural fact of its feature
        // axis.
        Some(SpliceTrackInfo {
            n_rows: 36000,
            n_delta_identified: 13000,
            nascent_count_fraction: 0.21,
            delta_base: DELTA_BASE_SPLICED.to_string(),
        }),
        &[0, 1, 2],
    );
    let path = dir.path().join("run.pinto.json");
    meta.write(&path).unwrap();
    let back = PintoMetadata::read(&path).unwrap();
    assert_eq!(back.command, "lc");
    assert_eq!(back.n_cells, 1234);
    assert_eq!(back.n_communities, Some(12));
    let levels = back.levels.expect("levels");
    // 3 cascade levels + final = 4 (final carries the merged consensus)
    assert_eq!(levels.len(), 4);
    assert_eq!(levels[0].tag, "L0");
    assert_eq!(levels[3].tag, "final");
    assert_eq!(levels[3].entropy_present, Some(true));
    let merge = back.outputs.dict_merge.as_ref().expect("dict_merge");
    assert_eq!(merge.cut_cosine, Some(0.25));
    assert!(merge
        .cut
        .as_deref()
        .is_some_and(|c| c.ends_with("run.dict_merges.cut.parquet")));
    assert_eq!(merge.centred_sections, Some(3));
    assert!(back.outputs.lr_activity.is_none());
    assert_eq!(
        back.outputs.coord_columns.as_deref(),
        Some(
            &[
                "pxl_row_in_fullres".to_string(),
                "pxl_col_in_fullres".to_string()
            ][..]
        )
    );

    // The splice block is what tells a consumer the axis was channelized, so a
    // round trip that drops it would be silent.
    let splice = back.splice.as_ref().expect("splice block must survive");
    assert_eq!(splice.n_rows, 36000);
    assert_eq!(splice.n_delta_identified, 13000);
    assert_eq!(splice.delta_base, DELTA_BASE_SPLICED);
}

#[test]
fn metadata_roundtrip_cage() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let data_files: Vec<Box<str>> = vec!["a.h5".into()];
    let coord_cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    let meta = create_cage_metadata(
        &RunInputs {
            prefix: &prefix,
            data_files: &data_files,
            coord_file: Some("a.csv"),
            coord_columns: &coord_cols,
            n_cells: 1000,
            n_features: 20000,
            n_edges: 5000,
            k: 16, // edge clusters
            graph: GraphParams::default(),
            ablation: None,
        },
        true,
        Some(SpliceTrackInfo {
            n_rows: 40000,
            n_delta_identified: 15000,
            nascent_count_fraction: 0.23,
            delta_base: DELTA_BASE_SPLICED.to_string(),
        }),
    );
    let path = dir.path().join("run.pinto.json");
    meta.write(&path).unwrap();
    let back = PintoMetadata::read(&path).unwrap();
    assert_eq!(back.command, "cage");
    assert_eq!(back.n_cells, 1000);
    assert_eq!(back.n_communities, Some(16));
    assert!(back.outputs.cell_embedding.is_some());
    // The encoder that placed the cells and pairs ships beside them, so
    // `predict` can place a new sample by the same map.
    assert_eq!(
        back.outputs.pair_encoder.as_deref(),
        Some(format!("{prefix}.pair_encoder.safetensors").as_str())
    );
    // The trained unit is the PB: pb tables + the cell->pb map ship,
    // and there is no per-cell bias to report.
    assert!(back.outputs.cell_bias.is_none());
    assert!(back.outputs.pb_embedding.is_some());
    assert!(back.outputs.pb_bias.is_some());
    assert!(back.outputs.cell_pb.is_some());
    assert!(back.outputs.feature_embedding.is_some());
    assert!(back.outputs.feature_bias.is_some());
    assert!(back.outputs.scores.is_some());
    // A channelized run reports FEATURES on `n_features` and keeps the matrix's
    // own row count in the splice block — reading `n_features` as a row count
    // is exactly the confusion the two-field split exists to prevent.
    let splice = back.splice.expect("splice block round-trips");
    assert_eq!(back.n_features, 20000);
    assert_eq!(splice.n_rows, 40000);
    assert_eq!(splice.n_delta_identified, 15000);
    assert_eq!(splice.delta_base, "spliced");
    assert!(back.outputs.batch_effects.is_some());
    assert!(back.outputs.clusters.is_none());
    let levels = back.levels.expect("levels");
    assert_eq!(levels.len(), 1);
    assert_eq!(levels[0].tag, "final");
    // The point of the pair projection: cage's level is the SAME shape lc and
    // dsvd publish — a real propensity (with entropy), a per-edge community
    // table, and a feature x community dictionary.
    assert!(levels[0].propensity.ends_with(".propensity.parquet"));
    assert!(levels[0]
        .link_community
        .as_deref()
        .unwrap()
        .ends_with(".link_community.parquet"));
    assert!(levels[0]
        .feature_community
        .as_deref()
        .unwrap()
        .ends_with(".feature_community.parquet"));
    assert_eq!(levels[0].entropy_present, Some(true));
}

#[test]
fn metadata_roundtrip_cage_no_batch() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let data_files: Vec<Box<str>> = vec!["a.h5".into()];
    let meta = create_cage_metadata(
        &RunInputs {
            prefix: &prefix,
            data_files: &data_files,
            coord_file: None,
            coord_columns: &[],
            n_cells: 100,
            n_features: 200,
            n_edges: 300,
            k: 8,
            graph: GraphParams::default(),
            ablation: None,
        },
        false,
        None,
    );
    // The encoder is part of every cage model, batches or not.
    assert_eq!(
        meta.outputs.pair_encoder.as_deref(),
        Some(format!("{prefix}.pair_encoder.safetensors").as_str())
    );
    let json = serde_json::to_string(&meta).unwrap();
    let back: PintoMetadata = serde_json::from_str(&json).unwrap();
    assert!(back.outputs.batch_effects.is_none());
    // Absent, not zeroed: "this input had no channels" and "this input had
    // channels that identified nothing" are different findings.
    assert!(back.splice.is_none());
    assert!(!json.contains("splice"));
}

#[test]
fn metadata_roundtrip_lc_merge_no_collapse() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let data_files: Vec<Box<str>> = vec!["a.h5".into()];
    let meta = create_lc_metadata(
        &RunInputs {
            prefix: &prefix,
            data_files: &data_files,
            coord_file: None,
            coord_columns: &[],
            n_cells: 100,
            n_features: 200,
            n_edges: 300,
            k: 8,
            graph: GraphParams::default(),
            ablation: None,
        },
        None,
        None,
        &[],
    );
    let path = dir.path().join("run.pinto.json");
    meta.write(&path).unwrap();
    let back = PintoMetadata::read(&path).unwrap();
    let levels = back.levels.expect("levels");
    // 0 cascade levels + final = 1
    assert_eq!(levels.len(), 1);
    assert_eq!(levels[0].tag, "final");
    assert!(back.outputs.dict_merge.is_none());
}

/// Round-trip one `GraphParams` through `create_lc_metadata` and the JSON.
///
/// The `TempDir` comes back with it: dropping it would delete the file the
/// read went through, and the two tests below differ only in the block and the
/// assertions, not the scaffolding.
fn roundtrip_graph(graph: GraphParams) -> (tempfile::TempDir, GraphParams) {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let data_files: Vec<Box<str>> = vec!["a.h5".into()];
    let coord_cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    let meta = create_lc_metadata(
        &RunInputs {
            prefix: &prefix,
            data_files: &data_files,
            coord_file: Some("a.tsv"),
            coord_columns: &coord_cols,
            n_cells: 1234,
            n_features: 18000,
            n_edges: 55555,
            k: 12,
            graph,
            ablation: None,
        },
        None,
        None,
        &[0],
    );
    let path = dir.path().join("run.pinto.json");
    meta.write(&path).unwrap();
    let back = PintoMetadata::read(&path)
        .unwrap()
        .graph
        .expect("graph parameters must round-trip");
    (dir, back)
}

/// Which graph produced the pairs must survive into the manifest.
///
/// Without this the flipped `--knn-expr` default is unreproducible from
/// artifacts: a run's pair count is recorded, but not the two k's that decided
/// it, so the only way to tell an augmented run from a spatial-only one is to
/// open the parquet and notice a missing column.
#[test]
fn metadata_records_the_graph_parameters() {
    let (_dir, g) = roundtrip_graph(GraphParams {
        knn_base: 5,
        knn_expr: 7,
        knn_expr_scope: Some("global".to_string()),
        reciprocal: false,
    });
    assert_eq!(g.knn_base, 5);
    assert_eq!(g.knn_expr, 7);
    assert_eq!(g.knn_expr_scope.as_deref(), Some("global"));
    assert!(g.knn_expr > 0, "this run unioned expression pairs in");
}

/// The default spatial run is the one that must be distinguishable. Its
/// omitted scope has to survive `skip_serializing_if` and come back `None`
/// rather than defaulting to a scope no search used.
#[test]
fn metadata_records_an_unaugmented_run_as_such() {
    let (_dir, g) = roundtrip_graph(GraphParams {
        knn_base: 5,
        knn_expr: 0,
        knn_expr_scope: None,
        reciprocal: false,
    });
    assert_eq!(g.knn_expr, 0, "and no union is recorded");
    assert_eq!(
        g.knn_expr_scope, None,
        "no search ran, so no scope is claimed"
    );
}

/// A `graph` block written by an older build must not sink the whole manifest.
///
/// This block gains fields over time, and `PintoMetadata::backfill_output`
/// swallows a read error at debug level — so a missing-field failure here is
/// silent, and `pinto plot --from` / `pinto lra --from` abort outright on a
/// manifest they should have read.
#[test]
fn an_older_graph_block_still_deserializes() {
    let json = r#"{
        "command": "lc",
        "version": "0.6.9",
        "timestamp": "0",
        "prefix": "run",
        "n_cells": 10,
        "n_features": 10,
        "graph": {"knn_base": 5, "knn_expr": 0, "augmented": false},
        "outputs": {}
    }"#;
    let back: PintoMetadata = serde_json::from_str(json).expect("must stay readable");
    let g = back.graph.expect("the block that was present must survive");
    assert_eq!(g.knn_base, 5);
    assert!(
        !g.reciprocal,
        "a field the writer never knew about defaults"
    );
    assert_eq!(g.knn_expr, 0, "no union, however the block was written");
}

/// The lra settings block is a record, not a schema: a block from another
/// version (a field renamed, a type changed) never fails the manifest.
#[test]
fn an_unknown_lra_settings_block_still_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r.pinto.json");
    let mut m = serde_json::to_value(PintoMetadata::default()).unwrap();
    m["outputs"]["lra"] = serde_json::json!({
        "table": "/x/r.lra.parquet",
        "settings": {"draws": 7, "renamed_field": [1, 2]},
        "a_later_slot": "/x/new.parquet"
    });
    std::fs::write(&path, m.to_string()).unwrap();
    let read = PintoMetadata::read(&path).unwrap();
    assert_eq!(read.outputs.lra.table.as_deref(), Some("/x/r.lra.parquet"));
    assert_eq!(read.outputs.lra.settings.as_ref().unwrap()["draws"], 7);
}

/// A manifest named in full but not `{prefix}.pinto.json` finds the run's
/// files through the prefix it records, beside itself.
#[test]
fn a_manifest_named_otherwise_finds_its_run_by_its_recorded_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let meta = PintoMetadata {
        prefix: "elsewhere/run1".into(),
        ..PintoMetadata::default()
    };
    let named = dir.path().join("run1.json");
    meta.write(&named).unwrap();
    let run = crate::util::metadata::Run::open(&named.to_string_lossy()).unwrap();
    assert_eq!(run.prefix, dir.path().join("run1").to_string_lossy());
    let usual = dir.path().join("copy.pinto.json");
    meta.write(&usual).unwrap();
    let run = crate::util::metadata::Run::open(&usual.to_string_lossy()).unwrap();
    assert_eq!(run.prefix, dir.path().join("copy").to_string_lossy());
}

/// The genes an lra identity left out are one line of the manifest, read
/// back as written; a name with a comma makes them an array, read too; a
/// manifest from before them reads as none.
#[test]
fn lra_identity_genes_are_one_line_and_read_back() {
    let options = |genes: &[&str]| LraOptions {
        identity_excluded: genes.iter().map(|g| g.to_string()).collect(),
        ..LraOptions::default()
    };
    let o = options(&["ENSG1_CFH", "GCLC", "NFYA"]);
    let json = serde_json::to_string_pretty(&o).unwrap();
    assert!(
        json.contains("\"identity_excluded\": \"ENSG1_CFH,GCLC,NFYA\""),
        "{json}"
    );
    assert_eq!(serde_json::from_str::<LraOptions>(&json).unwrap(), o);
    let odd = options(&["A,B", "C"]);
    let json = serde_json::to_string(&odd).unwrap();
    assert_eq!(serde_json::from_str::<LraOptions>(&json).unwrap(), odd);
    let old: LraOptions = serde_json::from_str(r#"{"seed": 42}"#).unwrap();
    assert!(old.identity_excluded.is_empty());
    assert!(!serde_json::to_string(&old).unwrap().contains("identity"));
}

/// `--marginal-baselines` is recorded with where its cell types came from
/// and its permutations; a run without it records neither, and a manifest
/// from before it reads as without.
#[test]
fn lra_marginal_baselines_are_recorded_only_when_asked() {
    let without = serde_json::to_string(&LraOptions::default()).unwrap();
    assert!(!without.contains("marginal"), "{without}");
    let with = LraOptions {
        marginal_baselines: Some("cell-types".into()),
        marginal_permutations: Some(500),
        ..LraOptions::default()
    };
    let json = serde_json::to_string(&with).unwrap();
    assert!(
        json.contains(r#""marginal_baselines":"cell-types""#),
        "{json}"
    );
    assert!(json.contains(r#""marginal_permutations":500"#), "{json}");
    assert_eq!(serde_json::from_str::<LraOptions>(&json).unwrap(), with);
    let old: LraOptions = serde_json::from_str(r#"{"seed": 42}"#).unwrap();
    assert_eq!(old.marginal_baselines, None);
}

/// A subset's identity genes are stored once: a subset made as the full
/// table was keeps no copy and reads the full list back; one made with
/// other pairs keeps its own; a subset recorded before any full table
/// keeps its own until one with the same list comes; and a new full table
/// with another list leaves what an inheriting subset reads unchanged.
#[test]
fn lra_subset_identity_genes_are_stored_once() {
    let options = |genes: &[&str], lr_pairs: Option<&str>| LraOptions {
        identity_excluded: genes.iter().map(|g| g.to_string()).collect(),
        lr_pairs: lr_pairs.map(String::from),
        ..LraOptions::default()
    };
    let subset = |table: &str, o: LraOptions| LraSubset {
        table: table.into(),
        options: Some(o),
        ..LraSubset::default()
    };
    let genes = |f: &LraFiles, i: usize| f.subset_options(&f.subsets[i]).unwrap().identity_excluded;
    let (full, other) = (
        options(&["A", "B"], None),
        options(&["A", "C"], Some("/p.tsv")),
    );

    let mut f = LraFiles::default();
    f.record_subset(subset("/early", options(&["A", "B"], None)));
    assert_eq!(
        f.subsets[0].options.as_ref().unwrap().identity_excluded,
        ["A", "B"]
    );
    f.record_options(full.clone());
    f.record_subset(subset("/same", options(&["A", "B"], None)));
    f.record_subset(subset("/other", other.clone()));
    for i in [0, 1] {
        assert!(f.subsets[i]
            .options
            .as_ref()
            .unwrap()
            .identity_excluded
            .is_empty());
        assert_eq!(genes(&f, i), ["A", "B"]);
    }
    assert_eq!(f.subsets[2].options, Some(other.clone()));

    // Through the manifest, then a full table made with another list.
    let json = serde_json::to_string(&f).unwrap();
    assert_eq!(json.matches("A,B").count(), 1, "{json}");
    let mut f: LraFiles = serde_json::from_str(&json).unwrap();
    f.record_options(options(&["A", "B", "D"], None));
    for i in [0, 1] {
        assert_eq!(genes(&f, i), ["A", "B"]);
    }
    assert_eq!(genes(&f, 2), ["A", "C"]);
    // And back to the first list: the copies are dropped again.
    f.record_options(full);
    assert!(f.subsets[0]
        .options
        .as_ref()
        .unwrap()
        .identity_excluded
        .is_empty());
    assert_eq!(genes(&f, 1), ["A", "B"]);

    // A subset from before the lists, beside a full table with one, reads
    // none unless it decides its receivers as the full table does.
    let mut f = LraFiles::default();
    f.record_options(options(&["A"], None));
    f.subsets.push(subset("/old", options(&[], Some("/p.tsv"))));
    assert!(genes(&f, 0).is_empty());
}
