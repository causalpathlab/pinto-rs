//! `--null-permutations`: the shuffle keeps each identity bin × tile's
//! ligand, a draw is the same wherever it runs, and a run writes, records
//! and (through shards) merges its null table.

use super::super::link_community_scope::shard_path;
use super::super::null_draws::{draw_seed, permuted_within_strata};
use super::super::output::null_beside;
use super::super::pipeline::merge;
use super::super::shard::Shard;
use super::lra_path;
use super::marginal_run::{run_on_the_test_tissue, TestRunChoices};
use super::simulate::fixture;
use crate::lr_activity::recorded;
use crate::util::metadata::PintoMetadata;
use legume_numeric::matrix::parquet::read_table_columns;
use std::collections::HashMap;
use std::path::Path;

/// Each stratum's values, sorted: what a shuffle within strata keeps.
fn by_stratum(bin: &[usize], tile: &[u32], values: &[f64]) -> HashMap<(usize, u32), Vec<u64>> {
    let mut out: HashMap<(usize, u32), Vec<u64>> = HashMap::new();
    for i in 0..values.len() {
        out.entry((bin[i], tile[i]))
            .or_default()
            .push(values[i].to_bits());
    }
    out.values_mut().for_each(|v| v.sort_unstable());
    out
}

#[test]
fn a_draw_shuffles_within_each_identity_bin_and_tile() {
    let fx = fixture();
    let t = &fx.tissue;
    let real = &fx.ligand;
    let draw = permuted_within_strata(t, real, 11);
    let (bin, tile) = (&t.identity_bin, &t.cf.cluster);
    assert_eq!(by_stratum(bin, tile, real), by_stratum(bin, tile, &draw));
    assert_ne!(&draw, real, "a shuffle moves the ligand");
    // the same seed, the same draw; another seed, another
    assert_eq!(draw, permuted_within_strata(t, real, 11));
    assert_ne!(draw, permuted_within_strata(t, real, 12));
    // a draw's seed follows the run, the ligand and the draw
    let s = draw_seed(7, "L1", 0);
    assert_eq!(s, draw_seed(7, "L1", 0));
    assert!([
        draw_seed(8, "L1", 0),
        draw_seed(7, "L2", 0),
        draw_seed(7, "L1", 1)
    ]
    .iter()
    .all(|&o| o != s));
}

fn run(dir: &Path, out: &str, manifest: &Path, shard: Option<Shard>) {
    let choices = TestRunChoices {
        only: &[],
        shard,
        marginal_baselines: false,
        link_community: None,
        null_permutations: 2,
    };
    run_on_the_test_tissue(dir, out, manifest, &choices);
}

/// The null table: each row of the effect table, per draw, per estimator,
/// recorded in the manifest; and the same table from shards merged.
#[test]
fn a_run_with_null_permutations_writes_records_and_merges_its_null_table() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("run.pinto.json");
    PintoMetadata {
        command: "lra".into(),
        ..Default::default()
    }
    .write(&manifest)
    .unwrap();
    let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let whole = at("whole");
    run(dir.path(), &whole, &manifest, None);
    let (table, null) = (lra_path(&whole), null_beside(&lra_path(&whole)));
    let (effect_rows, _) = read_table_columns(&table, &["ligand"], &[]).unwrap();
    let (s, x) = read_table_columns(&null, &["null", "estimator"], &["draw", "estimate"]).unwrap();
    assert_eq!(
        s[0].len(),
        effect_rows[0].len() * 2 * 2,
        "rows × draws × estimators"
    );
    assert!(s[0].iter().all(|n| &**n == "permutation"));
    let estimators: std::collections::BTreeSet<&str> = s[1].iter().map(|e| &**e).collect();
    assert_eq!(
        estimators.into_iter().collect::<Vec<_>>(),
        ["dr", "outcome"]
    );
    assert!(x[0].iter().all(|&d| d == 0.0 || d == 1.0));
    assert!(x[1].iter().any(|v| v.is_finite()), "some draw is estimable");
    let lra = PintoMetadata::read(&manifest).unwrap().outputs.lra;
    assert_eq!(
        lra.null.as_deref(),
        Some(recorded(Path::new(&null)).as_str())
    );
    assert_eq!(lra.options.and_then(|o| o.null_permutations), Some(2));

    let files: Vec<String> = (0..2)
        .map(|index| {
            let s = Shard { index, count: 2 };
            let out = at(&format!("shard{index}/o"));
            run(dir.path(), &out, &manifest, Some(s));
            shard_path(&out, s, None)
        })
        .collect();
    let merged = at("merged");
    merge(&files, &merged).unwrap();
    assert_eq!(
        std::fs::read(&null).unwrap(),
        std::fs::read(null_beside(&lra_path(&merged))).unwrap(),
        "a draw is the same in a shard as in the whole run"
    );
}
