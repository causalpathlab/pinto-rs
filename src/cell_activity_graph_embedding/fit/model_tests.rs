//! The module layer's size.

use super::{build_model, BuiltModel, ModelInputs};
use crate::cell_activity_graph_embedding::args::{CellActivityGraphEmbeddingArgs, ModulePlan};
use crate::test_support::{numbered, sparse_vec};
use crate::util::feature_axis::FeatureAxis;
use clap::Parser;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::candle::candle_nn::VarMap;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    cage: CellActivityGraphEmbeddingArgs,
}

const DIM: usize = 3;

/// Cells of each pseudobulk of the module fixture, and its pseudobulks.
const PB_CELLS: usize = 10;
const MODULE_PBS: usize = 64;
/// Features of each of its two planted groups, and those at one flat rate.
const MODULE_GROUP: usize = 30;
const MODULE_FLAT: usize = 6;
const MODULE_FEATURES: usize = 2 * MODULE_GROUP + MODULE_FLAT;

/// [`build_model`] under `plan` on counts with two planted feature groups
/// (the first [`MODULE_GROUP`] counted in the first half of the pseudobulks,
/// the next in the second) and [`MODULE_FLAT`] flat features after them.
/// There are more pseudobulks and more grouped features than the columns
/// the profiles are reduced to, so `auto` goes through the truncated SVD.
fn built_with(plan: ModulePlan) -> BuiltModel {
    let args = Cli::try_parse_from(["x", "data.zarr", "-o", "out"])
        .unwrap()
        .cage;
    let dir = tempfile::tempdir().unwrap();
    let n_cells = PB_CELLS * MODULE_PBS;
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for c in 0..n_cells {
        let first_half = c / PB_CELLS < MODULE_PBS / 2;
        for g in 0..MODULE_FEATURES {
            let x = match g / MODULE_GROUP {
                0 if first_half => 6 + (g + c) % 3,
                1 if !first_half => 6 + (g + c) % 3,
                0 | 1 => usize::from((g + c) % 5 == 0),
                _ => 2,
            };
            if x > 0 {
                triplets.push((g as u64, c as u64, x as f32));
            }
        }
    }
    let data = sparse_vec(
        dir.path(),
        "modules",
        &numbered("GENE", MODULE_FEATURES),
        &numbered("c", n_cells),
        &triplets,
    );
    let names = data.row_names().unwrap();
    let labels: Vec<usize> = (0..n_cells).map(|c| c / PB_CELLS).collect();
    build_model(ModelInputs {
        args: &args,
        profile_labels: &labels,
        data_vec: &data,
        feature_axis: &FeatureAxis::resolve(&names).unwrap(),
        feature_kind: &args.feature_name_mode.resolve_kind(&names),
        module_plan: plan,
        lora_spec: None,
        n_pb: MODULE_PBS,
        embedding_dim: DIM,
        varmap: &VarMap::new(),
        dev: &Device::Cpu,
    })
    .unwrap()
}

/// Every module table is sized by the count the run resolved, the one
/// handed back for training: the membership `[features x M]`, the module
/// dictionary `[M x H]` and the module bias `[M]`, for the count `auto`
/// finds (at least the two planted groups) and for a fixed one. The
/// profiles come back with the layer, features x pseudobulks. Under `auto`
/// the flat features start in one module, the last (the background), and
/// no grouped feature starts there.
#[test]
fn every_module_table_is_sized_by_the_resolved_count() {
    use crate::cell_activity_graph_embedding::fit::modules::PROFILE_SVD_RANK;
    const { assert!(MODULE_PBS > PROFILE_SVD_RANK && 2 * MODULE_GROUP > PROFILE_SVD_RANK) };
    for plan in [ModulePlan::Auto, ModulePlan::Fixed(3)] {
        let built = built_with(plan);
        let (cfg, profiles) = built.modules.as_ref().expect("a module layer");
        let m = cfg.n_modules;
        match plan {
            ModulePlan::Fixed(fixed) => assert_eq!(m, fixed),
            _ => assert!(m >= 2, "auto resolved {m}"),
        }
        assert_eq!(
            (profiles.nrows(), profiles.ncols()),
            (MODULE_FEATURES, MODULE_PBS)
        );
        let layer = built.model.modules.as_ref().expect("built with modules");
        assert_eq!(layer.logits.dims(), [MODULE_FEATURES, m], "{plan:?}");
        assert_eq!(layer.mu.dims(), [m, DIM], "{plan:?}");
        assert_eq!(layer.b_module.dims(), [m], "{plan:?}");
        let membership = layer.membership().unwrap();
        assert_eq!(membership.dims(), [MODULE_FEATURES, m], "{plan:?}");
        if plan == ModulePlan::Auto {
            // each feature's starting module holds most of its membership
            let start: Vec<usize> = membership
                .to_vec2::<f32>()
                .unwrap()
                .iter()
                .map(|row| (0..m).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap())
                .collect();
            let (grouped, flat) = start.split_at(2 * MODULE_GROUP);
            assert!(flat.iter().all(|&s| s == m - 1), "flat features: {flat:?}");
            assert!(grouped.iter().all(|&s| s != m - 1), "grouped: {grouped:?}");
        }
    }
}

/// A plan that is off builds no layer and hands none to training.
#[test]
fn a_plan_that_is_off_builds_no_module_layer() {
    let built = built_with(ModulePlan::Off);
    assert!(built.modules.is_none());
    assert!(built.model.modules.is_none());
}
