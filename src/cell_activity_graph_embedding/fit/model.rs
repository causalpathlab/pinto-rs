//! Building the model: the optional pre-trained feature side, then the one
//! match that constructs the model and its freeze state.

use super::modules::{pseudobulk_sizes, resolve_modules, ModuleStart};
use crate::cell_activity_graph_embedding::args::{
    CellActivityGraphEmbeddingArgs, FeatureEmbeddingMode, FeatureInitMode, ModulePlan,
};
use crate::cell_activity_graph_embedding::pretrained;
use crate::link_community::profiles::coarsen_cell_expression_dense;
use crate::util::common::*;

use crate::util::feature_axis::FeatureAxis;
use data_beans::aux::feature_names::FeatureNameKind;
use graph_embedding_util::model::{
    AdapterInit, JointEmbedModel, ModelArgs, ModelInit, ModuleInit, ModuleWarmStart,
};
use graph_embedding_util::{FeatureModuleConfig, LoraSpec};
use legume_numeric::candle::candle_core::Device;
use legume_numeric::candle::candle_core::Tensor;
use legume_numeric::candle::candle_nn::VarMap;

/// What the model is built from.
pub(super) struct ModelInputs<'a> {
    pub args: &'a CellActivityGraphEmbeddingArgs,
    /// Each cell's finest pseudobulk, which pools the feature profiles: the
    /// trained hierarchy's, or under cage-cell that of the coarsening built
    /// to seed its modules.
    pub profile_labels: &'a [usize],
    pub data_vec: &'a SparseIoVec,
    pub feature_axis: &'a FeatureAxis,
    pub feature_kind: &'a FeatureNameKind,
    /// What the run's flags ask of the module layer; the count is resolved
    /// here, where the profiles exist.
    pub module_plan: ModulePlan,
    pub lora_spec: Option<LoraSpec>,
    pub n_pb: usize,
    pub embedding_dim: usize,
    pub varmap: &'a VarMap,
    pub dev: &'a Device,
}

/// The model, its freeze state, and the module layer it was built with
/// (`None`: no layer): its settings at the resolved count, and the profiles
/// that seeded it (cage's module term pools them too).
pub(super) struct BuiltModel {
    pub model: JointEmbedModel,
    pub frozen_feature: Option<pretrained::FrozenGene>,
    pub modules: Option<(FeatureModuleConfig, Mat)>,
}

pub(super) fn build_model(inputs: ModelInputs<'_>) -> anyhow::Result<BuiltModel> {
    let ModelInputs {
        args,
        profile_labels,
        data_vec,
        feature_axis,
        feature_kind,
        module_plan,
        lora_spec,
        n_pb,
        embedding_dim,
        varmap,
        dev,
    } = inputs;
    let c = &args.common;
    let n_features = feature_axis.n_features();
    let feature_names = feature_axis.feature_names();
    // `JointEmbedModel.e_feat` / `b_feat` ARE the feature embedding
    // (cells and features share the same D-dim space). `n_features =
    // n_features` and `b_feat` is zero-init per feature; both are learned
    // alongside the cell side via AdamW over `varmap.all_vars()`.
    //
    // Pre-trained feature side, when requested. Loaded here, after the feature
    // axis is final and the coarsening exists (its finest level pools the
    // per-feature profiles that seed unmatched features), and before the model so
    // the init below can consume it.
    // Per-feature profiles over the finest pseudobulks `[G × n_pb]`: the seed for
    // unmatched pre-trained features, and the count rows the module term pools.
    // Lazy for the pre-trained path (a full pass over the data that runs only if
    // the loader finds a feature with no dictionary row); materialized once for the
    // module path, which reads it every step.
    let build_profiles = || -> anyhow::Result<Mat> {
        let n_pb = profile_labels.iter().copied().max().map_or(0, |m| m + 1);
        let row_profiles = coarsen_cell_expression_dense(data_vec, profile_labels, n_pb, None)?;
        Ok(feature_axis
            .pool_rows_opt(&row_profiles)
            .unwrap_or(row_profiles))
    };
    // The module count is known only against the profiles, so the layer is
    // resolved here: a fixed count, or the one `auto` finds (none when it
    // finds fewer than two modules).
    let module_start: Option<(ModuleStart, Mat)> = match module_plan {
        ModulePlan::Off => None,
        plan => {
            let profiles = build_profiles()?;
            anyhow::ensure!(
                profiles.nrows() == n_features,
                "module profiles of {} rows for {} features",
                profiles.nrows(),
                n_features
            );
            // The profiles are the finest pseudobulks' count sums; a
            // pseudobulk's size is its cell count.
            let sizes = pseudobulk_sizes(profile_labels, profiles.ncols());
            resolve_modules(plan, &profiles, &sizes, c.seed)?.map(|start| (start, profiles))
        }
    };
    let pretrained_feature = match args.feature_embedding.as_deref() {
        None => None,
        Some(dict_path) => {
            let pre = pretrained::load_pretrained_feature_embedding(pretrained::PretrainedArgs {
                dictionary_path: dict_path,
                bias_path: args.feature_embedding_bias.as_deref(),
                feature_names,
                name_kind: feature_kind.clone(),
                feature_profiles: &build_profiles,
                membership_init: match args.feature_init_mode {
                    FeatureInitMode::Membership => {
                        Some(graph_embedding_util::transfer::AlignKnobs {
                            k: graph_embedding_util::transfer::DEFAULT_INIT_NEIGHBOURS,
                            similarity_floor:
                                graph_embedding_util::transfer::DEFAULT_SIMILARITY_FLOOR,
                        })
                    }
                    FeatureInitMode::Neighbor => None,
                },
            })?;
            // The width was resolved from this file's footer; the loader's
            // count must agree, or the two read the file differently.
            if args.feature_embedding_mode != FeatureEmbeddingMode::Adapt {
                anyhow::ensure!(
                    pre.h() == embedding_dim,
                    "--feature-embedding loaded {} dimensions wide but the run resolved {}",
                    pre.h(),
                    embedding_dim
                );
            }
            pretrained::write_init_report(&c.out, &pre.records)?;
            info!(
                "Wrote {}.feature_embedding_init.parquet",
                crate::util::staging::final_str(&c.out)
            );
            Some(pre)
        }
    };
    let b_feat_init: Vec<f32> = pretrained_feature
        .as_ref()
        .map(|p| p.b_feat.clone())
        .unwrap_or_else(|| vec![0.0_f32; n_features]);

    // One match constructs the model AND, on the Freeze arm, the freeze state
    // in sequence — the fixed copy is taken directly after registration, so it
    // is structurally the loaded dictionary and never a gated view.
    let b_pb_init = vec![0.0_f32; n_pb];
    // The unit table (pseudobulks, or cells under cage-cell) starts from the
    // seeded random draw in every arm: nothing is fitted to the counts for it.
    let (model, frozen_feature) = match (&pretrained_feature, args.feature_embedding_mode) {
        // Learned feature modules: every feature row is `Σ_m π_gm μ_m + r_g`, warm-started
        // from the partition of the pseudobulk profiles resolved above. Same detached-snapshot
        // contract as the adapter arm.
        _ if module_start.is_some() => {
            let (ModuleStart { cfg: gm, labels }, _) =
                module_start.as_ref().expect("checked above");
            info!(
                "learned feature modules: {} features → {} modules (mixed membership), feature dropout {}, \
                 exact module term λ={}, balance λ={}",
                n_features, gm.n_modules, gm.feature_dropout, gm.lambda_module, gm.lambda_balance
            );
            (
                JointEmbedModel::new_with_modules(
                    ModuleInit {
                        n_features,
                        n_cells: n_pb,
                        embedding_dim,
                        n_modules: gm.n_modules,
                        warm: ModuleWarmStart::Labels {
                            labels,
                            own_mass: gm.init_own_mass,
                        },
                        b_feat: &b_feat_init,
                        b_cell: &b_pb_init,
                        seed: c.seed,
                    },
                    varmap,
                    dev,
                )?,
                None,
            )
        }
        // Adapter: the dictionary is a fixed constant and the feature side trains
        // one shared [h_src x H] map (plus the optional per-feature residual), so
        // every feature's gradient moves the same few parameters.
        (Some(p), FeatureEmbeddingMode::Adapt) => (
            JointEmbedModel::new_adapted(
                AdapterInit {
                    n_cells: n_pb,
                    embedding_dim,
                    rho: &p.e_feat,
                    b_feat: &b_feat_init,
                    b_cell: &b_pb_init,
                    seed: c.seed,
                    residual: args.feature_adapter_residual,
                },
                varmap,
                dev,
            )?,
            None,
        ),
        (pre, mode) => {
            let model = JointEmbedModel::new_with_init(
                ModelArgs {
                    n_features,
                    n_cells: n_pb,
                    embedding_dim,
                    seed: c.seed,
                },
                &ModelInit {
                    e_feat: pre.as_ref().map(|p| &p.e_feat),
                    e_cell: None,
                    b_feat: &b_feat_init,
                    b_cell: &b_pb_init,
                },
                varmap,
                dev,
            )?;
            let frozen = match (pre, mode) {
                (Some(p), FeatureEmbeddingMode::Freeze | FeatureEmbeddingMode::Lora) => {
                    let fetch = |name: &str| {
                        varmap
                            .data()
                            .lock()
                            .expect("varmap lock")
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| panic!("the free constructor registers {name}"))
                    };
                    let fixed = model.e_feat.copy()?;
                    let mask = Tensor::from_vec(p.frozen_row_mask(), (n_features, 1), dev)?;
                    let var = fetch(graph_embedding_util::model::E_FEAT_VAR_NAME);
                    // The per-feature bias is part of the dictionary contract only
                    // when a bias file was actually given; without one the
                    // zeros are a default, and freezing them would stop b_feat
                    // absorbing per-feature rates the way a baseline run does.
                    let bias = if args.feature_embedding_bias.is_some() {
                        let b_var = fetch("b_feat");
                        Some((b_var.as_tensor().copy()?, b_var))
                    } else {
                        None
                    };
                    let n_frozen = p.n_matched();
                    let residual = match lora_spec {
                        Some(l) => format!(
                            " under a rank-{} residual (LoRA+ ratio {}, ridge {} per row per epoch)",
                            l.rank, l.lr_ratio, l.ridge
                        ),
                        None => String::new(),
                    };
                    info!(
                        "Feature embedding PINNED: {} dictionary rows fixed{residual}, {} neighbor-seeded rows trainable{}",
                        n_frozen,
                        n_features - n_frozen,
                        if bias.is_some() {
                            "; loaded bias rows fixed too"
                        } else {
                            ""
                        }
                    );
                    Some(pretrained::FrozenGene::new(fixed, mask, var, bias))
                }
                _ => None,
            };
            // The residual rides on the pinned rows: the factors join the map
            // beside `e_feat`, `u` drawn on the matched features only.
            let model = match (pre, lora_spec) {
                (Some(p), Some(l)) => {
                    model.with_lora(varmap, dev, l.rank, &p.matched_ids(), c.seed)?
                }
                _ => model,
            };
            (model, frozen)
        }
    };
    // cage-cell trains a row per cell pair (outside the model) and no cell
    // table. geu's constructors make `e_cell` and `b_cell` together, the
    // biases being the cells', so the table leaves the map here, before any
    // optimizer is built over it, and the field holds an empty one.
    let mut model = model;
    if args.cell_level {
        varmap.data().lock().expect("varmap lock").remove("e_cell");
        model.e_cell = Tensor::zeros(
            (0, embedding_dim),
            legume_numeric::candle::candle_core::DType::F32,
            dev,
        )?;
    }
    Ok(BuiltModel {
        model,
        frozen_feature,
        modules: module_start.map(|(start, profiles)| (start.cfg, profiles)),
    })
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
