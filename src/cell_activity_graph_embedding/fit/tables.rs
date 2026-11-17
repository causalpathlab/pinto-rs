//! Writing the trained tables once the loop has ended.

use crate::util::common::*;
use crate::util::input::SrtInputArgs;

use crate::cell_activity_graph_embedding::pb_frame::PbFrame;
use graph_embedding_util::embedding_col_names;
use graph_embedding_util::eval::{save_bias, save_embedding};
use graph_embedding_util::model::JointEmbedModel;
use graph_embedding_util::transfer::{
    MODULE_BIAS_SUFFIX, MODULE_DICTIONARY_SUFFIX, MODULE_MEMBERSHIP_SUFFIX, MODULE_RESIDUAL_SUFFIX,
};
use legume_numeric::candle::candle_core::Tensor;

/// Writes the pseudobulk and feature tables, and returns the feature
/// embedding the pair projection scores against. `unit_table`: whether the
/// model trained a unit table (cage's pseudobulks; cage-cell trains its
/// pairs' rows instead, written as the latent).
pub(super) fn write_trained_tables(
    c: &SrtInputArgs,
    model: &mut JointEmbedModel,
    pb_frame: &PbFrame,
    cell_names: &[Box<str>],
    feature_names: &[Box<str>],
    embedding_dim: usize,
    unit_table: bool,
) -> anyhow::Result<Mat> {
    // Trained PB table [P × D] + bias [P] + the cell -> PB map. The
    // trained unit is the finest-level super-cell; the per-CELL embedding
    // ships separately, as each cell's own placement by the pair encoder,
    // written by the caller after the propensity pass.
    // Bare integer row keys (the writer's default), so pb_embedding,
    // pb_bias, and cell_pb's `pb` column all join directly on the id.
    if unit_table {
        let e_pb_mat = tensor_to_mat(&model.e_cell)?;
        e_pb_mat.to_parquet_with_names(
            &(c.out.to_string() + ".pb_embedding.parquet"),
            (None, Some("pb")),
            Some(&embedding_col_names(embedding_dim)),
        )?;
    }

    let b_pb_mat = tensor_to_mat_1d(&model.b_cell)?;
    b_pb_mat.to_parquet_with_names(
        &(c.out.to_string() + ".pb_bias.parquet"),
        (None, Some("pb")),
        Some(&[Box::from("b_pb")]),
    )?;

    crate::util::parquet_io::write_cell_pb(
        &c.out,
        cell_names,
        &pb_frame.cell_to_finest_pb,
        Vec::new(),
    )?;

    // Feature embedding [G × D] — same shared D-dim space as cells, over the FULL
    // feature axis. There is no HVG row subset to mirror any more: HVG weights the
    // projection, so every feature is trained and every feature gets a row here.
    //
    // What ships depends on which gate ran, because the two mean different
    // things and `e_feature_out` is ALSO the frozen dictionary `project_pairs`
    // scores every cell pair against. Shipping the wrong one would put the pair
    // latent — and so every link community — on a feature side the fit never
    // trained under, which is the defect commit 91c50a65 fixed in geu.
    //
    // A free feature table is already the trained Var; under the adapter or the
    // module layer the `e_feat` field is a construction-time snapshot, and
    // `materialize_e_feat` recomposes it from the live map (a no-op otherwise).
    model.materialize_e_feat()?;
    let e_feature_out = tensor_to_mat(&model.e_feat)?;
    let b_feature_out = tensor_to_mat_1d(&model.b_feat)?;

    e_feature_out.to_parquet_with_names(
        &(c.out.to_string() + ".feature_embedding.parquet"),
        (Some(feature_names), Some("feature")),
        Some(&embedding_col_names(embedding_dim)),
    )?;
    // Learned-module tables (no-op without modules); the feature embedding above
    // already holds the composed row.
    write_module_tables(&c.out, model, feature_names)?;

    // The adapter map itself, so the spatial refinement can be applied OUTSIDE
    // this run: any feature with a row in the source dictionary, panel or not,
    // maps into this run's frame as `rho_row . W`. Rows follow the source
    // dictionary's column order. The per-feature residual is deliberately not
    // exported on its own: it exists only for panel features and is already part
    // of feature_embedding.parquet.
    if let Some(a) = &model.adapter {
        let w_out = tensor_to_mat(&a.w)?;
        let src_names: Vec<Box<str>> = (0..w_out.nrows()).map(|i| format!("s{i}").into()).collect();
        w_out.to_parquet_with_names(
            &(c.out.to_string() + ".adapter.parquet"),
            (Some(&src_names), Some("source_dim")),
            Some(&embedding_col_names(embedding_dim)),
        )?;
        info!(
            "Wrote {}.adapter.parquet",
            crate::util::staging::final_str(&c.out)
        );
    }

    b_feature_out.to_parquet_with_names(
        &(c.out.to_string() + ".feature_bias.parquet"),
        (Some(feature_names), Some("feature")),
        Some(&[Box::from("b_feature")]),
    )?;
    Ok(e_feature_out)
}

/// The learned-module tables, as `graph_embedding_util::write_module_tables`
/// writes them (same files, same layouts), logged under the final prefix
/// rather than the staged one; nothing without modules.
///
/// * `{out}.module_membership.parquet`: `π [G × M]`, keyed on the features.
/// * `{out}.module_dictionary.parquet`: `μ [M × H]`, rows `m0..`.
/// * `{out}.module_residual.parquet`: `r [G × H]`, keyed on the features.
/// * `{out}.module_bias.parquet`: the per-module bias, rows `m0..`.
fn write_module_tables(
    out: &str,
    model: &JointEmbedModel,
    feature_names: &[Box<str>],
) -> anyhow::Result<()> {
    let Some(modules) = &model.modules else {
        return Ok(());
    };
    let pi = model
        .module_membership()?
        .ok_or_else(|| anyhow::anyhow!("a module model has no membership"))?;
    let m = modules.n_modules;
    let module_names: Vec<Box<str>> = (0..m).map(|i| format!("m{i}").into_boxed_str()).collect();
    pi.to_parquet_with_names(
        &format!("{out}.{MODULE_MEMBERSHIP_SUFFIX}"),
        (Some(feature_names), Some("feature")),
        Some(&module_names),
    )?;
    save_embedding(
        &format!("{out}.{MODULE_DICTIONARY_SUFFIX}"),
        &modules.mu.detach(),
        &module_names,
        "module",
    )?;
    save_embedding(
        &format!("{out}.{MODULE_RESIDUAL_SUFFIX}"),
        &modules.residual.detach(),
        feature_names,
        "feature",
    )?;
    save_bias(
        &format!("{out}.{MODULE_BIAS_SUFFIX}"),
        &modules.b_module.detach(),
        &module_names,
        "module",
    )?;
    info!(
        "Wrote {m} learned gene modules: {}.{MODULE_MEMBERSHIP_SUFFIX} (+ module_dictionary / module_residual / module_bias)",
        crate::util::staging::final_str(out)
    );
    Ok(())
}

/// Convert a 2-D `[R × C]` candle Tensor into an `nalgebra::DMatrix<f32>`
/// suitable for `to_parquet_with_names`.
fn tensor_to_mat(t: &Tensor) -> anyhow::Result<Mat> {
    let rows = t.dim(0)?;
    let cols = t.dim(1)?;
    let data: Vec<f32> = t.to_vec2::<f32>()?.into_iter().flatten().collect();
    // `from_row_slice` expects row-major, matching `to_vec2`.
    Ok(Mat::from_row_slice(rows, cols, &data))
}

fn tensor_to_mat_1d(t: &Tensor) -> anyhow::Result<Mat> {
    let n = t.dim(0)?;
    let data: Vec<f32> = t.to_vec1::<f32>()?;
    Ok(Mat::from_column_slice(n, 1, &data))
}
