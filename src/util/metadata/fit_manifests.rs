//! The manifest each fit writes: its inputs, the batch effects it wrote
//! beside it, and one builder per method (lc, dsvd, cage, propensity).

use super::*;
use crate::util::parquet_io::cells_table_path;

/// Build a `LevelInfo` for a per-cascade-level set of outputs at
/// `{prefix}.L{level_index}.*`.
pub fn lc_level_info(prefix: &str, level_index: usize) -> LevelInfo {
    let tag = format!("L{level_index}");
    LevelInfo {
        tag: tag.clone(),
        level_index,
        propensity: format!("{prefix}.{tag}.propensity.parquet"),
        link_community: Some(format!("{prefix}.{tag}.link_community.parquet")),
        feature_community: Some(format!("{prefix}.{tag}.feature_community.parquet")),
        community_names: None,
    }
}

/// Build the `final` `LevelInfo` every subcommand that publishes a propensity
/// ends on: `lc`'s tail level, and `dsvd`'s / `cage`'s only level. All three
/// write the same three parquets at the bare prefix with an `entropy` column,
/// so the shape lives here once rather than as three literals that have to be
/// kept in step. The sibling of [`lc_level_info`], which does the same job for
/// the `L*` cascade levels.
pub fn final_level_info(prefix: &str, level_index: usize) -> LevelInfo {
    LevelInfo {
        tag: "final".to_string(),
        level_index,
        propensity: format!("{prefix}.propensity.parquet"),
        link_community: Some(format!("{prefix}.link_community.parquet")),
        feature_community: Some(format!("{prefix}.feature_community.parquet")),
        community_names: None,
    }
}

/// Inputs shared by every metadata builder (`lc` / `dsvd`).
///
/// Bundled so call sites stay readable and the builders dodge
/// `clippy::too_many_arguments`.
pub struct RunInputs<'a> {
    pub prefix: &'a str,
    pub data_files: &'a [Box<str>],
    pub coord_file: Option<&'a str>,
    pub coord_columns: &'a [Box<str>],
    pub n_cells: usize,
    pub n_features: usize,
    pub n_edges: usize,
    /// Number of communities (lc) / clusters (dsvd) — same K dim either way.
    pub k: usize,
    /// Which graph produced `n_edges`. Not optional: every builder taking
    /// `RunInputs` builds a graph. `pinto apply propensity` re-cuts an existing pair table
    /// and has none, but it does not go through here.
    pub graph: GraphParams,
    /// The ablation this run is, if any.
    pub ablation: Option<crate::util::ablation::Ablation>,
    /// The δ the run wrote to `delta.parquet`; `None` when it wrote none.
    pub delta: Option<DeltaWritten>,
}

/// A δ a run wrote: what named its batches, and whether each cell's batch
/// was written beside it (a fit's; `predict` writes the δ alone).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaWritten {
    pub labels: BatchLabels,
    pub cells: bool,
    /// The state-aware fold written beside it.
    pub fold: crate::util::batch_fold::FoldWritten,
    /// Its labels carry covariate bins, and the covariate files were
    /// written beside it.
    pub covariates: bool,
}

impl DeltaWritten {
    /// A fit's δ, with its cells, for a run with inputs `c`: the fold it
    /// wrote beside it, and whether its labels carry covariate bins.
    pub fn fit(
        c: &crate::util::input::SrtInputArgs,
        fold: crate::util::batch_fold::FoldWritten,
        covariates: bool,
    ) -> Self {
        Self {
            labels: BatchLabels::of(&c.batch),
            cells: true,
            fold,
            covariates,
        }
    }
}

/// The covariate files a fit under `prefix` writes, relative to its
/// manifest.
pub(crate) fn covariate_files(prefix: &str) -> crate::util::metadata::CovariateFiles {
    use crate::util::covariates::files;
    let name = manifest_stem(prefix);
    crate::util::metadata::CovariateFiles {
        model: files::model(&name),
        cells: files::cells(&name),
        effect: files::effect(&name),
    }
}

/// The covariate files a run with `inputs` wrote, when its δ carries bins.
fn covariate_outputs(inputs: &RunInputs<'_>) -> Option<crate::util::metadata::CovariateFiles> {
    inputs
        .delta
        .filter(|d| d.covariates)
        .map(|_| covariate_files(inputs.prefix))
}

/// The fold files and what the profiles were divided by, for a run with
/// `inputs` ([`crate::util::batch_fold::BatchFoldFiles`]).
fn fold_outputs(
    inputs: &RunInputs<'_>,
) -> (
    Option<crate::util::batch_fold::BatchFoldFiles>,
    Option<crate::util::batch_fold::ProfilesDividedBy>,
) {
    use crate::util::batch_fold::{BatchFoldFiles, ProfilesDividedBy};
    let Some(delta) = inputs.delta else {
        return (None, None);
    };
    let files = BatchFoldFiles::written(&manifest_stem(inputs.prefix), delta.fold);
    let divided = if files.is_some() {
        ProfilesDividedBy::CellFold
    } else {
        ProfilesDividedBy::Delta
    };
    (files, Some(divided))
}

/// The manifest's own file name less its suffix, what files beside it are
/// recorded relative to: `-o out/` writes `out/.pinto.json` and
/// `out/.delta.parquet`.
fn manifest_stem(prefix: &str) -> String {
    let manifest = manifest_path(prefix);
    Path::new(&manifest)
        .file_name()
        .and_then(|f| f.to_str())
        .and_then(|f| f.strip_suffix(MANIFEST_SUFFIX))
        .unwrap_or_default()
        .to_string()
}

/// The δ outputs a run with `inputs` wrote: `delta.parquet`, and each
/// cell's batch when it was written. Recorded relative to the manifest
/// (which is written under the same prefix), so a later run finds them
/// from any working directory, and wherever the run is moved.
fn delta_outputs(inputs: &RunInputs<'_>) -> (Option<String>, Option<String>) {
    let Some(delta) = inputs.delta else {
        return (None, None);
    };
    let (delta_file, cells) = delta_files(inputs.prefix);
    (Some(delta_file), delta.cells.then_some(cells))
}

/// The δ table and each cell's batch beside it, as a fit under `prefix`
/// writes them, relative to its manifest.
pub(crate) fn delta_files(prefix: &str) -> (String, String) {
    let name = manifest_stem(prefix);
    (
        format!("{name}.delta.parquet"),
        crate::util::batch_effects::delta_cells_path(&name),
    )
}

/// Helper to create metadata for `pinto fit lc` runs.
///
/// `cascade_level_indices` is the list of `l` values for which
/// `{prefix}.L{l}.*` files were actually written by the cascade
/// (skipped levels are absent — the cascade drops levels with too few
/// super-edges, so indices need not be contiguous and may not start at 0).
/// `merge` is `Some` when the dictionary merge wrote its tree, with the cut
/// it was cut at, if any.
pub fn create_lc_metadata(
    inputs: &RunInputs<'_>,
    merge: Option<DictMergeSummary>,
    splice: Option<SpliceTrackInfo>,
    cascade_level_indices: &[usize],
) -> PintoMetadata {
    let prefix = inputs.prefix;
    let (batch_effects, delta_cells) = delta_outputs(inputs);
    let (batch_fold, profiles_divided_by) = fold_outputs(inputs);
    let dict_merge = merge.map(|m| DictMergeFiles {
        merges: format!("{prefix}.dict_merges.parquet"),
        cut: m
            .cut_cosine
            .map(|_| format!("{prefix}.dict_merges.cut.parquet")),
        min_nnz: Some(m.min_nnz),
        features_scored: Some(m.features_scored),
        cut_cosine: m.cut_cosine,
        centred_sections: m.centred_sections,
    });

    let mut levels: Vec<LevelInfo> = cascade_level_indices
        .iter()
        .map(|&l| lc_level_info(prefix, l))
        .collect();
    let tail_index = cascade_level_indices
        .iter()
        .copied()
        .max()
        .map_or(0, |m| m + 1);
    levels.push(final_level_info(prefix, tail_index));

    PintoMetadata {
        command: "lc".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: now_secs(),
        prefix: prefix.to_string(),
        data_files: Some(recorded_files(inputs.data_files)),
        coord_file: inputs.coord_file.map(|s| s.to_string()),
        n_cells: inputs.n_cells,
        n_features: inputs.n_features,
        n_edges: Some(inputs.n_edges),
        n_communities: Some(inputs.k),
        cell_level: false,
        graph: Some(inputs.graph.clone()),
        splice,
        batch_fold_placement: None,
        outputs: OutputFiles {
            coord_pairs: Some(format!("{prefix}.coord_pairs.parquet")),
            coord_columns: coord_columns_field(inputs.coord_columns),
            cells: inputs.coord_file.map(|_| cells_table_path(prefix)),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            link_community: Some(format!("{prefix}.link_community.parquet")),
            feature_community: Some(format!("{prefix}.feature_community.parquet")),
            feature_community_batch_adjusted: inputs.delta.is_some(),
            scores: Some(format!("{prefix}.scores.parquet")),
            batch_effects,
            delta_cells,
            batch_fold,
            profiles_divided_by,
            covariates: covariate_outputs(inputs),
            dict_merge,
            ..Default::default()
        },
        levels: Some(levels),
        ablation: inputs.ablation.clone(),
        holdout: None,
        feature_modules: None,
        batch_labels: inputs.delta.map(|d| d.labels),
        file_batches: file_batches(inputs.delta.map(|d| d.labels), inputs.data_files),
        baseline: None,
        max_threads: None,
        feature_names: None,
    }
}

fn coord_columns_field(cols: &[Box<str>]) -> Option<Vec<String>> {
    if cols.is_empty() {
        None
    } else {
        Some(cols.iter().map(|s| s.to_string()).collect())
    }
}

/// Helper for `pinto fit dsvd` runs. Only one "final" level is produced;
/// the cascade does not run.
pub fn create_dsvd_metadata(inputs: &RunInputs<'_>) -> PintoMetadata {
    let prefix = inputs.prefix;
    let (batch_effects, delta_cells) = delta_outputs(inputs);
    let (batch_fold, profiles_divided_by) = fold_outputs(inputs);
    let levels = vec![final_level_info(prefix, 0)];

    PintoMetadata {
        command: "dsvd".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: now_secs(),
        prefix: prefix.to_string(),
        data_files: Some(recorded_files(inputs.data_files)),
        coord_file: inputs.coord_file.map(|s| s.to_string()),
        n_cells: inputs.n_cells,
        n_features: inputs.n_features,
        n_edges: Some(inputs.n_edges),
        n_communities: Some(inputs.k),
        cell_level: false,
        graph: Some(inputs.graph.clone()),
        splice: None,
        batch_fold_placement: None,
        outputs: OutputFiles {
            coord_pairs: Some(format!("{prefix}.coord_pairs.parquet")),
            coord_columns: coord_columns_field(inputs.coord_columns),
            cells: inputs.coord_file.map(|_| cells_table_path(prefix)),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            feature_community: Some(format!("{prefix}.feature_community.parquet")),
            feature_community_batch_adjusted: inputs.delta.is_some(),
            batch_effects,
            delta_cells,
            batch_fold,
            profiles_divided_by,
            covariates: covariate_outputs(inputs),
            ..Default::default()
        },
        levels: Some(levels),
        ablation: inputs.ablation.clone(),
        holdout: None,
        feature_modules: None,
        batch_labels: inputs.delta.map(|d| d.labels),
        file_batches: file_batches(inputs.delta.map(|d| d.labels), inputs.data_files),
        baseline: None,
        max_threads: None,
        feature_names: None,
    }
}

/// Helper for `pinto fit cage` runs. One `final` level, in the same shape
/// `lc` / `dsvd` publish: cage projects every cell pair onto its frozen feature
/// embedding, clusters those pairs into link communities, and derives cell
/// propensity from incident-edge fractions — so the level's slots point at
/// the same three parquets, `entropy` included.
///
/// The pair encoder is part of every
/// cage model — `predict` and `impute` place a sample with it and nothing
/// else — so its slot is always filled. `inputs.k` is the number of edge
/// clusters, which is what `n_communities` reports — not the embedding
/// width, which is a different quantity and has no slot here.
pub fn create_cage_metadata(
    inputs: &RunInputs<'_>,
    splice: Option<SpliceTrackInfo>,
) -> PintoMetadata {
    let prefix = inputs.prefix;
    let (batch_effects, delta_cells) = delta_outputs(inputs);
    let (batch_fold, profiles_divided_by) = fold_outputs(inputs);

    let levels = vec![final_level_info(prefix, 0)];

    PintoMetadata {
        command: "cage".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: now_secs(),
        prefix: prefix.to_string(),
        data_files: Some(recorded_files(inputs.data_files)),
        coord_file: inputs.coord_file.map(|s| s.to_string()),
        n_cells: inputs.n_cells,
        n_features: inputs.n_features,
        n_edges: Some(inputs.n_edges),
        n_communities: Some(inputs.k),
        cell_level: false,
        graph: Some(inputs.graph.clone()),
        splice,
        batch_fold_placement: None,
        outputs: OutputFiles {
            coord_pairs: Some(format!("{prefix}.coord_pairs.parquet")),
            coord_columns: coord_columns_field(inputs.coord_columns),
            cells: inputs.coord_file.map(|_| cells_table_path(prefix)),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            link_community: Some(format!("{prefix}.link_community.parquet")),
            feature_community: Some(format!("{prefix}.feature_community.parquet")),
            feature_community_batch_adjusted: inputs.delta.is_some(),
            scores: Some(format!("{prefix}.scores.parquet")),
            batch_effects,
            delta_cells,
            batch_fold,
            profiles_divided_by,
            covariates: covariate_outputs(inputs),
            cell_embedding: Some(format!("{prefix}.cell_embedding.parquet")),
            pair_encoder: Some(format!("{prefix}.pair_encoder.safetensors")),
            pb_pair_embedding: Some(crate::util::parquet_io::pb_pair_embedding_path(prefix)),
            pb_bias: Some(format!("{prefix}.pb_bias.parquet")),
            cell_pb: Some(crate::util::parquet_io::cell_pb_path(prefix)),
            feature_embedding: Some(format!("{prefix}.feature_embedding.parquet")),
            feature_bias: Some(format!("{prefix}.feature_bias.parquet")),
            ..Default::default()
        },
        levels: Some(levels),
        ablation: inputs.ablation.clone(),
        holdout: None,
        feature_modules: None,
        batch_labels: inputs.delta.map(|d| d.labels),
        file_batches: file_batches(inputs.delta.map(|d| d.labels), inputs.data_files),
        baseline: None,
        max_threads: None,
        feature_names: None,
    }
}

/// Helper for the standalone `pinto apply propensity` command. Inputs are precomputed
/// latent + coord-pair files, not raw expression, so `data_files` is
/// optional.
pub fn create_prop_metadata(
    prefix: &str,
    expr_files: Option<&[Box<str>]>,
    coord_pair_file: Option<&str>,
    n_vertices: usize,
    n_clusters: usize,
) -> PintoMetadata {
    let levels = vec![LevelInfo {
        tag: "final".to_string(),
        level_index: 0,
        propensity: format!("{prefix}.propensity.parquet"),
        link_community: Some(format!("{prefix}.link_community.parquet")),
        feature_community: None,
        community_names: None,
    }];

    PintoMetadata {
        command: "prop".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: now_secs(),
        prefix: prefix.to_string(),
        data_files: expr_files.map(|fs| fs.iter().map(|s| s.to_string()).collect()),
        coord_file: coord_pair_file.map(|s| s.to_string()),
        n_cells: n_vertices,
        n_features: 0,
        n_edges: None,
        graph: None,
        n_communities: Some(n_clusters),
        cell_level: false,
        splice: None,
        batch_fold_placement: None,
        outputs: OutputFiles {
            coord_pairs: coord_pair_file.map(|s| s.to_string()),
            propensity: Some(format!("{prefix}.propensity.parquet")),
            ..Default::default()
        },
        levels: Some(levels),
        ablation: None,
        holdout: None,
        feature_modules: None,
        batch_labels: None,
        file_batches: None,
        baseline: None,
        max_threads: None,
        feature_names: None,
    }
}
