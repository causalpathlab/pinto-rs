//! How many feature modules a run trains, and the partition its membership
//! starts from.
//!
//! A fixed count partitions the features by k-means, as
//! `graph_embedding_util::partition_modules` does. `auto` takes the count
//! from the data: features with no grouping evidence share one background
//! module, and the rest are the communities Leiden finds in the
//! nearest-neighbour graph of their residual profiles. No features x
//! features table is built: the profiles are features x pseudobulks, their
//! reduction features x [`PROFILE_SVD_RANK`], and the graph holds
//! [`LEIDEN_KNN`] neighbours per feature.

use crate::cell_activity_graph_embedding::args::ModulePlan;
use crate::link_community::profiles::consensus_leiden;
use crate::util::common::*;
use crate::util::edge_clustering::{LEIDEN_KNN, LEIDEN_RUNS};
use data_beans::alg::feature_coarsening::{informative_features, residual_profiles};
use graph_embedding_util::{FeatureModuleArgs, FeatureModuleConfig};
use legume_numeric::matrix::traits::RandomizedAlgs;

/// Columns the residual profiles are reduced to before their
/// nearest-neighbour graph is built. Profiles no wider than this, or of no
/// more features than this, are used as they are.
pub(super) const PROFILE_SVD_RANK: usize = 50;

/// Leiden's modularity resolution for the feature graph: the default of
/// `--leiden-resolution`, which cuts the pair latent.
const LEIDEN_RESOLUTION: f64 = 1.0;

/// The module layer a run trains: its settings, and each feature's starting
/// module (`< cfg.n_modules`).
pub(super) struct ModuleStart {
    pub cfg: FeatureModuleConfig,
    pub labels: Vec<u32>,
}

/// A partition of the features whose module count the data decided.
pub(super) struct AutoPartition {
    /// Each feature's module.
    pub labels: Vec<u32>,
    /// Leiden's communities, plus the background module when there is one.
    pub n_modules: usize,
    /// The module of the features with no grouping evidence (the last), if
    /// any feature is in it.
    pub background: Option<u32>,
}

/// The module layer `plan` asks for on `profiles` (`[features x
/// pseudobulks]` count sums, the pseudobulks `sizes` cells each), or `None`
/// for a run without one: `plan` is off, or `auto` found fewer than two
/// modules.
pub(super) fn resolve_modules(
    plan: ModulePlan,
    profiles: &Mat,
    sizes: &[f32],
    seed: u64,
) -> anyhow::Result<Option<ModuleStart>> {
    let (labels, n_modules) = match plan {
        ModulePlan::Off => return Ok(None),
        ModulePlan::Fixed(m) => (
            graph_embedding_util::partition_modules(profiles, sizes, m, seed)?,
            m,
        ),
        ModulePlan::Auto => {
            let part = auto_partition(profiles, sizes, seed)?;
            if part.n_modules < 2 {
                info!(
                    "--feature-modules auto: fewer than two modules among {} features; \
                     training without the module layer",
                    profiles.nrows()
                );
                return Ok(None);
            }
            match part.background {
                Some(id) => info!(
                    "--feature-modules auto: M = {}, the background module is {id}",
                    part.n_modules
                ),
                None => info!("--feature-modules auto: M = {}", part.n_modules),
            }
            (part.labels, part.n_modules)
        }
    };
    // Every knob but the count is upstream's default.
    let cfg = FeatureModuleArgs {
        feature_modules: Some(n_modules),
    }
    .resolve(None)?
    .expect("a count resolves to a module layer");
    Ok(Some(ModuleStart { cfg, labels }))
}

/// Features grouped with their count left to the data.
///
/// The features of `counts` (`[features x pseudobulks]`, the pseudobulks
/// `sizes` cells each) that carry no grouping evidence
/// (`informative_features`) share one background module, numbered last;
/// with fewer than two features that do, every feature is in it. The
/// others are placed by their unit-length residual profiles, reduced to
/// [`PROFILE_SVD_RANK`] columns by a truncated SVD, and grouped by Leiden
/// (Traag, Waltman and van Eck 2019, "From Louvain to Leiden: guaranteeing
/// well-connected communities", Scientific Reports 9:5233) on the cosine
/// nearest-neighbour graph of those coordinates: a consensus of
/// [`LEIDEN_RUNS`] runs, as the pair latent is cut.
pub(super) fn auto_partition(
    counts: &Mat,
    sizes: &[f32],
    seed: u64,
) -> anyhow::Result<AutoPartition> {
    let n_features = counts.nrows();
    anyhow::ensure!(
        counts.ncols() == sizes.len(),
        "{} profile columns for {} pseudobulk sizes",
        counts.ncols(),
        sizes.len()
    );
    let informative = informative_features(counts, sizes);
    let mut rows: Vec<usize> = (0..n_features).filter(|&g| informative[g]).collect();
    // A lone feature with grouping evidence has nothing to be grouped with:
    // it joins the others, and the one module left is no layer.
    if rows.len() < 2 {
        rows.clear();
    }
    let community: Vec<usize> = if rows.is_empty() {
        Vec::new()
    } else {
        let coords = reduce_profiles(residual_profiles(counts, sizes, &rows, seed))?;
        consensus_leiden(
            &coords,
            LEIDEN_KNN,
            LEIDEN_RESOLUTION,
            None,
            LEIDEN_RUNS,
            seed,
        )?
    };
    let n_communities = community.iter().max().map_or(0, |&c| c + 1);
    let background = (rows.len() < n_features).then_some(n_communities as u32);
    let mut labels = vec![n_communities as u32; n_features];
    for (&g, &c) in rows.iter().zip(&community) {
        labels[g] = c as u32;
    }
    info!(
        "feature modules: {} of {n_features} features carry grouping evidence, in {n_communities} \
         Leiden communities; {} in the background module",
        rows.len(),
        n_features - rows.len()
    );
    Ok(AutoPartition {
        labels,
        n_modules: n_communities + usize::from(background.is_some()),
        background,
    })
}

/// The profiles' coordinates on their leading [`PROFILE_SVD_RANK`] singular
/// directions (`U·diag(s)`, one row per profile), or the profiles
/// themselves when they are already that small.
fn reduce_profiles(profiles: Mat) -> anyhow::Result<Mat> {
    if profiles.ncols().min(profiles.nrows()) <= PROFILE_SVD_RANK {
        return Ok(profiles);
    }
    let (mut u, s, _) = profiles.rsvd(PROFILE_SVD_RANK)?;
    for (mut col, &sv) in u.column_iter_mut().zip(s.iter()) {
        col *= sv;
    }
    Ok(u)
}

/// Cells per pseudobulk: `labels` is each cell's pseudobulk of `n_pb`.
pub(super) fn pseudobulk_sizes(labels: &[usize], n_pb: usize) -> Vec<f32> {
    let mut sizes = vec![0f32; n_pb];
    for &pb in labels {
        sizes[pb] += 1.0;
    }
    sizes
}

#[cfg(test)]
#[path = "modules_tests.rs"]
mod tests;
