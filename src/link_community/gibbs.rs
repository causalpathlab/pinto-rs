//! Collapsed Gibbs sampler for the link community model.
//!
//! Adapted from `hsblock/src/gibbs.rs` but simplified: flat K communities
//! (no tree/LCA), and edge-level rather than vertex-level assignments.

use crate::link_community::model::*;
use crate::util::cell_pairs::connected_components;
use crate::util::common::new_progress_bar;
use legume_numeric::matrix::knn_graph::KnnGraph;
use log::info;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// Controls how edges are reassigned in component-partitioned sweeps.
#[derive(Clone, Copy)]
enum MoveStrategy {
    /// Gibbs sampling: stochastic, runs all sweeps.
    Sample,
    /// Greedy argmax: deterministic, early exit when converged.
    Greedy,
}

/// Parameters for component-partitioned Gibbs/greedy methods.
pub struct ComponentGibbsArgs<'a> {
    pub graph: &'a KnnGraph,
    pub edges: &'a [(usize, usize)],
    pub k: usize,
    /// Dirichlet concentration for mixing weight prior (0.0 = no prior).
    pub alpha: f64,
    /// The `[n_cells × k]` table of `incidence::cell_incidence_scores`, computed
    /// once from the post-V-cycle labels and held fixed through EM and greedy.
    pub incidence: Option<&'a [f32]>,
}

/// Collapsed Gibbs sampler for link community assignments.
pub struct LinkGibbsSampler {
    rng: SmallRng,
    /// The most blocks a component splits into and the fewest edges a
    /// split-off block holds, which a test may set to split a small graph.
    #[cfg(test)]
    block_layout: (usize, usize),
    /// What the last component sweeps ran with and ended on.
    #[cfg(test)]
    pub(crate) last_run: LastComponentRun,
}

/// The blocks the last component sweeps ran with, and the global stats as
/// patched between sweeps when they ended, for tests to check.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct LastComponentRun {
    pub block_sizes: Vec<usize>,
    pub feature_sum: Vec<f64>,
    pub size_sum: Vec<f64>,
    pub edge_count: Vec<usize>,
}

impl LinkGibbsSampler {
    /// Create a new sampler with the given RNG.
    pub fn new(rng: SmallRng) -> Self {
        LinkGibbsSampler {
            rng,
            #[cfg(test)]
            block_layout: (GIBBS_BLOCKS, GIBBS_MIN_BLOCK_EDGES),
            #[cfg(test)]
            last_run: LastComponentRun::default(),
        }
    }

    /// The same sampler with at most `blocks` blocks a component of at least
    /// `min_block_edges` edges each, so a test can split a small graph.
    #[cfg(test)]
    pub fn with_blocks(mut self, blocks: usize, min_block_edges: usize) -> Self {
        self.block_layout = (blocks, min_block_edges);
        self
    }

    /// The most blocks a component splits into and the fewest edges a
    /// split-off block holds: [`GIBBS_BLOCKS`] and [`GIBBS_MIN_BLOCK_EDGES`].
    #[cfg(not(test))]
    fn block_layout(&self) -> (usize, usize) {
        (GIBBS_BLOCKS, GIBBS_MIN_BLOCK_EDGES)
    }

    /// The block layout, as a test set it.
    #[cfg(test)]
    fn block_layout(&self) -> (usize, usize) {
        self.block_layout
    }

    /// Run `num_sweeps` sequential Gibbs sweeps over all edges.
    ///
    /// Returns the total number of edge moves across all sweeps.
    #[cfg(test)]
    pub fn run(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        num_sweeps: usize,
    ) -> usize {
        self.run_with_observer(stats, profiles, num_sweeps, 0.0, |_, _| {})
    }

    /// One sequential sweep, each edge reassigned from its collapsed conditional
    /// given all the others as they stand. Returns the number of edges that moved.
    fn sweep(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        alpha: f64,
        strategy: MoveStrategy,
    ) -> usize {
        sweep_edges(stats, profiles, alpha, None, strategy, &mut self.rng)
    }

    /// One Gibbs sweep, as the posterior tests drive it.
    #[cfg(test)]
    pub fn gibbs_sweep(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        alpha: f64,
    ) -> usize {
        self.sweep(stats, profiles, alpha, MoveStrategy::Sample)
    }

    /// Up to `num_sweeps` sequential sweeps, calling `observer` with
    /// `(sweep_idx, &stats)` after each. `Greedy` stops once no edge moves.
    fn run_sequential<F>(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        num_sweeps: usize,
        alpha: f64,
        strategy: MoveStrategy,
        mut observer: F,
    ) -> usize
    where
        F: FnMut(usize, &LinkCommunityStats),
    {
        let name = match strategy {
            MoveStrategy::Sample => "Gibbs sweeps",
            MoveStrategy::Greedy => "greedy sweeps",
        };
        let prog_bar = new_progress_bar(num_sweeps as u64).with_message(name);
        let mut total_moves = 0;
        for sweep in 0..num_sweeps {
            let sweep_moves = self.sweep(stats, profiles, alpha, strategy);
            total_moves += sweep_moves;
            observer(sweep, stats);
            prog_bar.set_message(sweep_message(name, sweep, sweep_moves));
            prog_bar.inc(1);
            if matches!(strategy, MoveStrategy::Greedy) && sweep_moves == 0 {
                break;
            }
        }
        prog_bar.finish_and_clear();
        total_moves
    }

    /// Run `num_sweeps` sequential Gibbs sweeps, calling `observer` with
    /// `(sweep_idx, &stats)` after each one to record per-sweep diagnostics.
    ///
    /// For the cascade, whose super-edges are few. Updating them together against
    /// one snapshot would not do: two alike, each alone in its community, each see
    /// their own community empty once left out, and trade labels every sweep.
    ///
    /// `alpha` is the Dirichlet concentration on community mixing weights:
    /// `alpha > 0` enables the prior, `alpha == 0` disables it.
    ///
    /// Returns the total number of edge moves across all sweeps.
    pub fn run_with_observer<F>(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        num_sweeps: usize,
        alpha: f64,
        observer: F,
    ) -> usize
    where
        F: FnMut(usize, &LinkCommunityStats),
    {
        self.run_sequential(
            stats,
            profiles,
            num_sweeps,
            alpha,
            MoveStrategy::Sample,
            observer,
        )
    }

    /// Run greedy (argmax) sweeps over all edges with early exit on convergence.
    ///
    /// Deterministic, sequential argmax, used for per-level finalization in the
    /// V-cycle cascade. For full fine-resolution greedy, the blocks of each
    /// component in parallel, use [`Self::run_greedy_by_components`].
    ///
    /// `alpha` is the Dirichlet concentration on community mixing weights;
    /// pass `0.0` to disable the prior.
    ///
    /// Returns the total number of edge moves across all sweeps.
    #[cfg(test)]
    pub fn run_greedy_plain(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        max_sweeps: usize,
        alpha: f64,
    ) -> usize {
        self.run_greedy_plain_with_observer(stats, profiles, max_sweeps, alpha, |_, _| {})
    }

    /// `Self::run_greedy_plain` with a per-sweep observer closure called
    /// after each sweep; the observer fires even on the final converged
    /// sweep so the caller sees the end state of the early-exit.
    pub fn run_greedy_plain_with_observer<F>(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        max_sweeps: usize,
        alpha: f64,
        observer: F,
    ) -> usize
    where
        F: FnMut(usize, &LinkCommunityStats),
    {
        self.run_sequential(
            stats,
            profiles,
            max_sweeps,
            alpha,
            MoveStrategy::Greedy,
            observer,
        )
    }

    /// Memoized component-partitioned sweeps with configurable move strategy.
    ///
    /// Partitions read each other's moves only at the end of a sweep, so across
    /// partitions this only approximates the conditional: two alike edges, each
    /// alone in its community and in different partitions, can trade labels.
    ///
    /// Edges are partitioned by connected component, and a large component is
    /// split into near-equal blocks (see [`split_components_into_blocks`]).
    /// The layout depends on the edges alone, so the draws do not depend on
    /// the thread count.
    /// Each sweep: partitions run sequential edge updates in parallel using a
    /// frozen snapshot of global stats, then global stats are patched with
    /// per-partition deltas.
    ///
    /// `strategy` controls how edges are reassigned:
    /// - `Sample`: Gibbs sampling (stochastic)
    /// - `Greedy`: argmax (deterministic, early exit when no moves)
    fn run_components(
        &mut self,
        membership: &mut [usize],
        profiles: &LinkProfileStore,
        args: &ComponentGibbsArgs,
        num_sweeps: usize,
        strategy: MoveStrategy,
    ) -> usize {
        let ComponentGibbsArgs {
            graph,
            edges,
            k,
            alpha,
            incidence,
        } = args;
        let (k, alpha) = (*k, *alpha);
        let incidence_scores = *incidence;

        let (comp_labels, n_comp) = connected_components(graph);
        let (blocks, min_block_edges) = self.block_layout();
        let comp_edges =
            split_components_into_blocks(edges, &comp_labels, n_comp, blocks, min_block_edges);
        let n_parts = comp_edges.len();

        let sub_stores: Vec<LinkProfileStore> = comp_edges
            .iter()
            .map(|indices| profiles.subset(indices))
            .collect();

        // Per-component endpoint slice (indexed by local edge id), only
        // needed when incidence is enabled — avoids an unconditional
        // ~|E|·sizeof((usize,usize)) allocation in the baseline path.
        let local_edges: Vec<Vec<(usize, usize)>> = if incidence_scores.is_some() {
            comp_edges
                .iter()
                .map(|indices| indices.iter().map(|&e| edges[e]).collect())
                .collect()
        } else {
            Vec::new()
        };

        let label = match strategy {
            MoveStrategy::Sample => "EM Gibbs",
            MoveStrategy::Greedy => "Greedy",
        };

        if matches!(strategy, MoveStrategy::Sample) {
            info!(
                "Memoized {}: {} components -> {} partitions (edges: {})",
                label,
                n_comp,
                n_parts,
                comp_edges
                    .iter()
                    .map(|e| e.len().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        let mut global_stats = LinkCommunityStats::from_profiles(profiles, k, membership);

        let mut memo_stats: Vec<_> = comp_edges
            .iter()
            .map(|indices| LinkCommunityStats::component_stats(profiles, k, indices, membership))
            .collect();

        let mut total_moves = 0usize;
        let base_seed = self.rng.random::<u64>() | 1;

        let prog_bar =
            new_progress_bar((num_sweeps * n_parts) as u64).with_message(format!("{label} jobs"));

        for sweep in 0..num_sweeps {
            let sweep_seed = base_seed.wrapping_mul(sweep as u64 + 1);

            let local_results: Vec<ComponentResult> = (0..n_parts)
                .into_par_iter()
                .map(|c| {
                    let indices = &comp_edges[c];
                    let mut local_stats = LinkCommunityStats {
                        k,
                        m: profiles.m,
                        feature_sum: global_stats.feature_sum.clone(),
                        size_sum: global_stats.size_sum.clone(),
                        edge_count: global_stats.edge_count.clone(),
                        membership: indices.iter().map(|&e| membership[e]).collect(),
                        feature_table: global_stats.feature_table.clone(),
                    };

                    // Frozen incidence context: the cell scores are a
                    // global read-only table; the only per-component
                    // piece is the endpoint slice indexed by local edge id.
                    let inc_ctx = incidence_scores.map(|cell_scores| IncidenceCtx {
                        edges: local_edges[c].as_slice(),
                        cell_scores,
                    });

                    let comp_seed = sweep_seed ^ (c as u64).wrapping_mul(2654435761);
                    let mut rng = SmallRng::seed_from_u64(comp_seed);
                    let moves = sweep_edges(
                        &mut local_stats,
                        &sub_stores[c],
                        alpha,
                        inc_ctx,
                        strategy,
                        &mut rng,
                    );

                    let new_comp_stats =
                        LinkCommunityStats::local_stats(&sub_stores[c], k, &local_stats.membership);

                    prog_bar.inc(1);

                    ComponentResult {
                        membership: local_stats.membership,
                        new_stats: new_comp_stats,
                        moves,
                    }
                })
                .collect();

            let mut sweep_moves = 0usize;
            for (c, result) in local_results.into_iter().enumerate() {
                sweep_moves += result.moves;

                for (local_e, &global_e) in comp_edges[c].iter().enumerate() {
                    membership[global_e] = result.membership[local_e];
                }

                let old = (
                    memo_stats[c].0.as_slice(),
                    memo_stats[c].1.as_slice(),
                    memo_stats[c].2.as_slice(),
                );
                let new = (
                    result.new_stats.0.as_slice(),
                    result.new_stats.1.as_slice(),
                    result.new_stats.2.as_slice(),
                );
                global_stats.apply_delta(old, new);
                memo_stats[c] = result.new_stats;
            }

            total_moves += sweep_moves;
            prog_bar.set_message(sweep_message(&format!("{label} jobs"), sweep, sweep_moves));
            if matches!(strategy, MoveStrategy::Greedy) && sweep_moves == 0 {
                break;
            }
        }
        prog_bar.finish_and_clear();
        #[cfg(test)]
        {
            self.last_run = LastComponentRun {
                block_sizes: comp_edges.iter().map(Vec::len).collect(),
                feature_sum: global_stats.feature_sum,
                size_sum: global_stats.size_sum,
                edge_count: global_stats.edge_count,
            };
        }

        total_moves
    }

    /// Memoized EM Gibbs sampling, the blocks of each component in parallel.
    pub fn run_components_em(
        &mut self,
        membership: &mut [usize],
        profiles: &LinkProfileStore,
        args: &ComponentGibbsArgs,
        num_sweeps: usize,
    ) -> usize {
        self.run_components(membership, profiles, args, num_sweeps, MoveStrategy::Sample)
    }

    /// Memoized greedy finalization, the blocks of each component in parallel.
    pub fn run_greedy_by_components(
        &mut self,
        membership: &mut [usize],
        profiles: &LinkProfileStore,
        args: &ComponentGibbsArgs,
        max_sweeps: usize,
    ) -> usize {
        self.run_components(membership, profiles, args, max_sweeps, MoveStrategy::Greedy)
    }
}

/// One sequential sweep over the edges of `profiles`, each reassigned from its
/// collapsed conditional given all the others as they stand. Returns the moves.
fn sweep_edges(
    stats: &mut LinkCommunityStats,
    profiles: &LinkProfileStore,
    alpha: f64,
    incidence: Option<IncidenceCtx<'_>>,
    strategy: MoveStrategy,
    rng: &mut SmallRng,
) -> usize {
    let mut log_probs = vec![0.0f64; stats.k];
    let mut increments = vec![0.0f32; stats.k];
    let mut moves = 0;
    for e in 0..profiles.n_edges {
        let old_c = stats.membership[e];
        compute_log_probs_for_edge(
            e,
            stats,
            profiles,
            alpha,
            incidence,
            &mut log_probs,
            &mut increments,
        );
        let new_c = match strategy {
            MoveStrategy::Sample => sample_categorical_log(&log_probs, rng),
            MoveStrategy::Greedy => greedy_choice(&log_probs, old_c),
        };
        if new_c != old_c {
            stats.delta_move(e, old_c, new_c, profiles);
            moves += 1;
        }
    }
    moves
}

/// Result from a single component's Gibbs/greedy sweep.
struct ComponentResult {
    membership: Vec<usize>,
    new_stats: (Vec<f64>, Vec<f64>, Vec<usize>),
    moves: usize,
}

/// Partition edge indices by connected component.
///
/// Returns `n_comp` vectors, each containing the global edge indices belonging
/// to that component. Edge `(i, j)` is assigned to the component of node `i`.
fn partition_edges_by_component(
    edges: &[(usize, usize)],
    comp_labels: &[usize],
    n_comp: usize,
) -> Vec<Vec<usize>> {
    let mut comp_edges: Vec<Vec<usize>> = vec![Vec::new(); n_comp];
    for (e, &(i, _j)) in edges.iter().enumerate() {
        comp_edges[comp_labels[i]].push(e);
    }
    comp_edges
}

/// Most blocks the component sweeps split one component into.
///
/// A block runs one exact sequential sweep; blocks run in parallel, each
/// against one snapshot of the others, which they see only at the end of the
/// sweep. Fewer, larger blocks are closer to exact; more spread a component
/// over more threads. The count is fixed, not taken from the thread count, so
/// the draws do not depend on the machine. Sixteen matches pinto's default
/// thread cap today, so a default run on a machine with at least that many
/// cores keeps every thread busy on one large component, and holds one copy
/// of the community stats per thread, as when the blocks followed the thread
/// count. It is a literal, not the cap, so changing the cap does not change
/// lc's draws.
pub(crate) const GIBBS_BLOCKS: usize = 16;

/// Fewest edges a block split off a component holds: a component of fewer
/// than twice this many is one exact block. It is the floor the blocks had
/// when they followed the thread count, so at sixteen blocks a component
/// splits as it did at sixteen threads.
pub(crate) const GIBBS_MIN_BLOCK_EDGES: usize = 256;

/// Partition by connected component, then split each component into
/// `len / min_block_edges` near-equal blocks, at least one and at most
/// `blocks`, of consecutive edges.
pub(crate) fn split_components_into_blocks(
    edges: &[(usize, usize)],
    comp_labels: &[usize],
    n_comp: usize,
    blocks: usize,
    min_block_edges: usize,
) -> Vec<Vec<usize>> {
    let (blocks, min_block_edges) = (blocks.max(1), min_block_edges.max(1));
    let mut split = Vec::new();
    for group in partition_edges_by_component(edges, comp_labels, n_comp) {
        if group.is_empty() {
            continue;
        }
        let n = group.len();
        let n_blocks = (n / min_block_edges).clamp(1, blocks);
        split.extend(
            (0..n_blocks).map(|b| group[b * n / n_blocks..(b + 1) * n / n_blocks].to_vec()),
        );
    }
    split
}

/// A sweep bar's message: its name, which sweep, and the edges it moved
/// as the one `key=value`, the score `pinto run` draws.
pub(crate) fn sweep_message(name: &str, sweep: usize, moves: usize) -> String {
    format!("{name} {}: moves={moves}", sweep + 1)
}

/// The best community, if it strictly beats the current one. A tie stays put: a
/// lone edge ties with every empty community, and would otherwise keep moving.
fn greedy_choice(log_probs: &[f64], current: usize) -> usize {
    let best = argmax_log(log_probs);
    if log_probs[best] > log_probs[current] {
        best
    } else {
        current
    }
}

/// Pick the index with the highest value.
fn argmax_log(log_probs: &[f64]) -> usize {
    let mut best = 0;
    let mut best_val = log_probs[0];
    for (i, &v) in log_probs.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

/// Sample from a categorical distribution given log-probabilities.
///
/// Uses the log-sum-exp trick for numerical stability.
pub(crate) fn sample_categorical_log(log_probs: &[f64], rng: &mut SmallRng) -> usize {
    let max = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Compute total weight without allocating a Vec
    let total: f64 = log_probs.iter().map(|lp| (lp - max).exp()).sum();

    if total <= 0.0 || !total.is_finite() {
        return rng.random_range(0..log_probs.len());
    }

    let u: f64 = rng.random::<f64>() * total;
    let mut cum = 0.0;
    for (i, lp) in log_probs.iter().enumerate() {
        cum += (lp - max).exp();
        if cum >= u {
            return i;
        }
    }

    log_probs.len() - 1
}
