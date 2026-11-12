use crate::util::common::*;
use crate::util::knn_graph::KnnGraph;
use data_beans::alg::cell_pairs::collapse_pairs;
use data_beans::alg::union_find::UnionFind;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use std::collections::VecDeque;

/// Result of graph coarsening: a sequence of merges forming a dendrogram.
pub struct CoarsenResult {
    /// Each entry (a, b) records that groups rooted at a and b were merged.
    pub(super) merges: Vec<(usize, usize)>,
    /// Total number of original nodes.
    pub(super) n_nodes: usize,
}

/// Degree-corrected modularity-gain merge veto.
///
/// Reject a proposed merge `(i, j)` with similarity `sim(i,j)` when
/// `sim(i,j) < gamma · deg(i) · deg(j) / (2W)`, the standard Louvain/Leiden
/// merge-gain criterion adapted to a cosine-weighted adjacency. Weights are
/// clamped at zero so anti-correlated cell pairs never contribute to the
/// null.
///
/// `gamma = 1.0` is a natural default (modularity resolution). Set to `0.0`
/// to accept any proposal with non-negative similarity, or pass `None` to
/// the coarsener to disable the veto entirely.
#[derive(Clone, Copy, Debug)]
pub struct ModularityVeto {
    pub gamma: f32,
}

impl Default for ModularityVeto {
    fn default() -> Self {
        Self { gamma: 1.0 }
    }
}

/// Parallel graph-constrained agglomerative coarsening.
///
/// Each round merges every pair of neighbouring clusters that are each
/// other's cheapest merge:
/// 1. Every edge between two clusters is scored by the Ward cost of merging
///    them, `n_a · n_b / (n_a + n_b) · (1 − cos)` (parallel), `n` counting
///    cells.
/// 2. Each cluster names its cheapest neighbour; a pair that names each
///    other merges.
/// 3. The merges are applied: Union-Find, size-weighted centroids, edges.
/// 4. Repeat until no pair is left to merge.
///
/// The size term is what keeps a cut balanced. A centroid is an average, so
/// less noisy than a cell, and a noisy cell is therefore more like a large
/// cluster than like another cell. Scored on similarity alone, the clusters
/// around a large one all ask for it, one wins the round and the rest wait:
/// the large one grows a cell a round, and a cut hands back a few giants
/// and a crowd of single cells. With the size term a large cluster pays for
/// its size, so the sizes grow together.
///
/// Both ideas are adopted from existing work:
/// - The merge cost is Ward's: the rise in within-cluster sum of squares,
///   `n_a · n_b / (n_a + n_b) · ‖μ_a − μ_b‖²`. Here the centroids are unit
///   vectors, so `‖μ_a − μ_b‖² = 2 (1 − cos)`, and the constant 2 is dropped.
///   Ward, J. H. (1963). Hierarchical grouping to optimize an objective
///   function. Journal of the American Statistical Association, 58(301),
///   236-244. <https://doi.org/10.1080/01621459.1963.10500845>
/// - Merging reciprocal nearest neighbours, many pairs at once: for a
///   linkage like Ward's, a pair that are each other's nearest stays so
///   whatever else merges, so such pairs can be merged in any order, or
///   together, and give the hierarchy one merge at a time would. Taken here
///   on the graph's edges only, and with the cosine form of the cost, for
///   which that is an approximation.
///   Murtagh, F. (1983). A survey of recent advances in hierarchical
///   clustering algorithms. The Computer Journal, 26(4), 354-359.
///   <https://doi.org/10.1093/comjnl/26.4.354>
///
/// * `graph` - spatial KNN graph
/// * `cell_features` - `[proj_dim × n_nodes]` matrix with L2-normalized columns;
///   mutated in place as merged centroids replace absorbed columns.
/// * `sizes` - the cells each node stands for (spatial seeds); one apiece
///   when `None`.
/// * `veto` - optional [`ModularityVeto`]: when supplied, an edge is left
///   out if its similarity is no better than expected given each endpoint's
///   weighted degree in the current super-graph.
pub fn graph_coarsen(
    graph: &KnnGraph,
    cell_features: &mut Mat,
    sizes: Option<&[f32]>,
    veto: Option<&ModularityVeto>,
) -> CoarsenResult {
    let n = graph.n_nodes;
    let dim = cell_features.nrows();

    let mut uf = UnionFind::new(n);
    // Cells per cluster, read at a cluster's root.
    let mut size: Vec<f32> = sizes.map_or_else(|| vec![1.0; n], <[f32]>::to_vec);
    debug_assert_eq!(size.len(), n);
    let max_merges = n.saturating_sub(1);
    let mut merges = Vec::with_capacity(max_merges);

    let prog_bar = new_progress_bar(max_merges as u64).with_message("coarsening merges");

    // The super-graph's edges between cluster roots, each once as
    // `(min, max)`, sorted.
    let mut edges: Vec<(usize, usize)> = graph.edges.clone();
    let root_edges = |edges: &mut Vec<(usize, usize)>, uf: &UnionFind| {
        edges.par_iter_mut().for_each(|e| {
            let (a, b) = (uf.parent(e.0), uf.parent(e.1));
            *e = (a.min(b), a.max(b));
        });
        edges.retain(|&(a, b)| a != b);
        edges.par_sort_unstable();
        edges.dedup();
    };
    root_edges(&mut edges, &uf);

    // Each cluster's cheapest merge this round: `(cost, neighbour)`.
    const NONE: (f32, usize) = (f32::INFINITY, usize::MAX);
    let mut cheapest = vec![NONE; n];
    let mut sims: Vec<f32> = Vec::new();
    let mut deg_cos = vec![0.0f32; n];
    let mut total_vetoed = 0usize;
    let mut n_rounds = 0u32;

    while !edges.is_empty() {
        let features_ref = &*cell_features;
        edges
            .par_iter()
            .map(|&(i, j)| features_ref.column(i).dot(&features_ref.column(j)))
            .collect_into_vec(&mut sims);

        // Per-node weighted degree and total weight of the current
        // super-graph. Negative cosines clamp to zero so the null model
        // doesn't credit anti-correlated neighborhoods.
        let mut total_w = 0.0f32;
        if veto.is_some() {
            deg_cos.fill(0.0);
            for (&(i, j), &sim) in edges.iter().zip(&sims) {
                let s = sim.max(0.0);
                deg_cos[i] += s;
                deg_cos[j] += s;
                total_w += s;
            }
        }

        cheapest.fill(NONE);
        for (&(i, j), &sim) in edges.iter().zip(&sims) {
            if !sim.is_finite() {
                continue;
            }
            // Degree-corrected modularity-gain veto. Require strictly
            // positive similarity (no merging anti-correlated or orthogonal
            // pairs) AND strict gain over the null rate.
            // `total_w` is positive whenever some `sim` is.
            if let Some(v) = veto {
                if sim <= 0.0 || sim <= v.gamma * deg_cos[i] * deg_cos[j] / (2.0 * total_w) {
                    total_vetoed += 1;
                    continue;
                }
            }
            // A dot product of unit vectors can round above 1; a negative
            // distance would turn the size term into a reward.
            let cost = size[i] * size[j] / (size[i] + size[j]) * (1.0 - sim).max(0.0);
            // Ties go to the lower index, so the cheapest edge overall is
            // always named from both ends and a round always merges.
            if (cost, j) < cheapest[i] {
                cheapest[i] = (cost, j);
            }
            if (cost, i) < cheapest[j] {
                cheapest[j] = (cost, i);
            }
        }

        // Pairs that name each other, cheapest first: a cut may stop partway
        // through a round.
        let mut round_merges: Vec<(f32, usize, usize)> = edges
            .iter()
            .filter(|&&(i, j)| cheapest[i].1 == j && cheapest[j].1 == i)
            .map(|&(i, j)| (cheapest[i].0, i, j))
            .collect();
        if round_merges.is_empty() {
            break;
        }
        round_merges.sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then((x.1, x.2).cmp(&(y.1, y.2))));

        // UF merges + feature updates (sequential)
        for &(_, a, b) in &round_merges {
            merges.push((a, b));
            prog_bar.inc(1);

            let (size_a, size_b) = (size[a], size[b]);
            let total = size_a + size_b;
            let new_rep = uf.union(a, b);
            size[new_rep] = total;

            for r in 0..dim {
                cell_features[(r, new_rep)] =
                    (cell_features[(r, a)] * size_a + cell_features[(r, b)] * size_b) / total;
            }
            // Cells with no signal average to zero, which has no direction:
            // left at zero, it is merely unlike everything.
            cell_features
                .column_mut(new_rep)
                .try_normalize_mut(f32::MIN_POSITIVE);
        }

        // Each merged cluster is one hop from its new root.
        root_edges(&mut edges, &uf);

        n_rounds += 1;
    }

    prog_bar.finish_and_clear();
    let vetoed = veto.map_or_else(String::new, |_| format!(" ({total_vetoed} edges vetoed)"));
    info!(
        "Graph coarsening: {n} nodes, {} merges in {n_rounds} rounds{vetoed}",
        merges.len()
    );

    CoarsenResult { merges, n_nodes: n }
}

/// Map cell labels to pair sample indices.
///
/// Each pair (i,j) maps to a canonical sample key `(min(label[i], label[j]), max(...))`.
/// Returns `(pair_to_sample, n_samples)`.
///
/// The sample *keys* are discarded here; `build_super_edges` keeps them. Both
/// are views on the same collapse — see [`collapse_pairs`].
pub fn cell_labels_to_pair_samples(
    cell_labels: &[usize],
    pairs: &[(usize, usize)],
) -> (Vec<usize>, usize) {
    let (samples, pair_to_sample) = collapse_pairs(pairs, cell_labels);
    (pair_to_sample, samples.len())
}

/// Compute target cluster counts for multi-level extraction.
///
/// Linearly interpolates from coarsest (16 or fewer) to finest (`n_clusters`).
pub fn compute_level_n_clusters(n_clusters: usize, num_levels: usize) -> Vec<usize> {
    if num_levels <= 1 {
        return vec![n_clusters];
    }
    let coarsest = 16usize.min(n_clusters);
    let mut result = Vec::with_capacity(num_levels);
    for level in 0..num_levels {
        let t = level as f32 / (num_levels - 1) as f32;
        let nc = coarsest as f32 + t * (n_clusters - coarsest) as f32;
        result.push(nc.round() as usize);
    }
    result
}

/// Threshold: if n_nodes > this, use spatial seeding pre-pass.
pub const SEEDING_THRESHOLD: usize = 20_000;

/// Assign cells to spatially-seeded super-nodes via BFS flood-fill.
///
/// Places seeds on a regular grid over each group's bounding box (groups,
/// such as sections, have independent coordinate systems), then expands
/// all seeds simultaneously along graph edges.
///
/// Returns `(labels, num_super_nodes)` where `labels[cell] = super_node_id`.
pub(super) fn spatial_seed_labels(
    coordinates: &Mat,
    graph: &KnnGraph,
    target_super: usize,
    groups: Option<&[u32]>,
) -> (Vec<usize>, usize) {
    let n = graph.n_nodes;
    let n_dims = coordinates.ncols();

    // Partition cells by group (dense ids, single pass over cells)
    let group_cells: Vec<Vec<usize>> =
        groups.map_or_else(|| vec![(0..n).collect()], crate::util::ablation::group);
    let n_groups = group_cells.len();

    let mut seeds: Vec<usize> = Vec::new();
    let mut labels = vec![usize::MAX; n];
    let mut num_super = 0usize;

    for cells in &group_cells {
        let group_target = (target_super as f64 * cells.len() as f64 / n as f64).ceil() as usize;
        let group_target = group_target.max(1);

        // Bounding box for this group
        let mut mins = vec![f32::INFINITY; n_dims];
        let mut maxs = vec![f32::NEG_INFINITY; n_dims];
        for &i in cells {
            for d in 0..n_dims {
                let v = coordinates[(i, d)];
                if v < mins[d] {
                    mins[d] = v;
                }
                if v > maxs[d] {
                    maxs[d] = v;
                }
            }
        }

        // Grid dimensions
        let side = (group_target as f64).powf(1.0 / n_dims as f64).ceil() as usize;
        let side = side.max(1);
        let mut tile_strides = vec![1usize; n_dims];
        for d in (0..n_dims - 1).rev() {
            tile_strides[d] = tile_strides[d + 1] * side;
        }
        let total_tiles = tile_strides[0] * side;

        // Assign cells in this group to tiles, pick one seed per tile
        let mut tile_seed: Vec<Option<usize>> = vec![None; total_tiles];
        for &i in cells {
            let mut tile_id = 0usize;
            for d in 0..n_dims {
                let range = (maxs[d] - mins[d]).max(1e-12);
                let frac = ((coordinates[(i, d)] - mins[d]) / range).clamp(0.0, 0.999999);
                let bin = (frac * side as f32) as usize;
                tile_id += bin * tile_strides[d];
            }
            if tile_seed[tile_id].is_none() {
                tile_seed[tile_id] = Some(i);
            }
        }

        for seed in tile_seed.into_iter().flatten() {
            labels[seed] = num_super;
            seeds.push(seed);
            num_super += 1;
        }
    }

    // Capacity-limited multi-source BFS: each super-node absorbs at most
    // 2× the average cell count to keep sizes uniform across density variation.
    let max_cells = (n / num_super.max(1)) * 2;
    let mut seed_counts = vec![1usize; num_super]; // each seed is one cell

    let mut queue = VecDeque::with_capacity(n);
    let mut spillover: VecDeque<usize> = VecDeque::new();
    for &cell in &seeds {
        queue.push_back(cell);
    }
    let csc = &graph.adjacency;

    while let Some(cell) = queue.pop_front() {
        let label = labels[cell];
        if seed_counts[label] >= max_cells {
            continue;
        }
        let col = csc.col(cell);
        for &nb in col.row_indices() {
            if labels[nb] == usize::MAX {
                labels[nb] = label;
                seed_counts[label] += 1;
                if seed_counts[label] < max_cells {
                    queue.push_back(nb);
                } else {
                    // Seed is now full — collect overflow for uncapped spillover
                    spillover.push_back(nb);
                }
            }
        }
    }

    // Propagate remaining unassigned cells without capacity limit.
    while let Some(cell) = spillover.pop_front() {
        let label = labels[cell];
        let col = csc.col(cell);
        for &nb in col.row_indices() {
            if labels[nb] == usize::MAX {
                labels[nb] = label;
                spillover.push_back(nb);
            }
        }
    }

    // Truly disconnected cells get their own singleton super-node
    for label in labels.iter_mut() {
        if *label == usize::MAX {
            *label = num_super;
            num_super += 1;
        }
    }

    info!(
        "Spatial seeding: {} cells, {} groups → {} super-nodes (target {})",
        n, n_groups, num_super, target_super
    );

    (labels, num_super)
}

/// Fold a fine edge list by node labels into deduped `(min, max)` super
/// edges plus the fine-edge -> super-edge map (`None` = both endpoints in
/// one group; the edge folds into the node — Louvain / METIS semantics,
/// so callers that aggregate member counts onto the node already hold
/// that mass and a `(a, a)` self-loop would count it twice). The
/// edges-only core of [`build_super_graph`], callable without paying for
/// feature aggregation or the CSC adjacency. `keep`, when given, marks
/// fine edges that may NOT form or feed a super edge (cage: cross-batch
/// edges) — they map to `None` exactly like intra-group edges.
pub(crate) fn fold_edges_to_super(
    labels: &[usize],
    edges: &[(usize, usize)],
    keep: Option<&[bool]>,
) -> (Vec<(usize, usize)>, Vec<Option<usize>>) {
    // Keyed by INDEX, not just membership, so the fine->super map falls out
    // of the same pass instead of needing a second one.
    let mut edge_set: HashMap<(usize, usize), usize> =
        HashMap::with_capacity_and_hasher(edges.len(), Default::default());
    let mut super_edges: Vec<(usize, usize)> = Vec::with_capacity(edges.len());
    let mut fine_to_super: Vec<Option<usize>> = Vec::with_capacity(edges.len());
    for (e, &(i, j)) in edges.iter().enumerate() {
        let si = labels[i];
        let sj = labels[j];
        let dropped = keep.map(|k| !k[e]).unwrap_or(false);
        if si == sj || dropped {
            fine_to_super.push(None);
        } else {
            let key = (si.min(sj), si.max(sj));
            let idx = *edge_set.entry(key).or_insert_with(|| {
                super_edges.push(key);
                super_edges.len() - 1
            });
            fine_to_super.push(Some(idx));
        }
    }
    (super_edges, fine_to_super)
}

/// Coarsen the cell graph onto super-nodes: [`fold_edges_to_super`] for
/// the edge structure, plus per-super-node MEAN features and a CSC
/// adjacency. Returns `(super_graph, super_features, fine_to_super)`;
/// `fine_to_super[i]` is the super-edge index of input edge `i`, or
/// `None` when the edge is internal to one super-node.
pub(crate) fn build_super_graph(
    seed_labels: &[usize],
    num_super: usize,
    graph: &KnnGraph,
    cell_features: &Mat,
) -> (KnnGraph, Mat, Vec<Option<usize>>) {
    let dim = cell_features.nrows();

    // Aggregate features: sum columns per super-node via column ops
    let mut super_features = Mat::zeros(dim, num_super);
    let mut counts = vec![0usize; num_super];
    for (cell, &label) in seed_labels.iter().enumerate() {
        counts[label] += 1;
        let mut col = super_features.column_mut(label);
        col += cell_features.column(cell);
    }
    for (s, &c) in counts.iter().enumerate() {
        if c > 0 {
            super_features.column_mut(s).scale_mut(1.0 / c as f32);
        }
    }

    let (super_edges, fine_to_super) = fold_edges_to_super(seed_labels, &graph.edges, None);

    let distances = vec![1.0f32; super_edges.len()];

    let mut coo = CooMatrix::new(num_super, num_super);
    for &(i, j) in &super_edges {
        coo.push(i, j, 1.0f32);
        coo.push(j, i, 1.0f32);
    }
    let adjacency = CscMatrix::from(&coo);

    let super_graph = KnnGraph {
        adjacency,
        edges: super_edges,
        distances,
        n_nodes: num_super,
    };

    info!(
        "Super-graph: {} nodes, {} edges",
        num_super,
        super_graph.edges.len()
    );

    (super_graph, super_features, fine_to_super)
}

/// Optional spatial seeding parameters for large datasets.
pub struct SeedingParams<'a> {
    /// Positions in the tissue plane (no batch offset column).
    pub coordinates: &'a Mat,
    /// Each cell's group, seeded on its own grid: dense ids, 0..k.
    pub groups: Option<&'a [u32]>,
}

/// [`graph_coarsen`] over spatial seeds: `labels[cell]` is the cell's seed,
/// one of `num_super`, and `cell_features` has unit columns. The merges are
/// over seeds, each weighing the cells behind it, so the size term still
/// counts cells.
pub fn graph_coarsen_seeds(
    labels: &[usize],
    num_super: usize,
    graph: &KnnGraph,
    cell_features: &Mat,
    veto: Option<&ModularityVeto>,
) -> CoarsenResult {
    // A seed's centroid: the mean of its cells' unit columns, made unit.
    let (super_graph, mut super_features, _) =
        build_super_graph(labels, num_super, graph, cell_features);
    super_features.normalize_columns_inplace();
    let mut seed_cells = vec![0.0f32; num_super];
    labels.iter().for_each(|&l| seed_cells[l] += 1.0);
    graph_coarsen(&super_graph, &mut super_features, Some(&seed_cells), veto)
}

/// Configuration for multi-level graph coarsening.
pub struct CoarsenConfig<'a> {
    /// Finest-level target cluster count.
    pub n_clusters: usize,
    /// Number of dendrogram cuts to extract, linearly spaced coarsest → finest.
    pub num_levels: usize,
    /// Spatial grid seeding pre-pass for large datasets (see [`SeedingParams`]).
    pub seeding: Option<SeedingParams<'a>>,
    /// Optional degree-corrected modularity-gain veto on merge proposals.
    /// `None` disables the veto (legacy behaviour — accept any sim).
    pub modularity_veto: Option<ModularityVeto>,
}

/// Multi-level graph coarsening result.
pub struct MultiLevelCoarsenResult {
    /// Per-level pair-to-sample mapping (coarse → fine).
    pub all_pair_to_sample: Vec<Vec<usize>>,
    /// Per-level number of distinct samples.
    pub all_num_samples: Vec<usize>,
    /// Per-level cell cluster labels (coarse → fine).
    pub all_cell_labels: Vec<Vec<usize>>,
}

/// Graph-constrained agglomerative coarsening with multi-level extraction.
///
/// Coarsens cells along KNN graph edges ([`graph_coarsen`]), then cuts the
/// merge sequence at `config.num_levels` linearly-spaced cluster counts from
/// coarsest to `config.n_clusters`.
///
/// When `config.seeding` is provided and n > 20k, a spatial grid seeding
/// pre-pass reduces the problem to ~8× n_clusters super-nodes before
/// running the expression-guided agglomerative coarsening.
///
/// Returns per-level pair-to-sample mappings ready for the fused or
/// per-sample visitors.
///
/// # CELLS are collapsed; edges are INDUCED
///
/// No edge is ever collapsed directly. Cell-cell edges become
/// super-cell↔super-cell edges as a CONSEQUENCE of the cell partition, via
/// [`cell_labels_to_pair_samples`] here and `build_super_edges` downstream. So
/// the pseudobulk unit is a super-CELL, and super-edges are derived from it —
/// which is why `cage`'s sampler conditions on a per-super-cell `e_pb`.
///
/// # Levels are NESTED, and hold the cluster counts asked for
///
/// Every level is the same merge sequence stopped at a different point, so
/// each finer cluster lies inside exactly one coarser cluster, and a level
/// asked for `k` clusters has `k`. It has more only where the merging ran
/// out first: the graph has more than `k` connected pieces, or the veto
/// refused every edge that was left.
///
/// Nothing moves a cell after the cut: a moved cell would break the
/// nesting, and putting it back means splitting clusters at their parents'
/// boundaries, which leaves a level with more and smaller clusters than it
/// was asked for.
pub fn graph_coarsen_multilevel(
    graph: &KnnGraph,
    cell_features: &mut Mat,
    pairs: &[(usize, usize)],
    config: CoarsenConfig<'_>,
) -> MultiLevelCoarsenResult {
    let CoarsenConfig {
        n_clusters,
        num_levels,
        seeding,
        modularity_veto,
    } = config;
    let veto_ref = modularity_veto.as_ref();
    let n_original = graph.n_nodes;

    // Unit columns, for the seeds' centroids as for the cells'.
    cell_features.normalize_columns_inplace();

    // Optional seeding pre-pass for large datasets: the agglomeration then
    // runs on the seeds' super-graph, and `seed_labels` maps cells to seeds.
    let (coarsen_result, seed_labels) =
        if let Some(sp) = seeding.filter(|_| n_original > SEEDING_THRESHOLD) {
            let target_super = (n_clusters * 8).min(n_original / 4).max(n_clusters + 1);
            let (labels, num_super) =
                spatial_seed_labels(sp.coordinates, graph, target_super, sp.groups);
            let merged = graph_coarsen_seeds(&labels, num_super, graph, cell_features, veto_ref);
            (merged, Some(labels))
        } else {
            (graph_coarsen(graph, cell_features, None, veto_ref), None)
        };

    // One incremental UF pass over the levels, finest → coarsest:
    // `merge_idx` only ever advances, so the union-find can add merges but
    // never undo one.
    let n_coarsen = coarsen_result.n_nodes; // super-nodes if seeded, cells otherwise
    let level_n_clusters = compute_level_n_clusters(n_clusters, num_levels);
    let mut uf = UnionFind::new(n_coarsen);
    let mut merge_idx = 0usize;
    let mut out = MultiLevelCoarsenResult {
        all_pair_to_sample: Vec::with_capacity(level_n_clusters.len()),
        all_num_samples: Vec::with_capacity(level_n_clusters.len()),
        all_cell_labels: Vec::with_capacity(level_n_clusters.len()),
    };

    for &nc in level_n_clusters.iter().rev() {
        let target = n_coarsen
            .saturating_sub(nc)
            .min(coarsen_result.merges.len());
        for &(a, b) in &coarsen_result.merges[merge_idx..target.max(merge_idx)] {
            uf.union(a, b);
        }
        merge_idx = target.max(merge_idx);

        // Compact labels at the coarsened level
        let mut rep_to_label: HashMap<usize, usize> = Default::default();
        let coarse_labels: Vec<usize> = (0..n_coarsen)
            .map(|i| {
                let next = rep_to_label.len();
                *rep_to_label.entry(uf.find(i)).or_insert(next)
            })
            .collect();

        // Compose with seed labels to get original-cell-level labels
        let cell_labels: Vec<usize> = match seed_labels.as_deref() {
            Some(seed_of) => seed_of.iter().map(|&s| coarse_labels[s]).collect(),
            None => coarse_labels,
        };

        let (pair_to_sample, num_samples) = cell_labels_to_pair_samples(&cell_labels, pairs);
        out.all_pair_to_sample.push(pair_to_sample);
        out.all_num_samples.push(num_samples);
        out.all_cell_labels.push(cell_labels);
    }
    // Levels are reported coarse → fine.
    out.all_pair_to_sample.reverse();
    out.all_num_samples.reverse();
    out.all_cell_labels.reverse();

    info!(
        "Multi-level coarsening: {} levels, samples per level: {:?}",
        level_n_clusters.len(),
        out.all_num_samples
    );

    out
}
