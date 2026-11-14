//! Feature-module expression and module-pair profiles.

use crate::link_community::model::LinkProfileStore;
use crate::util::common::*;
use legume_numeric::matrix::pair_graph::FeaturePairGraph;
use legume_numeric::matrix::utils::generate_minibatch_intervals;
use rayon::prelude::*;

/// Feature-network-derived module-pair basis for per-cell-edge features.
///
/// Constructed once after feature-module resolution: walks the feature-feature edge
/// list, buckets each edge by its endpoints' module labels, and keeps the
/// canonical `(a ≤ b)` pairs with positive weight. Each kept pair gets a
/// contiguous index `0..n_pairs` and a precomputed null factor
/// `deg(a)·deg(b)/(2W)²` used in the per-edge residual.
/// Neighbor entry in `ModulePairBasis::pair_adj`.
#[derive(Copy, Clone, Debug)]
pub struct PairAdjEntry {
    /// Neighbor module index.
    pub b: u32,
    /// Contiguous pair index into profile columns.
    pub pair_idx: u32,
    /// Modularity null factor `deg(a)·deg(b) / (2W)²`.
    pub null_ab: f32,
}

pub struct ModulePairBasis {
    pub n_modules: usize,
    pub module_of_feature: Vec<Option<usize>>,
    /// `pair_adj[a]` is the sorted list of neighbor modules that form a
    /// canonical pair `(min(a,b), max(a,b))`. Stored under BOTH endpoints so
    /// the per-edge intersection walk can start from either side; a
    /// canonical-order guard (`a ≤ b`) suppresses double-visits.
    pub pair_adj: Vec<Vec<PairAdjEntry>>,
    pub n_pairs: usize,
}

impl ModulePairBasis {
    /// Build the basis from the feature network + per-feature module labels.
    ///
    /// Features with `module_of_feature[g] == None` contribute nothing. Feature-feature
    /// edges with both endpoints in some module accumulate to `B[a,b]`; the
    /// resulting module degrees seed the modularity null.
    pub fn build(graph: &FeaturePairGraph, module_of_feature: Vec<Option<usize>>) -> Self {
        let n_modules = module_of_feature
            .iter()
            .filter_map(|m| *m)
            .max()
            .map_or(0, |m| m + 1);

        // Canonical module-pair weights via sorted (a, b).
        let mut pair_weights: HashMap<(u32, u32), f64> = Default::default();
        let mut deg = vec![0.0f64; n_modules];
        for &(u, v) in &graph.feature_edges {
            let (Some(mu), Some(mv)) = (module_of_feature[u], module_of_feature[v]) else {
                continue;
            };
            let (a, b) = if mu <= mv {
                (mu as u32, mv as u32)
            } else {
                (mv as u32, mu as u32)
            };
            *pair_weights.entry((a, b)).or_insert(0.0) += 1.0;
            // Each undirected feature-feature edge contributes 1 to each endpoint's module degree.
            deg[mu] += 1.0;
            deg[mv] += 1.0;
        }
        let two_w: f64 = deg.iter().sum();
        let denom = two_w * two_w;

        // Assign contiguous pair indices in a deterministic order.
        let mut kept: Vec<((u32, u32), f64)> =
            pair_weights.into_iter().filter(|&(_, w)| w > 0.0).collect();
        kept.sort_by_key(|&((a, b), _)| (a, b));

        let mut pair_adj: Vec<Vec<PairAdjEntry>> = vec![Vec::new(); n_modules];
        for (pair_idx, &((a, b), _w)) in kept.iter().enumerate() {
            let null_ab = if denom > 0.0 {
                (deg[a as usize] * deg[b as usize] / denom) as f32
            } else {
                0.0
            };
            let pair_idx = pair_idx as u32;
            pair_adj[a as usize].push(PairAdjEntry {
                b,
                pair_idx,
                null_ab,
            });
            if a != b {
                pair_adj[b as usize].push(PairAdjEntry {
                    b: a,
                    pair_idx,
                    null_ab,
                });
            }
        }
        for adj in pair_adj.iter_mut() {
            adj.sort_by_key(|e| e.b);
        }

        let n_pairs = kept.len();
        info!(
            "ModulePairBasis: {} modules, {} retained pairs, 2W={:.1}",
            n_modules, n_pairs, two_w
        );

        ModulePairBasis {
            n_modules,
            module_of_feature,
            pair_adj,
            n_pairs,
        }
    }
}

/// Pre-collapse per-cell feature expression into per-cell module expression.
///
/// Returns `(module_expr, cell_totals)` where:
///   - `module_expr` is `n_modules × n_cells` dense (column-major): each
///     column is `x_{c,m} = Σ_{g ∈ m} x_{c,g}` for cell `c`. Modules with
///     no surviving features stay zero.
///   - `cell_totals[c] = Σ_m x_{c,m}` — the per-cell total used as the null
///     scale in the residual.
///
/// One streaming pass over the sparse expression matrix.
pub fn build_module_expression(
    data: &SparseIoVec,
    module_of_feature: &[Option<usize>],
    n_modules: usize,
    feature_weights: Option<&[f32]>,
    block_size: Option<usize>,
) -> anyhow::Result<(Mat, Vec<f32>)> {
    let n_cells = data.num_columns();
    let n_features = data.num_rows();
    debug_assert_eq!(module_of_feature.len(), n_features);
    if let Some(w) = feature_weights {
        debug_assert_eq!(w.len(), n_features);
    }

    // Dense column-major: rows = modules, columns = cells. Small compared
    // to the raw matrix (typical n_modules is 10² range).
    let jobs = generate_minibatch_intervals(n_cells, n_features, block_size);
    let prog_bar = new_progress_bar(jobs.len() as u64).with_message("module-expression blocks");

    let partials: Vec<(usize, Mat, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(prog_bar.clone())
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Mat, Vec<f32>)> {
            let x = data.read_columns_csc(lb..ub)?;
            let block_len = ub - lb;
            let mut block_expr = Mat::zeros(n_modules, block_len);
            let mut block_totals = vec![0.0f32; block_len];
            for col in 0..block_len {
                let s = x.col(col);
                for (&row, &val) in s.row_indices().iter().zip(s.values().iter()) {
                    if let Some(m) = module_of_feature[row] {
                        let v = match feature_weights {
                            Some(w) => val * w[row],
                            None => val,
                        };
                        block_expr[(m, col)] += v;
                        block_totals[col] += v;
                    }
                }
            }
            Ok((lb, block_expr, block_totals))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    prog_bar.finish_and_clear();

    let mut module_expr = Mat::zeros(n_modules, n_cells);
    let mut cell_totals = vec![0.0f32; n_cells];
    for (lb, block_expr, block_totals) in partials {
        let block_len = block_expr.ncols();
        for col in 0..block_len {
            module_expr
                .column_mut(lb + col)
                .copy_from(&block_expr.column(col));
            cell_totals[lb + col] = block_totals[col];
        }
    }
    Ok((module_expr, cell_totals))
}

/// Aggregate fine-cell module expression to pb-sample module expression.
///
/// For each fine cell `c` with pb-sample label `cell_labels[c] = sc`:
///   `super_expr[m, sc] += module_expr[m, c]`
/// Also returns per-pb-sample totals.
pub fn coarsen_module_expression(
    module_expr: &Mat,
    cell_labels: &[usize],
    n_pb_samples: usize,
) -> (Mat, Vec<f32>) {
    let n_modules = module_expr.nrows();
    let n_cells = module_expr.ncols();
    debug_assert_eq!(cell_labels.len(), n_cells);

    let mut super_expr = Mat::zeros(n_modules, n_pb_samples);
    let mut super_totals = vec![0.0f32; n_pb_samples];
    for c in 0..n_cells {
        let sc = cell_labels[c];
        for m in 0..n_modules {
            let v = module_expr[(m, c)];
            if v != 0.0 {
                super_expr[(m, sc)] += v;
                super_totals[sc] += v;
            }
        }
    }
    (super_expr, super_totals)
}

/// Build sparse module-pair profiles for a subset of edges.
///
/// For each edge `e = (i, j) = all_edges[edge_indices[e_idx]]` and canonical
/// module pair `(a, b)` with `a ≤ b`, emits
///
///   y = max(0, x_{i,a}·x_{j,b} + x_{i,b}·x_{j,a}
///              − X_i·X_j · deg(a)·deg(b)/(2W)²)
///
/// (with `a == b` using `x_{i,a}·x_{j,a}` to avoid double-counting).
///
/// A pair `(a, b)` can only produce a positive residual when its smaller
/// endpoint `a` is active (non-zero module expression) in at least one of
/// the two cells — otherwise `x_{i,a} = x_{j,a} = 0` forces `y_obs = 0`.
/// So the outer loop merges `A_i ∪ A_j` (sorted active-module lists per
/// cell) and walks `pair_adj[a]` only for `a` in the union, skipping
/// non-canonical entries via the `a ≤ b` guard.
pub fn build_module_pair_profiles_for_edges(
    module_expr: &Mat,
    cell_totals: &[f32],
    all_edges: &[(usize, usize)],
    edge_indices: &[usize],
    basis: &ModulePairBasis,
) -> LinkProfileStore {
    let n_modules = module_expr.nrows();
    let n_cells = module_expr.ncols();
    debug_assert_eq!(basis.n_modules, n_modules);

    // Per-cell sorted list of active (non-zero) module indices. One upfront
    // pass replaces the per-edge O(n_modules) sweep.
    let active_per_cell: Vec<Vec<u32>> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let col = module_expr.column(c);
            (0..n_modules)
                .filter(|&m| col[m] != 0.0)
                .map(|m| m as u32)
                .collect()
        })
        .collect();

    let rows: Vec<Vec<(u32, f32)>> = edge_indices
        .par_iter()
        .map(|&e| {
            let (i, j) = all_edges[e];
            let mass = cell_totals[i] as f64 * cell_totals[j] as f64;
            let a_i = &active_per_cell[i];
            let a_j = &active_per_cell[j];
            let mut out: Vec<(u32, f32)> = Vec::new();

            // Sorted-merge walk of A_i ∪ A_j with dedup (equal indices
            // advance both cursors so each `a` is visited once).
            let (mut pi, mut pj) = (0usize, 0usize);
            while pi < a_i.len() || pj < a_j.len() {
                let a = match (a_i.get(pi), a_j.get(pj)) {
                    (Some(&ai), Some(&aj)) if ai < aj => {
                        pi += 1;
                        ai
                    }
                    (Some(&ai), Some(&aj)) if ai > aj => {
                        pj += 1;
                        aj
                    }
                    (Some(&ai), Some(_)) => {
                        pi += 1;
                        pj += 1;
                        ai
                    }
                    (Some(&ai), None) => {
                        pi += 1;
                        ai
                    }
                    (None, Some(&aj)) => {
                        pj += 1;
                        aj
                    }
                    (None, None) => break,
                };

                let au = a as usize;
                let xi_a = module_expr[(au, i)] as f64;
                let xj_a = module_expr[(au, j)] as f64;

                for &entry in &basis.pair_adj[au] {
                    // Canonical guard: visit each (a, b) with a ≤ b once,
                    // from the smaller endpoint. Pairs where b < a will be
                    // visited (if at all) when we reach `b` in the union.
                    if a > entry.b {
                        continue;
                    }
                    let bu = entry.b as usize;
                    let y_obs = if bu == au {
                        xi_a * xj_a
                    } else {
                        let xi_b = module_expr[(bu, i)] as f64;
                        let xj_b = module_expr[(bu, j)] as f64;
                        xi_a * xj_b + xi_b * xj_a
                    };
                    let y = (y_obs - mass * entry.null_ab as f64).max(0.0) as f32;
                    if y > 0.0 {
                        out.push((entry.pair_idx, y));
                    }
                }
            }

            out
        })
        .collect();

    LinkProfileStore::from_sparse_rows(rows, basis.n_pairs)
}
