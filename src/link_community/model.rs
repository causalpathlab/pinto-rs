//! Link community model: sufficient statistics and the collapsed conditional.
//!
//! Each edge e has a profile vector y_e ∈ R^M (sparsely stored as non-zero
//! `(col, val)` pairs) and a community assignment z_e ∈ {0..K-1}. Let
//!   D_{kg} = Σ_{e: z_e=k} y_e^g    (edge-weighted feature degree in community k)
//!   V_k    = Σ_g D_{kg}             (community volume)
//!
//! With the rates integrated out under a symmetric Dirichlet(ε), the labels have
//! the joint [`LinkCommunityStats::log_marginal`], and with `⁻ᵉ` leaving edge e
//! out of the sums and `n_k` the edges in community k, its conditional is
//!
//!   ln p(z_e = k | rest) = Σ_{g: y_eg > 0} [ln Γ(D⁻ᵉ_{kg} + y_eg + ε) − ln Γ(D⁻ᵉ_{kg} + ε)]
//!                          − [ln Γ(V⁻ᵉ_k + s_e + Mε) − ln Γ(V⁻ᵉ_k + Mε)]
//!                          + ln(n⁻ᵉ_k + α/K) + const
//!
//! The plug-in score
//!   ℓ = Σ_kg f(D_{kg}) − Σ_k f(V_k),   f(x) = x · ln x
//! (equivalently −Σ_k V_k · H(p_k), where p_k = D_{k·}/V_k) is kept as a report.
//!
//! Profiles are stored in CSR: for each edge, `(indptr[e]..indptr[e+1])`
//! slices `indices` and `values` hold the non-zero entries. The hot Gibbs
//! loop iterates these pairs directly, avoiding zero-skip branches.

use crate::link_community::fast_math::{fast_ln, ln_gamma_increment, ln_gamma_increments};
use special::Gamma as SpecialGamma;

/// Prior pseudo-count `ε` of the symmetric Dirichlet on a community's feature
/// rates. At one, `ln Γ(y + ε) − ln Γ(ε)` goes to zero with `y`, so a vanishing
/// count says next to nothing; a tiny `ε` charges it `−ln ε` however small it is.
pub(crate) const PRIOR_COUNT: f64 = 1.0;

/// f(x) = x · ln(x), with f(0) = 0 by convention.
#[inline]
pub(crate) fn f_entropy(x: f64) -> f64 {
    if x > 0.0 {
        x * x.ln()
    } else {
        0.0
    }
}

/// CSR-backed per-edge profile store.
///
/// Row `e` has non-zero column indices in
/// `indices[indptr[e]..indptr[e+1]]` and matching values in
/// `values[indptr[e]..indptr[e+1]]`. Indices within a row are sorted
/// ascending (deterministic iteration) and unique.
pub struct LinkProfileStore {
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    /// Size factor per edge: Σ of non-zero values in that row.
    pub size_factors: Vec<f32>,
    pub n_edges: usize,
    pub m: usize,
}

impl LinkProfileStore {
    /// Build from per-edge lists of `(column, value)` pairs. Duplicate
    /// columns in a row are summed; zero values are dropped.
    pub fn from_sparse_rows(mut rows: Vec<Vec<(u32, f32)>>, m: usize) -> Self {
        let n_edges = rows.len();
        let mut indptr = Vec::with_capacity(n_edges + 1);
        indptr.push(0usize);

        let mut nnz_est = 0usize;
        for row in &rows {
            nnz_est += row.len();
        }
        let mut indices: Vec<u32> = Vec::with_capacity(nnz_est);
        let mut values: Vec<f32> = Vec::with_capacity(nnz_est);
        let mut size_factors: Vec<f32> = Vec::with_capacity(n_edges);

        for row in rows.iter_mut() {
            row.sort_by_key(|&(c, _)| c);
            // Merge duplicates and drop zeros.
            let mut sf = 0.0f32;
            let mut i = 0usize;
            while i < row.len() {
                let col = row[i].0;
                let mut val = row[i].1;
                let mut j = i + 1;
                while j < row.len() && row[j].0 == col {
                    val += row[j].1;
                    j += 1;
                }
                if val != 0.0 {
                    indices.push(col);
                    values.push(val);
                    sf += val;
                }
                i = j;
            }
            size_factors.push(sf);
            indptr.push(indices.len());
        }

        LinkProfileStore {
            indptr,
            indices,
            values,
            size_factors,
            n_edges,
            m,
        }
    }

    /// Build from a flat row-major `[n_edges × m]` dense buffer. Non-zero
    /// entries are converted to CSR (projection mode + tests).
    pub fn new(dense: Vec<f32>, n_edges: usize, m: usize) -> Self {
        debug_assert_eq!(dense.len(), n_edges * m);
        let mut rows: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n_edges);
        for e in 0..n_edges {
            let base = e * m;
            let mut row = Vec::new();
            for g in 0..m {
                let v = dense[base + g];
                if v != 0.0 {
                    row.push((g as u32, v));
                }
            }
            rows.push(row);
        }
        Self::from_sparse_rows(rows, m)
    }

    /// Sparse row slice for edge `e`: `(columns, values)`.
    #[inline]
    pub fn row(&self, e: usize) -> (&[u32], &[f32]) {
        let start = self.indptr[e];
        let end = self.indptr[e + 1];
        (&self.indices[start..end], &self.values[start..end])
    }

    /// Total non-zero entries across all rows.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Extract a sub-store for the given edge indices.
    pub fn subset(&self, edge_indices: &[usize]) -> Self {
        let n = edge_indices.len();
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0usize);

        let mut nnz_est = 0usize;
        for &e in edge_indices {
            nnz_est += self.indptr[e + 1] - self.indptr[e];
        }
        let mut indices = Vec::with_capacity(nnz_est);
        let mut values = Vec::with_capacity(nnz_est);
        let mut size_factors = Vec::with_capacity(n);

        for &e in edge_indices {
            let (cols, vals) = self.row(e);
            indices.extend_from_slice(cols);
            values.extend_from_slice(vals);
            size_factors.push(self.size_factors[e]);
            indptr.push(indices.len());
        }

        LinkProfileStore {
            indptr,
            indices,
            values,
            size_factors,
            n_edges: n,
            m: self.m,
        }
    }
}

/// Sufficient statistics for the link community model.
///
/// All accumulators are f64 to prevent drift during incremental updates.
pub struct LinkCommunityStats {
    pub k: usize,
    pub m: usize,
    /// Edge-weighted feature degree D_{kg}: `feature_sum[k*m + g] = Σ_{e: z_e=k} y_e^g`.
    pub feature_sum: Vec<f64>,
    /// Community volume V_k (size-factor sum): `size_sum[k] = Σ_{e: z_e=k} s_e`.
    pub size_sum: Vec<f64>,
    /// Per-community edge count.
    pub edge_count: Vec<usize>,
    pub membership: Vec<usize>,
    /// `feature_sum` plus the prior count, in `f32` and feature-major
    /// (`prior_x[g*k + k']`): what scoring reads, one short run per feature.
    pub(crate) prior_x: Vec<f32>,
}

impl LinkCommunityStats {
    /// Build sufficient statistics from profiles and initial labels.
    pub fn from_profiles(profiles: &LinkProfileStore, k: usize, labels: &[usize]) -> Self {
        debug_assert_eq!(labels.len(), profiles.n_edges);
        let (feature_sum, size_sum, edge_count) = Self::local_stats(profiles, k, labels);
        let m = profiles.m;
        let mut stats = LinkCommunityStats {
            k,
            m,
            feature_sum,
            size_sum,
            edge_count,
            membership: labels.to_vec(),
            prior_x: vec![0.0; k * m],
        };
        for t in 0..k {
            for g in 0..m {
                stats.refresh_prior_x(t, g);
            }
        }
        stats
    }

    /// Bring one cell of `prior_x` up to `feature_sum`.
    #[inline]
    fn refresh_prior_x(&mut self, t: usize, g: usize) {
        self.prior_x[g * self.k + t] = (self.feature_sum[t * self.m + g] + PRIOR_COUNT) as f32;
    }

    /// Move edge `e` from `old_k` to `new_k`, updating stats incrementally
    /// in O(nnz(y_e)).
    #[inline]
    pub fn delta_move(
        &mut self,
        e: usize,
        old_k: usize,
        new_k: usize,
        profiles: &LinkProfileStore,
    ) {
        debug_assert_eq!(self.membership[e], old_k);
        let m = self.m;
        let (cols, vals) = profiles.row(e);
        let sf = profiles.size_factors[e] as f64;

        for (&col, &y) in cols.iter().zip(vals.iter()) {
            let g = col as usize;
            for (t, delta) in [(old_k, -f64::from(y)), (new_k, f64::from(y))] {
                self.feature_sum[t * m + g] += delta;
                self.refresh_prior_x(t, g);
            }
        }

        self.size_sum[old_k] -= sf;
        self.size_sum[new_k] += sf;
        self.edge_count[old_k] -= 1;
        self.edge_count[new_k] += 1;
        self.membership[e] = new_k;
    }

    /// Aggregate stats for a subset of edges (identified by global indices).
    pub fn component_stats(
        profiles: &LinkProfileStore,
        k: usize,
        edge_indices: &[usize],
        membership: &[usize],
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let m = profiles.m;
        let mut feature_sum = vec![0.0f64; k * m];
        let mut size_sum = vec![0.0f64; k];
        let mut edge_count = vec![0usize; k];

        for &e in edge_indices {
            let c = membership[e];
            debug_assert!(c < k);
            let (cols, vals) = profiles.row(e);
            let base = c * m;
            for (&col, &v) in cols.iter().zip(vals.iter()) {
                feature_sum[base + col as usize] += v as f64;
            }
            size_sum[c] += profiles.size_factors[e] as f64;
            edge_count[c] += 1;
        }

        (feature_sum, size_sum, edge_count)
    }

    /// Aggregate stats for a contiguous sub-store with local membership.
    pub fn local_stats(
        profiles: &LinkProfileStore,
        k: usize,
        membership: &[usize],
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let m = profiles.m;
        let mut feature_sum = vec![0.0f64; k * m];
        let mut size_sum = vec![0.0f64; k];
        let mut edge_count = vec![0usize; k];

        for (e, &c) in membership.iter().enumerate() {
            debug_assert!(c < k);
            let (cols, vals) = profiles.row(e);
            let base = c * m;
            for (&col, &v) in cols.iter().zip(vals.iter()) {
                feature_sum[base + col as usize] += v as f64;
            }
            size_sum[c] += profiles.size_factors[e] as f64;
            edge_count[c] += 1;
        }

        (feature_sum, size_sum, edge_count)
    }

    /// Apply a delta to the sufficient statistics: self += (new - old).
    pub fn apply_delta(
        &mut self,
        old: (&[f64], &[f64], &[usize]),
        new: (&[f64], &[f64], &[usize]),
    ) {
        for (cell, (n, o)) in new.0.iter().zip(old.0).enumerate() {
            if n != o {
                self.feature_sum[cell] += n - o;
                self.refresh_prior_x(cell / self.m, cell % self.m);
            }
        }
        for c in 0..self.k {
            self.size_sum[c] += new.1[c] - old.1[c];
            self.edge_count[c] =
                (self.edge_count[c] as isize + new.2[c] as isize - old.2[c] as isize) as usize;
        }
    }

    /// Log joint of the labels and the profiles with every rate integrated out,
    /// up to terms free of the labels. What the collapsed conditional samples from:
    ///
    ///   Σ_k [ Σ_g ln Γ(D_kg + ε) − ln Γ(V_k + Mε) ] + Σ_k ln Γ(n_k + α/K)
    ///
    /// each term less its value at zero count, the second sum only when `alpha > 0`.
    /// Each `D_kg` reads as a count of a Dirichlet-multinomial; `n_k` is the
    /// community's edges.
    pub fn log_marginal(&self, alpha: f64) -> f64 {
        let lg = |x: f64| SpecialGamma::ln_gamma(x).0;
        let m_eps = self.m as f64 * PRIOR_COUNT;
        let features: f64 = self
            .feature_sum
            .iter()
            .map(|&d| lg(d + PRIOR_COUNT) - lg(PRIOR_COUNT))
            .sum();
        let sizes: f64 = self
            .size_sum
            .iter()
            .map(|&v| lg(v + m_eps) - lg(m_eps))
            .sum();
        let prior: f64 = if alpha > 0.0 {
            let a = alpha / self.k as f64;
            self.edge_count
                .iter()
                .map(|&n| lg(n as f64 + a) - lg(a))
                .sum()
        } else {
            0.0
        };
        features - sizes + prior
    }

    /// Plug-in Poisson DC-SBM log-likelihood Σ_kg f(D_kg) − Σ_k f(V_k),
    /// f(x)=x·ln x. Equivalently −Σ_k V_k · H(p_k). Higher = better.
    pub fn total_score(&self) -> f64 {
        let sum_f_d: f64 = self.feature_sum.iter().map(|&d| f_entropy(d)).sum();
        let sum_f_v: f64 = self.size_sum.iter().map(|&v| f_entropy(v)).sum();
        sum_f_d - sum_f_v
    }

    /// Mutual information between community assignment and feature profile, in nats.
    pub fn mutual_information(&self) -> f64 {
        self.score_and_mi().1
    }

    /// Fused `(total_score, mutual_information)` in one pass.
    pub fn score_and_mi(&self) -> (f64, f64) {
        let mut p_global = vec![0.0f64; self.m];
        let mut sum_f_d = 0.0f64;
        for chunk in self.feature_sum.chunks_exact(self.m) {
            for (acc, &d) in p_global.iter_mut().zip(chunk.iter()) {
                *acc += d;
                sum_f_d += f_entropy(d);
            }
        }
        let mut total = 0.0f64;
        let mut sum_f_v = 0.0f64;
        for &v in &self.size_sum {
            total += v;
            sum_f_v += f_entropy(v);
        }
        let score = sum_f_d - sum_f_v;
        if total <= 0.0 {
            return (score, 0.0);
        }
        let inv_total = 1.0 / total;
        let h_global: f64 = p_global
            .iter()
            .map(|&d| {
                let p = d * inv_total;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();
        (score, h_global + score * inv_total)
    }
}

/// Read-only context for the (frozen, VB) incidence term in per-edge scores.
///
/// `edges` is indexed the same way as the `LinkProfileStore` rows passed
/// alongside (so for a per-component sub-store, `edges` is the per-component
/// `Vec<(usize, usize)>` of cell endpoints, not the global edge list).
/// `cell_scores` is the frozen `[n_cells × k]` table of
/// `incidence::cell_incidence_scores`.
#[derive(Copy, Clone)]
pub(crate) struct IncidenceCtx<'a> {
    pub edges: &'a [(usize, usize)],
    pub cell_scores: &'a [f32],
}

/// The collapsed conditional of edge `e` over the communities (see the module
/// docs), its `ln Γ` increments as approximate as [`ln_gamma_increment`] is.
///
/// `log_probs[t] = ln p(z_e = t | rest) − ln p(z_e = current | rest)`, so the
/// current community's slot is 0. `alpha > 0` folds in the collapsed Dirichlet
/// prior on the community sizes, and `alpha == 0` leaves it out. When
/// `incidence_ctx` is `Some`, the frozen VB incidence term is added last.
/// `increments` is scratch of length `k`, the caller's so that no edge allocates.
pub(crate) fn compute_log_probs_for_edge(
    e: usize,
    stats: &LinkCommunityStats,
    profiles: &LinkProfileStore,
    alpha: f64,
    incidence_ctx: Option<IncidenceCtx<'_>>,
    log_probs: &mut [f64],
    increments: &mut [f32],
) {
    let k = stats.k;
    let m = stats.m;
    let current_c = stats.membership[e];
    let (cols, vals) = profiles.row(e);
    let sf = profiles.size_factors[e];
    let m_eps = m as f64 * PRIOR_COUNT;
    let prior_a = alpha / k as f64;
    // The size and prior terms, given the volume and edges without `e`.
    let tail = |volume: f64, count: f64| {
        let mut score = -f64::from(ln_gamma_increment((volume + m_eps) as f32, sf));
        if alpha > 0.0 {
            score += f64::from(fast_ln((count + prior_a) as f32));
        }
        score
    };

    // Every community as if it did not hold the edge, one feature at a time.
    for (t, lp) in log_probs.iter_mut().enumerate() {
        *lp = tail(stats.size_sum[t], stats.edge_count[t] as f64);
    }
    for (&col, &y) in cols.iter().zip(vals.iter()) {
        let g = col as usize;
        let xs = &stats.prior_x[g * k..(g + 1) * k];
        ln_gamma_increments(xs, y, increments);
        for (lp, &inc) in log_probs.iter_mut().zip(increments.iter()) {
            *lp += f64::from(inc);
        }
    }

    // The current community does hold it, so its score leaves the edge out.
    let features = &stats.feature_sum[current_c * m..(current_c + 1) * m];
    let mut at_current = tail(
        stats.size_sum[current_c] - f64::from(sf),
        stats.edge_count[current_c] as f64 - 1.0,
    );
    for (&col, &y) in cols.iter().zip(vals.iter()) {
        let rest = features[col as usize] - f64::from(y);
        at_current += f64::from(ln_gamma_increment((rest + PRIOR_COUNT) as f32, y));
    }
    for lp in log_probs.iter_mut() {
        *lp -= at_current;
    }
    log_probs[current_c] = 0.0;

    if let Some(ctx) = incidence_ctx {
        crate::link_community::incidence::add_incidence_to_log_probs(
            e,
            ctx.edges,
            ctx.cell_scores,
            k,
            current_c,
            log_probs,
        );
    }
}
