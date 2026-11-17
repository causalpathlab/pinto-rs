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
//! With a stage-1 divisor (`--offset`, from `pinto apply residual --residual
//! divide`) each edge also carries its expected profile `μ_e` under the
//! stage-1 model, dense over the `M` dimensions
//! ([`LinkProfileStore::offset`]), and the profile reads as
//! `y_eg ~ Poisson(λ_kg μ_eg)`, `λ_kg ~ Gamma(ε, b)`. With
//! `M_{kg} = Σ_{e: z_e=k} μ_eg` updated on each move as `D` is, the rates
//! integrate out to
//!
//!   ln p(y, z) = Σ_kg [ln Γ(D_kg + ε) − ln Γ(ε) + ε ln b − (D_kg + ε) ln(M_kg + b)]
//!                + Σ_k ln Γ(n_k + α/K) + const,
//!
//! so the conditional of edge e is
//!
//!   Σ_{g: y_eg > 0} [ln Γ(D⁻ᵉ_kg + y_eg + ε) − ln Γ(D⁻ᵉ_kg + ε)]
//!   + Σ_g [(D⁻ᵉ_kg + ε) ln(M⁻ᵉ_kg + b) − (D⁻ᵉ_kg + y_eg + ε) ln(M⁻ᵉ_kg + μ_eg + b)]
//!   + ln(n⁻ᵉ_k + α/K) + const:
//!
//! λ̂_kg = D_kg / M_kg in place of `D_kg / V_k`, the rate relative to the
//! stage-1 expectation. The second sum runs over every dimension, `μ_e`
//! being dense; the plug-in score becomes `Σ_kg [f(D_kg) − D_kg ln M_kg]`.
//!
//! Profiles are stored in CSR: for each edge, `(indptr[e]..indptr[e+1])`
//! slices `indices` and `values` hold the non-zero entries. The hot Gibbs
//! loop iterates these pairs directly, avoiding zero-skip branches.

use crate::link_community::fast_math::{fast_ln, ln_gamma_increment, ln_gamma_increments, LnTable};
use special::Gamma as SpecialGamma;

/// Prior pseudo-count `ε` of the symmetric Dirichlet on a community's feature
/// rates. At one, `ln Γ(y + ε) − ln Γ(ε)` goes to zero with `y`, so a vanishing
/// count says next to nothing; a tiny `ε` charges it `−ln ε` however small it is.
pub(crate) const PRIOR_COUNT: f64 = 1.0;

/// Prior rate `b` of the Gamma on a rate relative to the stage-1 offset
/// (see the module docs): one expected count's worth, so with `ε` = 1 the
/// prior mean of `λ_kg` is 1, the stage-1 expectation itself.
pub(crate) const PRIOR_RATE: f64 = 1.0;

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
    /// Under a stage-1 divisor, each edge's expected profile `μ_e`,
    /// row-major `[n_edges × m]`, every entry positive; `None` without one.
    pub offset: Option<Vec<f32>>,
}

impl LinkProfileStore {
    /// The same store with each edge's expected profile `offset`
    /// (`[n_edges × m]`, row-major, positive).
    pub fn with_offset(mut self, offset: Vec<f32>) -> Self {
        assert_eq!(
            offset.len(),
            self.n_edges * self.m,
            "one offset row of m per edge"
        );
        debug_assert!(offset.iter().all(|&x| x > 0.0));
        self.offset = Some(offset);
        self
    }

    /// Edge `e`'s expected profile, when the store has one.
    #[inline]
    pub fn offset_row(&self, e: usize) -> Option<&[f32]> {
        self.offset
            .as_deref()
            .map(|o| &o[e * self.m..(e + 1) * self.m])
    }

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
            offset: None,
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
        let offset = self.offset.as_ref().map(|_| {
            edge_indices
                .iter()
                .flat_map(|&e| self.offset_row(e).expect("an offset row").iter().copied())
                .collect()
        });

        LinkProfileStore {
            indptr,
            indices,
            values,
            size_factors,
            n_edges: n,
            m: self.m,
            offset,
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
    /// Under a stage-1 divisor, the expected degree
    /// `M_{kg} = Σ_{e: z_e=k} μ_eg`, laid out as `feature_sum`; empty
    /// without one.
    pub offset_sum: Vec<f64>,
    /// `1 / (M_kg + b)` and `ln(M_kg + b)`, laid out as `offset_sum` and
    /// kept with it: what the offset score reads for every community but
    /// the edge's own, so an edge costs no `ln` per community and feature.
    pub(crate) offset_inv: Vec<f32>,
    pub(crate) offset_ln: Vec<f32>,
    /// `feature_sum` plus the prior count, feature-major (`g*k + k'`): what scoring
    /// reads, one short run per feature.
    pub(crate) feature_table: LnTable,
}

impl LinkCommunityStats {
    /// Build sufficient statistics from profiles and initial labels.
    pub fn from_profiles(profiles: &LinkProfileStore, k: usize, labels: &[usize]) -> Self {
        debug_assert_eq!(labels.len(), profiles.n_edges);
        let (feature_sum, size_sum, edge_count) = Self::local_stats(profiles, k, labels);
        let m = profiles.m;
        let offset_sum = Self::local_offset(profiles, k, labels);
        let mut stats = LinkCommunityStats {
            k,
            m,
            feature_sum,
            size_sum,
            edge_count,
            membership: labels.to_vec(),
            offset_sum,
            offset_inv: Vec::new(),
            offset_ln: Vec::new(),
            feature_table: LnTable::new(k * m),
        };
        for t in 0..k {
            for g in 0..m {
                stats.refresh_cell(t, g);
            }
        }
        stats.rebuild_offset_cache();
        stats
    }

    /// `offset_inv` and `offset_ln` from `offset_sum`, every cell.
    pub(crate) fn rebuild_offset_cache(&mut self) {
        let n = self.offset_sum.len();
        self.offset_inv.resize(n, 0.0);
        self.offset_ln.resize(n, 0.0);
        for i in 0..n {
            self.refresh_offset(i);
        }
    }

    /// One cell of the offset cache brought up to `offset_sum`.
    #[inline]
    fn refresh_offset(&mut self, i: usize) {
        let x = self.offset_sum[i] + PRIOR_RATE;
        self.offset_inv[i] = (1.0 / x) as f32;
        self.offset_ln[i] = x.ln() as f32;
    }

    /// Bring one cell of the feature table up to `feature_sum`.
    #[inline]
    fn refresh_cell(&mut self, t: usize, g: usize) {
        let x = (self.feature_sum[t * self.m + g] + PRIOR_COUNT) as f32;
        self.feature_table.set(g * self.k + t, x);
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
                self.refresh_cell(t, g);
            }
        }

        if let Some(mu) = profiles.offset_row(e) {
            for (g, &x) in mu.iter().enumerate() {
                self.offset_sum[old_k * m + g] -= f64::from(x);
                self.offset_sum[new_k * m + g] += f64::from(x);
                self.refresh_offset(old_k * m + g);
                self.refresh_offset(new_k * m + g);
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

    /// The expected degree `M_{kg}` (`[k × m]`) of the edges `edge_indices`
    /// (global) under `membership`; empty when the store has no offset.
    pub fn component_offset(
        profiles: &LinkProfileStore,
        k: usize,
        edge_indices: &[usize],
        membership: &[usize],
    ) -> Vec<f64> {
        Self::offset_of(
            profiles,
            k,
            edge_indices.iter().map(|&e| (e, membership[e])),
        )
    }

    /// [`Self::component_offset`] of a sub-store with local membership.
    pub fn local_offset(profiles: &LinkProfileStore, k: usize, membership: &[usize]) -> Vec<f64> {
        Self::offset_of(profiles, k, membership.iter().copied().enumerate())
    }

    fn offset_of(
        profiles: &LinkProfileStore,
        k: usize,
        edges: impl Iterator<Item = (usize, usize)>,
    ) -> Vec<f64> {
        if profiles.offset.is_none() {
            return Vec::new();
        }
        let m = profiles.m;
        let mut sum = vec![0.0f64; k * m];
        for (e, c) in edges {
            let mu = profiles.offset_row(e).expect("an offset row");
            for (acc, &x) in sum[c * m..(c + 1) * m].iter_mut().zip(mu) {
                *acc += f64::from(x);
            }
        }
        sum
    }

    /// `self.offset_sum += new − old`, as [`Self::apply_delta`] patches the
    /// other sums; nothing without an offset.
    pub fn apply_offset_delta(&mut self, old: &[f64], new: &[f64]) {
        for ((acc, n), o) in self.offset_sum.iter_mut().zip(new).zip(old) {
            *acc += n - o;
        }
        self.rebuild_offset_cache();
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
                self.refresh_cell(cell / self.m, cell % self.m);
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
    ///
    /// Under a stage-1 divisor the size term is the offset's (module docs):
    /// `Σ_kg [(D_kg + ε) ln(M_kg + b) − ε ln b]` in place of `Σ_k ln Γ(V_k + Mε)`.
    pub fn log_marginal(&self, alpha: f64) -> f64 {
        let lg = |x: f64| SpecialGamma::ln_gamma(x).0;
        let m_eps = self.m as f64 * PRIOR_COUNT;
        let features: f64 = self
            .feature_sum
            .iter()
            .map(|&d| lg(d + PRIOR_COUNT) - lg(PRIOR_COUNT))
            .sum();
        let sizes: f64 = if self.offset_sum.is_empty() {
            self.size_sum
                .iter()
                .map(|&v| lg(v + m_eps) - lg(m_eps))
                .sum()
        } else {
            self.feature_sum
                .iter()
                .zip(&self.offset_sum)
                .map(|(&d, &mu)| {
                    (d + PRIOR_COUNT) * (mu + PRIOR_RATE).ln() - PRIOR_COUNT * PRIOR_RATE.ln()
                })
                .sum()
        };
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
    ///
    /// Under a stage-1 divisor, `Σ_kg [f(D_kg) − D_kg ln M_kg]`: the rates
    /// are `D_kg / M_kg`, relative to the stage-1 expectation.
    pub fn total_score(&self) -> f64 {
        let sum_f_d: f64 = self.feature_sum.iter().map(|&d| f_entropy(d)).sum();
        sum_f_d - self.volume_term()
    }

    /// What [`Self::total_score`] takes off `Σ f(D)`: `Σ_k f(V_k)`, or
    /// under a stage-1 divisor `Σ_kg D_kg ln M_kg`.
    fn volume_term(&self) -> f64 {
        if self.offset_sum.is_empty() {
            return self.size_sum.iter().map(|&v| f_entropy(v)).sum();
        }
        self.feature_sum
            .iter()
            .zip(&self.offset_sum)
            .map(|(&d, &mu)| {
                if d > 0.0 && mu > 0.0 {
                    d * mu.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Mutual information between community assignment and feature profile, in nats.
    pub fn mutual_information(&self) -> f64 {
        self.score_and_mi().1
    }

    /// Fused `(total_score, mutual_information)` in one pass. The mutual
    /// information is the labels' with the profiles as they are, an offset
    /// or not; the score is [`Self::total_score`].
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
        let plain = sum_f_d - sum_f_v;
        let score = if self.offset_sum.is_empty() {
            plain
        } else {
            sum_f_d - self.volume_term()
        };
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
        (score, h_global + plain * inv_total)
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

    // Under a stage-1 divisor the size term is the offset's (module docs):
    // community `t` without the edge scores
    // `Σ_g (D+ε)(ln M − ln(M+μ)) − Σ_g y ln(M+μ)` with `M` its offset sum
    // plus the prior rate. The first sum is dense, so it reads the cached
    // `1/M` as `−(D+ε) ln1p(μ/M)`; the second runs over the edge's nonzeros,
    // `ln(M+μ) = ln M + ln1p(μ/M)`.
    let offset = profiles.offset_row(e);
    let prior = |count: f64| {
        if alpha > 0.0 {
            f64::from(fast_ln((count + prior_a) as f32))
        } else {
            0.0
        }
    };
    let offset_tail = |mu: &[f32], t: usize| {
        let at = t * m;
        let (inv, lnm) = (&stats.offset_inv[at..at + m], &stats.offset_ln[at..at + m]);
        let feat = &stats.feature_sum[at..at + m];
        let mut dense = 0.0f64;
        for ((&x, &i), &d) in mu.iter().zip(inv).zip(feat) {
            dense += (d + PRIOR_COUNT) * f64::from(ln1p_small(x * i));
        }
        let mut score = -dense;
        for (&col, &y) in cols.iter().zip(vals.iter()) {
            let g = col as usize;
            score -= f64::from(y) * f64::from(lnm[g] + ln1p_small(mu[g] * inv[g]));
        }
        score + prior(stats.edge_count[t] as f64)
    };

    // Every community as if it did not hold the edge, one feature at a time.
    for (t, lp) in log_probs.iter_mut().enumerate() {
        *lp = match offset {
            None => tail(stats.size_sum[t], stats.edge_count[t] as f64),
            Some(_) if t == current_c => 0.0,
            Some(mu) => offset_tail(mu, t),
        };
    }
    for (&col, &y) in cols.iter().zip(vals.iter()) {
        let g = col as usize;
        ln_gamma_increments(stats.feature_table.run(g * k..(g + 1) * k), y, increments);
        for (lp, &inc) in log_probs.iter_mut().zip(increments.iter()) {
            *lp += f64::from(inc);
        }
    }

    // The current community does hold it, so its score leaves the edge out.
    let features = &stats.feature_sum[current_c * m..(current_c + 1) * m];
    let mut at_current = match offset {
        None => tail(
            stats.size_sum[current_c] - f64::from(sf),
            stats.edge_count[current_c] as f64 - 1.0,
        ),
        Some(mu) => {
            // the current community with the edge taken out: its rest plus
            // the edge's `μ` is its whole sum, so `ln(M+μ)` is exact here
            let at = current_c * m;
            let off = &stats.offset_sum[at..at + m];
            let mut score = 0.0;
            let mut y_at = cols.iter().zip(vals.iter()).peekable();
            for (g, (&x, &whole)) in mu.iter().zip(off).enumerate() {
                let y = y_at
                    .next_if(|&(&c, _)| c as usize == g)
                    .map_or(0.0, |(_, &y)| f64::from(y));
                let d = features[g] - y + PRIOR_COUNT;
                let after = (whole + PRIOR_RATE).ln();
                score += d * ((whole - f64::from(x) + PRIOR_RATE).ln() - after) - y * after;
            }
            score + prior(stats.edge_count[current_c] as f64 - 1.0)
        }
    };
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

/// `ln(1 + x)` for `x ≥ 0`: `2 atanh(z)`, `z = x / (2 + x)`, by its series
/// for `x ≤ 1` (`z ≤ 1/3`, within f32 rounding), libm beyond. An
/// edge's share of a community's expected mass is nearly always under 1,
/// and libm's `log1p` per community and feature was what an offset edge cost.
#[inline]
pub(crate) fn ln1p_small(x: f32) -> f32 {
    if x <= 1.0 {
        let z = x / (2.0 + x);
        let z2 = z * z;
        2.0 * z
            * (1.0
                + z2 * (1.0 / 3.0
                    + z2 * (1.0 / 5.0 + z2 * (1.0 / 7.0 + z2 * (1.0 / 9.0 + z2 * (1.0 / 11.0))))))
    } else {
        x.ln_1p()
    }
}
