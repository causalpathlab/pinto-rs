//! Latent edge incidence (frozen, variational-Bayes Poisson–Gamma block term).
//!
//! Treats each edge as a "visible" unit with a community label `z_e ∈ {0..K-1}`
//! and each incident vertex (cell) as a "hidden" unit with propensity vector
//! `θ[v, ·] ∈ Δ^K` (already produced by `compute_node_membership`). The K×K
//! incidence matrix `B[k, k']` is fitted **once** from the post-V-cycle edge
//! labels under a conjugate `Gamma(a, b)` prior and **frozen** for the
//! duration of the final EM-Gibbs and greedy passes.
//!
//! The cached value is the proper variational-Bayes expected log:
//!
//! ```text
//! E_q[log B[k, k']] = ψ(a + S[k, k']) − log(b + W[k'])
//! ```
//!
//! (digamma minus log of rate), where
//!
//! ```text
//! S[k, k'] = Σ_{e: z_e=k} (θ_L[k'] + θ_R[k'])
//! W[k']    = Σ_e          (θ_L[k'] + θ_R[k'])     (k-independent)
//! ```
//!
//! The score gains an additive term
//!
//! ```text
//! Σ_{k'} (θ_L(e)[k'] + θ_R(e)[k']) · E_q[log B[k, k']]
//! ```
//!
//! computed by `add_incidence_to_log_probs` from per-cell scores. There is no per-move
//! bookkeeping: `B` is a global read-only weight table, exactly the
//! "RBM-style frozen coupling" picture.

use rayon::prelude::*;
use special::Gamma as SpecialGamma;

/// Compute frozen variational `E_q[log B[k, k']]` from the current edge
/// labels and vertex propensity. Output is a length-`k*k` row-major matrix.
///
/// `propensity` is row-major `[n_cells × k]`.
pub fn fit_log_incidence(
    edges: &[(usize, usize)],
    membership: &[usize],
    propensity: &[f64],
    k: usize,
    a: f64,
    b: f64,
) -> Vec<f64> {
    debug_assert_eq!(edges.len(), membership.len());
    let mut s = vec![0.0f64; k * k];
    let mut w = vec![0.0f64; k];

    for (e, &(i, j)) in edges.iter().enumerate() {
        let c = membership[e];
        debug_assert!(c < k);
        let prop_i = &propensity[i * k..(i + 1) * k];
        let prop_j = &propensity[j * k..(j + 1) * k];
        let base = c * k;
        for kp in 0..k {
            let delta = prop_i[kp] + prop_j[kp];
            s[base + kp] += delta;
            w[kp] += delta;
        }
    }

    let mut log_b = vec![0.0f64; k * k];
    for c in 0..k {
        let base = c * k;
        for kp in 0..k {
            // E_q[log B] = ψ(a + S) − log(b + W)
            log_b[base + kp] = SpecialGamma::digamma(a + s[base + kp]) - (b + w[kp]).ln();
        }
    }
    log_b
}

/// Each cell's share of the incidence term per community, `[n_cells × k]`
/// row-major: `Σ_{k'} θ[v, k'] · log_B[k, k']`. An edge's term is the sum of its
/// two cells', so scoring it takes `O(K)` where the `K × K` form took `O(K²)`.
/// A cell's propensity is mostly zeros, which are skipped here.
pub fn cell_incidence_scores(propensity: &[f64], log_incidence: &[f64], k: usize) -> Vec<f32> {
    let mut table = vec![0.0f32; propensity.len()];
    table
        .par_chunks_mut(k)
        .zip(propensity.par_chunks(k))
        .for_each(|(scores, theta)| {
            for (kp, &share) in theta.iter().enumerate() {
                if share != 0.0 {
                    for (t, score) in scores.iter_mut().enumerate() {
                        *score += (share * log_incidence[t * k + kp]) as f32;
                    }
                }
            }
        });
    table
}

/// Pack a `[n_cells × k]` row-major `Vec<f64>` propensity from a
/// column-major `nalgebra::DMatrix<f32>`. The flat layout lets the hot
/// loop index a length-K cell row as a contiguous slice.
pub fn pack_propensity_row_major(propensity_mat: &crate::util::common::Mat) -> Vec<f64> {
    let n = propensity_mat.nrows();
    let k = propensity_mat.ncols();
    let mut out = vec![0.0f64; n * k];
    for i in 0..n {
        for kp in 0..k {
            out[i * k + kp] = propensity_mat[(i, kp)] as f64;
        }
    }
    out
}

/// Add the (frozen) incidence contribution to `log_probs` (delta form against
/// the current community), to be called immediately after
/// `compute_log_probs_for_edge`. `cell_scores` is [`cell_incidence_scores`].
#[inline]
pub fn add_incidence_to_log_probs(
    e: usize,
    edges: &[(usize, usize)],
    cell_scores: &[f32],
    k: usize,
    current_c: usize,
    log_probs: &mut [f64],
) {
    let (i, j) = edges[e];
    let scores_i = &cell_scores[i * k..(i + 1) * k];
    let scores_j = &cell_scores[j * k..(j + 1) * k];
    let at_current = f64::from(scores_i[current_c]) + f64::from(scores_j[current_c]);
    for ((lp, &a), &b) in log_probs.iter_mut().zip(scores_i).zip(scores_j) {
        *lp += f64::from(a) + f64::from(b) - at_current;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Summing per-cell scores gives an edge what the `K × K` form did.
    #[test]
    fn cell_scores_give_each_edge_its_incidence_term() {
        let k = 3usize;
        let propensity = vec![0.5, 0.5, 0.0, 0.0, 0.2, 0.8, 1.0, 0.0, 0.0];
        let log_incidence = vec![-0.5, -2.0, -1.5, -0.25, -3.0, -0.75, -1.0, -0.1, -2.5];

        let table = cell_incidence_scores(&propensity, &log_incidence, k);

        for (i, j) in [(0usize, 1usize), (1, 2), (0, 2)] {
            for t in 0..k {
                let want: f64 = (0..k)
                    .map(|kp| {
                        (propensity[i * k + kp] + propensity[j * k + kp])
                            * log_incidence[t * k + kp]
                    })
                    .sum();
                let got = f64::from(table[i * k + t]) + f64::from(table[j * k + t]);
                assert!(
                    (got - want).abs() < 1e-5,
                    "({i}, {j}), {t}: {got} against {want}"
                );
            }
        }
    }

    /// Two well-separated cell blocks, edges placed within blocks: the
    /// fitted `log B` should be diagonally dominant.
    #[test]
    fn diagonal_dominance_on_two_blocks() {
        let k = 2usize;
        let n_cells = 6usize;
        let mut prop = vec![0.0f64; n_cells * k];
        for i in 0..3 {
            prop[i * k] = 1.0;
        }
        for i in 3..6 {
            prop[i * k + 1] = 1.0;
        }
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let membership = vec![0, 0, 0, 1, 1, 1];

        let log_b = fit_log_incidence(&edges, &membership, &prop, k, 1.0, 1.0);

        assert!(log_b[0] > log_b[1]);
        assert!(log_b[k + 1] > log_b[k]);
    }

    /// Moving a within-block edge to the wrong community should be
    /// penalised by the incidence term.
    #[test]
    fn add_to_log_probs_penalises_wrong_block() {
        let k = 2usize;
        let n_cells = 6usize;
        let mut prop = vec![0.0f64; n_cells * k];
        for i in 0..3 {
            prop[i * k] = 1.0;
        }
        for i in 3..6 {
            prop[i * k + 1] = 1.0;
        }
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let membership = vec![0, 0, 0, 1, 1, 1];
        let log_b = fit_log_incidence(&edges, &membership, &prop, k, 1.0, 1.0);

        let table = cell_incidence_scores(&prop, &log_b, k);
        let mut log_probs = vec![0.0f64; k];
        add_incidence_to_log_probs(0, &edges, &table, k, 0, &mut log_probs);
        assert!(
            log_probs[1] < 0.0,
            "moving block-0 edge to community 1 should be penalised; got {:?}",
            log_probs
        );
    }
}
