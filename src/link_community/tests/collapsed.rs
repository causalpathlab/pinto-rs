//! The sampler against the posterior it claims: with few enough edges, every
//! labeling can be listed, so the target is known exactly.

use crate::link_community::gibbs::*;
use crate::link_community::model::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use special::Gamma;

const M: usize = 3;
const K: usize = 2;
/// The floor the old self-inclusive rule took its logs at.
const LOG_EPS: f64 = 1e-9;

/// Overlapping profiles, so that the posterior spreads over many labelings.
fn fixture() -> LinkProfileStore {
    let rows: [[f32; M]; 6] = [
        [2.0, 1.0, 0.0],
        [1.5, 0.5, 0.5],
        [0.0, 1.0, 2.0],
        [0.5, 0.5, 1.5],
        [1.0, 1.0, 1.0],
        [2.0, 0.0, 1.0],
    ];
    LinkProfileStore::new(rows.concat(), rows.len(), M)
}

/// The labeling numbered `state`, its edges as base-`K` digits.
fn labeling(state: usize, n_edges: usize) -> Vec<usize> {
    let mut rest = state;
    (0..n_edges)
        .map(|_| {
            let c = rest % K;
            rest /= K;
            c
        })
        .collect()
}

fn state_of(labels: &[usize]) -> usize {
    labels.iter().rev().fold(0, |acc, &c| acc * K + c)
}

fn enumerated_posterior(store: &LinkProfileStore, alpha: f64) -> Vec<f64> {
    let n_states = K.pow(store.n_edges as u32);
    let log_joint: Vec<f64> = (0..n_states)
        .map(|s| {
            LinkCommunityStats::from_profiles(store, K, &labeling(s, store.n_edges))
                .log_marginal(alpha)
        })
        .collect();
    let top = log_joint.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_joint.iter().map(|l| (l - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>()
}

const BURN_IN: usize = 500;
const SWEEPS: usize = 40_000;

/// Long-run share of each labeling under a chain advanced by `sweep`.
fn occupancy(
    store: &LinkProfileStore,
    mut sweep: impl FnMut(&mut LinkCommunityStats, &LinkProfileStore),
) -> Vec<f64> {
    let start = labeling(0b010101, store.n_edges);
    let mut stats = LinkCommunityStats::from_profiles(store, K, &start);
    let mut visits = vec![0usize; K.pow(store.n_edges as u32)];
    for s in 0..BURN_IN + SWEEPS {
        sweep(&mut stats, store);
        if s >= BURN_IN {
            visits[state_of(&stats.membership)] += 1;
        }
    }
    visits.iter().map(|&v| v as f64 / SWEEPS as f64).collect()
}

/// The marginal is what its docs say: a Dirichlet-multinomial per community,
/// and a Dirichlet-multinomial over the edge counts, up to constants in `z`.
#[test]
fn log_marginal_is_the_dirichlet_multinomial_by_hand() {
    let store = LinkProfileStore::new(vec![1.0, 2.0, 3.0, 0.5], 2, 2);
    let stats = LinkCommunityStats::from_profiles(&store, 2, &[0, 1]);
    let lg = |x: f64| Gamma::ln_gamma(x).0;
    let (eps, m_eps) = (PRIOR_COUNT, 2.0 * PRIOR_COUNT);
    let community =
        |a: f64, b: f64| lg(a + eps) + lg(b + eps) - 2.0 * lg(eps) - lg(a + b + m_eps) + lg(m_eps);
    let a = 0.6 / 2.0;
    let want = community(1.0, 2.0) + community(3.0, 0.5) + 2.0 * (lg(1.0 + a) - lg(a));
    let got = stats.log_marginal(0.6);
    assert!((got - want).abs() < 1e-9, "{got} against {want}");
}

/// Every conditional, from every labeling, against the ratio of joints.
fn assert_conditional_is_the_joint_ratio(store: &LinkProfileStore, alpha: f64) {
    let mut log_probs = vec![0.0; K];
    for state in 0..K.pow(store.n_edges as u32) {
        let labels = labeling(state, store.n_edges);
        let stats = LinkCommunityStats::from_profiles(store, K, &labels);
        let here = stats.log_marginal(alpha);
        for e in 0..store.n_edges {
            compute_log_probs_for_edge(
                e,
                &stats,
                store,
                alpha,
                None,
                &mut log_probs,
                &mut [0.0; K],
            );
            // A conditional sums two increments per feature, per size and per
            // prior, each held to a percent plus `0.03` in `tests/fast_math.rs`.
            let n_increments = 2 * (store.row(e).0.len() + 2);
            let slack = 0.03 * n_increments as f64;
            for (t, &got) in log_probs.iter().enumerate() {
                let mut moved = labels.clone();
                moved[e] = t;
                let want =
                    LinkCommunityStats::from_profiles(store, K, &moved).log_marginal(alpha) - here;
                assert!(
                    (got - want).abs() <= 1e-2 * want.abs() + slack,
                    "state {state}, edge {e} to {t}: {got} against {want}"
                );
            }
        }
    }
}

#[test]
fn conditional_is_the_joint_ratio() {
    assert_conditional_is_the_joint_ratio(&fixture(), 0.0);
    assert_conditional_is_the_joint_ratio(&fixture(), 1.5);
    assert_conditional_is_the_joint_ratio(&fixture(), 0.04);
}

/// A count near zero says next to nothing, so it may not decide a move. A tiny
/// prior count fails this: `ln Γ(y + ε) − ln Γ(ε)` jumps by `−ln ε` at `y > 0`.
#[test]
fn a_vanishing_count_costs_next_to_nothing() {
    let store = LinkProfileStore::new(vec![1e-4, 0.0, 5.0, 5.0, 5.0, 5.0], 3, 2);
    // Alone in its community, so its count meets an empty feature.
    let stats = LinkCommunityStats::from_profiles(&store, 2, &[0, 1, 1]);
    let mut log_probs = vec![0.0; 2];
    compute_log_probs_for_edge(0, &stats, &store, 0.0, None, &mut log_probs, &mut [0.0; 2]);
    assert!(
        log_probs[1].abs() < 0.05,
        "moving a near-empty edge scored {}",
        log_probs[1]
    );
}

#[test]
fn the_sampler_reaches_the_enumerated_posterior() {
    let store = fixture();
    // At this K, the last gives each community the `α/K` a default of one does at 50.
    for alpha in [0.0, 1.5, 0.04] {
        let target = enumerated_posterior(&store, alpha);
        let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(7));
        let visits = occupancy(&store, |stats, store| {
            sampler.gibbs_sweep(stats, store, alpha);
        });
        let tv = total_variation(&visits, &target);
        assert!(tv < 0.05, "α = {alpha}: total variation {tv}");
    }
}

/// The check above must be able to fail: the self-inclusive plug-in rule, which
/// scores an edge's own community with the edge still in it, is a broken move.
#[test]
fn a_self_inclusive_move_fails_the_posterior_check() {
    let store = fixture();
    let target = enumerated_posterior(&store, 0.0);
    let mut rng = SmallRng::seed_from_u64(7);
    let visits = occupancy(&store, |stats, store| {
        let mut log_probs = [0.0; K];
        for e in 0..store.n_edges {
            let (cols, vals) = store.row(e);
            let sf = f64::from(store.size_factors[e]);
            for (t, lp) in log_probs.iter_mut().enumerate() {
                let rate = |g: usize| (stats.feature_sum[t * M + g] + LOG_EPS).ln();
                *lp = cols
                    .iter()
                    .zip(vals)
                    .map(|(&g, &y)| f64::from(y) * rate(g as usize))
                    .sum::<f64>()
                    - sf * (stats.size_sum[t] + M as f64 * LOG_EPS).ln();
            }
            let new_c = sample_categorical_log(&log_probs, &mut rng);
            let old_c = stats.membership[e];
            if new_c != old_c {
                stats.delta_move(e, old_c, new_c, store);
            }
        }
    });
    let tv = total_variation(&visits, &target);
    assert!(tv > 0.15, "the broken move went unnoticed: {tv}");
}

/// An edge alone in its community ties with an empty one, as it should, so greedy
/// has no reason to go: moving would keep a converged sweep from ever reading zero.
#[test]
fn greedy_leaves_a_lone_edge_beside_an_empty_community() {
    for scale in [0.37f32, 1.3, 7.1, 53.9, 811.0] {
        let rows: [[f32; M]; 4] = [
            [scale, 0.3 * scale, 0.0],
            [0.0, 0.1, 40.0],
            [0.0, 0.2, 40.0],
            [0.0, 0.1, 41.0],
        ];
        let store = LinkProfileStore::new(rows.concat(), rows.len(), M);
        let mut stats = LinkCommunityStats::from_profiles(&store, 3, &[1, 2, 2, 2]);
        let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(7));
        let moves = sampler.run_greedy_plain(&mut stats, &store, 5, 0.0);
        assert_eq!(moves, 0, "scale {scale}: {:?}", stats.membership);
    }
}

/// Both read the same arguments, an empty community and an edge's own once it is
/// left out, so nothing but the path through the arithmetic could tell them apart.
#[test]
fn a_lone_edge_ties_exactly_with_an_empty_community() {
    const WIDE: usize = 40;
    for i in 0..400 {
        let scale = 1e-3 * 1.04f32.powi(i);
        let row: Vec<f32> = (0..WIDE)
            .map(|g| scale * (0.05 + ((g + i as usize) as f32 * 0.618).fract()))
            .collect();
        let store = LinkProfileStore::new(row, 1, WIDE);
        let stats = LinkCommunityStats::from_profiles(&store, 2, &[1]);
        let mut log_probs = vec![0.0; 2];
        compute_log_probs_for_edge(0, &stats, &store, 0.0, None, &mut log_probs, &mut [0.0; 2]);
        assert_eq!(log_probs[0], log_probs[1], "scale {scale}");
    }
}
