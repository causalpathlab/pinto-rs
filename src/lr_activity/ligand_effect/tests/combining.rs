//! Combining analyses: Rubin's rules over fixed choices, and the Bayesian
//! average over exposure thresholds.

use crate::lr_activity::ligand_effect::averaging::{
    abf_average, abf_summary, background_comparison, combine,
};
use crate::lr_activity::ligand_effect::inference::Estimate;
use crate::lr_activity::ligand_effect::inference::ONE_CLUSTER_DF;
use crate::lr_activity::ligand_effect::options::EffectOpts;

fn est(theta: f64, se: f64) -> Estimate {
    Estimate {
        theta,
        se,
        n: 100,
        n_treated: 40,
        mean_h: 0.2,
        ess: 80.0,
        df: 50.0,
    }
}

#[test]
fn rubin_adds_the_between_analysis_spread() {
    let same = combine(&[est(0.2, 0.1), est(0.2, 0.1), est(0.2, 0.1)]);
    assert!((same.theta - 0.2).abs() < 1e-12 && (same.se - 0.1).abs() < 1e-12);
    let spread = combine(&[est(0.0, 0.1), est(0.2, 0.1), est(0.4, 0.1)]);
    // W = 0.01, B = 0.04, T = 0.01 + (4/3)·0.04.
    assert!((spread.theta - 0.2).abs() < 1e-12);
    assert!((spread.se - (0.01f64 + 4.0 / 3.0 * 0.04).sqrt()).abs() < 1e-12);
    let with_nan = combine(&[est(f64::NAN, f64::NAN), est(0.3, 0.1)]);
    assert!((with_nan.theta - 0.3).abs() < 1e-12);
    assert!(combine(&[est(f64::NAN, 0.1)]).theta.is_nan());
}

/// A scope inside one variance cluster: its SE is a rounding residue with
/// no degrees of freedom. The guard makes it not estimable (df 0 says
/// why), and Rubin's rules never read a NaN df as infinite.
#[test]
fn one_variance_cluster_is_not_estimable() {
    let one = Estimate {
        se: 1e-17,
        df: f64::NAN,
        ..est(0.3, 0.0)
    };
    let (g, sums) = EffectOpts::default().guard((one, vec![1e-17, 0.0]));
    assert!(g.theta.is_nan() && g.se.is_nan());
    assert_eq!(g.df, ONE_CLUSTER_DF);
    assert!(sums.iter().all(|&s| s == 0.0));

    let c = combine(&[one, one, one]);
    assert!(c.theta.is_nan() && c.p().is_nan(), "{c:?}");
    let c = combine(&[g, g, g]);
    assert!(c.theta.is_nan());
    assert_eq!(c.df, ONE_CLUSTER_DF);
    // A NaN-df estimate beside estimable ones is left out, not read as ∞.
    let mixed = combine(&[one, est(0.1, 0.05), est(0.12, 0.05)]);
    assert!(mixed.df.is_finite() && mixed.df < 50.0, "{}", mixed.df);
    assert!((mixed.theta - 0.11).abs() < 1e-12);
}

/// The prior SD in multiples of the median standard error: with standard
/// errors of 0.1, a prior variance of 0.04.
const PRIOR: f64 = 2.0;

#[test]
fn the_average_is_the_mixture_of_its_thresholds() {
    // One usable estimate is itself; none stays not estimable.
    let one = est(0.3, 0.1);
    assert_eq!(abf_average(&[one], PRIOR).theta, one.theta);
    let beside_nan = abf_average(&[est(f64::NAN, f64::NAN), one], PRIOR);
    assert_eq!((beside_nan.theta, beside_nan.se), (one.theta, one.se));
    assert!(abf_average(&[est(f64::NAN, f64::NAN)], PRIOR)
        .theta
        .is_nan());

    // Equal evidence: equal weights, and the spread widens the variance.
    let (a, b) = (est(0.3, 0.1), est(-0.3, 0.1));
    let m = abf_average(&[a, b], PRIOR);
    assert!(m.theta.abs() < 1e-12);
    assert!((m.se - (0.01f64 + 0.09).sqrt()).abs() < 1e-12, "{}", m.se);

    // Identical thresholds change nothing but the evidence's bookkeeping.
    let m = abf_average(&[a, a, a], PRIOR);
    assert!((m.theta - 0.3).abs() < 1e-12 && (m.se - 0.1).abs() < 1e-12);
    assert!((m.df - 50.0).abs() < 1e-9);
}

/// The weights follow the Bayes factor's closed form, and the threshold with
/// the evidence carries the average.
#[test]
fn the_weights_follow_the_evidence() {
    let parts = [est(0.0, 0.1), est(0.5, 0.1)];
    let prior_var = PRIOR * PRIOR * 0.1 * 0.1;
    let bf = |e: &Estimate| {
        let (v, z) = (e.se * e.se, e.theta / e.se);
        (v / (v + prior_var)).sqrt() * (0.5 * z * z * prior_var / (v + prior_var)).exp()
    };
    let (b0, b1) = (bf(&parts[0]), bf(&parts[1]));
    let w1 = b1 / (b0 + b1);
    let m = abf_average(&parts, PRIOR);
    assert!((m.theta - w1 * 0.5).abs() < 1e-12, "{}", m.theta);
    let s = abf_summary(&parts, &[0.0, 2.0], PRIOR);
    assert_eq!(s.top_tau, 2.0);
    assert!((s.top_weight - w1).abs() < 1e-12);
    assert!((s.log10_bf - ((b0 + b1) / 2.0).log10()).abs() < 1e-9);
    // No effect anywhere: the evidence leans against one.
    let s = abf_summary(&[est(0.0, 0.01), est(0.001, 0.01)], &[0.0, 1.0], PRIOR);
    assert!(s.log10_bf < 0.0, "{}", s.log10_bf);
}

/// Each estimand gets its own weights: a control's largest threshold need not
/// be the programme's.
#[test]
fn a_control_is_averaged_by_its_own_evidence() {
    let programme = [est(0.5, 0.1), est(0.0, 0.1)];
    let control = [est(0.0, 0.1), est(0.5, 0.1)];
    let a = abf_summary(&programme, &[0.0, 1.0], PRIOR);
    let b = abf_summary(&control, &[0.0, 1.0], PRIOR);
    assert_eq!((a.top_tau, b.top_tau), (0.0, 1.0));
    assert!(
        (abf_average(&programme, PRIOR).theta - abf_average(&control, PRIOR).theta).abs() < 1e-12
    );
}

/// The effect against its background programmes is a predictive test: the
/// spread of the background programmes, with `K − 1` degrees of freedom; their
/// sampling variance is its floor.
#[test]
fn the_effect_is_compared_with_the_spread_of_its_backgrounds() {
    let backgrounds = [est(0.0, 0.01), est(0.2, 0.01), est(0.4, 0.01)];
    let (mean, over) = background_comparison(&est(1.0, 0.01), &backgrounds);
    assert!((mean.theta - 0.2).abs() < 1e-12);
    // s² = 0.04, K = 3.
    assert!((over.theta - 0.8).abs() < 1e-12);
    assert!((over.se - (0.04f64 * 4.0 / 3.0).sqrt()).abs() < 1e-12);
    assert!((mean.se - (0.04f64 / 3.0).sqrt()).abs() < 1e-12);
    assert_eq!(over.df, 2.0);
    // Identical background programmes are not certainty.
    let (_, over) = background_comparison(&est(1.0, 0.1), &[est(0.2, 0.1); 4]);
    assert!((over.se - (0.01f64 * 1.25).sqrt()).abs() < 1e-12);
    // An effect less certain than its background programmes sets the width and df.
    let (_, over) = background_comparison(&est(1.0, 0.5), &backgrounds);
    assert!((over.se - (0.25f64 + 0.04 / 3.0).sqrt()).abs() < 1e-12);
    assert_eq!(over.df, 50.0);
    // Too few usable background programmes, or no effect, compare nothing.
    assert!(background_comparison(&est(1.0, 0.1), &[est(0.2, 0.1)])
        .1
        .theta
        .is_nan());
    let (mean, over) = background_comparison(&est(f64::NAN, f64::NAN), &backgrounds);
    assert!(mean.theta.is_finite() && over.theta.is_nan());
}
