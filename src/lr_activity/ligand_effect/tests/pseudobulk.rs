//! Random pseudobulk: units never mix arms, identities or tiles; Rubin's
//! rules widen the interval by the between-draw spread; and the effect is
//! still recovered, with type I error held, on the synthetic tissue.

use super::run_sim;
use super::simulate::{base, fixture, opts};
use crate::lr_activity::ligand_effect::estimate::{EffectOpts, Exposure};
use crate::lr_activity::ligand_effect::overlap::Estimate;
use crate::lr_activity::ligand_effect::pseudobulk::{combine, partition, unit_response};
use crate::lr_activity::ligand_effect::simulate::Scenario;

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
fn units_stay_inside_their_stratum() {
    let f = fixture();
    let t = &f.tissue;
    let n = t.n();
    let frame = Exposure {
        tau: 0.0,
        t: (0..n).map(|i| i % 3 == 0).collect(),
        t_ring: (0..n).map(|i| i % 5 == 0).collect(),
        own: vec![0.0; n],
        niche: vec![0.0; n],
    };
    let (unit_of, n_units) = partition(t, &frame, 6, 3);
    assert!(n_units > n / 12 && n_units <= n, "{n_units} units");
    let mut lead = vec![usize::MAX; n_units];
    let mut size = vec![0usize; n_units];
    for (i, &u) in unit_of.iter().enumerate() {
        let u = u as usize;
        size[u] += 1;
        if lead[u] == usize::MAX {
            lead[u] = i;
            continue;
        }
        let j = lead[u];
        assert_eq!(frame.t[i], frame.t[j], "arms mixed");
        assert_eq!(frame.t_ring[i], frame.t_ring[j], "ring arms mixed");
        assert_eq!(t.identity_bin[i], t.identity_bin[j], "identities mixed");
        assert_eq!(t.cf.cluster[i], t.cf.cluster[j], "tiles mixed");
    }
    assert!(size.iter().all(|&s| s >= 1));
    // Same seed, same partition; another seed, another one.
    assert_eq!(partition(t, &frame, 6, 3).0, unit_of);
    assert_ne!(partition(t, &frame, 6, 4).0, unit_of);
}

#[test]
fn unit_response_sums_counts_before_the_log() {
    let unit_of = [0u32, 0, 1];
    let sf = [2.0, 1.0];
    let gene: &[(u32, f32)] = &[(0, 1.0), (1, 3.0), (2, 2.0)];
    let y = unit_response(&[(2.0, gene)], &unit_of, &sf);
    assert!((y[0] - (4.0f64 / 2.0).ln_1p()).abs() < 1e-12);
    assert!((y[1] - (2.0f64).ln_1p()).abs() < 1e-12);
}

#[test]
fn rubin_adds_the_between_draw_spread() {
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

fn pb() -> EffectOpts {
    EffectOpts {
        pseudobulk: 4,
        draws: 3,
        ..opts()
    }
}

#[test]
fn pseudobulk_recovers_a_planted_effect() {
    let f = fixture();
    let scen = Scenario {
        effect: 0.5,
        confound_ligand: 1.0,
        confound_response: 1.0,
        scale: 32.0,
        ..Scenario::default()
    };
    let r = run_sim(&f.tissue, &base(&f), &scen, &pb(), 21).unwrap();
    assert!(r.truth > 0.05, "truth {}", r.truth);
    assert!(
        ((r.estimate.theta - r.truth) / r.estimate.se).abs() < 3.0,
        "estimate {:.3} ± {:.3}, truth {:.3}",
        r.estimate.theta,
        r.estimate.se,
        r.truth
    );
}

#[test]
fn pseudobulk_holds_type_one_error_under_confounding() {
    let f = fixture();
    let scen = Scenario {
        effect: 0.0,
        confound_ligand: 1.5,
        confound_response: 1.5,
        scale: 32.0,
        ..Scenario::default()
    };
    let (mut rej, mut crude) = (0, 0);
    for rep in 0..4 {
        let r = run_sim(&f.tissue, &base(&f), &Scenario { rep, ..scen }, &pb(), 11).unwrap();
        rej += r.rejected() as usize;
        crude += r.crude_rejected() as usize;
    }
    assert!(crude >= 3, "crude rejected {crude}/4");
    assert!(rej <= 1, "pseudobulk estimate rejected {rej}/4 null runs");
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
    assert_eq!(
        g.df,
        crate::lr_activity::ligand_effect::estimate::ONE_CLUSTER_DF
    );
    assert!(sums.iter().all(|&s| s == 0.0));

    let c = combine(&[one, one, one]);
    assert!(c.theta.is_nan() && c.p().is_nan(), "{c:?}");
    let c = combine(&[g, g, g]);
    assert!(c.theta.is_nan());
    assert_eq!(
        c.df,
        crate::lr_activity::ligand_effect::estimate::ONE_CLUSTER_DF
    );
    // A NaN-df draw beside estimable ones is left out, not read as ∞.
    let mixed = combine(&[one, est(0.1, 0.05), est(0.12, 0.05)]);
    assert!(mixed.df.is_finite() && mixed.df < 50.0, "{}", mixed.df);
    assert!((mixed.theta - 0.11).abs() < 1e-12);
}
