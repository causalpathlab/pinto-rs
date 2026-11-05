//! Random pseudobulk: units never mix arms, identities or tiles; Rubin's
//! rules widen the interval by the between-draw spread; and the effect is
//! still recovered, with type I error held, on the synthetic tissue.

use super::run_sim;
use super::simulate::{base, fixture, opts};
use crate::lr_activity::ligand_effect::estimate::{EffectOpts, Exposure};
use crate::lr_activity::ligand_effect::overlap::Estimate;
use crate::lr_activity::ligand_effect::pseudobulk::{combine, draw, partition, unit_response};
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
    let (unit_of, n_units, random) = partition(t, &frame, 6, 3);
    assert!(random);
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

/// A partition whose strata all hold fewer than two units' worth of
/// receivers is the same under every seed, and says so.
#[test]
fn a_partition_no_seed_changes_is_fixed() {
    let f = fixture();
    let t = &f.tissue;
    let n = t.n();
    let frame = Exposure {
        tau: 0.0,
        t: (0..n).map(|i| i % 3 == 0).collect(),
        t_ring: vec![false; n],
        own: vec![0.0; n],
        niche: vec![0.0; n],
    };
    let big = n;
    let (a, _, random) = partition(t, &frame, big, 1);
    assert!(!random);
    assert_eq!(a, partition(t, &frame, big, 2).0);
    assert!(partition(t, &frame, 1, 1).2);
}

/// The guards count units, the estimator's points: twenty exposed
/// receivers in one stratum make two units of ten, an arm the guard
/// rejects at `min_arm` 20, and the skip of hopeless fits reads the same
/// two.
#[test]
fn the_guard_counts_units_not_their_members() {
    use crate::lr_activity::ligand_effect::estimate::{Reads, Rows};
    use crate::lr_activity::ligand_effect::overlap::{subset_estimate, OverlapTerms};
    use crate::lr_activity::ligand_effect::pseudobulk::draw;
    let f = fixture();
    let t = &f.tissue;
    let n = t.n();
    // The largest group of one tile and identity bin; twenty of it exposed.
    let mut groups: std::collections::HashMap<(u32, usize), Vec<usize>> = Default::default();
    for i in 0..n {
        groups
            .entry((t.cf.cluster[i], t.identity_bin[i]))
            .or_default()
            .push(i);
    }
    let group = groups.into_values().max_by_key(|g| g.len()).unwrap();
    assert!(group.len() >= 20, "{} receivers", group.len());
    let mut exposed = vec![false; n];
    for &i in &group[..20] {
        exposed[i] = true;
    }
    let frame = Exposure {
        tau: 0.0,
        t: exposed,
        t_ring: vec![false; n],
        own: vec![0.0; n],
        niche: vec![0.0; n],
    };
    let u = draw(t, &frame, &vec![0.0; n], 10, 5);
    let rows: Vec<u32> = (0..u.lead.len() as u32).collect();
    let terms = OverlapTerms {
        h: vec![0.25; rows.len()],
        num: vec![0.01; rows.len()],
    };
    let (e, sums) = subset_estimate(&terms, &u.exposure.t, &rows, &u.cf);
    assert_eq!(e.n_treated, 2);
    assert_eq!(e.n, u.lead.len());
    let o = opts();
    assert_eq!(o.min_arm, 20);
    assert!(o.thin_arm(e.n, e.n_treated));
    assert!(o.guard((e, sums)).0.theta.is_nan());
    // The skip reads the same count: the fit of that arm is not run.
    let every = vec![vec![true; u.lead.len()]];
    let reads = Reads::of(Rows::units(&u), &u.exposure, &every, &o);
    assert!(!reads.competent);
}

/// A unit's pooled response depends on its pooled depth and member count, and
/// arms differ in both, so every adjustment step adjusts for them: a step that
/// leaves them out reports the unit-size artefact as an effect.
#[test]
fn every_adjustment_step_carries_the_unit_size() {
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
    let u = draw(t, &frame, &vec![0.0; n], 6, 3);
    let corr = |a: &[f64], j: usize| {
        let b: Vec<f64> = u.design.x.column(j).iter().map(|&v| v as f64).collect();
        let k = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / k, b.iter().sum::<f64>() / k);
        let cov: f64 = a.iter().zip(&b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
        let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
        cov / (va * vb).sqrt()
    };
    let depth: Vec<f64> = u.size_factor.iter().map(|s| s.ln()).collect();
    let members: Vec<f64> = u.size.iter().map(|&s| (s as f64).ln()).collect();
    assert!(corr(&depth, 1) > 0.999, "column 1 is the pooled depth");
    assert!(corr(&members, 2) > 0.999, "column 2 is the member count");
    assert_eq!(u.design.p(), t.design.p() + 2);
    for (s, cells) in u.steps.iter().zip(&t.steps) {
        assert_eq!(s.design.p(), cells.design.p() + 2, "{}", s.name);
    }
}
