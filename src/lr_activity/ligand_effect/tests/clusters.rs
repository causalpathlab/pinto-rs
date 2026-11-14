//! The effective number of variance clusters a t reference is read on.

use crate::lr_activity::ligand_effect::inference::{clustered_sd, effective_clusters};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// Clusters alike count as about their number; a subgroup whose variance
/// sits in a few of them counts as few, whatever the rest number.
#[test]
fn effective_clusters_count_the_clusters_the_variance_rests_on() {
    let mut rng = SmallRng::seed_from_u64(1);
    let alike: Vec<f64> = (0..60).map(|_| StandardNormal.sample(&mut rng)).collect();
    let c = effective_clusters(&alike);
    assert!((40.0..=60.0).contains(&c), "{c}");
    // Three tiles carry it, fifty more hold a trace.
    let few: Vec<f64> = [1.0, -0.8, 0.9]
        .into_iter()
        .chain(std::iter::repeat_n(1e-3, 50))
        .collect();
    let c = effective_clusters(&few);
    assert!((2.0..=9.0).contains(&c), "{c}");
    let (_, df) = clustered_sd(&few);
    assert!(df < 9.0, "df {df}, not the 52 of C - 1");
    // Never above the clusters that carry any, never below two.
    assert_eq!(effective_clusters(&[1.0, 1.0, 0.0]), 2.0);
    assert_eq!(effective_clusters(&[5.0, 1e-9]), 2.0);
    assert!(effective_clusters(&[1.0, 0.0]).is_nan());
    assert!(clustered_sd(&[1.0]).1.is_nan());
}

/// A tile left out is in no fold, and the design's Grams are those of the
/// design built on the same columns without it.
#[test]
fn a_left_out_tile_leaves_every_fold_and_gram() {
    use crate::lr_activity::ligand_effect::crossfit::{CrossFit, Design, LEFT_OUT};
    let mut rng = SmallRng::seed_from_u64(3);
    let n = 120;
    let x = nalgebra::DMatrix::<f32>::from_fn(n, 4, |_, j| {
        if j == 0 {
            1.0
        } else {
            let e: f64 = StandardNormal.sample(&mut rng);
            e as f32
        }
    });
    let fold: Vec<u8> = (0..n).map(|i| (i % 3) as u8).collect();
    let cluster: Vec<u32> = (0..n).map(|i| (i / 10) as u32).collect();
    let cf = CrossFit::new(fold, 3, cluster);
    let design = Design::new(x, &cf);
    let out: Vec<usize> = (40..50).collect();
    let left = cf.leaving_out(&out);
    assert!(out.iter().all(|&i| left.fold[i] == LEFT_OUT));
    for k in 0..3 {
        assert!(left.train[k]
            .iter()
            .chain(&left.test[k])
            .all(|i| !out.contains(i)));
        assert_eq!(left.train[k].len() + left.test[k].len(), n - out.len());
    }
    let by_subtraction = design.leaving_out(&cf, &out);
    let rebuilt = Design::new((*design.x).clone(), &left);
    for (a, b) in by_subtraction.view().gram.iter().zip(rebuilt.view().gram) {
        assert!((a - b).abs().max() < 1e-6, "{}", (a - b).abs().max());
    }
}

#[test]
fn jackknife_se_by_hand() {
    use crate::lr_activity::ligand_effect::jackknife::jackknife_se;
    // Replicates 1, 2, 3: mean 2, (G - 1)/G * 2 = 4/3.
    assert!((jackknife_se(&[1.0, 2.0, 3.0]) - (4.0f64 / 3.0).sqrt()).abs() < 1e-12);
    assert!(jackknife_se(&[1.0]).is_nan());
    assert!(jackknife_se(&[1.0, f64::NAN]).is_nan());
}

/// A replicate fold started from the full fit's factor less the left-out
/// rows reaches the fresh fit's optimum on the rows kept.
#[test]
fn a_downdated_start_reaches_the_fresh_optimum() {
    use crate::lr_activity::ligand_effect::crossfit::LEFT_OUT;
    use crate::lr_activity::ligand_effect::regression::{logistic_base, logistic_from};
    let mut rng = SmallRng::seed_from_u64(9);
    let n = 400;
    let x = nalgebra::DMatrix::<f32>::from_fn(n, 4, |_, j| {
        if j == 0 {
            1.0
        } else {
            let e: f64 = StandardNormal.sample(&mut rng);
            e as f32
        }
    });
    let t: Vec<f64> = (0..n)
        .map(|i| {
            let eta = 0.8 * x[(i, 1)] as f64 - 0.5 * x[(i, 2)] as f64;
            let u: f64 = StandardNormal.sample(&mut rng);
            ((eta + u) > 0.0) as u8 as f64
        })
        .collect();
    let rows: Vec<usize> = (0..n).collect();
    let full = logistic_base(&x, 4, &[], &t, &rows, 1.0, None).unwrap();
    let mut fold = vec![0u8; n];
    for f in fold.iter_mut().take(60).skip(20) {
        *f = LEFT_OUT;
    }
    let start = full.leaving_out(&fold, &x, 4, &[]).expect("a replicate");
    let kept: Vec<usize> = (0..n).filter(|&i| fold[i] != LEFT_OUT).collect();
    let from = logistic_from(&x, 4, &[], &t, 1.0, Some(&full.beta), start).unwrap();
    let fresh = logistic_base(&x, 4, &[], &t, &kept, 1.0, None).unwrap();
    assert!(
        (&from.beta - &fresh.beta).amax() < 1e-6,
        "{} vs {}",
        from.beta,
        fresh.beta
    );
    // Nothing left out: not a replicate.
    assert!(full.leaving_out(&vec![0u8; n], &x, 4, &[]).is_none());
}

/// Worked by hand. Ten tiles with sums 4, 1, 1, 1, 1, 1, 1, 1, 1, 1:
/// Σ s² = 16 + 9 = 25, Σ s⁴ = 256 + 9 = 265, so C_eff = 3 · 625 / 265
/// = 7.0755 and the t reference has 6.0755 degrees of freedom, where the
/// old rule, every carrying tile less one, gave 9. The SE is unchanged:
/// sqrt(25 · 10 / 9) = 5.2705. Three equal sums give 3 · 81 / 27 = 9,
/// clamped to the 3 tiles: df 2, as before.
#[test]
fn effective_clusters_by_hand() {
    let sums = [4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let c = effective_clusters(&sums);
    assert!((c - 1875.0 / 265.0).abs() < 1e-12, "{c}");
    let (sd, df) = clustered_sd(&sums);
    assert!((df - (1875.0 / 265.0 - 1.0)).abs() < 1e-12, "{df}");
    assert!((sd - (25.0f64 * 10.0 / 9.0).sqrt()).abs() < 1e-12, "{sd}");
    let (_, df3) = clustered_sd(&[3.0, 3.0, 3.0]);
    assert_eq!(df3, 2.0);
}

/// Widening scales the headline, the negative-control contrast and the
/// contrast with the rest, never down; a jackknife whose replicates ran but
/// gave no ratio says so.
#[test]
fn widening_covers_the_control_contrast_and_failure_is_flagged() {
    use crate::lr_activity::ligand_effect::effects::ScopeEffects;
    use crate::lr_activity::ligand_effect::inference::Estimate;
    use crate::lr_activity::ligand_effect::jackknife::Jackknife;
    let est = |se: f64| Estimate {
        theta: 0.1,
        se,
        n: 100,
        n_treated: 40,
        mean_h: 0.2,
        ess: 80.0,
        df: 8.0,
    };
    let mut e = ScopeEffects {
        competent: est(1.0),
        effect: est(2.0),
        versus_rest: est(3.0),
        ..ScopeEffects::zip_with(&[], |_| Estimate::nan(0, 0))
    };
    Jackknife {
        competent: 1.5,
        versus_rest: 0.5,
        replicates: 10,
    }
    .widen(&mut e);
    assert_eq!(
        (e.competent.se, e.effect.se, e.versus_rest.se),
        (1.5, 3.0, 3.0)
    );
    let failed = Jackknife {
        competent: f64::NAN,
        ..Jackknife::NONE
    };
    assert!(!failed.failed(), "never ran");
    assert!(Jackknife {
        replicates: 4,
        ..failed
    }
    .failed());
}
