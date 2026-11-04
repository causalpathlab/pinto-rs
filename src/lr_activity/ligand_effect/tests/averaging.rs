//! Choices averaged rather than asked for, and what each pair reports
//! beside its headline: the competence grid combined by Rubin's rules, the
//! automatic number of pseudobulk draws, the threshold–response curve with
//! its simultaneous band, the matched decoy programme, the two-scale
//! moments and the per-cell-type rows.

use super::estimate_pair;
use crate::lr_activity::ligand_effect::estimate::*;
use crate::lr_activity::ligand_effect::overlap::{Estimate, Z_975};
use crate::lr_activity::ligand_effect::pseudobulk::{combine, enough_draws, MAX_DRAWS, MIN_DRAWS};
use crate::lr_activity::ligand_effect::tissue::{Tissue, TissueInputs, TissueOpts};
use crate::util::common::Mat;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Poisson, StandardNormal};

const SIDE: usize = 40;

fn est(theta: f64, se: f64, df: f64) -> Estimate {
    Estimate {
        theta,
        se,
        n: 100,
        n_treated: 40,
        mean_h: 0.2,
        ess: 80.0,
        df,
    }
}

struct World {
    tissue: Tissue,
    ligand: Vec<f64>,
    receptor: Vec<f64>,
    response: Vec<(u32, f32)>,
    /// Genes unrelated to the ligand, at a spread of mean levels.
    noise: Vec<Vec<(u32, f32)>>,
}

fn poisson(rng: &mut SmallRng, rate: f64) -> f32 {
    Poisson::new(rate.max(1e-9)).unwrap().sample(rng) as f32
}

/// A grid whose cells touch their four neighbours, one section, three
/// identity bins: bin 0 detects the receptor in every cell, bin 1 in about
/// 15% (competent at the grid's 0.05 and 0.1, not at 0.2), bin 2 never.
/// A random 30% of cells carry ligand. Competent receivers' programme rate
/// rises with dose, `2 exp(dose · a)`, `a` the number of ligand-carrying
/// contacts. Two cell types in 8×8 patches, named when `named`.
fn world(seed: u64, dose: f64, named: bool) -> World {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = SIDE * SIDE;
    let coords: Vec<Option<(f32, f32)>> = (0..n)
        .map(|i| Some(((i % SIDE) as f32 * 10.0, (i / SIDE) as f32 * 10.0)))
        .collect();
    let mut pairs = Vec::new();
    for i in 0..n {
        if i % SIDE + 1 < SIDE {
            pairs.push((i, i + 1));
        }
        if i / SIDE + 1 < SIDE {
            pairs.push((i, i + SIDE));
        }
    }
    let section: Vec<Box<str>> = vec!["s".into(); n];
    let identity_bin: Vec<usize> = (0..n).map(|i| (i * 7 + i / SIDE) % 3).collect();
    let ty: Vec<u32> = (0..n)
        .map(|i| (((i % SIDE) / 8 + (i / SIDE) / 8) % 2) as u32)
        .collect();
    let identity = Mat::from_fn(4, n, |_, _| {
        let e: f64 = StandardNormal.sample(&mut rng);
        e as f32
    });
    let log1p_depth = vec![(500.0f32).ln_1p(); n];
    let mut tissue = Tissue::build(
        &TissueInputs {
            coords: &coords,
            section: &section,
            pairs: &pairs,
            identity: &identity,
            identity_bin: &identity_bin,
            log1p_depth: &log1p_depth,
            cell_type: Some(&ty),
        },
        &TissueOpts {
            folds: 4,
            tile: 6.0,
            ..TissueOpts::default()
        },
    )
    .unwrap();
    if named {
        tissue = tissue.with_cell_type_names(vec!["epi".into(), "fib".into()]);
    }
    let receptor: Vec<f64> = identity_bin
        .iter()
        .map(|&b| match b {
            0 => 1.0,
            1 => (rng.random::<f64>() < 0.15) as u8 as f64,
            _ => 0.0,
        })
        .collect();
    let ligand: Vec<f64> = (0..n)
        .map(|_| if rng.random::<f64>() < 0.3 { 3.0 } else { 0.0 })
        .collect();
    let response: Vec<(u32, f32)> = (0..n)
        .filter_map(|i| {
            let a = tissue.contacts.adj[i]
                .iter()
                .filter(|&&j| ligand[j as usize] > 0.0)
                .count() as f64;
            let competent = identity_bin[i] < 2;
            let rate = 2.0 * if competent { (dose * a).exp() } else { 1.0 };
            let x = poisson(&mut rng, rate);
            (x > 0.0).then_some((i as u32, x))
        })
        .collect();
    let noise: Vec<Vec<(u32, f32)>> = (0..30)
        .map(|g| {
            let rate = 0.2 * 1.15f64.powi(g);
            (0..n)
                .filter_map(|i| {
                    let x = poisson(&mut rng, rate);
                    (x > 0.0).then_some((i as u32, x))
                })
                .collect()
        })
        .collect();
    World {
        tissue,
        ligand,
        receptor,
        response,
        noise,
    }
}

fn data<'a>(w: &'a World, targets: &'a [(f64, &'a [(u32, f32)])]) -> PairData<'a> {
    PairData {
        ligand: "L",
        receptor: "R",
        ligand_counts: &w.ligand,
        receptor_counts: &w.receptor,
        targets,
        reweighted: None,
        n_removed: 0,
    }
}

fn opts(competences: &'static [f64], curve: bool) -> EffectOpts {
    EffectOpts {
        threshold: None,
        competences,
        curve,
        ..EffectOpts::default()
    }
}

// * Competence grid

/// Averaging over the grid is Rubin's rules over the single-threshold
/// estimates, so the spread between thresholds widens the interval.
#[test]
fn competence_grid_combines_the_single_estimates() {
    use crate::lr_activity::ligand_effect::exposure::competence;
    let w = world(1, 0.15, false);
    let targets = [(1.0, w.response.as_slice())];
    let pick =
        |o: EffectOpts| estimate_pair(&w.tissue, &data(&w, &targets), &o).unwrap()[0].clone();
    static GRID: [f64; 2] = [0.05, 0.2];
    let both = pick(opts(&GRID, false));
    // The grid is one joint fit; its per-threshold estimates, combined by
    // Rubin's rules, are the reported one.
    let detected: Vec<bool> = w.receptor.iter().map(|&r| r > 0.0).collect();
    let levels: Vec<Vec<bool>> = GRID
        .iter()
        .map(|&c| competence(&w.tissue.identity_bin, &detected, c))
        .collect();
    let (x, _) = Exposure::build(&w.tissue, &w.ligand, &opts(&[0.1], false));
    let r = Response::build(&w.tissue, &targets);
    let f = PairFrame {
        exposure: &x,
        response: &r,
    };
    let per = effects_on(
        Rows::cells(&w.tissue).pooled(),
        f,
        &levels,
        Sensitivity::ALL,
        &opts(&GRID, false),
    )
    .unwrap();
    let (ja, jb) = (
        per.levels[0].scopes[0].competent,
        per.levels[1].scopes[0].competent,
    );
    let want = combine(&[ja, jb]);
    let got = both.effects.competent;
    assert!((got.theta - want.theta).abs() < 1e-9, "{got:?} vs {want:?}");
    assert!((got.se - want.se).abs() < 1e-9);
    let within = (0.5 * (ja.se.powi(2) + jb.se.powi(2))).sqrt();
    assert!(got.se >= within - 1e-12);
    // Competence differs across the grid: bin 1 is in at 0.05, out at 0.2.
    assert_ne!(ja.n, jb.n);
    // Each threshold's estimate from the joint fit stays close to its own
    // single-threshold fit: the joint models nest the single ones.
    let a = pick(opts(&[0.05], false)).effects.competent;
    let b = pick(opts(&[0.2], false)).effects.competent;
    for (j, s) in [(ja, a), (jb, b)] {
        assert_eq!(j.n, s.n);
        assert!(
            (j.theta - s.theta).abs() < 0.25 * s.se,
            "joint {j:?} single {s:?}"
        );
    }
}

// * Automatic draws

#[test]
fn draws_stop_when_monte_carlo_error_is_small() {
    let same = vec![est(0.3, 0.1, 30.0); MIN_DRAWS];
    assert!(
        !enough_draws(&same[..MIN_DRAWS - 1]),
        "fewer than the minimum"
    );
    assert!(enough_draws(&same), "no spread between draws");
    // Spread sd 0.1 against within-SE 0.1: sd/√R < 0.01 needs R > 100.
    let spread: Vec<Estimate> = (0..MAX_DRAWS)
        .map(|d| est(0.3 + if d % 2 == 0 { 0.1 } else { -0.1 }, 0.1, 30.0))
        .collect();
    for r in MIN_DRAWS..MAX_DRAWS {
        assert!(!enough_draws(&spread[..r]), "stopped at {r}");
    }
    assert!(enough_draws(&spread), "the cap");
    // Small spread beside the SE: sd 0.002, sd/√3 ≈ 0.0012 < 0.01.
    let small = [
        est(0.300, 0.1, 30.0),
        est(0.302, 0.1, 30.0),
        est(0.298, 0.1, 30.0),
    ];
    assert!(enough_draws(&small));
    let nan = vec![Estimate::nan(10, 2); MIN_DRAWS];
    assert!(
        enough_draws(&nan),
        "not estimable: more draws will not help"
    );
}

/// Pseudobulk with automatic draws runs to a finite headline.
#[test]
fn pseudobulk_with_automatic_draws() {
    let w = world(2, 0.3, false);
    let targets = [(1.0, w.response.as_slice())];
    let o = EffectOpts {
        pseudobulk: 4,
        draws: AUTO_DRAWS,
        ..opts(&[0.1], false)
    };
    let r = &estimate_pair(&w.tissue, &data(&w, &targets), &o).unwrap()[0];
    assert!(
        r.effects.competent.theta.is_finite(),
        "{:?}",
        r.effects.competent
    );
    assert!(r.effects.competent.z() > 3.0, "{:?}", r.effects.competent);
}

// * Threshold–response curve

#[test]
fn curve_thresholds_are_increasing_and_leave_both_arms() {
    let a: Vec<f64> = (0..400)
        .map(|i| if i < 100 { 0.0 } else { (i % 10) as f64 + 1.0 })
        .collect();
    let all = vec![true; a.len()];
    let taus = curve_taus(&a, 0.0, &all, 20);
    assert_eq!(taus[0], (0.0, 0.0), "τ = 0 first");
    assert!(taus.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].1));
    assert!(taus.len() >= 3, "{taus:?}");
    for &(tau, q) in &taus {
        let above = a.iter().filter(|&&v| v > tau).count();
        assert!(above >= 20 && a.len() - above >= 20);
        assert!((0.0..=1.0).contains(&q));
    }
    // A headline outside the quartiles joins the grid.
    let taus = curve_taus(&a, 9.0, &all, 20);
    assert!(taus.iter().any(|&(t, _)| t == 9.0));
    // A strict arm minimum drops the high thresholds.
    let few = curve_taus(&a, 0.0, &all, 250);
    assert!(few
        .iter()
        .all(|&(t, _)| a.iter().filter(|&&v| v > t).count() >= 250));
}

#[test]
fn sup_t_critical_value_lies_between_pointwise_and_bonferroni() {
    let mut rng = SmallRng::seed_from_u64(9);
    let rows = 200;
    let indep: Vec<Vec<f64>> = (0..rows)
        .map(|_| {
            (0..4)
                .map(|_| {
                    let e: f64 = StandardNormal.sample(&mut rng);
                    e
                })
                .collect()
        })
        .collect();
    let c4 = sup_t_crit(&indep, 4, 1);
    // Šidák for four independent: 2.49.
    assert!((c4 - 2.49).abs() < 0.12, "independent {c4}");
    let same: Vec<Vec<f64>> = indep.iter().map(|r| vec![r[0]; 4]).collect();
    let c1 = sup_t_crit(&same, 4, 1);
    assert!((c1 - Z_975).abs() < 0.12, "identical columns {c1}");
    assert!(sup_t_crit(&[vec![0.0, 0.0]], 2, 1).is_nan());
}

/// On a planted dose response the curve rises with τ, every band holds
/// its pointwise interval, and the headline is one of its points.
#[test]
fn curve_rises_with_dose_and_band_covers_pointwise() {
    let w = world(3, 0.25, false);
    let targets = [(1.0, w.response.as_slice())];
    let r = &estimate_pair(&w.tissue, &data(&w, &targets), &opts(&[0.1], true)).unwrap()[0];
    let c = &r.curve;
    assert!(c.len() >= 3, "{c:?}");
    assert!(c.windows(2).all(|w| w[0].tau < w[1].tau));
    for p in c {
        assert!(p.effect.theta.is_finite(), "{p:?}");
        assert!(p.band_lo <= p.lo && p.band_hi >= p.hi, "{p:?}");
        assert!(
            p.band_hi > p.hi,
            "a band over several points is wider: {p:?}"
        );
    }
    let head = c
        .iter()
        .find(|p| p.tau == r.threshold)
        .expect("headline on the curve");
    assert!((head.effect.theta - r.effects.competent.theta).abs() < 1e-9);
    for w in c.windows(2) {
        assert!(
            w[1].effect.theta > w[0].effect.theta - w[0].effect.se,
            "not rising: {c:?}"
        );
    }
    assert!(c.last().unwrap().effect.theta > c[0].effect.theta);
}

// * Decoy programme

#[test]
fn decoy_genes_are_matched_by_expression_and_used_once() {
    let n = 100;
    let sf = vec![1.0; n];
    let gene =
        |count: usize, x: f32| -> Vec<(u32, f32)> { (0..count).map(|r| (r as u32, x)).collect() };
    // Pool means 0.1, 0.2, 0.4, 0.8 (count rows of 1 over 100 rows).
    let pool_genes: Vec<Vec<(u32, f32)>> = [10, 20, 40, 80].iter().map(|&c| gene(c, 1.0)).collect();
    let pool = DecoyPool::new(&sf, pool_genes.iter().map(|g| g.as_slice()).collect());
    let t1 = gene(38, 1.0); // 0.38: nearest 0.4
    let t2 = gene(42, 1.0); // 0.42: nearest 0.4 is used, then 0.2 or 0.8 (log 0.42→0.2 is 0.74, →0.8 is 0.64)
    let targets = [(2.0, t1.as_slice()), (-1.0, t2.as_slice())];
    let d = pool.matched(&sf, &targets);
    assert_eq!(d.len(), 2);
    assert_eq!(
        (d[0].0, d[0].1.len()),
        (2.0, 40),
        "heaviest target first, same weight"
    );
    assert_eq!(
        (d[1].0, d[1].1.len()),
        (-1.0, 80),
        "nearest unused in log mean"
    );
    // A pool holding fewer than half the programme gives no decoy.
    let small = DecoyPool::new(&sf, vec![pool_genes[0].as_slice()]);
    let three = [
        (1.0, t1.as_slice()),
        (1.0, t1.as_slice()),
        (1.0, t2.as_slice()),
    ];
    assert!(small.matched(&sf, &three).is_empty());
    let picks = decoy_pool_genes(1000, &[1, 2, 3], 7);
    assert_eq!(picks.len(), 199);
    assert!(picks.windows(2).all(|w| w[0] < w[1]) && !picks.contains(&2));
    assert_eq!(picks, decoy_pool_genes(1000, &[1, 2, 3], 7), "seeded");
}

/// Genes unrelated to the ligand make a null decoy beside a real effect;
/// a decoy that responds fails the control.
#[test]
fn decoy_is_null_on_unrelated_genes_and_fails_the_control_when_not() {
    let w = world(4, 0.3, false);
    let programme: Vec<(f64, &[(u32, f32)])> = vec![(1.0, w.response.as_slice()); 3];
    let pool = DecoyPool::new(
        &w.tissue.size_factor,
        w.noise.iter().map(|g| g.as_slice()).collect(),
    );
    let o = opts(&[0.1], false);
    let mut rows = estimate_pair_with(&w.tissue, &data(&w, &programme), &pool, &o).unwrap();
    let r = &rows[0];
    assert_eq!(r.n_decoy_genes, 3);
    assert!(r.effects.competent.z() > 4.0, "{:?}", r.effects.competent);
    assert!(r.decoy_effect.z().abs() < 3.0, "decoy {:?}", r.decoy_effect);
    report(&mut rows, &ReportOpts::default());
    assert!(!rows[0].warnings.contains("decoy"), "{}", rows[0].warnings);

    // The responding programme itself in the pool: its decoy responds.
    let pool = DecoyPool::new(&w.tissue.size_factor, vec![w.response.as_slice(); 3]);
    let mut rows = estimate_pair_with(&w.tissue, &data(&w, &programme), &pool, &o).unwrap();
    report(&mut rows, &ReportOpts::default());
    assert!(rows[0].decoy_q < 0.05);
    assert!(rows[0]
        .warnings
        .contains("a matched decoy programme responds too"));
}

// * Moments

#[test]
fn moments_on_both_scales() {
    let e = est(0.2, 0.05, f64::INFINITY);
    let m = Moments::of(&e);
    assert!((m.lo - (0.2 - Z_975 * 0.05)).abs() < 1e-9);
    assert!((m.pct - 100.0 * (0.2f64.exp() - 1.0)).abs() < 1e-9);
    assert!((m.pct_sd - 100.0 * 0.2f64.exp() * 0.05).abs() < 1e-9);
    assert!((m.pct_lo - 100.0 * m.lo.exp_m1()).abs() < 1e-9);
    assert!((m.pct_hi - 100.0 * m.hi.exp_m1()).abs() < 1e-9);
    assert!(m.pct_lo < m.pct && m.pct < m.pct_hi);
    // Few clusters: the t reference widens the interval.
    let t = Moments::of(&est(0.2, 0.05, 5.0));
    assert!(t.hi - t.lo > m.hi - m.lo);
    assert!(Moments::of(&Estimate::nan(0, 0)).pct.is_nan());
}

// * By receiver cell type

#[test]
fn named_cell_types_add_one_row_each() {
    let w = world(5, 0.15, true);
    let targets = [(1.0, w.response.as_slice())];
    let rows = estimate_pair(&w.tissue, &data(&w, &targets), &opts(&[0.1], false)).unwrap();
    let labels: Vec<&str> = rows.iter().map(|r| &*r.section).collect();
    assert_eq!(labels, ["all", "type:epi", "type:fib"]);
    let n: usize = rows[1..].iter().map(|r| r.n_receivers).sum();
    assert_eq!(n, rows[0].n_receivers, "types partition the receivers");
    for r in &rows[1..] {
        assert!(r.effects.competent.theta.is_finite(), "{}", r.section);
        assert!(r.effects.steps[0].theta.is_nan(), "steps pooled only");
        assert!(r.curve.is_empty() && r.decoy_effect.theta.is_nan());
    }
    // Without names, no type rows.
    let w = world(5, 0.15, false);
    let rows = estimate_pair(&w.tissue, &data(&w, &targets), &opts(&[0.1], false)).unwrap();
    assert_eq!(rows.len(), 1);
}
