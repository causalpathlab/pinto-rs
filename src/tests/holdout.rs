use super::*;

fn profiles(rows: &[&[f64]]) -> DMatrix<f64> {
    // rows[k] is community k's distribution over features.
    DMatrix::from_fn(rows.len(), rows[0].len(), |k, g| rows[k][g])
}

/// A plan over `g` features, everything fitted and scored, flat null.
struct Fixture {
    fit: Vec<bool>,
    scored: Vec<bool>,
    log_flat: Vec<f64>,
    shares: Vec<f64>,
}

impl Fixture {
    fn new(g: usize, k: usize) -> Self {
        Fixture {
            fit: vec![true; g],
            scored: vec![true; g],
            log_flat: vec![(1.0 / g as f64).ln(); g],
            shares: vec![1.0 / k as f64; k],
        }
    }

    fn plan(&self, split: Split) -> Plan<'_> {
        Plan {
            split,
            log_null: &self.log_flat,
            log_ceiling: &self.log_flat,
            shares: &self.shares,
            agreement: None,
            iters: 100,
        }
    }

    fn profiles(&self, p_t: &DMatrix<f64>) -> BatchProfiles {
        BatchProfiles::new(p_t, &vec![1.0; p_t.ncols()], &self.fit, &self.scored)
    }
}

fn wide(c: &Compact) -> Counts {
    widen(c.iter())
}

/// Every data row matched to the slot of the same index.
fn identity(n: usize) -> Vec<Option<usize>> {
    (0..n).map(Some).collect()
}

#[test]
fn thinning_splits_every_count_and_is_seeded() {
    let rows = [(0, 10.0f32), (1, 3.0), (2, 0.0), (3, 5.0)];
    let (fit, scored) = (vec![true; 4], vec![true; 4]);
    let m = identity(4);
    let split = Split::Thinning(0.5);
    let (f, t) = split_cell(rows.iter().copied(), &m, &split, &fit, &scored, 1);
    let total = |v: &Compact| v.iter().map(|o| f64::from(o.1)).sum::<f64>();
    assert_eq!(total(&f) + total(&t), 18.0);
    assert!(f.iter().chain(&t).all(|o| o.1 > 0.0 && o.0 != 2));
    assert_eq!(
        split_cell(rows.iter().copied(), &m, &split, &fit, &scored, 1),
        (f, t)
    );
}

#[test]
fn the_split_does_not_depend_on_which_features_a_model_has() {
    // Model A matches rows 0..4; model B lacks row 1. On the rows both have,
    // the two must split alike, or their scores could not be paired.
    let rows = [(0, 12.0f32), (1, 9.0), (2, 7.0), (3, 30.0)];
    let all = vec![true; 4];
    let a = identity(4);
    let b = vec![Some(0), None, Some(1), Some(2)];
    let split = Split::Thinning(0.5);
    let (fa, ta) = split_cell(rows.iter().copied(), &a, &split, &all, &all, 9);
    let (fb, tb) = split_cell(rows.iter().copied(), &b, &split, &all, &all, 9);
    let on = |v: &Compact, slot: u32| v.iter().find(|o| o.0 == slot).map(|o| o.1);
    for (sa, sb) in [(0, 0), (2, 1), (3, 2)] {
        assert_eq!(on(&fa, sa), on(&fb, sb));
        assert_eq!(on(&ta, sa), on(&tb, sb));
    }
}

#[test]
fn a_pairs_parts_are_its_cells_parts() {
    let parts: Vec<CellParts> = vec![
        (vec![(0, 2.0), (1, 1.0)], vec![(1, 3.0)]),
        (vec![(1, 4.0)], vec![(0, 1.0), (1, 2.0)]),
    ];
    let (fit, test) = unit_parts(&parts, 0, Some(1));
    assert_eq!(fit, vec![(0, 2.0), (1, 5.0)]);
    assert_eq!(test, vec![(0, 1.0), (1, 5.0)]);
    assert_eq!(
        unit_parts(&parts, 1, None),
        (vec![(1, 4.0)], vec![(0, 1.0), (1, 2.0)])
    );
}

#[test]
fn feature_ablation_fits_on_the_rest_and_scores_the_hidden() {
    let rows = [(0, 30.0f32), (1, 5.0), (2, 28.0), (3, 6.0)];
    let fit = vec![true, true, false, false];
    let scored = vec![false, false, true, true];
    let (f, t) = split_cell(
        rows.iter().copied(),
        &identity(4),
        &Split::Ablate,
        &fit,
        &scored,
        0,
    );
    assert_eq!(f, vec![(0, 30.0), (1, 5.0)]);
    assert_eq!(t, vec![(2, 28.0), (3, 6.0)]);

    // Fitted to community 0 on features 0-1, which predicts feature 2 over 3.
    let p_t = profiles(&[&[0.4, 0.1, 0.4, 0.1], &[0.1, 0.4, 0.1, 0.4]]);
    let mut fx = Fixture::new(4, 2);
    fx.fit = fit;
    fx.scored = scored;
    let s = score_unit(
        &wide(&f),
        &wide(&t),
        &fx.profiles(&p_t),
        &fx.plan(Split::Ablate),
    );
    assert_eq!((s.n_fit, s.n_test), (35.0, 34.0));
    assert!(s.loglik > s.loglik_pooled);
}

#[test]
fn the_score_is_the_mixture_renormalised_over_the_scored_features() {
    let p_t = profiles(&[&[0.5, 0.5, 0.0], &[0.0, 0.5, 0.5]]);
    let all = vec![true; 3];
    let prof = BatchProfiles::new(&p_t, &[1.0; 3], &all, &all);
    let ll = prof.loglik(&[(0, 2.0), (1, 1.0)], &[1.0, 0.0]);
    assert!((ll - 3.0 * 0.5f64.ln()).abs() < 1e-12);
    // Scoring only feature 1 renormalises onto it: probability 1.
    let only1 = vec![false, true, false];
    let prof = BatchProfiles::new(&p_t, &[1.0; 3], &all, &only1);
    assert!(prof.loglik(&[(1, 4.0)], &[0.5, 0.5]).abs() < 1e-12);
    // A feature the mixture gives no mass is floored, not -inf.
    let prof = BatchProfiles::new(&p_t, &[1.0; 3], &all, &all);
    assert!(prof.loglik(&[(2, 1.0)], &[1.0, 0.0]).is_finite());
}

#[test]
fn proportions_fitted_on_part_of_the_axis_become_weights_on_the_whole() {
    // Community 0 has little mass on the fit features, so the same fitted share
    // means more of it on the full profile.
    let p_t = profiles(&[&[0.1, 0.9], &[0.5, 0.5]]);
    let prof = BatchProfiles::new(&p_t, &[1.0, 1.0], &[true, false], &[false, true]);
    let w = prof.weights(&[0.5, 0.5]);
    // π / m = (0.5/0.1, 0.5/0.5) = (5, 1) → (5/6, 1/6)
    assert!((w[0] - 5.0 / 6.0).abs() < 1e-12, "{w:?}");
}

#[test]
fn a_unit_with_nothing_to_score_keeps_weights_on_the_full_profiles() {
    let p_t = profiles(&[&[0.1, 0.9], &[0.5, 0.5]]);
    let mut fx = Fixture::new(2, 2);
    fx.fit = vec![true, false];
    fx.scored = vec![false, true];
    let prof = fx.profiles(&p_t);
    let s = score_unit(&[(0, 10.0)], &[], &prof, &fx.plan(Split::Ablate));
    assert_eq!(s.n_test, 0.0);
    let mut pi = vec![0.0; 2];
    crate::util::profile_match::em_proportions(&[(0, 10.0)], &prof.p_fit, 100, &mut pi);
    assert_eq!(
        s.pi,
        prof.weights(&pi),
        "weights on the full profiles, not π"
    );
}

#[test]
fn delta_shifts_a_feature_and_keeps_rows_normalised() {
    let p_t = profiles(&[&[0.5, 0.5]]);
    let all = vec![true; 2];
    let prof = BatchProfiles::new(&p_t, &[3.0, 1.0], &all, &all);
    assert!((prof.q[(0, 0)] - 0.75).abs() < 1e-12);
    assert!((prof.q.row(0).sum() - 1.0).abs() < 1e-12);
}

#[test]
fn a_delta_round_moves_toward_what_was_observed() {
    let mut delta = vec![1.0, 1.0, 1.0];
    let fit = vec![true, true, false];
    update_delta(&mut delta, &[99.0, 9.0, 50.0], &[49.0, 49.0, 1.0], &fit);
    assert!(delta[0] > 1.0 && delta[1] < 1.0, "{delta:?}");
    assert_eq!(delta[2], 1.0, "a feature outside the fit keeps δ = 1");
    assert!(
        ((delta[0] + delta[1]) / 2.0 - 1.0).abs() < 1e-12,
        "mean 1 over the fit"
    );
}

#[test]
fn a_unit_from_one_community_beats_the_pooled_profile() {
    let p_t = profiles(&[&[0.45, 0.45, 0.05, 0.05], &[0.05, 0.05, 0.45, 0.45]]);
    let fx = Fixture::new(4, 2);
    let rows = [(0, 2.0f32), (2, 40.0), (3, 38.0)];
    let (f, t) = split_cell(
        rows.iter().copied(),
        &identity(4),
        &Split::Thinning(0.5),
        &fx.fit,
        &fx.scored,
        7,
    );
    let s = score_unit(
        &wide(&f),
        &wide(&t),
        &fx.profiles(&p_t),
        &fx.plan(Split::Thinning(0.5)),
    );
    assert!(s.n_test > 0.0 && s.n_fit > 0.0);
    assert!(
        s.loglik > s.loglik_pooled,
        "{} vs {}",
        s.loglik,
        s.loglik_pooled
    );
    assert!(
        s.loglik > s.loglik_null,
        "{} vs flat {}",
        s.loglik,
        s.loglik_null
    );
    assert!(s.pi[1] > 0.9, "{:?}", s.pi);
    assert_eq!(s.n_fit + s.n_test, 80.0);
}

#[test]
fn agreement_is_scored_on_the_listed_features() {
    let p_t = profiles(&[&[0.6, 0.3, 0.1]]);
    let fx = Fixture::new(3, 1);
    let prof = fx.profiles(&p_t);
    let mut plan = fx.plan(Split::Thinning(0.5));
    plan.agreement = Some(Agreement::new(vec![0, 1, 2], 3));
    let counts = [(0, 300.0), (1, 150.0), (2, 50.0)];
    let s = score_unit(&counts, &counts, &prof, &plan);
    assert!(s.spearman > 0.9, "{}", s.spearman);
    let plan = fx.plan(Split::Thinning(0.5));
    assert!(score_unit(&[(0, 6.0)], &[(0, 6.0)], &prof, &plan)
        .spearman
        .is_nan());
}

#[test]
fn a_unit_without_counts_is_not_scored() {
    let p_t = profiles(&[&[0.5, 0.5]]);
    let fx = Fixture::new(2, 1);
    let s = score_unit(&[], &[], &fx.profiles(&p_t), &fx.plan(Split::Thinning(0.5)));
    assert_eq!(s.n_test, 0.0);
    assert!(s.pi.iter().all(|&p| p == 0.0));
}

#[test]
fn a_composition_is_normalised_over_the_scored_features() {
    let lc = log_composition(&[2.0, 2.0, 4.0], &[true, true, false]);
    assert!((lc[0] - 0.5f64.ln()).abs() < 1e-12);
    assert!(lc[2].is_nan());
}

#[test]
fn a_manifest_with_nothing_scored_still_reads() {
    let s = SectionScore {
        section: "s".into(),
        n_scored: 0,
        eval_count: 0.0,
        eval_llik_per_count: f64::NAN,
        eval_null_llik_per_count: f64::NAN,
        eval_pooled_k1_llik_per_count: f64::NAN,
        eval_ceiling_llik_per_count: f64::NAN,
    };
    let json = serde_json::to_string(&s).unwrap();
    assert!(json.contains("null"));
    let back: SectionScore = serde_json::from_str(&json).unwrap();
    assert!(back.eval_llik_per_count.is_nan());
}

#[test]
fn cell_seeds_differ_between_neighbours() {
    assert_ne!(cell_seed(42, 0), cell_seed(42, 1));
    assert_eq!(cell_seed(42, 5), cell_seed(42, 5));
    assert_ne!(cell_seed(42, 5), cell_seed(43, 5));
}
