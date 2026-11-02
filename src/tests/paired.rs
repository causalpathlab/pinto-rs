use super::*;

fn side(units: &[(&str, &str, f64, f64)]) -> HashMap<Box<str>, (String, UnitScore)> {
    units
        .iter()
        .map(|&(k, sec, llik, count)| (k.into(), (sec.to_string(), UnitScore { llik, count })))
        .collect()
}

#[test]
fn sign_test_matches_the_binomial() {
    // scipy.stats.binomtest(1, 6).pvalue, binomtest(5, 5).pvalue
    assert!((sign_test(1, 6) - 0.21875).abs() < 1e-12);
    assert!((sign_test(5, 5) - 0.0625).abs() < 1e-12);
    assert_eq!(sign_test(0, 0), 1.0);
}

#[test]
fn wilcoxon_is_exact() {
    // scipy.stats.wilcoxon([1, 2, 3, -0.5, 4]).pvalue
    let p = wilcoxon(&[1.0, 2.0, 3.0, -0.5, 4.0]).unwrap();
    assert!((p - 0.125).abs() < 1e-12, "{p}");
    // Symmetric in sign, and zeros dropped.
    assert_eq!(p, wilcoxon(&[-1.0, -2.0, -3.0, 0.5, -4.0, 0.0]).unwrap());
    assert!(wilcoxon(&[1.0]).is_none());
    // Ties take mid-ranks and stay a valid probability.
    let t = wilcoxon(&[1.0, 2.0, 2.0, -2.0, 3.0, 4.0]).unwrap();
    assert!(t > 0.0 && t <= 1.0);
}

#[test]
fn donors_are_the_replicates() {
    // Two sections of donor D1 (mapped), one of D2; A is better on D1, worse on D2.
    let a = side(&[
        ("p1", "GSM1_d1inf", -100.0, 10.0),
        ("p2", "GSM2_d1t1", -90.0, 10.0),
        ("p3", "GSM3_d2", -50.0, 10.0),
        ("p4", "GSM3_d2", -60.0, 10.0),
    ]);
    let b = side(&[
        ("p1", "GSM1_d1inf", -110.0, 10.0),
        ("p2", "GSM2_d1t1", -100.0, 10.0),
        ("p3", "GSM3_d2", -40.0, 10.0),
        ("p4", "GSM3_d2", -60.0, 10.0),
        ("p5", "GSM3_d2", -1.0, 1.0),
    ]);
    let map = vec![
        ("d1inf".to_string(), "D1".to_string()),
        ("d1t1".to_string(), "D1".to_string()),
    ];
    let r = paired(&[], &[], &a, &b, &map, 200, 1).unwrap();
    assert_eq!((r.n_units, r.n_only_b, r.n_donors), (4, 1, 2));
    let d1 = r.donors.iter().find(|d| d.donor == "D1").unwrap();
    assert_eq!(d1.sections.len(), 2);
    assert!((d1.delta_per_count - 1.0).abs() < 1e-12);
    let d2 = r.donors.iter().find(|d| d.donor == "GSM3_d2").unwrap();
    assert!((d2.delta_per_count + 0.5).abs() < 1e-12);
    assert_eq!((r.donors_a_better, r.donors_b_better), (1, 1));
    // Pooled: (20 - 10) / 40.
    assert!((r.delta_per_count - 0.25).abs() < 1e-12);
    assert!(r.bootstrap.ci95[0] <= r.bootstrap.ci95[1]);
    assert!(r.notes.iter().any(|n| n.contains("leave-one-donor-out")));
}

#[test]
fn units_split_differently_are_refused() {
    let a = side(&[("p1", "s", -10.0, 10.0)]);
    let b = side(&[("p1", "s", -10.0, 12.0)]);
    let e = paired(&[], &[], &a, &b, &[], 10, 1)
        .unwrap_err()
        .to_string();
    assert!(e.contains("split"), "{e}");
}
