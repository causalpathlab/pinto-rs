use crate::simulate::score::{score, TruthSpots};

/// Two donors, two shared domains (A, B) and a patch (P) owned by donor 1.
/// Donor 0: 10 A, 10 B. Donor 1: 10 A, 6 B, 4 P.
fn truth() -> TruthSpots {
    let mut domain = Vec::new();
    let mut donor = Vec::new();
    for (d, counts) in [(0usize, [10, 10, 0]), (1, [10, 6, 4])] {
        for (k, &n) in counts.iter().enumerate() {
            domain.extend(std::iter::repeat_n(k, n));
            donor.extend(std::iter::repeat_n(d, n));
        }
    }
    TruthSpots {
        cells: (0..domain.len()).map(|i| format!("c{i}").into()).collect(),
        domain,
        donor,
        domain_labels: vec!["A".into(), "B".into(), "P".into()],
        domain_owner: vec![None, None, Some(1)],
        n_donors: 2,
    }
}

/// Each shared domain's dependence, rounded to 12 places.
fn by_domain(s: &crate::simulate::score::Score) -> Vec<Option<f64>> {
    s.donor_dependence_by_domain
        .iter()
        .map(|d| d.dependence.map(|x| (x * 1e12).round() / 1e12))
        .collect()
}

fn labels_from(t: &TruthSpots, f: impl Fn(usize) -> i64) -> Vec<(Box<str>, i64)> {
    t.cells
        .iter()
        .enumerate()
        .map(|(i, c)| (c.clone(), f(i)))
        .collect()
}

#[test]
fn the_truth_itself_scores_perfectly() {
    let t = truth();
    let s = score(&t, &labels_from(&t, |i| t.domain[i] as i64)).unwrap();
    assert!((s.shared_ari - 1.0).abs() < 1e-12);
    assert_eq!(s.donor_dependence, 0.0);
    assert_eq!(by_domain(&s), [Some(0.0), Some(0.0)]);
    let p = &s.patches[0];
    assert_eq!(
        (p.f1, p.precision, p.recall, p.owner_share),
        (1.0, 1.0, 1.0, 1.0)
    );
}

/// Labels that follow the donor depend wholly on the donor within each
/// domain.
#[test]
fn labels_that_follow_the_donor_fail_test_one() {
    let t = truth();
    let s = score(&t, &labels_from(&t, |i| t.donor[i] as i64)).unwrap();
    assert_eq!(by_domain(&s), [Some(1.0), Some(1.0)]);
    assert!(
        (s.donor_dependence - 1.0).abs() < 1e-12,
        "{}",
        s.donor_dependence
    );
    assert!(s.shared_ari < 0.1);
}

/// The patch merged into domain B: test two fails (low precision), test
/// one is unhurt.
#[test]
fn a_patch_folded_into_a_shared_domain_fails_test_two() {
    let t = truth();
    let s = score(
        &t,
        &labels_from(&t, |i| {
            if t.domain[i] == 2 {
                1
            } else {
                t.domain[i] as i64
            }
        }),
    )
    .unwrap();
    assert_eq!(s.donor_dependence, 0.0);
    // test one looks at shared spots only: the folded patch does not count
    assert!((s.shared_ari - 1.0).abs() < 1e-12, "{}", s.shared_ari);
    let p = &s.patches[0];
    assert!(p.precision < 0.3, "{}", p.precision);
    assert!(p.f1 < 0.5, "{}", p.f1);
    assert_eq!(p.recall, 1.0);
}

/// Domain A split by donor (its own label in each): A's own dependence is
/// 1 and B's 0, and the pooled dependence is A's donor entropy over the
/// shared domains' total, B adding entropy but no information.
#[test]
fn a_shared_domain_split_by_donor_is_flagged() {
    let t = truth();
    let s = score(
        &t,
        &labels_from(&t, |i| match t.domain[i] {
            0 => 10 + t.donor[i] as i64,
            k => k as i64,
        }),
    )
    .unwrap();
    assert_eq!(by_domain(&s), [Some(1.0), Some(0.0)]);
    // A: 10 + 10 spots, labels set by donor; B: 10 + 6 spots, labels not
    let h = |p: f64| -p * p.ln() - (1.0 - p) * (1.0 - p).ln();
    let want = 20.0 * h(0.5) / (20.0 * h(0.5) + 16.0 * h(10.0 / 16.0));
    assert!(
        (s.donor_dependence - want).abs() < 1e-12,
        "{} vs {want}",
        s.donor_dependence
    );
}

/// Unlabelled spots (−1, or absent) are left out and counted.
#[test]
fn unlabelled_spots_are_left_out() {
    let t = truth();
    let mut l = labels_from(&t, |i| t.domain[i] as i64);
    l[0].1 = -1;
    l.pop();
    let s = score(&t, &l).unwrap();
    assert_eq!(s.n_scored, t.cells.len() - 2);
    assert_eq!(s.donor_dependence, 0.0);
}

/// A domain that truly sits mostly in one donor is not a split: the truth
/// shows no donor dependence, and with one donor there is none.
#[test]
fn a_domain_mostly_in_one_donor_is_not_a_split() {
    // donor 0: 18 A, 10 B; donor 1: 2 A, 10 B
    let domain: Vec<usize> = [vec![0; 18], vec![1; 10], vec![0; 2], vec![1; 10]].concat();
    let donor: Vec<usize> = [vec![0; 28], vec![1; 12]].concat();
    let t = TruthSpots {
        cells: (0..domain.len()).map(|i| format!("c{i}").into()).collect(),
        domain,
        donor,
        domain_labels: vec!["A".into(), "B".into()],
        domain_owner: vec![None, None],
        n_donors: 2,
    };
    let s = score(&t, &labels_from(&t, |i| t.domain[i] as i64)).unwrap();
    assert_eq!(s.donor_dependence, 0.0);
    let one = TruthSpots {
        donor: vec![0; t.cells.len()],
        n_donors: 1,
        ..t
    };
    let s = score(&one, &labels_from(&one, |i| one.domain[i] as i64)).unwrap();
    assert_eq!(s.donor_dependence, 0.0);
    // in one donor a domain has no donor entropy: no dependence to give
    assert_eq!(by_domain(&s), [None, None]);
}

/// A table is read by its `cell` and label columns; empty, NA, -1 and
/// unassigned labels are none. A repeated cell is an error when scored.
#[test]
fn label_tables_mark_unlabelled_spots() {
    use crate::simulate::score::read_labels;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tsv").to_string_lossy().to_string();
    let text = "cell\tcell_type\tscore\nc1\tX\t1\nc2\tNA\t1\nc3\t\t1\nc4\t-1\t1\nc5\tunassigned\t1\nc6\tY\t1\n";
    std::fs::write(&path, text).unwrap();
    let ok = read_labels(&path, "final", "cell_type").unwrap();
    let labelled: Vec<&str> = ok
        .iter()
        .filter(|(_, l)| *l >= 0)
        .map(|(c, _)| c.as_ref())
        .collect();
    assert_eq!(labelled, ["c1", "c6"]);
    assert!(read_labels(&path, "final", "label").is_err());
    let t = truth();
    let mut twice = labels_from(&t, |i| t.domain[i] as i64);
    twice.push(twice[0].clone());
    assert!(score(&t, &twice).is_err());
}

/// The best match is by F1, not by overlap: a small label holding part of
/// the patch beats a large shared label holding more of it.
#[test]
fn the_best_match_is_by_f1_not_overlap() {
    let t = truth();
    // patch spots are the last 4 cells; 3 go to B's label (1), 1 to its own
    let n = t.cells.len();
    let s = score(
        &t,
        &labels_from(&t, |i| match t.domain[i] {
            2 if i == n - 1 => 9,
            2 => 1,
            k => k as i64,
        }),
    )
    .unwrap();
    let p = &s.patches[0];
    // label 9: precision 1, recall 1/4, F1 = 2/5; label 1: 3 of 19, F1 = 6/23
    assert!((p.f1 - 0.4).abs() < 1e-12, "{}", p.f1);
    assert_eq!(p.precision, 1.0);
}

/// An empty truth is refused, not read from the filesystem root.
#[test]
fn an_empty_truth_is_refused() {
    for given in ["", "  "] {
        let Err(e) = TruthSpots::read(given) else {
            panic!("{given:?} was read");
        };
        let e = e.to_string();
        assert!(e.contains("no truth given"), "{e}");
    }
}

/// A simulation is named by its out folder, its truth folder, its truth
/// run's manifest or that run's prefix.
#[test]
fn a_simulation_is_named_any_way_a_run_is() {
    use crate::simulate::score::sim_folder;
    let dir = tempfile::tempdir().unwrap();
    let sim = dir.path().join("sim");
    std::fs::create_dir_all(sim.join("truth")).unwrap();
    std::fs::write(sim.join("truth/truth.pinto.json"), "{}").unwrap();
    let want = sim.to_string_lossy().to_string();
    for given in ["", "/truth", "/truth/truth.pinto.json", "/truth/truth"] {
        assert_eq!(sim_folder(&format!("{want}{given}")), want, "{given:?}");
    }
}

/// Each whitespace-separated field of `line`, as its start and end column.
pub(super) fn spans(line: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, ch) in line.chars().chain(std::iter::once(' ')).enumerate() {
        match (ch == ' ', start) {
            (false, None) => start = Some(i),
            (true, Some(s)) => {
                out.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    out
}

/// A long patch name leaves the columns after it where the header puts
/// them: `patch` starts where the name does, and f1, prec and owner end
/// where their values do.
#[test]
fn the_score_table_keeps_its_columns_past_a_long_patch_name() {
    use crate::simulate::score::{table_header, table_rows, PatchScore, Score};
    let s = Score {
        n_spots: 4,
        n_scored: 4,
        shared_ari: 1.0,
        donor_dependence: 0.0,
        donor_dependence_by_domain: Vec::new(),
        patches: vec![PatchScore {
            domain: "D1_specific_domain_1".into(),
            owner: "D1".into(),
            f1: 1.0,
            precision: 1.0,
            recall: 1.0,
            owner_share: 1.0,
        }],
    };
    let header = spans(&table_header());
    let rows = table_rows("run", &s);
    let row = spans(&rows[0]);
    let show = format!("{}\n{}", table_header(), rows[0]);
    assert_eq!(header[4].0, row[4].0, "patch\n{show}");
    for f in 5..8 {
        assert_eq!(header[f].1, row[f].1, "field {f}\n{show}");
    }
}
