use crate::simulate::pairs::{
    auroc, average_precision, read_vectors, score_pairs, PairOptions, PairScore, SpotLayout,
};
use crate::simulate::score::TruthSpots;
use crate::util::common::Mat;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

const SIDE: usize = 24;

/// Two donors, one section each, on a 24 x 24 grid 100 um apart. Shared
/// domains A, B, C in vertical stripes of 8 columns; donor 1 has a patch P,
/// an 8 x 8 block inside B.
fn toy() -> (TruthSpots, SpotLayout) {
    let (mut domain, mut donor, mut section, mut xy) = (vec![], vec![], vec![], vec![]);
    for d in 0..2 {
        for r in 0..SIDE {
            for c in 0..SIDE {
                let k = if d == 1 && r < 8 && (8..16).contains(&c) {
                    3
                } else {
                    c / 8
                };
                domain.push(k);
                donor.push(d);
                section.push(d);
                xy.push([r as f32 * 100.0, c as f32 * 100.0]);
            }
        }
    }
    let t = TruthSpots {
        cells: (0..domain.len()).map(|i| format!("c{i}").into()).collect(),
        domain,
        donor,
        domain_labels: vec!["A".into(), "B".into(), "C".into(), "P".into()],
        domain_owner: vec![None, None, None, Some(1)],
        n_donors: 2,
    };
    (t, SpotLayout { section, xy })
}

fn one_hot(n: usize, k: usize, of: impl Fn(usize) -> usize) -> Mat {
    Mat::from_fn(n, k, |i, j| (of(i) == j) as u8 as f32)
}

fn opts() -> PairOptions {
    PairOptions {
        pairs: 20_000,
        seed: 7,
        knn: 6,
    }
}

fn scored(t: &TruthSpots, l: &SpotLayout, x: &Mat) -> PairScore {
    score_pairs(t, l, &t.cells, x, &opts()).unwrap()
}

fn near(x: f64, want: f64, tol: f64, what: &str) {
    assert!((x - want).abs() <= tol, "{what}: {x} vs {want}");
}

// * AUROC and AUPRC

/// Hand-computed: one positive below one negative; ties count a half.
#[test]
fn auroc_matches_hand_computed_cases() {
    // positives 0.9, 0.7; negatives 0.8, 0.6: 3 of 4 pairs ordered
    near(
        auroc(&[0.9, 0.8, 0.7, 0.6], &[true, false, true, false]),
        0.75,
        1e-12,
        "untied",
    );
    // positives 1, 0; negative 1: one tie (1/2) and one loss
    near(
        auroc(&[1.0, 1.0, 0.0], &[true, false, true]),
        0.25,
        1e-12,
        "tied",
    );
    near(
        auroc(&[0.5; 4], &[true, false, true, false]),
        0.5,
        1e-12,
        "all tied",
    );
    near(
        auroc(&[0.1, 0.2, 0.9], &[false, false, true]),
        1.0,
        1e-12,
        "perfect",
    );
    assert!(auroc(&[0.1, 0.2], &[true, true]).is_nan(), "no negative");
}

/// Hand-computed average precision, tied scores taken as one threshold.
#[test]
fn average_precision_matches_hand_computed_cases() {
    // ranks: P(1/1) at recall 1/2, P(2/3) at recall 1
    near(
        average_precision(&[0.9, 0.8, 0.7, 0.6], &[true, false, true, false]),
        0.5 + 0.5 * 2.0 / 3.0,
        1e-12,
        "untied",
    );
    // threshold 1: recall 1/2, precision 1/2; threshold 0: recall 1, precision 2/3
    near(
        average_precision(&[1.0, 1.0, 0.0], &[true, false, true]),
        0.25 + 0.5 * 2.0 / 3.0,
        1e-12,
        "tied",
    );
    near(
        average_precision(&[0.5; 4], &[true, false, true, false]),
        0.5,
        1e-12,
        "all tied",
    );
    assert!(average_precision(&[0.1], &[false]).is_nan(), "no positive");
}

// * The four pair sets

/// The truth's own domain one-hot vectors separate every pair set.
#[test]
fn the_truth_domains_score_one_everywhere() {
    let (t, l) = toy();
    let x = one_hot(t.cells.len(), 4, |i| t.domain[i]);
    let s = scored(&t, &l, &x);
    for (name, set) in [
        ("cross", &s.cross_donor),
        ("within", &s.within_donor),
        ("spatial", &s.spatial),
        ("patch", &s.patches[0].pairs),
    ] {
        assert_eq!(set.auroc, 1.0, "{name}");
        assert_eq!(set.auprc, 1.0, "{name}");
        assert!(set.n_pairs > 0 && set.n_positive > 0, "{name}");
        near(
            set.prevalence,
            set.n_positive as f64 / set.n_pairs as f64,
            1e-12,
            name,
        );
    }
    assert_eq!(s.patches[0].closest_auroc, 1.0);
    assert_eq!(s.donor_gap, 0.0);
}

/// Vectors that follow the donor: no better than chance across donors,
/// and the within-donor reference at least as high.
#[test]
fn donor_vectors_fail_across_donors() {
    let (t, l) = toy();
    let x = one_hot(t.cells.len(), 2, |i| t.donor[i]);
    let s = scored(&t, &l, &x);
    near(s.cross_donor.auroc, 0.5, 0.02, "cross");
    assert!(s.within_donor.auroc >= s.cross_donor.auroc - 1e-12);
    // within a donor every pair is alike: chance again
    near(s.within_donor.auroc, 0.5, 0.02, "within");
}

/// Random vectors are at chance everywhere.
#[test]
fn random_vectors_are_at_chance() {
    let (t, l) = toy();
    let mut r = SmallRng::seed_from_u64(11);
    let x = Mat::from_fn(t.cells.len(), 16, |_, _| StandardNormal.sample(&mut r));
    let s = scored(&t, &l, &x);
    near(s.cross_donor.auroc, 0.5, 0.03, "cross");
    near(s.within_donor.auroc, 0.5, 0.03, "within");
    near(s.spatial.auroc, 0.5, 0.06, "spatial");
    near(s.patches[0].pairs.auroc, 0.5, 0.08, "patch");
}

/// The patch folded into B, its host domain: the shared sets are unhurt,
/// the patch set falls, and against B alone the patch is at chance.
#[test]
fn a_patch_folded_into_its_host_fails_the_patch_set() {
    let (t, l) = toy();
    let x = one_hot(t.cells.len(), 3, |i| match t.domain[i] {
        3 => 1,
        k => k,
    });
    let s = scored(&t, &l, &x);
    assert_eq!(s.cross_donor.auroc, 1.0);
    assert_eq!(s.within_donor.auroc, 1.0);
    let p = &s.patches[0];
    assert_eq!((p.domain.as_str(), p.owner.as_str()), ("P", "D2"));
    assert!(p.pairs.auroc < 0.9, "{}", p.pairs.auroc);
    assert_eq!(p.closest_domain, "B");
    near(p.closest_auroc, 0.5, 1e-12, "closest");
    assert!(s.spatial.auroc < 1.0);
}

/// The same seed draws the same pairs; another seed, others.
#[test]
fn pairs_are_drawn_from_the_seed() {
    let (t, l) = toy();
    let mut r = SmallRng::seed_from_u64(3);
    let x = Mat::from_fn(t.cells.len(), 4, |_, _| StandardNormal.sample(&mut r));
    let a = scored(&t, &l, &x);
    let b = scored(&t, &l, &x);
    assert_eq!(a.cross_donor.auroc, b.cross_donor.auroc);
    assert_eq!(a.patches[0].pairs.auroc, b.patches[0].pairs.auroc);
    let other = PairOptions { seed: 8, ..opts() };
    let c = score_pairs(&t, &l, &t.cells, &x, &other).unwrap();
    assert_ne!(a.cross_donor.auroc, c.cross_donor.auroc);
}

/// Spatial edges join each spot to its nearest neighbours in its own
/// section only: two sections on the same grid share no edge.
#[test]
fn spatial_edges_stay_within_a_section() {
    let (t, l) = toy();
    // one section per donor, so a vector of the donor tells the sections
    // apart: within sections every edge is a tie at cosine 1 (chance),
    // while an edge across them (the same grid point, at distance 0) would
    // be a same-domain pair at cosine 0, below chance
    let donors = one_hot(t.cells.len(), 2, |i| t.donor[i]);
    let s = scored(&t, &l, &donors);
    near(s.spatial.auroc, 0.5, 1e-12, "spatial");
    // 24 x 24 grid, 6 nearest: at least the 4 lattice neighbours per spot
    assert!(s.spatial.n_pairs >= 2 * 2 * SIDE * (SIDE - 1));
    assert!(s.spatial.n_pairs <= 2 * SIDE * SIDE * 6);
}

/// Cells join the truth by name, in any order; truth spots without a
/// vector are left out and counted, and an unknown or repeated cell is
/// refused.
#[test]
fn vectors_join_the_truth_by_name() {
    let (t, l) = toy();
    let n = t.cells.len();
    let keep: Vec<usize> = (0..n).rev().filter(|i| i % 10 != 0).collect();
    let cells: Vec<Box<str>> = keep.iter().map(|&i| t.cells[i].clone()).collect();
    let x = one_hot(keep.len(), 4, |r| t.domain[keep[r]]);
    let s = score_pairs(&t, &l, &cells, &x, &opts()).unwrap();
    assert_eq!((s.n_spots, s.n_scored), (n, keep.len()));
    assert_eq!(s.cross_donor.auroc, 1.0);
    assert_eq!(s.spatial.auroc, 1.0);
    let mut twice = cells.clone();
    twice[1] = twice[0].clone();
    assert!(score_pairs(&t, &l, &twice, &x, &opts()).is_err());
    let mut nan = x.clone();
    nan[(2, 0)] = f32::NAN;
    let e = format!(
        "{:#}",
        score_pairs(&t, &l, &cells, &nan, &opts()).unwrap_err()
    );
    assert!(e.contains(cells[2].as_ref()), "{e}");
    let none: Vec<Box<str>> = (0..keep.len()).map(|i| format!("x{i}").into()).collect();
    assert!(score_pairs(&t, &l, &none, &x, &opts()).is_err());
}

/// A table is read by its `cell` column and every other column as a
/// number; a column that is not numeric is refused by name.
#[test]
fn vector_tables_are_read_by_their_cell_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.tsv").to_string_lossy().to_string();
    std::fs::write(&path, "x\tcell\ty\n1\tc1\t0.5\n-2\tc2\t3e-1\n").unwrap();
    let (cells, x) = read_vectors(&path, "final").unwrap();
    assert_eq!(cells, ["c1".into(), "c2".into()] as [Box<str>; 2]);
    assert_eq!(x, Mat::from_row_slice(2, 2, &[1.0, 0.5, -2.0, 0.3]));
    let bad = dir.path().join("b.tsv").to_string_lossy().to_string();
    std::fs::write(&bad, "cell\tlabel\ty\nc1\tT\t1\n").unwrap();
    let e = format!("{:#}", read_vectors(&bad, "final").unwrap_err());
    assert!(e.contains("label"), "{e}");
    let none = dir.path().join("n.tsv").to_string_lossy().to_string();
    std::fs::write(&none, "name\ty\nc1\t1\n").unwrap();
    assert!(read_vectors(&none, "final").is_err());
}

/// On a written simulation: its truth run at level `domain` gives 1 on
/// every set, read through the run, its sections and coordinates.
#[test]
fn a_simulations_truth_run_scores_one() {
    use crate::simulate::run::{run_simulate, truth_run_path, SimulateArgs};
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim").to_string_lossy().to_string();
    let set = ["grid_side=12", "n_genes=40", "programme_genes=6", "seed=5"];
    run_simulate(&SimulateArgs {
        scenario: "L2".into(),
        set: set.iter().map(|s| s.to_string()).collect(),
        coords: Vec::new(),
        coord_column_names: vec!["row_um".into(), "col_um".into()],
        out: out.as_str().into(),
    })
    .unwrap();
    let truth = truth_run_path(&out);
    let t = TruthSpots::read(&truth).unwrap();
    let l = SpotLayout::read(&truth, &t).unwrap();
    assert!(l.section.iter().max().unwrap() + 1 >= t.n_donors);
    let (cells, x) = read_vectors(&truth, "domain").unwrap();
    let s = score_pairs(&t, &l, &cells, &x, &opts()).unwrap();
    assert_eq!(s.n_scored, t.cells.len());
    assert_eq!(s.cross_donor.auroc, 1.0);
    assert_eq!(s.within_donor.auroc, 1.0);
    assert_eq!(s.spatial.auroc, 1.0);
    assert_eq!(s.patches.len(), 1);
    assert_eq!(s.patches[0].pairs.auroc, 1.0);
}
