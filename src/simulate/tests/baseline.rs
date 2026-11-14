use crate::simulate::baseline::{
    cluster, domains_per_donor, gene_mean_dispersion, hvg, match_to_reference, normalise_log,
    normalised_dispersion, read_counts, reference_donor, run_baseline, scale, scanpy_pcs,
    BaselineArgs, Method, KNN, N_HVG, N_PCS, SEED,
};
use crate::simulate::harmony::HarmonyParams;
use crate::simulate::score::{read_labels, score, Score, TruthSpots};
use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;
use tempfile::TempDir;

/// genes × spots, from dense rows.
fn csc(rows: &[&[f32]]) -> CscMatrix<f32> {
    CscMatrix::from(&DMatrix::from_fn(rows.len(), rows[0].len(), |g, s| {
        rows[g][s]
    }))
}

/// Each spot is scaled to 1e4 counts and then log1p'd; an empty spot stays
/// empty.
#[test]
fn normalise_log_scales_each_spot_to_1e4_then_log1p() {
    let mut x = csc(&[&[1.0, 3.0, 0.0], &[1.0, 1.0, 0.0]]);
    normalise_log(&mut x);
    let d = DMatrix::from(&x);
    let want = [[5000f32, 7500.0, 0.0], [5000.0, 2500.0, 0.0]];
    for g in 0..2 {
        for s in 0..3 {
            let w = want[g][s].ln_1p();
            assert!((d[(g, s)] - w).abs() < 1e-4, "{g},{s}: {}", d[(g, s)]);
        }
    }
}

/// Mean and dispersion are taken on the counts scale (expm1 of the log
/// data), the variance unbiased; the mean is returned log1p'd, the
/// dispersion log'd, and a gene never seen has none.
#[test]
fn mean_and_dispersion_are_on_the_counts_scale() {
    // expm1 values: gene 0 = (1, 3), gene 1 = (2, 2), gene 2 = (0, 0)
    let l = |v: f32| v.ln_1p();
    let x = csc(&[&[l(1.0), l(3.0)], &[l(2.0), l(2.0)], &[0.0, 0.0]]);
    let (mean, disp) = gene_mean_dispersion(&x);
    assert!((mean[0] - 2f64.ln_1p()).abs() < 1e-5);
    // var (ddof 1) of (1, 3) = 2, over mean 2 = 1, log = 0
    assert!(disp[0].abs() < 1e-5, "{}", disp[0]);
    assert!(disp[1].is_nan(), "zero variance has no dispersion");
    assert!(disp[2].is_nan());
}

/// The dispersion is normalised within its mean bin, so a gene that stands
/// out among genes of its own mean outranks one that is high only because
/// its bin is; a gene alone in its bin scores 1.
#[test]
fn dispersion_is_normalised_within_mean_bins() {
    // two bins (low and high mean) of four genes, and one gene alone
    let mean = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 5.0];
    let disp = [1.0, 1.0, 1.0, 3.0, 10.0, 10.0, 12.0, 12.0, 7.0];
    let z = normalised_dispersion(&mean, &disp, 20);
    // low bin: mean 1.5, sd 1; high bin: mean 11, sd 2/sqrt(3)
    assert!((z[3] - 1.5).abs() < 1e-9, "{}", z[3]);
    assert!((z[6] - 3f64.sqrt() / 2.0).abs() < 1e-9, "{}", z[6]);
    assert!((z[8] - 1.0).abs() < 1e-9, "{}", z[8]);
    assert!(z[3] > z[6]);
    // a lone gene scores 1 whatever the sign of its dispersion
    let z = normalised_dispersion(&[0.0, 0.0, 0.0, 10.0], &[1.0, 2.0, 3.0, -0.5], 20);
    assert!((z[3] - 1.0).abs() < 1e-9, "{}", z[3]);
}

/// The top genes by normalised dispersion, in gene order; genes without a
/// dispersion are never picked, so fewer may come back than asked.
#[test]
fn hvg_takes_the_top_genes_by_normalised_dispersion() {
    let l = |v: f32| v.ln_1p();
    // same mean (2) for genes 0..3: variances 1/3, 16/3, 4/3; gene 3 is
    // flat, and flat gene 4 only widens the mean range so genes 0..3 share
    // a bin despite rounding
    let x = csc(&[
        &[l(1.5), l(2.5), l(1.5), l(2.5)],
        &[l(0.0), l(4.0), l(0.0), l(4.0)],
        &[l(1.0), l(3.0), l(1.0), l(3.0)],
        &[l(2.0), l(2.0), l(2.0), l(2.0)],
        &[l(50.0), l(50.0), l(50.0), l(50.0)],
    ]);
    assert_eq!(hvg(&x, 2), vec![1, 2]);
    assert_eq!(hvg(&x, 10), vec![0, 1, 2]);
}

/// `knn` counts the spot itself, as scanpy's `n_neighbors`: three far-apart
/// pairs at `knn` 2 give each spot only its partner, so the graph has three
/// components and Leiden cannot merge them even when tuned toward one
/// cluster.
#[test]
fn cluster_knn_counts_the_spot_itself() {
    let pcs = DMatrix::from_row_slice(6, 1, &[0.0, 0.1, 10.0, 10.1, 20.0, 20.1]);
    let labels = cluster(&pcs, 2, 1, 1).unwrap();
    assert_eq!(
        labels,
        vec![labels[0], labels[0], labels[2], labels[2], labels[4], labels[4]]
    );
    assert!(labels[0] != labels[2] && labels[2] != labels[4] && labels[0] != labels[4]);
}

/// Scaling gives each gene zero mean and unit (unbiased) SD, a flat gene
/// stays zero, and values are clipped at 10.
#[test]
fn scale_centres_unit_variance_and_clips() {
    let mut x = DMatrix::from_row_slice(4, 2, &[1.0, 5.0, 2.0, 5.0, 3.0, 5.0, 4.0, 5.0]);
    scale(&mut x);
    let c = x.column(0);
    assert!(c.mean().abs() < 1e-6);
    assert!((c.variance() * 4.0 / 3.0 - 1.0).abs() < 1e-5);
    assert!(x.column(1).iter().all(|&v| v == 0.0));
    // one outlier among many zeros goes past 10 SD before clipping
    let mut y = DMatrix::zeros(200, 1);
    y[(0, 0)] = 1.0;
    scale(&mut y);
    assert_eq!(y[(0, 0)], 10.0);
}

/// A clean run (L0, no nuisance) small enough for a test, its cell-state
/// signal raised until the domains are plain in the counts.
const CLEAN: &[&str] = &[
    "grid_side=16",
    "n_genes=200",
    "programme_genes=50",
    "pve_type=0.6",
    "seed=3",
];

/// The clean domains under a large donor-wide effect.
const DONOR_WIDE: &[&str] = &[
    "grid_side=16",
    "n_genes=200",
    "programme_genes=50",
    "pve_type=0.5",
    "pve_donor=0.3",
    "seed=3",
];

/// An L0 simulation with `set`, in a fresh folder: the folder (kept alive)
/// and the simulation's path.
fn simulate(set: &[&str]) -> (TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let truth = dir.path().join("sim").to_string_lossy().to_string();
    crate::simulate::run::run_simulate(&crate::simulate::run::SimulateArgs {
        scenario: "L0".into(),
        set: set.iter().map(|s| s.to_string()).collect(),
        coords: Vec::new(),
        coord_column_names: vec!["row_um".into(), "col_um".into()],
        out: truth.clone().into(),
    })
    .unwrap();
    (dir, truth)
}

/// `method` on the simulation at `truth` with the CLI defaults, writing to
/// `{truth}.{method}.tsv`.
fn baseline_args(truth: &str, method: Method) -> BaselineArgs {
    BaselineArgs {
        truth: truth.into(),
        method,
        k: None,
        n_pcs: N_PCS,
        n_hvg: N_HVG,
        knn: KNN,
        seed: SEED,
        harmony: HarmonyParams::default(),
        out: format!("{truth}.{method:?}.tsv").into(),
    }
}

/// The baseline run and its labels scored, every spot labelled.
fn scored(args: &BaselineArgs) -> Score {
    run_baseline(args).unwrap();
    let t = TruthSpots::read(&args.truth).unwrap();
    let s = score(&t, &read_labels(&args.out, "final", "label").unwrap()).unwrap();
    assert_eq!(s.n_scored, t.cells.len(), "every spot labelled");
    s
}

/// On L0 (no donor or section nuisance; every domain dominated by its own
/// cell state with its own programme genes), pooling the sections is all a
/// method needs. A run this small has few spots, so the cell-state signal
/// is raised (pve_type 0.6) until the domains are plain in the counts:
/// there the pooled scanpy recipe at the true K must recover the shared
/// domains with ARI >= 0.9 (the margin is for counting noise, so a miss
/// means the pipeline is broken). Its output is a cell<TAB>label table the
/// scorer reads, naming every spot.
#[test]
fn pooled_recovers_the_domains_of_a_clean_run() {
    let (_dir, truth) = simulate(CLEAN);
    let args = baseline_args(&truth, Method::Pooled);
    let s = scored(&args);
    let text = std::fs::read_to_string(args.out.as_ref()).unwrap();
    assert!(text.starts_with("cell\tlabel\n"));
    eprintln!("pooled shared ARI on L0: {:.3}", s.shared_ari);
    assert!(s.shared_ari >= 0.9, "shared ARI {}", s.shared_ari);
}

/// A large donor-wide effect (pve_donor 0.3) on the clean L0 domains:
/// pooled labels then follow the donor within a domain, and Harmony on the
/// donors' PCs must remove most of that. Its donor dependence (the score's
/// I(label; donor | domain) / H(donor | domain) over shared domains) must
/// be at most half of pooled's, and pooled's at least 0.2, so the test
/// stays one where there is a donor effect to remove.
#[test]
fn harmony_removes_most_of_the_donor_dependence() {
    // A smaller grid than DONOR_WIDE's (a later --set wins): Harmony's cost
    // grows with the spots, and the donor effect is still there to remove.
    let (_dir, truth) = simulate(&[DONOR_WIDE, &["grid_side=12"]].concat());
    let pooled = scored(&baseline_args(&truth, Method::Pooled));
    let harmony = scored(&baseline_args(&truth, Method::Harmony));
    let (p, h) = (pooled.donor_dependence, harmony.donor_dependence);
    eprintln!(
        "donor dependence: pooled {p:.3} (ARI {:.3}), harmony {h:.3} (ARI {:.3})",
        pooled.shared_ari, harmony.shared_ari
    );
    assert!(p >= 0.2, "pooled donor dependence {p}");
    assert!(h <= 0.5 * p, "harmony {h} vs pooled {p}");
}

/// A donor whose clusters are the reference's in another order, plus a
/// large donor-wide shift that differs by gene, maps each cluster to the
/// reference cluster it copies: centring each donor by its own mean takes
/// the shift out before the correlation. The reference keeps its own
/// labels.
#[test]
fn matching_sees_through_a_donor_wide_shift() {
    // reference: three clusters, each high on its own marker gene
    let reference = DMatrix::from_row_slice(
        3,
        5,
        &[
            3.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 3.0, 0.0, 0.0, 1.0, //
            0.0, 0.0, 3.0, 1.0, 1.0,
        ],
    );
    // donor 1 holds clusters (2, 0, 1) of the reference, shifted
    let shift = [8.0, 0.5, 0.0, 6.0, 2.0];
    let order = [2usize, 0, 1];
    let copy = DMatrix::from_fn(3, 5, |c, g| reference[(order[c], g)] + shift[g]);
    let matched = match_to_reference(&[copy, reference], 1);
    assert_eq!(matched[1], vec![0, 1, 2], "the reference keeps its labels");
    assert_eq!(matched[0], order.to_vec());
}

/// Per donor on the large donor-wide effect of [`DONOR_WIDE`]: each donor
/// alone has no donor effect, and the matching centres each donor before
/// comparing, so the shared domains must come back with ARI >= 0.8 (a bar
/// set before measuring; lower than pooled's 0.9 on a clean run because
/// each donor's graph has a third of the spots).
#[test]
fn per_donor_runs_and_is_scored() {
    let (_dir, truth) = simulate(DONOR_WIDE);
    let s = scored(&baseline_args(&truth, Method::PerDonor));
    eprintln!(
        "per-donor shared ARI {:.3}, donor dependence {:.3}",
        s.shared_ari, s.donor_dependence
    );
    assert!(s.shared_ari >= 0.8, "shared ARI {}", s.shared_ari);
}

/// With fewer HVGs asked for (150) than genes with a dispersion (all 200),
/// the pipeline uses exactly [`hvg`]'s top 150, and on those alone the
/// pooled recipe still recovers the clean run's shared domains with ARI >=
/// 0.9, the bar of the all-genes test. The signal is spread over many genes
/// in a run this small, so far fewer HVGs lose it.
#[test]
fn hvg_selection_keeps_the_top_genes_and_still_recovers_the_domains() {
    let (_dir, truth) = simulate(CLEAN);
    let mut args = baseline_args(&truth, Method::Pooled);
    args.n_hvg = 150;
    let (mut log, _) = read_counts(&truth).unwrap();
    normalise_log(&mut log);
    let (mean, disp) = gene_mean_dispersion(&log);
    let z = normalised_dispersion(&mean, &disp, 20);
    let with_dispersion = z.iter().filter(|v| !v.is_nan()).count();
    assert!(with_dispersion > args.n_hvg, "{with_dispersion} genes");
    let (_, genes) = scanpy_pcs(&log, &args).unwrap();
    assert_eq!(genes, hvg(&log, args.n_hvg));
    let s = scored(&args);
    eprintln!("pooled shared ARI on L0 with 150 HVGs: {:.3}", s.shared_ari);
    assert!(s.shared_ari >= 0.9, "shared ARI {}", s.shared_ari);
}

/// The reference donor is the one with the most spots, ties to the first.
#[test]
fn the_reference_donor_is_the_largest_and_ties_go_to_the_first() {
    assert_eq!(reference_donor(&[10, 30, 20]), 1);
    assert_eq!(reference_donor(&[30, 30, 30]), 0);
    assert_eq!(reference_donor(&[5, 30, 30]), 1);
}

/// Each donor is tuned toward the domains it holds, as the pooled run is
/// toward all of them: shared ones and its own.
#[test]
fn each_donor_is_tuned_toward_the_domains_it_holds() {
    // donor 0: domains 0, 1; donor 1: domains 0, 1, 2 (its own)
    let domain = [0, 1, 1, 0, 1, 2, 2];
    let donor = [0, 0, 0, 1, 1, 1, 1];
    assert_eq!(domains_per_donor(&domain, &donor), vec![2, 3]);
}

/// A donor with one cluster has no profile left once centred by its own
/// mean, so it is matched on its mean as it is.
#[test]
fn a_donor_with_one_cluster_is_matched_on_its_mean() {
    let reference = DMatrix::from_row_slice(3, 3, &[5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0]);
    let lone = DMatrix::from_row_slice(1, 3, &[0.0, 4.0, 1.0]);
    let matched = match_to_reference(&[reference, lone], 0);
    assert_eq!(matched[1], vec![1]);
}

/// A gene at one level in every spot has no variance, however large the
/// level: its dispersion is NaN, not a rounding error's logarithm.
#[test]
fn a_flat_gene_has_no_dispersion_at_any_level() {
    let level = (5000f32).ln_1p();
    let x = csc(&[&[level; 997], &[1.0; 997]]);
    let (_, disp) = gene_mean_dispersion(&x);
    assert!(disp[0].is_nan() && disp[1].is_nan(), "{disp:?}");
}

/// Settings the pipeline cannot run with are refused before any data is
/// read: the simulation named here does not exist.
#[test]
fn bad_settings_are_refused_before_reading() {
    let base = || baseline_args("no/such/sim", Method::Harmony);
    let mut cases = vec![
        BaselineArgs {
            k: Some(0),
            ..base()
        },
        BaselineArgs { knn: 1, ..base() },
        BaselineArgs { n_pcs: 0, ..base() },
        BaselineArgs { n_hvg: 1, ..base() },
    ];
    let mut no_ridge = base();
    no_ridge.harmony.lambda = 0.0;
    cases.push(no_ridge);
    for args in cases {
        let e = run_baseline(&args).unwrap_err().to_string();
        assert!(e.contains("must be"), "{e}");
    }
}
