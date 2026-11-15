//! `pinto lra --marginal-baselines`: the co-expression statistics
//! `lra-simulate` benchmarks, read on the run's own cell types, and the
//! per-type-pair table beside the effects table.

use super::{lattice, type_names};
use crate::lr_activity::ligand_effect::baselines::{
    baselines, marginals, CellTypeTest, TypeLabels, TypeSource, PERMUTATIONS,
};
use crate::lr_activity::ligand_effect::effects::ScopeEffects;
use crate::lr_activity::ligand_effect::inference::Estimate;
use crate::lr_activity::ligand_effect::output::{
    celltype_rows, write_celltype, write_effects, Q_SCOPE_ALL,
};
use crate::lr_activity::ligand_effect::rows::EffectRow;
use crate::lr_activity::ligand_effect::tissue::{Tissue, TissueInputs, TissueOpts};
use crate::util::common::Mat;
use legume_numeric::matrix::parquet::{peek_parquet_field_names, read_table_columns};

const SIDE: usize = 30;

/// Each cell's type code: CT1, CT2 and CT3 in vertical bands of eight
/// columns, the last six columns unassigned.
fn type_code(i: usize) -> u32 {
    ((i % SIDE) / 8).min(3) as u32
}

/// A grid touching its four neighbours, with identity bins in horizontal
/// bands, so that bins and types cut the tissue differently; with
/// `types`, the cell types of [`type_code`].
fn tissue(types: bool) -> Tissue {
    let n = SIDE * SIDE;
    let (coords, pairs) = lattice(SIDE);
    let bin: Vec<usize> = (0..n).map(|i| (i / SIDE) / 10).collect();
    let identity = Mat::from_fn(3, n, |k, i| (bin[i] as f32 - 1.0) * (k as f32 + 1.0));
    let section: Vec<Box<str>> = vec!["s0".into(); n];
    let log1p_depth = vec![500f32.ln_1p(); n];
    let codes: Vec<u32> = (0..n).map(type_code).collect();
    let t = Tissue::build(
        &TissueInputs {
            coords: &coords,
            section: &section,
            pairs: &pairs,
            identity: &identity,
            identity_bin: &bin,
            log1p_depth: &log1p_depth,
            cell_type: types.then_some(codes.as_slice()),
            region: None,
        },
        &TissueOpts {
            folds: 4,
            tile: 6.0,
            ..TissueOpts::default()
        },
    )
    .unwrap();
    if types {
        t.with_cell_type_names(type_names())
    } else {
        t
    }
}

/// Ligand planted in CT1 and receptor in CT3: half of each type's cells
/// carry four counts, every other type one count in a fifth of its cells
/// (detected often enough to be tested). The unassigned cells carry both,
/// so a pair with them would score highest were they ever tested.
fn planted(t: &Tissue) -> (Vec<f64>, Vec<f64>) {
    let draw = |hot: u32| -> Vec<f64> {
        t.cell
            .iter()
            .map(|&c| match (type_code(c), c % 2 == 0, c % 5 == 0) {
                (3, _, _) => 6.0,
                (ty, true, _) if ty == hot => 4.0,
                (_, _, true) => 1.0,
                _ => 0.0,
            })
            .collect()
    };
    (draw(0), draw(2))
}

fn planted_test(permutations: usize) -> (TypeLabels, CellTypeTest, f64, f64, usize) {
    let t = tissue(true);
    let (l, r) = planted(&t);
    let labels = TypeLabels::of(&t);
    let (b, test) = marginals(&t, &l, &r, 11, &labels, permutations, false);
    (
        labels,
        test,
        b.celltype_score,
        b.celltype_p,
        b.celltype_tested,
    )
}

/// The planted sender-receiver type pair is the one the permutations
/// single out, and the pair's summary is its p, Bonferroni-corrected over
/// the tested type pairs, as `lra-simulate` reports it.
#[test]
fn the_planted_type_pair_has_the_smallest_p() {
    let (labels, test, score, p, tested) = planted_test(PERMUTATIONS);
    assert_eq!(labels.source, TypeSource::CellTypes);
    // Three named types each way; the unassigned cells are never a type.
    assert_eq!(test.pairs.len(), 9);
    assert_eq!(tested, 9);
    let best = test
        .pairs
        .iter()
        .min_by(|a, b| a.p.total_cmp(&b.p))
        .unwrap();
    assert_eq!(
        (
            &*labels.names[best.sender as usize],
            &*labels.names[best.receiver as usize]
        ),
        ("CT1", "CT3")
    );
    for o in test.pairs.iter().filter(|o| !std::ptr::eq(*o, best)) {
        assert!(best.p < o.p, "{} not below {}", best.p, o.p);
    }
    assert_eq!(p, (best.p * 9.0).min(1.0));
    assert!(p < 0.05, "{p}");
    let top = test.pairs.iter().map(|o| o.score).fold(f64::MIN, f64::max);
    assert_eq!(score, top);
    assert_eq!(best.score, top);
    assert_eq!(test.permutations, PERMUTATIONS);
}

/// Given cell types, the statistic groups cells by them, not by the
/// identity bins; without, it falls back to the bins, and is then exactly
/// what the benchmark computes.
#[test]
fn real_cell_types_are_used_when_given() {
    let t = tissue(true);
    let (l, r) = planted(&t);
    let labels = TypeLabels::of(&t);
    assert_eq!(labels.source, TypeSource::CellTypes);
    assert_eq!(labels.names, type_names());
    let (_, test) = marginals(&t, &l, &r, 3, &labels, 200, false);
    let sf = &t.size_factor;
    for o in &test.pairs {
        let rows: Vec<usize> = (0..t.n())
            .filter(|&i| type_code(t.cell[i]) == o.sender)
            .collect();
        let mean = rows.iter().map(|&i| (l[i] / sf[i]).ln_1p()).sum::<f64>() / rows.len() as f64;
        assert!(
            (o.mean_ligand - mean).abs() < 1e-12,
            "{} {mean}",
            o.mean_ligand
        );
        assert_eq!(o.n_sender, rows.len());
        assert!(o.sender < 3 && o.receiver < 3, "unassigned tested");
    }

    let bare = tissue(false);
    let bins = TypeLabels::of(&bare);
    assert_eq!(bins.source, TypeSource::IdentityBins);
    let (b, test) = marginals(&bare, &l, &r, 5, &bins, PERMUTATIONS, true);
    // Three bands of bins, every pair of them tested.
    assert_eq!(test.pairs.len(), 9);
    let bench = baselines(&bare, &l, &r, 5);
    assert_eq!(
        [
            b.celltype_score,
            b.celltype_p,
            b.moran_i,
            b.moran_p,
            b.contact_p
        ],
        [
            bench.celltype_score,
            bench.celltype_p,
            bench.moran_i,
            bench.moran_p,
            bench.contact_p
        ]
    );
    assert_eq!(b.celltype_tested, bench.celltype_tested);
}

/// The log says where the types came from, and why when they are the
/// identity bins: the run has no cell types, or its cell types label none
/// of the receivers; it lists only the types that hold cells.
#[test]
fn the_log_says_why_the_types_are_the_identity_bins() {
    let t = tissue(true);
    let note = TypeLabels::of(&t).note(&t);
    assert_eq!(note, "the run's cell types (3 types: CT1, CT2, CT3)");
    let bare = tissue(false);
    let note = TypeLabels::of(&bare).note(&bare);
    assert_eq!(
        note,
        "the identity bins, the run having no cell types (3 types: bin0, bin1, bin2)"
    );
    let mut none = tissue(true);
    none.cell_type = Some(vec![3; none.n()]);
    let note = TypeLabels::of(&none).note(&none);
    assert_eq!(
        note,
        "the identity bins, the run's cell types labelling none of the receivers \
         (3 types: bin0, bin1, bin2)"
    );
}

/// `--marginal-permutations` sets how many permutations run: a rejecting
/// type pair runs all of them and sits at the floor `1 / (1 + P)`.
#[test]
fn the_permutation_count_is_the_one_asked_for() {
    let (_, test, _, _, _) = planted_test(400);
    assert_eq!(test.permutations, 400);
    let p_min = test.pairs.iter().map(|o| o.p).fold(f64::MAX, f64::min);
    assert_eq!(p_min, 1.0 / 401.0);
}

/// The table's names and the rows of its one tested pair.
fn table_columns(path: &str) -> Vec<String> {
    peek_parquet_field_names(path)
        .unwrap()
        .iter()
        .map(|c| c.to_string())
        .collect()
}

const MARGINAL_COLUMNS: [&str; 7] = [
    "celltype_score",
    "celltype_p",
    "celltype_tested",
    "moran_i",
    "moran_p",
    "contact_score",
    "contact_p",
];

/// Without the flag the effects table is exactly what it was; with it,
/// each pair's pooled row carries the baselines, every other row NaN.
#[test]
fn the_flag_adds_the_marginal_columns_and_nothing_else() {
    let none = Estimate::nan(0, 0);
    let row = || {
        EffectRow::pending(
            ScopeEffects::zip_with(&[], |_| none),
            (f64::NAN, f64::NAN),
            f64::NAN,
        )
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("o.lra.parquet");
    let path = path.to_str().unwrap();
    write_effects(path, &[row()], Q_SCOPE_ALL, false).unwrap();
    let before = [
        "row",
        "section",
        "ligand",
        "receptor",
        "n_targets",
        "n_targets_removed",
        "n_receivers",
        "n_competent",
        "n_exposed",
        "exposure_threshold",
        "coexpr_log_or",
        "coexpr_se",
        "marginal_cor",
        "marginal_cor_se",
        "unadjusted_diff",
        "unadjusted_diff_se",
        "effect_ipw",
        "effect_ipw_se",
        "effect_ipw_lo",
        "effect_ipw_hi",
        "effect_outcome",
        "effect_competent",
        "effect_competent_se",
        "effect_competent_df",
        "effect_p",
        "effect_q",
        "coexpr_p",
        "coexpr_q",
        "n_pairs_same_ligand",
        "ligand_q",
        "effect_competent_section",
        "effect_competent_section_se",
        "effect_competent_spatial",
        "effect_competent_spatial_se",
        "effect_competent_identity",
        "effect_competent_identity_se",
        "effect_competent_nospill",
        "effect_competent_nospill_se",
        "effect_incompetent",
        "effect_incompetent_se",
        "effect_incompetent_q",
        "effect_contrast",
        "effect_contrast_se",
        "effect_vs_rest",
        "effect_vs_rest_se",
        "effect_vs_rest_df",
        "effect_vs_rest_p",
        "jackknife_ratio",
        "jackknife_vs_rest_ratio",
        "jackknife_tiles",
        "effect_log10_bf",
        "tau_top",
        "tau_top_weight",
        "effect_contrast_q",
        "mean_overlap",
        "overlap_ess",
        "near_determined_share",
        "exposure_eta2",
        "n_competent_bins",
        "n_incompetent_bins",
        "verdict",
        "verdict_note",
        "warnings",
        "effect_lo",
        "effect_hi",
        "effect_pct",
        "effect_pct_sd",
        "effect_pct_lo",
        "effect_pct_hi",
        "q_scope",
    ];
    assert_eq!(table_columns(path), before);

    let t = tissue(true);
    let (l, r) = planted(&t);
    let (b, _) = marginals(&t, &l, &r, 1, &TypeLabels::of(&t), 200, false);
    let pooled = EffectRow {
        marginal: Some(b),
        ..row()
    };
    let section = EffectRow {
        section: "s0".into(),
        ..row()
    };
    write_effects(path, &[pooled, section], Q_SCOPE_ALL, true).unwrap();
    let cols = table_columns(path);
    let mut want: Vec<&str> = before[..before.len() - 1].to_vec();
    want.extend(MARGINAL_COLUMNS);
    want.push("q_scope");
    assert_eq!(cols, want);
    let (_, x) = read_table_columns(path, &["ligand"], &MARGINAL_COLUMNS).unwrap();
    let got: Vec<f64> = x.iter().map(|c| c[0]).collect();
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs().max(1.0);
    for (g, w) in got.iter().zip([
        b.celltype_score,
        b.celltype_p,
        b.celltype_tested as f64,
        b.moran_i,
        b.moran_p,
        b.contact_score,
        b.contact_p,
    ]) {
        assert!(close(*g, w), "{g} {w}");
    }
    assert!(x.iter().all(|c| c[1].is_nan() || c[1] == 0.0));
}

/// One row per tested sender-receiver type pair, named, with the pair's
/// BH q over its own type pairs.
#[test]
fn the_type_pair_table_has_a_row_per_tested_type_pair() {
    let (labels, test, _, _, tested) = planted_test(300);
    let mut rows = celltype_rows("GENE1", "GENE2", &test, &labels);
    rows.extend(celltype_rows("GENE3", "GENE2", &test, &labels));
    assert_eq!(rows.len(), 2 * tested);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("o.lra.celltype.parquet");
    let path = path.to_str().unwrap();
    write_celltype(path, &rows).unwrap();
    let strs = [
        "ligand",
        "receptor",
        "sender_type",
        "receiver_type",
        "labels",
    ];
    let nums = [
        "n_sender",
        "n_receiver",
        "mean_ligand_sender",
        "mean_receptor_receiver",
        "ligand_detected_sender",
        "receptor_detected_receiver",
        "score",
        "p",
        "q",
        "permutations",
    ];
    let (s, x) = read_table_columns(path, &strs, &nums).unwrap();
    assert_eq!(s[0].len(), 2 * tested);
    assert!(s[2]
        .iter()
        .chain(&s[3])
        .all(|n| ["CT1", "CT2", "CT3"].contains(&&**n)));
    assert!(s[4].iter().all(|n| &**n == "cell-types"));
    let (p, q) = (&x[7], &x[8]);
    assert!(p.iter().zip(q).all(|(p, q)| q >= p && *q <= 1.0));
    // BH within each pair's own type pairs, not over the table.
    let own = crate::lr_activity::ligand_effect::report::bh(&p[..tested]);
    for (a, b) in q[..tested].iter().zip(&own) {
        assert!((a - b).abs() < 1e-6, "{a} {b}");
    }
    let k = (0..tested).min_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
    assert!(q[k] <= 9.0 * p[k] + 1e-6);
    assert!(x[9].iter().all(|&n| n == 300.0));
    assert!(x[4].iter().chain(&x[5]).all(|&d| d >= 0.1));
}

/// With many tested type pairs the Bonferroni summary can stop permuting
/// early, but the per-type-pair table's p and q need every permutation
/// asked for: with 144 type pairs, early stopping stops short, and the
/// table's run does not.
#[test]
fn the_type_pair_table_gets_every_permutation() {
    let t = tissue(false);
    let k = 12;
    let labels = TypeLabels {
        codes: t.cell.iter().map(|&c| (c % k) as u32).collect(),
        names: (1..=k).map(|i| format!("CT{i}").into()).collect(),
        testable: vec![true; k],
        source: TypeSource::CellTypes,
    };
    let ones = vec![1.0; t.n()];
    let (_, summary) = marginals(&t, &ones, &ones, 3, &labels, 2000, true);
    assert_eq!(summary.pairs.len(), k * k, "every type pair is tested");
    assert!(summary.permutations < 2000, "the summary stops early");
    let (_, table) = marginals(&t, &ones, &ones, 3, &labels, 2000, false);
    assert_eq!(table.permutations, 2000);
}

/// A run restarted from its recorded options (the viewer's rerun, MCP's
/// run_lra) asks for the baselines again, with their permutation count when
/// it is not the default.
#[cfg(feature = "view")]
#[test]
fn a_rerun_keeps_the_marginal_baselines() {
    use crate::lr_activity::args::LraMethod;
    use crate::util::metadata::LraOptions;
    let o = LraOptions {
        marginal_baselines: Some("cell-types".into()),
        marginal_permutations: Some(500),
        ..LraOptions::default()
    };
    let argv = o.argv(Some("r.pinto.json"), None, LraMethod::default(), &[]);
    assert!(argv.iter().any(|a| a == "--marginal-baselines"), "{argv:?}");
    assert!(
        argv.iter().any(|a| a == "--marginal-permutations=500"),
        "{argv:?}"
    );
    let plain = LraOptions::default().argv(Some("r.pinto.json"), None, LraMethod::default(), &[]);
    assert!(
        !plain.iter().any(|a| a.starts_with("--marginal")),
        "{plain:?}"
    );
}

/// The baselines are contact-effect's: asked for with a co-expression
/// method they are refused, as --only, --region and --cell-types are.
#[test]
fn the_baselines_are_refused_with_a_coexpression_method() {
    let parse = |extra: &[&str]| -> crate::lr_activity::args::SrtLrActivityArgs {
        let argv = ["pinto", "lra", "--from", "r.pinto.json"]
            .iter()
            .chain(extra)
            .copied();
        match <crate::Cli as clap::Parser>::try_parse_from(argv)
            .unwrap()
            .commands
        {
            crate::Commands::LrActivity(a) => *a,
            _ => unreachable!("an lra command line"),
        }
    };
    let coexpr = parse(&["--method", "coexpression-test", "--marginal-baselines"]);
    let e = crate::lr_activity::fit::check_method_flags(&coexpr)
        .unwrap_err()
        .to_string();
    assert!(e.contains("--marginal-baselines"), "{e}");
    let ok = parse(&["--marginal-baselines"]);
    assert!(crate::lr_activity::fit::check_method_flags(&ok).is_ok());
}

/// `--marginal-permutations` is refused as the command line is parsed,
/// before any data is read: without `--marginal-baselines` (and so with a
/// co-expression method too), and at 0.
#[test]
fn the_permutations_need_the_baselines_and_one_at_least() {
    let parse = |extra: &[&str]| {
        let argv = ["pinto", "lra", "--from", "r.pinto.json"]
            .iter()
            .chain(extra)
            .copied();
        <crate::Cli as clap::Parser>::try_parse_from(argv).map(|_| ())
    };
    for refused in [
        &["--marginal-permutations", "5000"][..],
        &[
            "--method",
            "coexpression-test",
            "--marginal-permutations",
            "10",
        ],
        &["--marginal-baselines", "--marginal-permutations", "0"],
    ] {
        assert!(parse(refused).is_err(), "{refused:?}");
    }
    assert!(parse(&["--marginal-baselines", "--marginal-permutations", "1"]).is_ok());
    assert!(parse(&["--marginal-baselines"]).is_ok());
    assert!(parse(&[]).is_ok(), "the default needs no flag");
}
