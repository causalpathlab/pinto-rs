//! The lra layer: the run's tables, the pair list filtered and sorted, the
//! curve drawn whatever its points, the commands it starts, and `view_lra`
//! driven as an agent drives it, a pair selected and then drawn by
//! `view_map`.

use crate::lr_activity::input::Contacts;
use crate::lr_activity::ligand_effect::output::{PairRow, Verdict, Q_SCOPE_ALL};
use crate::lr_activity::ligand_effect::receivers::RunFrame;
use crate::util::feature_axis::FeatureAxis;
use crate::util::metadata::{LraOptions, LraSubset};
use crate::view::color::Theme;
use crate::view::curve::{self, ticks, Curve};
use crate::view::lra::{
    lra_command, lra_table, next_verdict, visible, Filter, LraView, Receivers, Sort,
};
use crate::view::Base;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use std::path::Path;

// * A run with an lra table

/// Grid side of the fixture's cells.
const SIDE: usize = 10;

fn strs(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|&x| x.into()).collect()
}

/// `{out}.lra.parquet` of `rows` (ligand, receptor, verdict, q, pct,
/// threshold), each pooled, with a section row that is not; `scope` its
/// `q_scope`.
fn write_table(path: &Path, rows: &[(&str, &str, &str, f32, f32, f32)], scope: &str) {
    let n = rows.len() + 1;
    fn col(n: usize, f: impl Fn(usize) -> String) -> Vec<Box<str>> {
        (0..n).map(|i| f(i).into()).collect()
    }
    let section = col(n, |i| {
        if i < rows.len() { "all" } else { "section:s1" }.into()
    });
    let at = |i: usize| rows[i.min(rows.len() - 1)];
    let ligand = col(n, |i| at(i).0.into());
    let receptor = col(n, |i| at(i).1.into());
    let verdict = col(n, |i| at(i).2.into());
    let note = col(n, |i| at(i).2.parse::<Verdict>().unwrap().note().into());
    let warnings = col(n, |_| String::new());
    let scope = col(n, |_| scope.into());
    let num = |f: &dyn Fn(usize) -> f32| -> Vec<f32> { (0..n).map(f).collect() };
    let q = num(&|i| at(i).3);
    let pct = num(&|i| at(i).4);
    let effect = num(&|i| (1.0 + at(i).4 / 100.0).ln());
    let tau = num(&|i| at(i).5);
    let se = num(&|_| 0.05);
    let lo = num(&|i| at(i).4 - 5.0);
    let hi = num(&|i| at(i).4 + 5.0);
    let counts = num(&|_| 10.0);
    let names: Vec<Box<str>> = (0..n).map(|i| i.to_string().into()).collect();
    write_named_table(
        &path.to_string_lossy(),
        "row",
        &names,
        &[
            ("section".into(), Column::Str(&section)),
            ("ligand".into(), Column::Str(&ligand)),
            ("receptor".into(), Column::Str(&receptor)),
            ("verdict".into(), Column::Str(&verdict)),
            ("verdict_note".into(), Column::Str(&note)),
            ("warnings".into(), Column::Str(&warnings)),
            ("effect_competent".into(), Column::F32(&effect)),
            ("effect_competent_se".into(), Column::F32(&se)),
            ("effect_p".into(), Column::F32(&q)),
            ("effect_q".into(), Column::F32(&q)),
            ("coexpr_q".into(), Column::F32(&q)),
            ("effect_pct".into(), Column::F32(&pct)),
            ("effect_pct_lo".into(), Column::F32(&lo)),
            ("effect_pct_hi".into(), Column::F32(&hi)),
            ("exposure_threshold".into(), Column::F32(&tau)),
            ("n_receivers".into(), Column::F32(&counts)),
            ("n_competent".into(), Column::F32(&counts)),
            ("n_exposed".into(), Column::F32(&counts)),
            ("q_scope".into(), Column::Str(&scope)),
        ],
    )
    .unwrap();
}

/// `{out}.lra.curve.parquet`: four thresholds for `pair`, the last not
/// estimable.
fn write_curve(path: &Path, pair: (&str, &str)) {
    let ligand = strs(&[pair.0; 4]);
    let receptor = strs(&[pair.1; 4]);
    let tau = [0.0f32, 1.0, 2.0, 3.0];
    let quantile = [0.0f32, 0.25, 0.5, 0.75];
    let effect = [0.1f32, 0.2, 0.25, f32::NAN];
    let lo = [0.0f32, 0.1, 0.1, f32::NAN];
    let hi = [0.2f32, 0.3, 0.4, f32::NAN];
    let band_lo = [-0.05f32, 0.05, 0.05, f32::NAN];
    let band_hi = [0.25f32, 0.35, 0.45, f32::NAN];
    let names = strs(&["0", "1", "2", "3"]);
    write_named_table(
        &path.to_string_lossy(),
        "row",
        &names,
        &[
            ("ligand".into(), Column::Str(&ligand)),
            ("receptor".into(), Column::Str(&receptor)),
            ("tau".into(), Column::F32(&tau)),
            ("quantile".into(), Column::F32(&quantile)),
            ("effect".into(), Column::F32(&effect)),
            ("lo".into(), Column::F32(&lo)),
            ("hi".into(), Column::F32(&hi)),
            ("band_lo".into(), Column::F32(&band_lo)),
            ("band_hi".into(), Column::F32(&band_hi)),
        ],
    )
    .unwrap();
}

/// A run of `SIDE²` cells on a grid with its expression (ligand `L1` on
/// the left, receptor `R1` in every other cell, filler genes), an lra table
/// of three pairs and a subset table of one, recorded in its manifest; its
/// prefix, and the run's cells, data and contacts as lra reads them.
pub(crate) fn lra_run(dir: &Path) -> (String, Contacts) {
    use legume_numeric::matrix::traits::IoOps;
    let n = SIDE * SIDE;
    let prefix = dir.join("run").to_string_lossy().into_owned();
    let genes = [
        "L1", "R1", "G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9",
    ];
    let mut triplets = Vec::new();
    for c in 0..n {
        let (x, y) = (c % SIDE, c / SIDE);
        if x < 3 {
            triplets.push((0u64, c as u64, 2.0f32));
        }
        if c % 2 == 0 {
            triplets.push((1, c as u64, 1.0));
        }
        for g in 2..genes.len() {
            let v = ((c * 7 + g * 13 + x * y) % 5) as f32;
            if v > 0.0 {
                triplets.push((g as u64, c as u64, v));
            }
        }
    }
    let zarr = crate::test_support::write_zarr(dir, "d", &genes, n, &triplets);
    let names: Vec<Box<str>> = (0..n).map(|j| format!("d_{j}").into()).collect();
    let xy = nalgebra::DMatrix::<f32>::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i % SIDE) as f32
        } else {
            (i / SIDE) as f32
        }
    });
    xy.to_parquet_with_names(
        &format!("{prefix}.cells.parquet"),
        (Some(&names), Some("cell")),
        Some(&strs(&["x", "y"])),
    )
    .unwrap();
    let prop = nalgebra::DMatrix::<f32>::from_fn(n, 2, |i, j| {
        if ((i % SIDE) < SIDE / 2) == (j == 0) {
            0.9
        } else {
            0.1
        }
    });
    prop.to_parquet_with_names(
        &format!("{prefix}.propensity.parquet"),
        (Some(&names), Some("cell")),
        Some(&strs(&["C0", "C1"])),
    )
    .unwrap();
    let table = format!("{prefix}.lra.parquet");
    let curve = format!("{prefix}.lra.curve.parquet");
    write_table(
        Path::new(&table),
        &[
            ("L1", "G0", "no_evidence", 0.6, 1.0, 0.0),
            ("L1", "R1", "supported", 0.01, 25.0, 0.0),
            (
                "G1",
                "R1",
                "insufficient_data",
                f32::NAN,
                f32::NAN,
                f32::NAN,
            ),
        ],
        Q_SCOPE_ALL,
    );
    write_curve(Path::new(&curve), ("L1", "R1"));
    let only = format!("{prefix}.lra.only-l1-r1.parquet");
    write_table(
        Path::new(&only),
        &[("L1", "R1", "supported", 0.01, 25.0, 0.0)],
        "q corrected within the --only subset (1 of 3 pairs)",
    );
    let subset = LraSubset {
        only: vec!["L1:R1".into()],
        pairs: 1,
        of: 3,
        table: only.clone(),
        curve: format!("{prefix}.lra.only-l1-r1.curve.parquet"),
        options: None,
    };
    let manifest = serde_json::json!({
        "command": "lc",
        "version": "0",
        "timestamp": "",
        "prefix": prefix,
        "n_cells": n,
        "n_features": genes.len(),
        "data_files": [zarr.to_string()],
        "outputs": {
            "cells": format!("{prefix}.cells.parquet"),
            "propensity": format!("{prefix}.propensity.parquet"),
            "lra": {
                "table": table,
                "curve": curve,
                "options": LraOptions::default(),
                "subsets": [subset],
            },
        },
    });
    std::fs::write(format!("{prefix}.pinto.json"), manifest.to_string()).unwrap();

    let data = crate::lr_activity::fit::load_expr_data(&[zarr], false).unwrap();
    let axis = FeatureAxis::resolve_or_identity(&data.row_names().unwrap()).unwrap();
    let mut pairs = Vec::new();
    for c in 0..n {
        if c % SIDE + 1 < SIDE {
            pairs.push((c, c + 1));
        }
        if c / SIDE + 1 < SIDE {
            pairs.push((c, c + SIDE));
        }
    }
    // The run's cell pairs, as lra reads them from the run.
    let left: Vec<Box<str>> = pairs.iter().map(|&(i, _)| names[i].clone()).collect();
    let right: Vec<Box<str>> = pairs.iter().map(|&(_, j)| names[j].clone()).collect();
    let ids: Vec<Box<str>> = (0..pairs.len()).map(|k| k.to_string().into()).collect();
    write_named_table(
        &format!("{prefix}.coord_pairs.parquet"),
        "cell_pair",
        &ids,
        &[
            ("left_cell".into(), Column::Str(&left)),
            ("right_cell".into(), Column::Str(&right)),
        ],
    )
    .unwrap();
    let contacts = Contacts {
        data,
        axis,
        pairs,
        cell_section: vec!["all".into(); n],
        coords: (0..n)
            .map(|c| Some(((c % SIDE) as f32, (c / SIDE) as f32)))
            .collect(),
        cell_type: None,
    };
    (prefix, contacts)
}

/// The run's receivers, read without the reference the tests lack.
pub(crate) fn receivers(base: &Base, contacts: Contacts) -> Receivers {
    let frame = RunFrame::without_reference(contacts, &[]).unwrap();
    Receivers::new(frame, base, &LraOptions::default())
}

// * The list

fn row(ligand: &str, receptor: &str, verdict: Verdict, q: f64, pct: f64) -> PairRow {
    PairRow {
        ligand: ligand.into(),
        receptor: receptor.into(),
        key: (ligand.to_uppercase().into(), receptor.to_uppercase().into()),
        verdict,
        note: verdict.note().into(),
        warnings: "".into(),
        effect: (1.0 + pct / 100.0).ln(),
        effect_se: 0.1,
        effect_p: q / 2.0,
        effect_q: q,
        coexpr_q: q,
        pct,
        pct_lo: pct - 1.0,
        pct_hi: pct + 1.0,
        threshold: 0.0,
        n_receivers: 100,
        n_competent: 40,
        n_exposed: 20,
    }
}

fn names(rows: &[PairRow], shown: &[usize]) -> Vec<String> {
    shown.iter().map(|&i| rows[i].name()).collect()
}

fn rows() -> Vec<PairRow> {
    vec![
        row("TGFB1", "TGFBR1_TGFBR2", Verdict::NoEvidence, 0.4, -2.0),
        row("SPP1", "CD44", Verdict::Supported, 0.001, 12.0),
        row("SPP1", "ITGB1", Verdict::ColocationOnly, 0.03, -30.0),
        row(
            "CXCL12",
            "CXCR4",
            Verdict::InsufficientData,
            f64::NAN,
            f64::NAN,
        ),
        row("TGFB1", "ACVRL1", Verdict::Supported, 0.02, 5.0),
    ]
}

#[test]
fn the_list_sorts_by_q_effect_or_name_unestimable_last() {
    let r = rows();
    let by = |sort| {
        names(
            &r,
            &visible(
                &r,
                &Filter {
                    sort,
                    ..Filter::default()
                },
            ),
        )
    };
    assert_eq!(
        by(Sort::Q),
        [
            "SPP1:CD44",
            "TGFB1:ACVRL1",
            "SPP1:ITGB1",
            "TGFB1:TGFBR1_TGFBR2",
            "CXCL12:CXCR4"
        ]
    );
    assert_eq!(
        by(Sort::Effect),
        [
            "SPP1:ITGB1",
            "SPP1:CD44",
            "TGFB1:ACVRL1",
            "TGFB1:TGFBR1_TGFBR2",
            "CXCL12:CXCR4"
        ]
    );
    assert_eq!(
        by(Sort::Name),
        [
            "CXCL12:CXCR4",
            "SPP1:CD44",
            "SPP1:ITGB1",
            "TGFB1:ACVRL1",
            "TGFB1:TGFBR1_TGFBR2"
        ]
    );
    assert_eq!(Sort::Q.next().next().next(), Sort::Q);
}

#[test]
fn the_list_filters_by_verdict_and_genes() {
    let r = rows();
    let f = |verdict, search: &str| {
        names(
            &r,
            &visible(
                &r,
                &Filter {
                    verdict,
                    search: search.into(),
                    sort: Sort::Name,
                },
            ),
        )
    };
    assert_eq!(
        f(Some(Verdict::Supported), ""),
        ["SPP1:CD44", "TGFB1:ACVRL1"]
    );
    // A gene of the ligand or the receptor, a complex's subunit too, case aside.
    assert_eq!(f(None, "tgfbr2"), ["TGFB1:TGFBR1_TGFBR2"]);
    assert_eq!(f(None, "spp1"), ["SPP1:CD44", "SPP1:ITGB1"]);
    // Every word must match.
    assert_eq!(f(None, "spp1 cd44"), ["SPP1:CD44"]);
    assert_eq!(f(Some(Verdict::Supported), "spp1"), ["SPP1:CD44"]);
    assert!(f(Some(Verdict::ControlFailed), "").is_empty());
    let counts = Verdict::counts(r.iter().map(|r| r.verdict));
    assert_eq!(counts.len(), Verdict::ALL.len());
    assert_eq!(counts[0], (Verdict::Supported, 2));
    // The verdict filter steps through every verdict, then all.
    let mut v = None;
    for _ in 0..=Verdict::ALL.len() {
        v = next_verdict(v, 1);
    }
    assert_eq!(v, None);
    assert_eq!(next_verdict(None, -1), Some(Verdict::InsufficientData));
}

// * The curve

#[test]
fn the_curve_draws_whatever_its_points() {
    use crate::lr_activity::ligand_effect::output::CurveRow;
    let p = |q: f64, e: f64| CurveRow {
        tau: q * 4.0,
        quantile: q,
        effect: e,
        lo: e - 0.1,
        hi: e + 0.1,
        band_lo: e - 0.2,
        band_hi: e + 0.2,
    };
    let nan = CurveRow {
        effect: f64::NAN,
        lo: f64::NAN,
        hi: f64::NAN,
        band_lo: f64::NAN,
        band_hi: f64::NAN,
        ..p(0.5, 0.0)
    };
    let cases: Vec<Vec<CurveRow>> = vec![
        Vec::new(),
        vec![nan],
        vec![p(0.0, 0.1)],
        vec![p(0.0, 0.1), nan, p(0.5, -0.3), p(0.75, f64::INFINITY)],
        vec![p(0.25, 0.0), p(0.0, 0.0)],
    ];
    for (k, pts) in cases.iter().enumerate() {
        let c = Curve {
            title: format!("case {k}"),
            points: pts,
            headline: Some(0.0),
        };
        for (w, h) in [(1, 1), (40, 12), (600, 360)] {
            let f = curve::render(&c, (w, h), Theme::Dark, 16.);
            assert_eq!((f.w, f.h), (w, h), "case {k}");
        }
        let f = curve::render(&c, (300, 200), Theme::Light, 12.);
        assert_eq!(f.rgba.len(), 300 * 200 * 4);
        assert!(curve::page(&c, Theme::Dark).is_ok());
    }
    // Something is drawn in the curve's colour where it has points.
    let pts = [p(0.0, 0.1), p(0.5, 0.3)];
    let c = Curve {
        title: "x".into(),
        points: &pts,
        headline: None,
    };
    let f = curve::render(&c, (400, 240), Theme::Dark, 12.);
    let bg = Theme::Dark.background();
    let coloured = f
        .rgba
        .chunks(4)
        .filter(|p| p[2] > 200 && i32::from(p[2]) > i32::from(p[0]) + 60)
        .count();
    assert!(coloured > 50, "{coloured} blue pixels");
    assert!(f.rgba.chunks(4).any(|p| p[..3] == bg));
}

#[test]
fn ticks_are_round_and_cover_the_range() {
    assert_eq!(ticks(0.0, 10.0), [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    assert_eq!(ticks(-3.0, 4.0), [-2.0, 0.0, 2.0, 4.0]);
    let t = ticks(-0.04, 0.03);
    assert!(t.len() >= 3 && t.len() <= 6, "{t:?}");
    assert_eq!(ticks(1.0, 1.0), [1.0]);
    assert_eq!(ticks(f64::NAN, 1.0), [0.0]);
}

// * Commands

#[test]
fn the_layer_starts_lra_beside_the_run() {
    let opts = LraOptions::default();
    let argv = lra_command(Path::new("run.pinto.json"), &opts, &[]);
    assert_eq!(argv, ["lra", "--from", "run.pinto.json"]);
    let opts = LraOptions {
        lr_pairs: Some("/p/pairs.tsv".into()),
        targets: vec!["/p/a.gmt".into(), "/p/b.tsv".into()],
        species: "mouse".into(),
        seed: 7,
        identity_excluded: vec!["G0".into()],
    };
    let only = vec!["SPP1:CD44".to_string(), "TGFB1".to_string()];
    let argv = lra_command(Path::new("run.pinto.json"), &opts, &only);
    assert_eq!(
        argv,
        [
            "lra",
            "--from",
            "run.pinto.json",
            "--lr-pairs",
            "/p/pairs.tsv",
            "--targets",
            "/p/a.gmt,/p/b.tsv",
            "--species",
            "mouse",
            "--seed",
            "7",
            "--only",
            "SPP1:CD44,TGFB1"
        ]
    );
    // pinto takes it as written.
    let mut cli = <crate::Cli as clap::CommandFactory>::command();
    cli.build();
    crate::run::form::check(&cli, &argv).unwrap();
    // The table it writes, beside the run.
    let t = lra_table("/r/run", &only, LraOptions::default());
    assert_eq!(t.path, Path::new("/r/run.lra.only-spp1-cd44+tgfb1.parquet"));
    assert_eq!(
        t.curve.as_deref(),
        Some(Path::new("/r/run.lra.only-spp1-cd44+tgfb1.curve.parquet"))
    );
    assert_eq!(t.label(), "only SPP1:CD44,TGFB1");
    // Without --only, the full table and its curve.
    let t = lra_table("/r/run", &[], LraOptions::default());
    assert_eq!(t.path, Path::new("/r/run.lra.parquet"));
    assert_eq!(
        t.curve.as_deref(),
        Some(Path::new("/r/run.lra.curve.parquet"))
    );
    assert_eq!(t.label(), "full table");
}

// * The run's tables and view_lra

#[test]
fn the_runs_tables_are_the_full_one_then_its_subsets() {
    let dir = tempfile::tempdir().unwrap();
    let (prefix, _) = lra_run(dir.path());
    let run = crate::view::data::Run::open(&prefix).unwrap();
    let tables = run.tables();
    assert_eq!(tables.len(), 2);
    assert_eq!(tables[0].label(), "full table");
    assert!(tables[0].curve.is_some());
    assert_eq!(tables[1].label(), "only L1:R1");
    // Its curve was never written: none.
    assert!(tables[1].curve.is_none());
    let mut v = LraView::new(tables).unwrap().unwrap();
    assert_eq!(v.rows().len(), 3, "pooled rows only");
    assert!(!v.table().subset_corrected());
    v.select("l1:r1").unwrap();
    let e = v.select("L1:R2").unwrap_err().to_string();
    assert!(e.contains("nearest: L1:R1"), "{e}");
    assert_eq!(v.curve().unwrap().points.len(), 4);
    // The subset keeps the pair selected, its q subset-corrected.
    v.load(1).unwrap();
    assert_eq!(v.selected_row().unwrap().name(), "L1:R1");
    assert!(v.table().subset_corrected());
    assert!(v.table().q_scope.as_deref() != Some(Q_SCOPE_ALL));
    assert!(v.curve().unwrap().points.is_empty());
    assert!(v.curve().unwrap().title.contains("subset"));
    // A run without a table has no layer.
    assert!(LraView::new(Vec::new()).unwrap().is_none());
}

/// A `run_lra --only` that ends after the run was opened: its table is listed
/// and can be switched to, the ones read kept.
#[test]
fn view_lra_finds_a_table_written_after_the_run_was_opened() {
    use crate::view::mcp::{LraArgs, Open};
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let (prefix, _) = lra_run(dir.path());
    let manifest = format!("{prefix}.pinto.json");
    let text = std::fs::read_to_string(&manifest).unwrap();
    let mut held: serde_json::Value = serde_json::from_str(&text).unwrap();
    let subsets = held["outputs"]["lra"]["subsets"].take();
    held["outputs"]["lra"]["subsets"] = json!([]);
    std::fs::write(&manifest, held.to_string()).unwrap();

    let mut open = Open::new(&prefix, None).unwrap();
    let call = |open: &mut Open, a: serde_json::Value| {
        let a: LraArgs = serde_json::from_value(a).unwrap();
        open.lra_with(a, &|_| {})
    };
    call(&mut open, json!({})).unwrap();
    assert_eq!(open.lra.as_ref().unwrap().tables.len(), 1);

    std::fs::write(&manifest, text).unwrap();
    assert!(subsets.as_array().is_some_and(|s| !s.is_empty()));
    call(&mut open, json!({"table": "L1:R1"})).unwrap();
    let v = open.lra.as_ref().unwrap();
    assert_eq!(v.tables.len(), 2);
    assert_eq!(v.tables[v.at].label(), "only L1:R1");
}

#[test]
fn view_lra_lists_selects_and_view_map_draws_the_pair() {
    use crate::view::mcp::{LraArgs, MapArgs, Open};
    use legume_mcp::Content;
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let (prefix, contacts) = lra_run(dir.path());
    let mut open = Open::new(&prefix, None).unwrap();
    let lra = |a: serde_json::Value| -> LraArgs { serde_json::from_value(a).unwrap() };
    let call = |open: &mut Open, a| open.lra_with(lra(a), &|_| {});

    let r = call(&mut open, json!({})).unwrap();
    assert_eq!(r["table"], "full table");
    assert_eq!(r["pairs"], 3);
    assert_eq!(r["q_scope"], "all pairs");
    assert_eq!(r["verdicts"]["supported"], 1);
    assert_eq!(r["list"][0]["pair"], "L1:R1", "by q");
    assert_eq!(r["list"][0]["change"], "+25% (95% CI +20% to +30%)");
    assert_eq!(r["tables"].as_array().unwrap().len(), 2);
    assert!(r["selected"].is_null());

    // Filters stay set until changed.
    let r = call(&mut open, json!({"verdict": "no_evidence"})).unwrap();
    assert_eq!(r["shown"], 1);
    let r = call(&mut open, json!({"limit": 0})).unwrap();
    assert_eq!(r["filter"]["verdict"], "no_evidence");
    assert_eq!(r["list"].as_array().unwrap().len(), 0);
    let r = call(
        &mut open,
        json!({"verdict": "", "search": "g1", "sort": "name"}),
    )
    .unwrap();
    assert_eq!(r["shown"], 1);
    assert_eq!(r["list"][0]["pair"], "G1:R1");
    let e = call(&mut open, json!({"verdict": "maybe"}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("supported, effect_without_coexpression"), "{e}");
    let e = call(&mut open, json!({"sort": "p"}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("q, effect, name"), "{e}");

    // A pair selected is drawn by view_map: its classes, then its exposure.
    let base = Base::load(&prefix, "auto", Theme::Dark).unwrap();
    let rx = receivers(&base, contacts);
    open.lra.as_mut().unwrap().receivers = Some(rx);
    let r = call(&mut open, json!({"pair": "L1:R1", "search": ""})).unwrap();
    let sel = &r["selected"];
    assert_eq!(sel["pair"], "L1:R1");
    assert_eq!(sel["verdict"], "supported");
    assert_eq!(sel["curve_points"], 4);
    assert_eq!(sel["drawn"], true);
    let map = &sel["map"]["classes"];
    let senders = map["sender (ligand+)"].as_u64().unwrap();
    assert_eq!(senders, 30, "the three left columns carry L1");
    let total: u64 = map
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_u64().unwrap())
        .sum();
    assert_eq!(total, (SIDE * SIDE) as u64);
    // The table's threshold (0): every receiver touching a sender.
    assert_eq!(sel["map"]["threshold"], 0.0);
    assert_eq!(
        sel["map"]["exposed"], 40,
        "the senders and the column beside them"
    );

    let state = |content: Vec<Content>| -> (serde_json::Value, bool) {
        let mut png = false;
        let mut state = serde_json::Value::Null;
        for c in content {
            match c {
                Content::Png(_) => png = true,
                Content::Text(t) => state = serde_json::from_str(&t).unwrap(),
                _ => {}
            }
        }
        (state, png)
    };
    let (s, png) = state(open.map(MapArgs::default()).unwrap());
    assert!(png);
    assert_eq!(s["pair"], "L1:R1");
    assert_eq!(s["pair_show"], "classes");
    assert_eq!(s["legend"]["groups"].as_array().unwrap().len(), 5);
    call(&mut open, json!({"show": "exposure"})).unwrap();
    let (s, _) = state(open.map(MapArgs::default()).unwrap());
    assert_eq!(s["pair_show"], "exposure");
    assert_eq!(s["legend"]["ramp"], "L1 contact exposure");
    // A layer asked of view_map draws the groups again.
    let args: MapArgs = serde_json::from_value(json!({"layer": "argmax"})).unwrap();
    let (s, _) = state(open.map(args).unwrap());
    assert!(s["pair"].is_null());
    assert_eq!(s["layer"], "argmax");

    // The subset table, by its list: the pair stays, its q subset-corrected.
    let r = call(&mut open, json!({"table": "l1:r1", "pair": "L1:R1"})).unwrap();
    assert_eq!(r["table"], "only L1:R1");
    assert_eq!(r["selected"]["subset_corrected"], true);
    assert!(r["q_scope"].as_str().unwrap().contains("--only subset"));
    let e = call(&mut open, json!({"pair": "X:Y"}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("no pair"), "{e}");
}

// * Reading and drawing that fails, or changes the receivers

/// A table that does not read leaves the layer as it was: a path new to
/// it is not added, and the table shown, read again and broken now, stays
/// shown as it was read.
#[test]
fn a_table_that_does_not_read_changes_nothing() {
    use crate::view::lra::TableRef;
    let dir = tempfile::tempdir().unwrap();
    let (prefix, _) = lra_run(dir.path());
    let run = crate::view::data::Run::open(&prefix).unwrap();
    let mut v = LraView::new(run.tables()).unwrap().unwrap();
    v.select("L1:R1").unwrap();
    let missing = TableRef {
        path: dir.path().join("none.lra.parquet"),
        curve: None,
        subset: None,
        options: LraOptions::default(),
    };
    assert!(v.load_path(missing).is_err());
    assert_eq!((v.tables.len(), v.at), (2, 0), "no table added");
    // The table shown, written over with what does not read.
    let shown = v.tables[0].clone();
    std::fs::write(&shown.path, "not a table").unwrap();
    assert!(v.load_path(shown).is_err());
    assert_eq!(v.table().r.label(), "full table");
    assert_eq!(v.rows().len(), 3);
    assert_eq!(v.selected_row().unwrap().name(), "L1:R1");
    assert!(v.curve().is_some());
}

/// The receivers are read for the options of the table shown: a table made
/// with other options reads them again (from the genes its manifest
/// records the identity left out), and a pair drawn on the others is not
/// shown on it. The FDR alone does not change them.
#[test]
fn receivers_are_read_again_for_a_table_made_with_other_options() {
    let dir = tempfile::tempdir().unwrap();
    let (prefix, contacts) = lra_run(dir.path());
    let base = Base::load(&prefix, "auto", Theme::Dark).unwrap();
    let mut tables = base.run.tables();
    tables[1].options = LraOptions {
        identity_excluded: vec!["L1".into(), "R1".into()],
        ..LraOptions::default()
    };
    let mut v = LraView::new(tables).unwrap().unwrap();
    v.receivers = Some(receivers(&base, contacts));
    v.select("L1:R1").unwrap();
    assert!(!v.needs_receivers());
    assert!(v.on_map(&base, 0).unwrap().is_some());
    v.load(1).unwrap();
    assert!(v.needs_receivers());
    assert!(v.shown().is_none(), "drawn on the other receivers");
    // Read from the run with the genes recorded: no reference needed.
    let on = v.on_map(&base, 0).unwrap().unwrap();
    assert_eq!(on.legend().iter().map(|l| l.2).sum::<usize>(), SIDE * SIDE);
    assert!(!v.needs_receivers() && v.shown().is_some());
}

/// A manifest from before the genes were recorded reads the pairs' files
/// again; one that moved is named.
#[test]
fn receivers_name_a_pair_file_that_is_gone() {
    let dir = tempfile::tempdir().unwrap();
    let (prefix, _) = lra_run(dir.path());
    let manifest = Path::new(&format!("{prefix}.pinto.json")).to_path_buf();
    let gone = dir.path().join("moved/pairs.tsv");
    let options = LraOptions {
        lr_pairs: Some(gone.to_string_lossy().into()),
        ..LraOptions::default()
    };
    let e = match RunFrame::open(&manifest, &options) {
        Ok(_) => panic!("the pairs file is gone"),
        Err(e) => format!("{e:#}"),
    };
    assert!(e.contains(&*gone.to_string_lossy()), "{e}");
}

/// `view_lra` checks all it is asked before it keeps any of it: a pair the
/// table switched to lacks changes neither the table, nor the filter, nor
/// what the map shows. A pair drawn stays drawn in the table switched to,
/// at that table's threshold.
#[test]
fn view_lra_changes_nothing_when_a_call_fails_and_redraws_on_a_table_switch() {
    use crate::view::mcp::{LraArgs, Open};
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let (prefix, contacts) = lra_run(dir.path());
    let mut open = Open::new(&prefix, None).unwrap();
    let call = |open: &mut Open, a: serde_json::Value| {
        let a: LraArgs = serde_json::from_value(a).unwrap();
        open.lra_with(a, &|_| {})
    };
    call(&mut open, json!({})).unwrap();
    let base = Base::load(&prefix, "auto", Theme::Dark).unwrap();
    open.lra.as_mut().unwrap().receivers = Some(receivers(&base, contacts));
    let r = call(&mut open, json!({"pair": "L1:R1"})).unwrap();
    assert_eq!(r["selected"]["drawn"], true);

    let e = call(
        &mut open,
        json!({"table": "l1:r1", "verdict": "no_evidence", "show": "exposure", "pair": "L1:G0"}),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("no pair"), "{e}");
    let r = call(&mut open, json!({})).unwrap();
    assert_eq!(r["table"], "full table");
    assert!(r["filter"]["verdict"].is_null());
    assert_eq!(r["show"], "classes");
    assert_eq!(r["selected"]["pair"], "L1:R1");
    assert_eq!(r["selected"]["drawn"], true);

    // Another table of the pair, at another threshold: drawn there too.
    let other = dir.path().join("other.lra.only-l1-r1.parquet");
    write_table(
        &other,
        &[("L1", "R1", "supported", 0.01, 25.0, 0.5)],
        "q corrected within the --only subset (1 of 3 pairs)",
    );
    let r = call(&mut open, json!({"table": other.to_string_lossy()})).unwrap();
    assert_eq!(r["selected"]["pair"], "L1:R1");
    assert_eq!(r["selected"]["drawn"], true);
    assert_eq!(r["selected"]["map"]["threshold"], 0.5);
}

/// On a real run: every pair's map counts the exposed and competent
/// receivers its table row counts, ties at a median threshold included.
/// `PINTO_LRA_MANIFEST` names the run's `.pinto.json`, `PINTO_LRA_TABLE`
/// its table (else the manifest's), `PINTO_LRA_EVERY` a stride over the
/// pairs: `cargo test --release real_tissue -- --ignored --nocapture`.
#[test]
#[ignore]
fn real_tissue_maps_count_what_the_table_counts() {
    use crate::lr_activity::ligand_effect::output::read_pair_rows;
    use crate::util::metadata::PintoMetadata;
    let Ok(manifest) = std::env::var("PINTO_LRA_MANIFEST") else {
        eprintln!("PINTO_LRA_MANIFEST unset: nothing to check");
        return;
    };
    let meta = PintoMetadata::read(Path::new(&manifest)).unwrap();
    let lra = &meta.outputs.lra;
    let table = std::env::var("PINTO_LRA_TABLE")
        .ok()
        .or_else(|| lra.table.clone())
        .expect("a table");
    let every: usize = std::env::var("PINTO_LRA_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let options = lra.options.clone().unwrap_or_default();
    eprintln!(
        "receivers by {}",
        if options.identity_excluded.is_empty() {
            "the pairs' files".to_string()
        } else {
            format!("{} genes recorded", options.identity_excluded.len())
        }
    );
    let frame = RunFrame::open(Path::new(&manifest), &options).unwrap();
    let (rows, _) = read_pair_rows(&table).unwrap();
    let (mut checked, mut median, mut wrong) = (0, 0, Vec::new());
    for row in rows.iter().step_by(every.max(1)) {
        let threshold = row.threshold.is_finite().then_some(row.threshold);
        let map = frame.pair(&row.ligand, &row.receptor, threshold).unwrap();
        let count = |f: &[bool]| f.iter().filter(|&&x| x).count();
        let got = (count(&map.exposed), count(&map.competent), map.n());
        let want = (row.n_exposed, row.n_competent, row.n_receivers);
        checked += 1;
        median += usize::from(row.threshold > 0.0);
        if got != want {
            wrong.push(format!("{}: map {got:?}, table {want:?}", row.name()));
        }
    }
    eprintln!(
        "{checked} pairs checked ({median} at a threshold above 0), {} differ",
        wrong.len()
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
