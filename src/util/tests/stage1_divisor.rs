//! The stage-1 divisor a stage-2 fit reads (`--offset`): its tables round
//! trip, a constant one changes nothing (the divide and the projection both),
//! a planted per-cell × row fold is divided out, and the batch options are
//! refused beside it.

use super::ignore_batch::two_batches;
use crate::util::common::*;
use crate::util::input::SrtInputArgs;
use crate::util::srt_pipeline::{
    preprocess_srt, BatchEstimate, FeatureAxisMode, SrtPreprocessConfig,
};
use crate::util::stage1_divisor::{Stage1Divisor, Stage1Tables};
use clap::Parser;
use data_beans::aux::feature_names::FeatureNameKind;

/// Tables over `cells` and `rows` with `k` communities in batches
/// `batches` (cell `i` in `batch_of(i)`): `θ` and the profiles as given,
/// `p̄` the `n`-weighted mean share, unit totals.
pub(crate) fn tables(
    cells: &[Box<str>],
    rows: &[Box<str>],
    k: usize,
    batches: &[&str],
    batch_of: impl Fn(usize) -> usize,
    theta: impl Fn(usize, usize) -> f32,
    profile: impl Fn(usize, usize, usize) -> f32,
) -> Stage1Tables {
    let (n, g, nb) = (cells.len(), rows.len(), batches.len());
    let theta: Vec<f32> = (0..n * k).map(|x| theta(x / k, x % k)).collect();
    let mut profiles = vec![0f32; nb * g * k];
    for b in 0..nb {
        for r in 0..g {
            for t in 0..k {
                profiles[(b * g + r) * k + t] = profile(b, r, t);
            }
        }
    }
    let batch: Vec<u32> = (0..n).map(|i| batch_of(i) as u32).collect();
    let pbar: Vec<f32> = (0..g)
        .map(|r| {
            (0..n)
                .map(|i| {
                    let b = batch[i] as usize;
                    (0..k)
                        .map(|t| theta[i * k + t] * profiles[(b * g + r) * k + t])
                        .sum::<f32>()
                })
                .sum::<f32>()
                / n as f32
        })
        .collect();
    Stage1Tables {
        cells: cells.to_vec(),
        n: vec![1.0; n],
        batch,
        theta,
        k,
        batches: batches.iter().map(|&b| b.into()).collect(),
        features: rows.to_vec(),
        profiles,
        pbar,
        phi: vec![0.25; g],
        sigma2: vec![1.0; g * nb],
    }
}

fn open(files: &[Box<str>]) -> SparseIoVec {
    crate::util::input::open_fit_data(files, &FeatureNameKind::Exact).unwrap()
}

/// What is written is what is read back.
#[test]
fn the_tables_read_back_as_written() {
    let dir = tempfile::tempdir().unwrap();
    let cells: Vec<Box<str>> = ["a", "b", "c"].map(Into::into).to_vec();
    let rows: Vec<Box<str>> = ["g0", "g1"].map(Into::into).to_vec();
    let t = tables(
        &cells,
        &rows,
        2,
        &["D1", "D2"],
        |i| i % 2,
        |i, k| (i + k + 1) as f32 / 4.0,
        |b, r, k| (b + 2 * r + k + 1) as f32 / 10.0,
    );
    let d = dir.path().to_str().unwrap();
    t.write(d).unwrap();
    let back = Stage1Tables::read(d, &crate::util::memory::Budget::new(None, None)).unwrap();
    assert_eq!(back, t);
}

/// Each column of `x` scaled to a total of 1e4.
fn at_library_scale(x: &nalgebra_sparse::CscMatrix<f32>) -> Vec<Vec<(usize, f32)>> {
    x.col_iter()
        .map(|c| {
            let total: f32 = c.values().iter().sum();
            c.row_indices()
                .iter()
                .zip(c.values())
                .map(|(&r, &v)| (r, v * 1e4 / total))
                .collect()
        })
        .collect()
}

fn assert_columns(x: &nalgebra_sparse::CscMatrix<f32>, want: &[Vec<(usize, f32)>]) {
    for (j, (col, want)) in x.col_iter().zip(want).enumerate() {
        let got: Vec<(usize, f32)> = col
            .row_indices()
            .iter()
            .copied()
            .zip(col.values().iter().copied())
            .collect();
        assert_eq!(got.len(), want.len(), "cell {j}");
        for (&(r, v), &(wr, w)) in got.iter().zip(want) {
            assert_eq!(r, wr);
            assert!(
                (v - w).abs() <= 1e-3 * w.max(1.0),
                "cell {j} row {r}: {v} vs {w}"
            );
        }
    }
}

/// One community in one batch, every share its pooled share: `d = 1`, so
/// the division only puts each cell at 1e4, as no divisor and a
/// normalisation to 1e4 would.
#[test]
fn a_constant_divisor_equals_none() {
    let dir = tempfile::tempdir().unwrap();
    let files = two_batches(dir.path(), 30);
    let data = open(&files);
    let (cells, rows) = (data.column_names().unwrap(), data.row_names().unwrap());
    let g = rows.len();
    let t = tables(
        &cells,
        &rows,
        1,
        &["all"],
        |_| 0,
        |_, _| 1.0,
        |_, _, _| 1.0 / g as f32,
    );
    let s1 = Stage1Divisor::align(&t, &cells, &rows).unwrap();
    for i in 0..cells.len() {
        for r in 0..g {
            assert!((s1.at(i, r) - 1.0).abs() < 1e-6);
        }
    }
    let raw = data.read_columns_csc(0..cells.len()).unwrap();
    let mut divided = raw.clone();
    s1.divide_columns(&mut divided, |j| j);
    assert_columns(&divided, &at_library_scale(&raw));
}

/// Counts planted as a per-cell scale times a base times a per-cell × row
/// fold `d_gi` (two communities mixed per cell, profiles per batch) come
/// back as the base at 1e4: the fold is divided out and the scale absorbed,
/// so the projection does not see the scale either.
#[test]
fn a_planted_multiplicative_divisor_is_divided_out() {
    let dir = tempfile::tempdir().unwrap();
    let (n, g, k) = (20usize, 6usize, 2usize);
    let rows: Vec<String> = (0..g).map(|r| format!("GENE{r}")).collect();
    let rows_ref: Vec<&str> = rows.iter().map(String::as_str).collect();
    let theta = |i: usize, t: usize| {
        if t == 0 {
            (i % 5) as f32 / 4.0
        } else {
            1.0 - (i % 5) as f32 / 4.0
        }
    };
    let profile = |b: usize, r: usize, t: usize| (1 + r + 3 * t + b) as f32 / 40.0;
    let base = |i: usize, r: usize| (2 + (i + r) % 4) as f32;
    let scale = |i: usize| 1.0 + (i % 7) as f32;
    // the planted fold is the tables' own d, so plant with them
    let names: Vec<Box<str>> = (0..n).map(|i| format!("c{i}").into()).collect();
    let row_names: Vec<Box<str>> = rows.iter().map(|r| r.as_str().into()).collect();
    let planted = tables(
        &names,
        &row_names,
        k,
        &["D1", "D2"],
        |i| i % 2,
        theta,
        profile,
    );
    let s_planted = Stage1Divisor::align(&planted, &names, &row_names).unwrap();
    let write = |name: &str, scaled: bool| {
        let triplets: Vec<(u64, u64, f32)> = (0..n)
            .flat_map(|i| {
                let s = &s_planted;
                let c = if scaled { scale(i) } else { 1.0 };
                (0..g).map(move |r| (r as u64, i as u64, c * base(i, r) * s.at(i, r)))
            })
            .collect();
        crate::test_support::write_zarr(dir.path(), name, &rows_ref, n, &triplets)
    };
    let (plain, scaled) = (open(&[write("x", false)]), open(&[write("y", true)]));
    let (cells, data_rows) = (plain.column_names().unwrap(), plain.row_names().unwrap());
    // the same tables, on the data's own names for the cells
    let t = Stage1Tables {
        cells: cells.clone(),
        ..planted.clone()
    };
    let s1 = Stage1Divisor::align(&t, &cells, &data_rows).unwrap();
    let base_x = nalgebra_sparse::CscMatrix::from(&{
        let mut coo = nalgebra_sparse::CooMatrix::new(g, n);
        for i in 0..n {
            for r in 0..g {
                coo.push(r, i, base(i, r));
            }
        }
        coo
    });
    let want = at_library_scale(&base_x);
    for data in [&plain, &scaled] {
        let mut x = data.read_columns_csc(0..n).unwrap();
        s1.divide_columns(&mut x, |j| j);
        assert_columns(&x, &want);
    }
    let (a, b) = (
        s1.project_columns(&plain, 4, None).unwrap(),
        s1.project_columns(&scaled, 4, None).unwrap(),
    );
    let gap = (&a.proj - &b.proj).abs().max();
    assert!(gap < 1e-4, "a per-cell scale moved the projection by {gap}");
}

/// A cell of the data the tables lack is refused, by name.
#[test]
fn a_cell_the_tables_lack_is_refused() {
    let cells: Vec<Box<str>> = ["a", "b"].map(Into::into).to_vec();
    let rows: Vec<Box<str>> = ["g0"].map(Into::into).to_vec();
    let t = tables(&cells, &rows, 1, &["all"], |_| 0, |_, _| 1.0, |_, _, _| 1.0);
    let wanted: Vec<Box<str>> = ["a", "zz"].map(Into::into).to_vec();
    let e = Stage1Divisor::align(&t, &wanted, &rows)
        .unwrap_err()
        .to_string();
    assert!(e.contains("cell zz"), "{e}");
}

/// `--offset` refuses the options that set a batch-effect estimate up,
/// each named, before anything is read.
#[test]
fn the_batch_options_are_refused_beside_an_offset() {
    let dir = tempfile::tempdir().unwrap();
    let files = two_batches(dir.path(), 10);
    let labels = dir.path().join("labels.txt");
    std::fs::write(&labels, "a\n".repeat(10)).unwrap();
    let batch_files = format!("{0},{0}", labels.to_str().unwrap());
    let clones = dir.path().join("x.clones.parquet");
    let out = dir.path().join("out").to_str().unwrap().to_string();
    let offset = dir.path().join("nowhere").to_str().unwrap().to_string();
    for (flag, extra) in [
        ("--batch-files", vec!["--batch-files", batch_files.as_str()]),
        ("--auto-batch", vec!["--auto-batch"]),
        (
            "--cnv-clones",
            vec!["--cnv-clones", clones.to_str().unwrap()],
        ),
    ] {
        let mut argv = vec!["pinto"];
        argv.extend(files.iter().map(|f| f.as_ref()));
        argv.extend_from_slice(&["-o", &out, "--no-qc"]);
        argv.extend_from_slice(&extra);
        let args = SrtInputArgs::try_parse_from(argv).unwrap();
        let e = preprocess_srt(SrtPreprocessConfig {
            common: &args,
            offset: Some(&offset),
            ablation: None,
            batch_effects: BatchEstimate::Fit,
            feature_axis: FeatureAxisMode::Rows,
            feature_kind: None,
            cell_projection: false,
        })
        .err()
        .expect("refused")
        .to_string();
        assert!(
            e.starts_with(&format!("{flag} sets up a batch-effect estimate"))
                && e.contains("a stage-2 fit takes no batch effect")
                && e.ends_with(&format!("Drop {flag}")),
            "{e}"
        );
    }
}
