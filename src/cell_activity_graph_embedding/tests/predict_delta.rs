//! `pinto predict` divides a query by the model's own batch effect for the
//! batches the model trained on, and estimates only the new ones.

use super::cell_level::fit;
use crate::test_support::section;
use crate::util::batch_effects::read_delta;
use crate::util::common::Mat;
use clap::Parser;
use legume_numeric::matrix::traits::IoOps;

/// `pinto predict` of `model` on `data` to `{dir}/{name}`, the per-feature
/// totals from the training files; its prefix.
fn predict(
    dir: &std::path::Path,
    (data, coords): (&str, &str),
    model: &str,
    training: &str,
    name: &str,
) -> String {
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        predict: crate::predict::PredictArgs,
    }
    let out = dir.join(name).to_str().unwrap().to_string();
    let mut argv = vec![
        "x",
        data,
        "-c",
        coords,
        "--coord-column-names",
        "x,y",
        "--model",
        model,
        "-o",
        &out,
        "--null-from",
    ];
    argv.extend(training.split(','));
    crate::predict::predict(&Cli::try_parse_from(argv).unwrap().predict).unwrap();
    out
}

/// Each cell's embedding row, by its name without the `@section` tag a
/// multi-file run adds.
fn embedding_by_cell(prefix: &str) -> std::collections::HashMap<String, Vec<f32>> {
    let m = Mat::from_parquet(&format!("{prefix}.cell_embedding.parquet")).unwrap();
    m.rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let name = r.split('@').next().unwrap().to_string();
            (name, m.mat.row(i).iter().copied().collect())
        })
        .collect()
}

/// How far `query`'s cells sit from where `model` placed the same cells
/// (the cells the model has; a new section's are left out).
fn moved(model: &str, query: &str) -> f32 {
    let (fitted, placed) = (embedding_by_cell(model), embedding_by_cell(query));
    let shared: Vec<(&Vec<f32>, &Vec<f32>)> = placed
        .iter()
        .filter_map(|(cell, row)| fitted.get(cell).map(|f| (row, f)))
        .collect();
    assert!(!shared.is_empty(), "{query} shares no cell with {model}");
    shared
        .iter()
        .flat_map(|(row, f)| row.iter().zip(f.iter()).map(|(a, b)| (a - b).abs()))
        .fold(0.0, f32::max)
}

#[test]
fn predict_divides_known_batches_by_the_models_delta_and_estimates_new_ones() {
    let dir = tempfile::tempdir().unwrap();
    let side = 8;
    let s1 = section(dir.path(), "s1", side, &|_| 1.0);
    let s2 = section(dir.path(), "s2", side, &|g| {
        if g % 3 == 0 {
            4.0
        } else {
            1.0
        }
    });
    // a new section, folded on other features than s2
    let s3 = section(dir.path(), "s3", side, &|g| {
        if g % 3 == 1 {
            8.0
        } else {
            1.0
        }
    });
    let training = format!("{},{}", s1.0, s2.0);
    let coords = format!("{},{}", s1.1, s2.1);
    let model = fit(dir.path(), &training, &coords, "model", &[]);
    let model_delta = read_delta(&format!("{model}.delta.parquet")).unwrap();

    // the training files again: every cell where the model put it
    let again = predict(dir.path(), (&training, &coords), &model, &training, "again");
    let d = moved(&model, &again);
    assert!(d < 1e-4, "predict moved the training cells by up to {d}");

    // one training section alone: one batch, nothing to estimate, so only
    // the model's δ can correct it
    let alone = predict(dir.path(), (&s1.0, &s1.1), &model, &training, "alone");
    let d = moved(&model, &alone);
    assert!(
        d < 1e-4,
        "predict moved a known section's cells by up to {d}"
    );

    // a known section beside a new one: the known one keeps the model's δ,
    // the new one is estimated and carried into the model's frame
    let mixed = predict(
        dir.path(),
        (&format!("{},{}", s1.0, s3.0), &format!("{},{}", s1.1, s3.1)),
        &model,
        &training,
        "mixed",
    );
    let d = moved(&model, &mixed);
    assert!(
        d < 1e-4,
        "predict moved the known section's cells by up to {d}"
    );
    let used = read_delta(&format!("{mixed}.delta.parquet")).unwrap();
    assert_eq!(used.batches.len(), 2);
    let model_row = model_delta.row_index(&used.rows);
    for (g, r) in model_row.iter().enumerate() {
        let r = r.expect("every row is a model row");
        assert_eq!(used.mean[(g, 0)], model_delta.mean[(r, 0)], "row {g}");
        // the fits' table in full: the known batch's spread is the model's
        assert_eq!(used.sd[(g, 0)], model_delta.sd[(r, 0)], "row {g}");
        assert!(
            model_delta.sd[(r, 0)] > 0.0,
            "row {g}: the model has no spread"
        );
        // the new section's fold over the known one is s3's over s1's
        let fold = used.mean[(g, 1)] / used.mean[(g, 0)];
        let want = if g % 3 == 1 { 8.0 } else { 1.0 };
        assert!(
            (fold / want - 1.0).abs() < 0.2,
            "row {g}: the new section's fold is {fold}, not {want}"
        );
    }
}

/// `pinto impute` with a cage model divides each reference cell's counts by
/// the run's δ for its batch before pooling: with one neighbour per query
/// cell, every imputed row is one training cell's counts so divided.
#[test]
fn impute_divides_the_reference_by_the_runs_delta() {
    let dir = tempfile::tempdir().unwrap();
    let side = 8;
    let s1 = section(dir.path(), "s1", side, &|_| 1.0);
    let s2 = section(dir.path(), "s2", side, &|g| {
        if g % 3 == 0 {
            4.0
        } else {
            1.0
        }
    });
    let training = format!("{},{}", s1.0, s2.0);
    let coords = format!("{},{}", s1.1, s2.1);
    let model = fit(dir.path(), &training, &coords, "model", &[]);
    let delta = read_delta(&format!("{model}.delta.parquet")).unwrap();

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        impute: crate::impute::ImputeArgs,
    }
    let out = dir.path().join("imputed").to_str().unwrap().to_string();
    let argv = [
        "x",
        &s2.0,
        "-c",
        &s2.1,
        "--coord-column-names",
        "x,y",
        "--model",
        &model,
        "-o",
        &out,
        "--impute-knn",
        "1",
    ];
    crate::impute::run_impute(&Cli::try_parse_from(argv).unwrap().impute).unwrap();
    let imputed = Mat::from_parquet(&format!("{out}.imputed.parquet"))
        .unwrap()
        .mat;

    // every training cell's counts divided by its file's δ column ("0", "1")
    let mut divided: Vec<Vec<f32>> = Vec::new();
    for (b, (file, _)) in [&s1, &s2].iter().enumerate() {
        let data = data_beans::convert::try_open_or_convert(file).unwrap();
        let x = data.read_columns_csc((0..side * side).collect()).unwrap();
        for col in x.col_iter() {
            let mut v = vec![0.0f32; 12];
            for (&g, &c) in col.row_indices().iter().zip(col.values()) {
                v[g] = c / delta.mean[(g, b)];
            }
            divided.push(v);
        }
    }
    for q in 0..imputed.nrows() {
        let row: Vec<f32> = imputed.row(q).iter().copied().collect();
        if row.iter().all(|&v| v == 0.0) {
            continue;
        }
        assert!(
            divided
                .iter()
                .any(|d| d.iter().zip(&row).all(|(a, b)| (a - b).abs() < 1e-3)),
            "imputed row {q} is no training cell's divided counts: {row:?}"
        );
    }
}
