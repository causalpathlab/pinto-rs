//! `pinto apply predict` divides a query by the model's own batch effect for the
//! batches the model trained on, and estimates only the new ones.

use super::cell_level::fit;
use crate::test_support::section;
use crate::util::batch_effects::read_delta;
use crate::util::common::Mat;
use clap::Parser;
use legume_numeric::matrix::traits::IoOps;

/// `pinto apply predict` of `model` on `data` to `{dir}/{name}`, the per-feature
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
    crate::predict::predict(
        &Cli::try_parse_from(argv).unwrap().predict,
        &Default::default(),
    )
    .unwrap();
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

/// Cells per side of each section.
const SIDE: usize = 8;

/// One cage model fitted on s1 and s2 (s2 folded by [`s2_fold`]), and s3,
/// a new section folded on other features than s2.
struct Trained {
    dir: std::path::PathBuf,
    s1: (String, String),
    s2: (String, String),
    s3: (String, String),
    training: String,
    coords: String,
    model: String,
}

/// The model is fitted once; each step reads it, and the steps that
/// rewrite it run after those that read what they rewrite.
#[test]
fn predict_and_impute_on_one_model() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let s1 = section(&dir, "s1", SIDE, &|_| 1.0);
    let s2 = section(&dir, "s2", SIDE, &s2_fold);
    let s3 = section(&dir, "s3", SIDE, &|g| if g % 3 == 1 { 8.0 } else { 1.0 });
    let training = format!("{},{}", s1.0, s2.0);
    let coords = format!("{},{}", s1.1, s2.1);
    let model = fit(&dir, &training, &coords, "model", &[]);
    let t = Trained {
        dir,
        s1,
        s2,
        s3,
        training,
        coords,
        model,
    };
    file_batches_are_named_by_their_stems(&t);
    predict_divides_known_batches_by_the_models_delta_and_estimates_new_ones(&t);
    impute_divides_the_reference_by_the_runs_delta(&t);
    a_model_without_a_delta_divides_nothing(&t);
}

/// `pinto apply predict` divides a query by the model's δ for the batches it
/// trained on and estimates only the new ones.
fn predict_divides_known_batches_by_the_models_delta_and_estimates_new_ones(t: &Trained) {
    let Trained {
        dir,
        s1,
        s3,
        training,
        coords,
        model,
        ..
    } = t;
    let (dir, model) = (dir.as_path(), model.as_str());
    let model_delta = read_delta(&format!("{model}.delta.parquet")).unwrap();

    // the training files again: every cell where the model put it
    let again = predict(dir, (training, coords), model, training, "again");
    let d = moved(model, &again);
    assert!(d < 1e-4, "predict moved the training cells by up to {d}");

    // one training section alone: one batch, nothing to estimate, so only
    // the model's δ can correct it
    let alone = predict(dir, (&s1.0, &s1.1), model, training, "alone");
    let d = moved(model, &alone);
    assert!(
        d < 1e-4,
        "predict moved a known section's cells by up to {d}"
    );

    // a known section beside a new one: the known one keeps the model's δ,
    // the new one is estimated and carried into the model's frame
    let mixed = predict(
        dir,
        (&format!("{},{}", s1.0, s3.0), &format!("{},{}", s1.1, s3.1)),
        model,
        training,
        "mixed",
    );
    let d = moved(model, &mixed);
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

/// A model whose manifest records no δ trained on undivided counts: a known
/// and a new section are neither estimated nor divided. It rewrites the
/// model's manifest, so it runs last.
fn a_model_without_a_delta_divides_nothing(t: &Trained) {
    let Trained {
        dir,
        s1,
        s3,
        training,
        model,
        ..
    } = t;
    let manifest = crate::util::metadata::manifest_path(model);
    let mut meta =
        crate::util::metadata::PintoMetadata::read(std::path::Path::new(&manifest)).unwrap();
    meta.outputs.batch_effects = None;
    meta.write(std::path::Path::new(&manifest)).unwrap();
    let undivided = predict(
        dir,
        (&format!("{},{}", s1.0, s3.0), &format!("{},{}", s1.1, s3.1)),
        model,
        training,
        "undivided",
    );
    assert!(
        !std::path::Path::new(&format!("{undivided}.delta.parquet")).exists(),
        "a model without a δ had its query estimated"
    );
}

/// `pinto apply impute` with a cage model divides each reference cell's counts by
/// the run's δ for its batch before pooling: with one neighbour per query
/// cell, every imputed row is one training cell's counts so divided. The
/// cells' batches come from the table the fit wrote beside its δ, or, for
/// a model fitted before it wrote one, from its coord_pairs: the same.
///
/// It removes the cells table and its record, so it runs after the steps
/// that read them.
fn impute_divides_the_reference_by_the_runs_delta(t: &Trained) {
    let Trained {
        dir, s1, s2, model, ..
    } = t;
    let delta = read_delta(&format!("{model}.delta.parquet")).unwrap();

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        impute: crate::impute::ImputeArgs,
    }
    let impute = |name: &str| {
        let out = dir.join(name).to_str().unwrap().to_string();
        let argv = [
            "x",
            &s2.0,
            "-c",
            &s2.1,
            "--coord-column-names",
            "x,y",
            "--model",
            model,
            "-o",
            &out,
            "--impute-knn",
            "1",
        ];
        crate::impute::run_impute(&Cli::try_parse_from(argv).unwrap().impute, &[])?;
        anyhow::Ok(
            Mat::from_parquet(&format!("{out}.imputed.parquet"))
                .unwrap()
                .mat,
        )
    };
    let manifest = crate::util::metadata::manifest_path(model);
    let mut meta =
        crate::util::metadata::PintoMetadata::read(std::path::Path::new(&manifest)).unwrap();
    assert!(
        meta.outputs.delta_cells.is_some(),
        "the fit records each cell's batch"
    );
    let imputed = impute("imputed").unwrap();

    // every training cell's counts divided by its file's δ column ("s1", "s2")
    let mut divided: Vec<Vec<f32>> = Vec::new();
    for (b, (file, _)) in [s1, s2].iter().enumerate() {
        let data = data_beans::convert::try_open_or_convert(file).unwrap();
        let x = data.read_columns_csc((0..SIDE * SIDE).collect()).unwrap();
        for col in x.col_iter() {
            let mut v = vec![0.0f32; 12];
            for (&g, &c) in col.row_indices().iter().zip(col.values()) {
                v[g] = c / delta.mean[(g, b)];
            }
            divided.push(v);
        }
    }
    let mut n_divided = 0;
    for q in 0..imputed.nrows() {
        let row: Vec<f32> = imputed.row(q).iter().copied().collect();
        if row.iter().all(|&v| v == 0.0) {
            continue;
        }
        n_divided += 1;
        assert!(
            divided
                .iter()
                .any(|d| d.iter().zip(&row).all(|(a, b)| (a - b).abs() < 1e-3)),
            "imputed row {q} is no training cell's divided counts: {row:?}"
        );
    }
    assert!(n_divided > 0, "nothing was imputed");

    // the table recorded but gone: an error, not the coord_pairs labels
    std::fs::remove_file(crate::util::batch_effects::delta_cells_path(model)).unwrap();
    assert!(
        impute("gone").is_err(),
        "a recorded cells table that is gone fell back to coord_pairs"
    );

    // an older model: no table of the cells' batches, so its coord_pairs
    meta.outputs.delta_cells = None;
    meta.write(std::path::Path::new(&manifest)).unwrap();
    let older = impute("older").unwrap();
    assert!(
        (&older - &imputed).abs().max() < 1e-6,
        "the coord_pairs labels divide as the fit's table does"
    );
}

/// The fold of the second training section on feature `g`.
fn s2_fold(g: usize) -> f32 {
    if g.is_multiple_of(3) {
        4.0
    } else {
        1.0
    }
}

/// The δ column of `delta` named `batch`.
fn column(delta: &crate::util::batch_effects::RunDelta, batch: &str) -> Vec<f32> {
    let b = delta
        .batches
        .iter()
        .position(|n| n.as_ref() == batch)
        .unwrap_or_else(|| panic!("no batch {batch} in {:?}", delta.batches));
    delta.mean.column(b).iter().copied().collect()
}

/// The predicted δ of a one-file query `prefix` is the model's column
/// `batch`, row for row.
fn takes_models_column(
    prefix: &str,
    model: &crate::util::batch_effects::RunDelta,
    batch: &str,
) -> bool {
    let used = read_delta(&format!("{prefix}.delta.parquet")).unwrap();
    let want = column(model, batch);
    used.batches.len() == 1
        && model
            .row_index(&used.rows)
            .iter()
            .enumerate()
            .all(|(g, r)| r.is_some_and(|r| used.mean[(g, 0)] == want[r]))
}

/// A data file left to be its own batch is named by its stem: in the δ,
/// each cell's batch beside it and the cells table, whatever order the
/// files come in, and a query file of the same stem takes that batch's δ.
fn file_batches_are_named_by_their_stems(t: &Trained) {
    use std::collections::BTreeSet;
    let Trained {
        dir,
        s1,
        s2,
        training,
        model,
        ..
    } = t;
    let delta = read_delta(&format!("{model}.delta.parquet")).unwrap();
    let stems = ["s1", "s2"].map(Box::<str>::from).to_vec();
    assert_eq!(delta.batches, stems);

    let by_cell = crate::util::batch_effects::read_cell_batches(
        &crate::util::batch_effects::delta_cells_path(model),
    )
    .unwrap();
    assert_eq!(by_cell.len(), 2 * SIDE * SIDE);
    for (cell, batch) in &by_cell {
        assert!(cell.starts_with(&format!("{batch}_")), "{cell} in {batch}");
    }
    let manifest = crate::util::metadata::manifest_path(model);
    let meta = crate::util::metadata::PintoMetadata::read(std::path::Path::new(&manifest)).unwrap();
    let cells = crate::util::parquet_io::read_cells_table(
        std::path::Path::new(meta.outputs.cells.as_deref().unwrap()),
        None,
    )
    .unwrap();
    let in_table: BTreeSet<&str> = cells.batches.iter().flatten().map(|b| &**b).collect();
    assert_eq!(in_table, BTreeSet::from(["s1", "s2"]));
    let json = std::fs::read_to_string(&manifest).unwrap();
    assert!(
        json.contains("\"batch_labels\": \"file-names\""),
        "the manifest records how its batches were named: {json}"
    );

    // the files the other way round: the same batches, by name
    let swapped = fit(
        dir,
        &format!("{},{}", s2.0, s1.0),
        &format!("{},{}", s2.1, s1.1),
        "swapped",
        &[],
    );
    let swapped = read_delta(&format!("{swapped}.delta.parquet")).unwrap();
    assert_eq!(swapped.batches, stems);
    for b in ["s1", "s2"] {
        for (x, y) in column(&delta, b).iter().zip(column(&swapped, b)) {
            assert!((x / y - 1.0).abs() < 1e-3, "{b}: {x} against {y}");
        }
    }

    // s2's sample again, elsewhere: the model's s2, by its stem
    let elsewhere = dir.join("elsewhere");
    std::fs::create_dir(&elsewhere).unwrap();
    let again = section(&elsewhere, "s2", SIDE, &s2_fold);
    let query = predict(dir, (&again.0, &again.1), model, training, "query");
    assert!(
        takes_models_column(&query, &delta, "s2"),
        "a query file named s2 did not take the model's s2"
    );
}

/// Two data files of one stem would be one batch name: refused.
#[test]
fn data_files_of_one_stem_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = (dir.path().join("a"), dir.path().join("b"));
    std::fs::create_dir(&a).unwrap();
    std::fs::create_dir(&b).unwrap();
    let s1 = section(&a, "s1", 8, &|_| 1.0);
    let s2 = section(&b, "s1", 8, &|_| 2.0);
    let e = super::cell_level::try_fit(
        dir.path(),
        &format!("{},{}", s1.0, s2.0),
        &format!("{},{}", s1.1, s2.1),
        "model",
        &[],
    )
    .unwrap_err();
    let e = format!("{e:#}");
    assert!(e.contains("same name s1"), "{e}");
}
