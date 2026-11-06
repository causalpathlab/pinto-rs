//! Predicting the held-out counts from a fitted run's profiles.

use super::{
    cell_seed, fit_side, log_composition, per_section, score_unit, split_cell, unit_parts,
    unit_side, update_delta, Agreement, BatchProfiles, CellParts, HoldoutInfo, Plan, Scored, Split,
    SplitKind, Totals, Unit,
};
use crate::link_community::outputs::write_propensity_matrix;
use crate::predict::PredictArgs;
use crate::util::common::*;
use crate::util::metadata::PintoMetadata;
use crate::util::profile_match::{em_proportions, match_profiles, training_feature_totals};
use crate::util::srt_pipeline::{
    preprocess_srt, FeatureAxisMode, SrtPreprocessConfig, SrtPreprocessed,
};
use legume_numeric::matrix::common_io::{mkdir_parent, read_name_list};
use legume_numeric::matrix::parquet::{write_named_table, Column};
use std::path::Path;

/// The communities' training shares, from the model's link communities (one
/// share per profile column); equal shares when the model has none.
fn community_shares(model: &str, k: usize, notes: &mut Vec<String>) -> Vec<f64> {
    let path = format!("{model}.link_community.parquet");
    let counts = crate::util::parquet_io::read_link_community(Path::new(&path))
        .ok()
        .map(|(_, _, counts)| counts)
        .filter(|c| c.len() == k && c.iter().sum::<usize>() > 0);
    match counts {
        Some(c) => {
            let total = c.iter().sum::<usize>() as f64;
            notes.push(
                "pooled K=1: one profile, the communities' profiles weighted by their \
                 training edge shares, under the same δ"
                    .into(),
            );
            c.iter().map(|&x| x as f64 / total).collect()
        }
        None => {
            notes.push(
                "pooled K=1: one profile, the communities' profiles weighted equally \
                 (the model lists no link communities), under the same δ"
                    .into(),
            );
            vec![1.0 / k as f64; k]
        }
    }
}

/// The slots a name list picks, by canonical name.
fn slots_named(
    path: &str,
    flag: &str,
    slot_names: &[Box<str>],
    kind: &data_beans::aux::feature_names::FeatureNameKind,
) -> anyhow::Result<Vec<bool>> {
    let listed: HashSet<Box<str>> = read_name_list(path)
        .map_err(|e| anyhow::anyhow!("reading {flag} {path}: {e}"))?
        .iter()
        .map(|n| kind.canonicalize(n))
        .collect();
    let on: Vec<bool> = slot_names.iter().map(|n| listed.contains(n)).collect();
    let n_on = on.iter().filter(|&&x| x).count();
    anyhow::ensure!(
        n_on > 0,
        "{flag} {path}: none of its {} names is a feature both the model and this sample have",
        listed.len()
    );
    info!(
        "{flag}: {n_on} of {} listed features are scored here",
        listed.len()
    );
    Ok(on)
}

/// `pinto predict` for an lc / dsvd model. Returns the per-cell propensity
/// and cell names it writes.
pub fn predict_profiles(
    args: &PredictArgs,
    meta: &PintoMetadata,
) -> anyhow::Result<(Mat, Vec<Box<str>>)> {
    let c = &args.common;
    let h = &args.holdout;
    let model = args.model.as_ref();
    mkdir_parent(&c.out)?;

    let gc_path = format!("{model}.feature_community.parquet");
    anyhow::ensure!(
        Path::new(&gc_path).is_file(),
        "{model}: {gc_path} does not exist, so there are no community profiles to score"
    );
    let (profiles, model_features) =
        crate::util::parquet_io::read_feature_community(Path::new(&gc_path))?;
    let k = profiles.ncols();
    info!(
        "{} model: scoring held-out units ({:?}) against {} feature × {} community profiles",
        meta.command,
        h.holdout_unit,
        profiles.nrows(),
        k
    );

    let peek_names = data_beans::convert::try_open_or_convert(&c.data_files[0])?.row_names()?;
    let feature_kind = args.feature_name_mode.resolve_kind(&peek_names);
    let SrtPreprocessed {
        data_vec,
        graph,
        batch_membership,
        ..
    } = preprocess_srt(SrtPreprocessConfig {
        common: c,
        ablation: None,
        fisher_weights: false,
        // The model's profiles are what is scored; δ is estimated below from
        // the fitting parts, not from this sample's whole counts.
        batch_effects: false,
        feature_axis: FeatureAxisMode::Rows,
        feature_kind: Some(feature_kind.clone()),
        cell_projection: false,
    })?;
    let cells = data_vec.column_names()?;
    let n_cells = cells.len();
    let row_names = data_vec.row_names()?;

    let matched = match_profiles(
        &row_names,
        &profiles,
        &model_features,
        &feature_kind,
        "held-out sample",
        n_cells,
    )?;
    let p_t = &matched.p_t;
    let g = p_t.ncols();
    let coverage = g as f32 / model_features.len().max(1) as f32;
    anyhow::ensure!(
        coverage >= args.min_feature_overlap,
        "the sample carries {:.1}% of the model's features, below --min-feature-overlap {:.1}%",
        100.0 * coverage,
        100.0 * args.min_feature_overlap
    );
    let mut slot_names: Vec<Box<str>> = vec![Box::from(""); g];
    for (row, slot) in matched.row_to_matched.iter().enumerate() {
        if let Some(s) = *slot {
            if slot_names[s].is_empty() {
                slot_names[s] = feature_kind.canonicalize(&row_names[row]);
            }
        }
    }

    // What is fitted on and what is scored.
    let named = |path: &str, flag: &str| slots_named(path, flag, &slot_names, &feature_kind);
    let ablated = h
        .ablate_features
        .as_deref()
        .map(|p| named(p, "--ablate-features"))
        .transpose()?;
    let eval_path = args
        .eval_features
        .as_deref()
        .or(h.ablate_features.as_deref());
    let eval = eval_path.map(|p| named(p, "--eval-features")).transpose()?;
    let (split, fit, scored): (Split, Vec<bool>, Vec<bool>) = match &ablated {
        Some(hidden) => {
            let scored: Vec<bool> = (0..g)
                .map(|j| hidden[j] && eval.as_ref().is_none_or(|e| e[j]))
                .collect();
            anyhow::ensure!(
                scored.iter().any(|&s| s),
                "--eval-features shares no feature with --ablate-features: nothing to score"
            );
            (Split::Ablate, hidden.iter().map(|&x| !x).collect(), scored)
        }
        None => (
            Split::Thinning(h.holdout_fit_fraction),
            vec![true; g],
            eval.clone().unwrap_or_else(|| vec![true; g]),
        ),
    };

    // Every cell split once into its fitting and scored parts, and the
    // sample's own totals on the matched window.
    let block = c.block_size.unwrap_or(1000).max(1);
    let blocks: Vec<(Vec<CellParts>, Vec<f64>)> = (0..n_cells)
        .step_by(block)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|lb| -> anyhow::Result<_> {
            let csc = data_vec.read_columns_csc(lb..(lb + block).min(n_cells))?;
            let mut totals = vec![0f64; g];
            let parts = (0..csc.ncols())
                .map(|j| {
                    let col = csc.col(j);
                    let nonzeros = || {
                        col.row_indices()
                            .iter()
                            .copied()
                            .zip(col.values().iter().copied())
                    };
                    for (row, v) in nonzeros() {
                        if let Some(s) = matched.row_to_matched[row] {
                            totals[s] += f64::from(v);
                        }
                    }
                    split_cell(
                        nonzeros(),
                        &matched.row_to_matched,
                        &split,
                        &fit,
                        &scored,
                        cell_seed(c.seed, lb + j),
                    )
                })
                .collect();
            Ok((parts, totals))
        })
        .collect::<anyhow::Result<_>>()?;
    let mut sample_totals = vec![0f64; g];
    let mut parts: Vec<CellParts> = Vec::with_capacity(n_cells);
    for (p, t) in blocks {
        parts.extend(p);
        sample_totals.iter_mut().zip(t).for_each(|(a, b)| *a += b);
    }

    let mut notes = vec![
        "ceiling: the held-out sample's own composition, one profile for every unit; it knows \
         the sample's marginal, but a per-unit mixture can still score above it"
            .to_string(),
    ];
    // The null: the training data's composition. `--null-from` names it; else
    // the model's own data files, when they are where its manifest says.
    let recorded: Vec<Box<str>> = meta
        .data_files
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(Into::into)
        .collect();
    let (null_totals, null_from) = match args.null_from.as_deref() {
        Some(files) => (
            training_feature_totals(files, c, &feature_kind, &slot_names)?,
            "--null-from".to_string(),
        ),
        None if !recorded.is_empty() && recorded.iter().all(|f| Path::new(f.as_ref()).exists()) => {
            info!("null: the model's own training data, as its manifest lists it");
            (
                training_feature_totals(&recorded, c, &feature_kind, &slot_names)?,
                "the model's training data".to_string(),
            )
        }
        None => {
            log::warn!(
                "no --null-from and the model's training data is not where its manifest says: \
                 the null is this sample's own composition, a second ceiling rather than a floor. \
                 Pass the training data"
            );
            notes.push("the null is the held-out sample's own composition (an oracle)".into());
            (sample_totals.clone(), "the held-out sample".to_string())
        }
    };
    let log_null = log_composition(&null_totals, &scored);
    let log_ceiling = log_composition(&sample_totals, &scored);
    let shares = community_shares(model, k, &mut notes);
    let agreement = eval
        .as_ref()
        .map(|_| Agreement::new((0..g).filter(|&j| scored[j]).collect(), g));
    if agreement.is_none() {
        notes.push(
            "spearman / pearson_log1p are written only with --eval-features: a per-pair \
             correlation over every feature costs a sort of the feature axis per pair"
                .into(),
        );
    }
    let plan = Plan {
        split,
        log_null: &log_null,
        log_ceiling: &log_ceiling,
        shares: &shares,
        agreement,
        iters: h.holdout_em_iters,
    };

    let units: Vec<(usize, Option<usize>)> = match h.holdout_unit {
        Unit::Pair => graph.edges.iter().map(|&(i, j)| (i, Some(j))).collect(),
        Unit::Cell => (0..n_cells).map(|i| (i, None)).collect(),
    };
    anyhow::ensure!(
        !units.is_empty(),
        "the held-out sample has no units to score"
    );

    // Batches: one δ each. A pair's batch is its first cell's (the graph keeps
    // sections apart).
    let mut batch_names: Vec<&str> = batch_membership.iter().map(|b| b.as_ref()).collect();
    batch_names.sort_unstable();
    batch_names.dedup();
    let batch_of: Vec<usize> = batch_membership
        .iter()
        .map(|b| batch_names.binary_search(&b.as_ref()).expect("listed"))
        .collect();
    let n_batches = batch_names.len();
    let mut delta = vec![vec![1.0f64; g]; n_batches];
    let build = |delta: &[Vec<f64>]| -> Vec<BatchProfiles> {
        delta
            .iter()
            .map(|d| BatchProfiles::new(p_t, d, &fit, &scored))
            .collect()
    };
    let mut profs = build(&delta);

    for round in 0..h.holdout_delta_iters {
        info!(
            "δ round {}/{} over the fitting parts ({} batches)...",
            round + 1,
            h.holdout_delta_iters,
            n_batches
        );
        // Per batch: observed counts per feature, and Σ_u n_u π_u (the
        // predicted totals are then Σ_k p_fit_k(g) A_k, no per-unit G-sweep).
        let zero = || {
            (
                vec![vec![0f64; g]; n_batches],
                vec![vec![0f64; k]; n_batches],
            )
        };
        let (obs_tot, alloc) = units
            .par_iter()
            .fold(zero, |(mut o, mut a), &(i, j)| {
                let b = batch_of[i];
                let fit_part = unit_side(&parts, i, j, fit_side);
                let mut pi = vec![0.0; k];
                if em_proportions(&fit_part, &profs[b].p_fit, plan.iters, &mut pi) {
                    let n: f64 = fit_part.iter().map(|o| o.1).sum();
                    for &(s, x) in &fit_part {
                        o[b][s] += x;
                    }
                    for (ac, &p) in a[b].iter_mut().zip(&pi) {
                        *ac += n * p;
                    }
                }
                (o, a)
            })
            .reduce(zero, |(mut o1, mut a1), (o2, a2)| {
                for b in 0..n_batches {
                    o1[b].iter_mut().zip(&o2[b]).for_each(|(x, y)| *x += y);
                    a1[b].iter_mut().zip(&a2[b]).for_each(|(x, y)| *x += y);
                }
                (o1, a1)
            });
        for b in 0..n_batches {
            let pred: Vec<f64> = (0..g)
                .map(|s| {
                    (0..k)
                        .map(|kk| profs[b].p_fit[(kk, s)] * alloc[b][kk])
                        .sum()
                })
                .collect();
            update_delta(&mut delta[b], &obs_tot[b], &pred, &fit);
        }
        profs = build(&delta);
    }
    if h.holdout_delta_iters == 0 {
        notes.push(
            "no δ: the model's profiles are applied unshifted, so a batch shift of this \
             sample counts against the model"
                .into(),
        );
    } else {
        notes.push(format!(
            "δ per batch from the fitting parts only ({} rounds); features hidden by \
             --ablate-features keep δ = 1",
            h.holdout_delta_iters
        ));
    }

    info!("Scoring {} units...", units.len());
    let scored_units: Vec<Scored> = units
        .par_iter()
        .map(|&(i, j)| {
            let (fit_part, test_part) = unit_parts(&parts, i, j);
            score_unit(&fit_part, &test_part, &profs[batch_of[i]], &plan)
        })
        .collect();

    // A cell's propensity: its own weights (cell units), or the mean of its
    // pairs' (pair units), the incident-edge reading lc publishes.
    let mut prop = Mat::zeros(n_cells, k);
    let mut degree = vec![0usize; n_cells];
    for (&(i, j), s) in units.iter().zip(&scored_units) {
        // A unit whose fit failed has no proportions and does not count.
        if s.pi.iter().sum::<f64>() <= 0.0 {
            continue;
        }
        for cell in std::iter::once(i).chain(j) {
            degree[cell] += 1;
            for (c, &p) in s.pi.iter().enumerate() {
                prop[(cell, c)] += p as f32;
            }
        }
    }
    for (i, &d) in degree.iter().enumerate() {
        if d > 0 {
            prop.row_mut(i).scale_mut(1.0 / d as f32);
        }
    }
    write_propensity_matrix(&c.out, &prop, &cells)?;
    write_predictive(&c.out, &units, &scored_units, &cells)?;

    let kept: Vec<&Scored> = scored_units.iter().filter(|s| s.n_test > 0.0).collect();
    let mut total = Totals::default();
    kept.iter().for_each(|s| total.add(s));
    let mean_of = |f: fn(&Scored) -> f32| -> Option<f64> {
        let v: Vec<f64> = kept
            .iter()
            .map(|s| f64::from(f(s)))
            .filter(|x| x.is_finite())
            .collect();
        (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
    };
    let info = HoldoutInfo {
        model: model.to_string(),
        model_command: meta.command.clone(),
        held_out: h.held_out,
        unit: h.holdout_unit,
        split: match plan.split {
            Split::Thinning(_) => SplitKind::Thinning,
            Split::Ablate => SplitKind::FeatureAblation,
        },
        fit_fraction: matches!(plan.split, Split::Thinning(_)).then_some(h.holdout_fit_fraction),
        seed: c.seed,
        n_units: units.len(),
        n_scored: total.n,
        n_features_matched: g,
        n_features_scored: scored.iter().filter(|&&s| s).count(),
        n_features_ablated: ablated.as_ref().map(|a| a.iter().filter(|&&x| x).count()),
        null_from,
        delta_iters: h.holdout_delta_iters,
        eval_count: total.count,
        eval_llik_per_count: total.per_count(total.llik),
        eval_null_llik_per_count: total.per_count(total.null),
        eval_pooled_k1_llik_per_count: total.per_count(total.pooled),
        eval_ceiling_llik_per_count: total.per_count(total.ceiling),
        per_section: per_section(
            &units
                .iter()
                .map(|&(i, _)| crate::util::ablation::section_of(&cells[i]))
                .collect::<Vec<_>>(),
            &scored_units,
        ),
        mean_spearman: mean_of(|s| s.spearman),
        mean_pearson_log1p: mean_of(|s| s.pearson_log1p),
        notes,
    };
    info!(
        "Held-out llik/count {:.4} vs training null {:.4} (gain {:+.4}), pooled K=1 {:.4}, \
         ceiling {:.4}, over {} of {} units",
        info.eval_llik_per_count,
        info.eval_null_llik_per_count,
        info.eval_llik_per_count - info.eval_null_llik_per_count,
        info.eval_pooled_k1_llik_per_count,
        info.eval_ceiling_llik_per_count,
        info.n_scored,
        info.n_units
    );

    let out_meta = PintoMetadata {
        command: "predict".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: crate::util::metadata::now_secs(),
        prefix: c.out.to_string(),
        data_files: Some(c.data_files.iter().map(|s| s.to_string()).collect()),
        coord_file: c.coord_files_joined(),
        n_cells,
        n_features: g,
        n_edges: Some(graph.edges.len()),
        n_communities: Some(k),
        outputs: crate::util::metadata::OutputFiles {
            cells: c
                .has_coordinates()
                .then(|| crate::util::parquet_io::cells_table_path(&c.out)),
            propensity: Some(format!("{}.propensity.parquet", c.out)),
            predictive: Some(format!("{}.predictive.parquet", c.out)),
            ..Default::default()
        },
        holdout: Some(info),
        ..Default::default()
    };
    let meta_path = format!("{}.pinto.json", c.out);
    out_meta.write(Path::new(&meta_path))?;
    info!("Wrote {meta_path}");
    Ok((prop, cells))
}

/// `{out}.predictive.parquet`, one row per unit, under the column names
/// cage's `pinto predict` and `senna predict` use for the same quantities:
/// `eval_llik` and `eval_count` over the scored part, `eval_llik_per_count`,
/// `eval_null_llik_per_count` under the training composition, `spearman` and
/// `pearson_log1p`; then `eval_pooled_k1_llik_per_count` under the pooled
/// profile. A unit with no scored counts has NaN per count, never 0, which
/// would read as perfect.
fn write_predictive(
    out: &str,
    units: &[(usize, Option<usize>)],
    scored: &[Scored],
    cells: &[Box<str>],
) -> anyhow::Result<()> {
    let key: Vec<Box<str>> = units
        .iter()
        .map(|&(i, j)| match j {
            Some(j) => format!("{}|{}", cells[i], cells[j]).into(),
            None => cells[i].clone(),
        })
        .collect();
    let left: Vec<Box<str>> = units.iter().map(|&(i, _)| cells[i].clone()).collect();
    let right: Vec<Box<str>> = units
        .iter()
        .map(|&(_, j)| j.map_or_else(|| Box::from(""), |j| cells[j].clone()))
        .collect();
    let f =
        |g: &dyn Fn(&Scored) -> f64| -> Vec<f32> { scored.iter().map(|s| g(s) as f32).collect() };
    let per = |x: f64, s: &Scored| {
        if s.n_test > 0.0 {
            x / s.n_test
        } else {
            f64::NAN
        }
    };
    let llik = f(&|s| if s.n_test > 0.0 { s.loglik } else { f64::NAN });
    let count = f(&|s| s.n_test);
    let llik_per = f(&|s| per(s.loglik, s));
    let null_per = f(&|s| per(s.loglik_null, s));
    let pooled_per = f(&|s| per(s.loglik_pooled, s));
    let spearman: Vec<f32> = scored.iter().map(|s| s.spearman).collect();
    let pearson: Vec<f32> = scored.iter().map(|s| s.pearson_log1p).collect();
    let n_fit = f(&|s| s.n_fit);
    let community: Vec<i64> = scored
        .iter()
        .map(|s| {
            s.pi.iter()
                .enumerate()
                .filter(|(_, &p)| p > 0.0)
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(-1, |(c, _)| c as i64)
        })
        .collect();
    let path = format!("{out}.predictive.parquet");
    write_named_table(
        &path,
        if units.first().is_some_and(|u| u.1.is_some()) {
            "cell_pair"
        } else {
            "cell"
        },
        &key,
        &[
            ("left".into(), Column::Str(&left)),
            ("right".into(), Column::Str(&right)),
            ("eval_llik".into(), Column::F32(&llik)),
            ("eval_count".into(), Column::F32(&count)),
            ("eval_llik_per_count".into(), Column::F32(&llik_per)),
            ("spearman".into(), Column::F32(&spearman)),
            ("pearson_log1p".into(), Column::F32(&pearson)),
            ("eval_null_llik_per_count".into(), Column::F32(&null_per)),
            (
                "eval_pooled_k1_llik_per_count".into(),
                Column::F32(&pooled_per),
            ),
            ("fit_count".into(), Column::F32(&n_fit)),
            ("community".into(), Column::I64(&community)),
        ],
    )?;
    info!("Wrote {path}");
    Ok(())
}
