//! Pseudobulk + propensity-stratified sample-permutation LR activity test.
//!
//! Cells are collapsed into pseudobulk samples = (batch × propensity-bin),
//! where the propensity bin is the sign-LSH binary-sort code from an
//! SVD'd random projection of gene expression (within-batch centred).
//! Each cell carries soft membership over the link communities (the
//! fraction of its within-community edge instances falling in each), and
//! per (community, sample) we accumulate membership-weighted gene sums for
//! the LR genes, giving one pseudobulk profile per sample per community.
//! Strata ARE link communities: every within-community edge is enumerated
//! in both orientations, so every statistic here is symmetric in the pair
//! and no endpoint plays a privileged role. (An earlier directional design
//! paired communities into a -> b strata; it was removed after its arm
//! measured statistically inert and its roles proved to descend from the
//! edge list's arbitrary serialization order — see `orientation`.)
//!
//! Per-(community, sample) gene rates are estimated as `Gamma(num + a0,
//! denom + b0)` posteriors with a `Gamma(1, 1)` prior, calibrated via
//! `legume_numeric::param::dmatrix_gamma::GammaMatrix`. The variational log mean
//! `E[log λ] = ψ(a) − log(b)` (rather than `log1p(num/denom)`) is the
//! input to the test: a sample with `num = 0` and small `denom` lands
//! well above one with `num = 0` and large `denom`, breaking the
//! zero-count tie pile-up. NB-Fisher-info gene weights are baked into
//! these log-mean matrices once at the collapse boundary.
//!
//! For each (batch, community, LR pair) the statistic is a **weighted
//! covariance** between the pseudobulk log-mean of L and of R across
//! samples (sample weight = the membership mass behind the sample).
//! Covariance (rather than correlation) preserves L-R magnitude
//! information so pairs separate cleanly under restandardization.
//!
//! The null is sample-level permutation of L within propensity-stratified
//! buckets — shuffles are restricted to samples sharing the top
//! `shuffle_stratify_dim` bits of the propensity code, so the cell-type
//! marginal is preserved across permutations. Per shuffle, a *fresh*
//! log-posterior sample (delta method: `Normal(ψ(a) − log(b), ψ'(a))`) is
//! drawn and used as the per-permutation log expression for both genes.
//! The same draw is shared across all pairs in a given shuffle so
//! cross-pair dependence (and the WY guarantee) is preserved; sparse
//! pseudobulks correctly contribute a wider null than dense ones.
//!
//! Inference layers (per stratum):
//! - `p_empirical` / `p_z` — per-pair permutation diagnostics.
//! - `z_re` / `p_re` — Efron-Tibshirani restandardization: `(stat_obs - μ) / σ`
//!   with `(μ, σ)` = robust (median, 1.4826·MAD) of `stat_obs` across pairs.
//!   Two-sided p; sign restriction (active LR ⇒ positive `z_re`) applied at
//!   the reporting layer. Strata with MAD ≤ 1e-4 are flagged uncalibrated.
//! - `fwer_wy` — Westfall-Young single-step minP (FWER); same shuffle σ_k
//!   applied to every pair so cross-pair dependence is preserved.

use super::{lra_outputs, recorded};
use crate::lr_activity::args::{LraMethod, LraSimulateArgs, SrtLrActivityArgs};
use crate::lr_activity::input::{load_contacts, record_manifest, LraInput};
use crate::lr_activity::io::*;
use crate::lr_activity::orientation::CommunityStrata;
use crate::lr_activity::outputs::{
    pvalue_histogram, write_lr_activity, write_lr_activity_json, LrActivityRow, StratumEntry,
};
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use data_beans::alg::gene_weighting::fisher_weights_from_stats;
use data_beans::alg::random_projection::{binary_sort_columns, RandProjOps};
use legume_numeric::matrix::common_io::mkdir_parent;
use legume_numeric::matrix::membership::GeneIndexResolver;
use pseudobulk::{assign_samples, collapse_pseudobulk, load_edges, stratum_present};
use score::{score_pairs_for_stratum, ScoreInputs, StratumJob};

mod pseudobulk;
mod score;
pub(crate) use pseudobulk::{derive_cell_batch_labels, load_expr_data};
#[cfg(test)]
pub(crate) use score::{restandardization_scale, testable_weighted_cov, weighted_cov};

/// Pseudo-batch label written when no per-edge batch is on file.
pub const BATCH_LABEL_ALL: &str = "all";
/// Pseudo-batch label for pooled-across-batches rows emitted alongside per-batch rows.
pub const BATCH_LABEL_META: &str = "pooled";

/// Don't compute a statistic when the stratum has fewer than this many
/// samples — correlation + permutation are too noisy below this.
const MIN_SAMPLES_PER_STRATUM: usize = 4;

pub fn fit_srt_lr_activity(args: &SrtLrActivityArgs) -> anyhow::Result<()> {
    let input = args.input.resolve()?;
    let out = input.out(args.out.as_deref())?;
    mkdir_parent(&out)?;
    if args.method == LraMethod::ContactEffect {
        return contact_effect(args, &input, &out);
    }
    anyhow::ensure!(
        args.only.is_empty(),
        "--only applies to --method contact-effect; the co-expression methods test the \
         pairs of --lr-pairs"
    );
    let lc_prefix = input.lc_prefix()?;
    let lc_prefix = lc_prefix.as_str();
    // The `--from` run's manifest, which every method records into.
    let manifest = input
        .run
        .as_ref()
        .expect("lc_prefix checked")
        .manifest
        .as_path();
    let edge_scores_only = args.method == LraMethod::CoexpressionScores;

    /////////////////////////////////////////////////
    // 1. Load expression + resolve gene index map //
    /////////////////////////////////////////////////
    info!("Loading expression data...");
    let data_vec = load_expr_data(&input.data_files, input.preload_data)?;

    let row_names = data_vec.row_names()?;
    let cell_names = data_vec.column_names()?;
    let n_cells = data_vec.num_columns();

    // Ligand and receptor names name GENES, so they are resolved against the
    // gene axis. Against raw row names they cannot resolve at all on a
    // splice-channelized matrix: the resolver aliases on `--gene-delimiter`
    // (default `_`) and never on `/`, so `GENE1` does not reach
    // `GENE1/count/spliced` and every pair is dropped.
    let feature_axis = FeatureAxis::resolve_or_identity(&row_names)?;
    let gene_names: Vec<Box<str>> = feature_axis.feature_names().to_vec();
    let n_genes = feature_axis.n_features();
    if feature_axis.is_channelized() {
        info!(
            "Feature axis: {} rows carry splice channels over {} genes; a \
             ligand or receptor is scored on its gene, both tracks summed",
            row_names.len(),
            n_genes
        );
    }

    let gene_resolver =
        GeneIndexResolver::build(&gene_names, args.gene_delimiter, args.gene_allow_prefix);
    if gene_resolver.alias_collisions() > 0 {
        warn!(
            "{} gene-name aliases are shared by more than one row (duplicated \
             symbols, most likely); an LR pair naming one resolves to a single \
             arbitrary row. Name genes by their full row name to disambiguate.",
            gene_resolver.alias_collisions()
        );
    }
    let cell_to_col: HashMap<Box<str>, usize> = cell_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    //////////////////////////////////////
    // 2. Parse LR pairs, resolve genes //
    //////////////////////////////////////
    let lr_pairs_file = args
        .lr_pairs
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--lr-pairs is required by the co-expression methods"))?;
    info!("Reading LR pairs from {}...", lr_pairs_file);
    let raw_pairs = read_lr_pairs(lr_pairs_file)?;
    let mut resolved_pairs: Vec<(Box<str>, Box<str>, usize, usize)> = Vec::new();
    let mut missing = 0usize;
    for (l, r) in raw_pairs {
        match (gene_resolver.resolve(&l), gene_resolver.resolve(&r)) {
            (Some(li), Some(ri)) => resolved_pairs.push((l, r, li, ri)),
            _ => missing += 1,
        }
    }
    if missing > 0 {
        // warn, not info: at the default log level an info line is invisible,
        // and a pairs file in the wrong naming convention would silently run
        // on whatever fraction happened to match.
        warn!(
            "Skipped {} of {} LR pairs with unresolved gene names; \
             check the pair file's naming against the matrix row names \
             (--gene-delimiter aliases compound names)",
            missing,
            missing + resolved_pairs.len()
        );
    }
    anyhow::ensure!(
        !resolved_pairs.is_empty(),
        "no LR pairs resolved against expression row names"
    );
    info!("Resolved {} LR pairs", resolved_pairs.len());

    let (anchor_edges, edges) = load_edges(lc_prefix, &cell_to_col)?;

    // Over the anchor list: a community that exists only on expression pairs
    // still has to be a valid stratum label.
    let n_communities = (anchor_edges.iter().map(|e| e.2).max().unwrap_or(0) as usize) + 1;

    // One stratum per link community; edges bridging two communities sit
    // out. See `orientation` for why the directional design was removed.
    let strata_map = CommunityStrata::from_edge_modes(&anchor_edges, &edges, n_cells);
    let n_strata_total = strata_map.n_strata();
    info!(
        "{} cells, {} edges, {} communities -> {} within-community strata",
        n_cells,
        edges.len(),
        n_communities,
        n_strata_total,
    );

    // Sparse-stratum filter: strata with too few edges can't calibrate (most
    // pseudobulk samples will be empty / have constant L or R, collapsing
    // stat_obs to 0 and breaking restandardization).
    let active_strata: HashSet<u32> = (0..n_strata_total as u32)
        .filter(|&s| strata_map.edges_in(s as usize) >= args.min_edges_per_community)
        .collect();
    let n_skipped = n_strata_total - active_strata.len();
    if n_skipped > 0 {
        info!(
            "Skipping {} sparse strata (< {} edge instances each)",
            n_skipped, args.min_edges_per_community
        );
    }

    ////////////////////////////////////////////////
    // 4. Per-gene total counts (filter LR pairs) //
    ////////////////////////////////////////////////
    // One pass, two consumers. `sum()` on the gene-axis statistics IS the
    // per-gene total the count filter needs, and the same statistics give the
    // NB precisions further down, so a separate totals pass would be a second
    // full read of the matrix for a number already in hand.
    info!("Computing per-gene statistics...");
    let (_, gene_stats) = feature_axis.running_stats(&data_vec, input.block_size, "NB-Fisher")?;
    let gene_sum: Vec<f32> = gene_stats.sum().to_vec();
    let fisher_all = fisher_weights_from_stats(&gene_stats, n_cells);
    let pre_filter_n = resolved_pairs.len();
    let real_pairs: Vec<(Box<str>, Box<str>, usize, usize)> = resolved_pairs
        .into_iter()
        .filter(|(_, _, li, ri)| {
            gene_sum[*li] >= args.min_gene_count && gene_sum[*ri] >= args.min_gene_count
        })
        .collect();
    if pre_filter_n - real_pairs.len() > 0 {
        info!(
            "Dropped {} LR pairs whose L or R has < {} total counts",
            pre_filter_n - real_pairs.len(),
            args.min_gene_count
        );
    }
    anyhow::ensure!(
        !real_pairs.is_empty(),
        "no LR pairs survive --min-gene-count={}",
        args.min_gene_count
    );

    //////////////////////////////////////////////////
    // 8. Read just the LR-gene rows (dense, small) //
    //////////////////////////////////////////////////
    let mut lr_genes: Vec<usize> = Vec::new();
    let mut gene_to_local: HashMap<usize, usize> = HashMap::default();
    for &(_, _, li, ri) in &real_pairs {
        for g in [li, ri] {
            if let std::collections::hash_map::Entry::Vacant(e) = gene_to_local.entry(g) {
                e.insert(lr_genes.len());
                lr_genes.push(g);
            }
        }
    }
    // `lr_genes` are gene ids; the backend is read by row, so expand each gene
    // to the rows carrying it and fold them back afterwards. On a matrix
    // without channels this is the identity and the read is unchanged.
    let mut rows_to_read: Vec<usize> = Vec::with_capacity(lr_genes.len());
    let mut row_owner: Vec<usize> = Vec::with_capacity(lr_genes.len());
    for r in 0..row_names.len() {
        if let Some(&local) = gene_to_local.get(&feature_axis.feature_of_row(r)) {
            rows_to_read.push(r);
            row_owner.push(local);
        }
    }
    info!(
        "Reading {} rows for {} LR genes from backend...",
        rows_to_read.len(),
        lr_genes.len()
    );
    let x_rows = data_vec.read_rows_dmatrix(rows_to_read.iter().copied())?;
    // Column-outer and parallel, because `DMatrix` is column-major: a row-outer
    // walk strides by `nrows` and takes a cache miss per element.
    let x_lr = build_columns_par(lr_genes.len(), n_cells, |col, dst| {
        let src = x_rows.column(col);
        for (r, &local) in row_owner.iter().enumerate() {
            dst[local] += src[r];
        }
    });

    ///////////////////////////////////////////////////////////
    // Descriptive mode: per-batch edge scores, then done.   //
    ///////////////////////////////////////////////////////////
    if edge_scores_only {
        use crate::lr_activity::edge_scores::*;

        let distinct: HashSet<&str> = edges.iter().filter_map(|e| e.3.as_deref()).collect();
        // A label set that is entirely numeric is the fingerprint of a
        // coord_pairs table from before batch labels were exported as
        // strings: the numbers are a spatial offset, not identities, and a
        // score matrix keyed by them cannot be joined to anything.
        if !distinct.is_empty() && distinct.iter().all(|b| b.parse::<f64>().is_ok()) {
            warn!(
                "batch labels {:?} are all numeric; this looks like a run from \
                 before batch labels were exported. The scores will be keyed by \
                 a meaningless offset; re-run `pinto lc` to get identifiable \
                 batch labels.",
                distinct
            );
        }

        info!("Computing per-cell log1p depth...");
        let log_depth = per_cell_log1p_depth(&data_vec, input.block_size)?;

        info!(
            "Scoring {} LR pairs on the edges of {} strata...",
            real_pairs.len(),
            strata_map.n_strata()
        );
        let (score_rows, n_straddling) = compute_edge_scores(&EdgeScoresInput {
            edges: &edges,
            strata: &strata_map,
            pairs: &real_pairs,
            gene_to_local: &gene_to_local,
            x_lr: &x_lr,
            log_depth: &log_depth,
        });
        if n_straddling > 0 {
            info!(
                "Dropped {} straddling edges (endpoint batch labels differ); \
                 a contact belonging to no single batch scores in none",
                n_straddling
            );
        }
        let n_unmeasurable = score_rows.iter().filter(|r| r.log_or.is_nan()).count();
        if n_unmeasurable > 0 {
            info!(
                "{} of {} rows are prior dominated (no co-detection observed, \
                 none expected): log_or and log_or_se are NaN there, not zero",
                n_unmeasurable,
                score_rows.len()
            );
        }
        let out_path = lra_outputs(LraMethod::CoexpressionScores, &out, &[]).remove(0);
        write_edge_scores(&out_path.to_string_lossy(), &score_rows)?;
        info!(
            "Wrote {} score rows to {}",
            score_rows.len(),
            out_path.display()
        );

        // Back-fill the upstream manifest, the way the test path does for
        // its JSON sidecar, so a downstream reader can find this table
        // from the prefix alone rather than guessing the name.
        crate::util::metadata::PintoMetadata::backfill_output(manifest, |o| {
            o.lr_scores = Some(recorded(&out_path))
        });
        return Ok(());
    }

    //////////////////////////////////////////
    // 5. Cell→community soft membership    //
    //////////////////////////////////////////
    info!("Building cell→community soft membership...");
    let p_member = strata_map.memberships(&edges, n_cells);

    ////////////////////////////////////////////////////
    // 6. Per-cell batch label (modal incident batch) //
    ////////////////////////////////////////////////////
    let cell_batch = derive_cell_batch_labels(&edges, n_cells);

    ///////////////////////////////////////////////////
    // 7. Random projection + propensity binary-sort //
    ///////////////////////////////////////////////////
    info!(
        "Random projection (dim {}) + propensity binary-sort...",
        args.propensity_dim
    );
    let proj = data_vec.project_columns_with_batch_correction(
        args.propensity_dim,
        input.block_size,
        Some(&cell_batch),
    )?;
    let propbin = binary_sort_columns(&proj.proj, args.propensity_dim)?;
    let (sample_id_per_cell, sample_batch_label, sample_propbin) =
        assign_samples(&cell_batch, &propbin);
    let n_samples = sample_batch_label.len();
    info!(
        "{} pseudobulk samples across {} batches",
        n_samples,
        sample_batch_label
            .iter()
            .map(|b| b.as_ref())
            .collect::<HashSet<_>>()
            .len()
    );

    // Fisher weight is multiplicative on pb_mean *before* log1p — without
    // the non-linear log step it would cancel out of any correlation /
    // covariance and have no effect.
    // Per GENE, off the statistics computed above. A Fisher weight is a
    // function of abundance and mean and is not additive, so no arithmetic on
    // two rows' weights gives the pooled gene's weight.
    let fisher_lr: Vec<f32> = lr_genes.iter().map(|&g| fisher_all[g]).collect();
    info!(
        "Fisher w (LR genes): min={:.3e}, mean={:.3e}, max={:.3e}",
        fisher_lr.iter().cloned().fold(f32::INFINITY, f32::min),
        fisher_lr.iter().sum::<f32>() / (fisher_lr.len().max(1) as f32),
        fisher_lr.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );

    /////////////////////////////////////////////////////////
    // 9. Collapse into per-(community, sample) pseudobulk //
    /////////////////////////////////////////////////////////
    info!(
        "Collapsing into pseudobulk: {} strata × {} samples × {} LR genes...",
        n_strata_total,
        n_samples,
        lr_genes.len()
    );
    let mut collapse = collapse_pseudobulk(
        &x_lr,
        &sample_id_per_cell,
        &p_member,
        n_strata_total,
        n_samples,
        &active_strata,
    );

    // Bake Fisher weights into log-mean matrices once (per-gene row scale).
    // Under weighted_cov, per-gene scalars on each side stack as `w_L · w_R`
    // on stat_obs; precomputing here saves an n_pairs × n_samples mul per
    // stratum.
    for c in 0..n_strata_total {
        if !active_strata.contains(&(c as u32)) {
            continue;
        }
        apply_feature_weights(&mut collapse.log_mean[c], &fisher_lr);
    }

    ////////////////////////////////////////////////////
    // 10. Per-stratum scoring (per-batch and pooled) //
    ////////////////////////////////////////////////////
    let mut rows: Vec<LrActivityRow> = Vec::new();
    let mut strata: Vec<StratumEntry> = Vec::new();

    let mut unique_batches: Vec<Box<str>> = sample_batch_label
        .iter()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    unique_batches.sort();

    let n_real_batches = unique_batches
        .iter()
        .filter(|b| b.as_ref() != BATCH_LABEL_ALL)
        .count();

    let base_seed = args.input.seed;

    let mut plan: Vec<StratumJob> = Vec::new();
    for batch_label in &unique_batches {
        let samples_in_batch: Vec<usize> = (0..n_samples)
            .filter(|&s| sample_batch_label[s].as_ref() == batch_label.as_ref())
            .collect();
        if samples_in_batch.len() < MIN_SAMPLES_PER_STRATUM {
            continue;
        }
        for stratum in 0..(n_strata_total as u32) {
            if !active_strata.contains(&stratum) {
                continue;
            }
            if !stratum_present(&collapse, stratum as usize, &samples_in_batch) {
                continue;
            }
            plan.push(StratumJob {
                batch_label: batch_label.clone(),
                stratum,
                samples: samples_in_batch.clone(),
                pooled: false,
            });
        }
    }
    if n_real_batches > 1 {
        let all_samples: Vec<usize> = (0..n_samples).collect();
        let meta_label: Box<str> = BATCH_LABEL_META.into();
        for stratum in 0..(n_strata_total as u32) {
            if !active_strata.contains(&stratum) {
                continue;
            }
            if !stratum_present(&collapse, stratum as usize, &all_samples) {
                continue;
            }
            plan.push(StratumJob {
                batch_label: meta_label.clone(),
                stratum,
                samples: all_samples.clone(),
                pooled: true,
            });
        }
    }

    let n_strata = plan.len();
    info!(
        "Scoring {} strata × {} LR pairs ({} permutations each)...",
        n_strata,
        real_pairs.len(),
        args.n_permutations
    );

    // Sequential deliberately. `score_pairs_for_stratum` parallelises over
    // permutations and, measured on a full-database run, already holds ~83%
    // of all cores; a second parallel axis here would chase the remainder
    // while multiplying the in-flight posterior draws, which are what set
    // peak memory.
    let score_inputs = ScoreInputs {
        real_pairs: &real_pairs,
        gene_to_local: &gene_to_local,
        gene_names: &gene_names,
        collapse: &collapse,
        fisher_lr: &fisher_lr,
        strata_map: &strata_map,
        sample_batch_label: &sample_batch_label,
        sample_propbin: &sample_propbin,
        shuffle_stratify_dim: args.shuffle_stratify_dim,
        n_perm: args.n_permutations,
    };
    let pb = new_progress_bar(n_strata as u64).with_message("strata");
    for (k, job) in plan.into_iter().enumerate() {
        let (batch_label, stratum, is_meta) = (&job.batch_label, job.stratum, job.pooled);
        let stratum_id = strata.len();
        strata.push(stratum_entry(
            batch_label,
            stratum as usize,
            &strata_map,
            &edges,
            &cell_names,
            if is_meta {
                None
            } else {
                Some(batch_label.as_ref())
            },
        ));
        let seed = if is_meta {
            base_seed
                .wrapping_add(0xDEAD_BEEF)
                .wrapping_add(stratum as u64)
        } else {
            base_seed.wrapping_add(stratum as u64 * 1_000_003)
        };
        let t0 = std::time::Instant::now();
        let mut br = score_pairs_for_stratum(&score_inputs, &job, seed, stratum_id);
        info!(
            "  [{}/{}] batch={} stratum={} n_samples={} → {} rows ({:.1}s)",
            k + 1,
            n_strata,
            batch_label,
            strata_map.label(stratum as usize),
            job.samples.len(),
            br.len(),
            t0.elapsed().as_secs_f32()
        );
        rows.append(&mut br);
        pb.inc(1);
    }
    pb.finish_and_clear();

    if log::log_enabled!(log::Level::Info) {
        eprintln!();
        eprintln!("{}", pvalue_histogram(&rows, 50));
        eprintln!();
    }

    //////////////////////
    // 11. Write output //
    //////////////////////
    let [out_path, json_path]: [std::path::PathBuf; 2] =
        lra_outputs(LraMethod::CoexpressionTest, &out, &[])
            .try_into()
            .expect("a table and its sidecar");
    let (out_path, json_path) = (
        out_path.to_string_lossy().into_owned(),
        json_path.to_string_lossy().into_owned(),
    );
    info!("Writing {} rows to {}", rows.len(), out_path);
    write_lr_activity(&out_path, &rows)?;

    if args.emit_json {
        let upstream_meta_path = manifest.to_string_lossy().into_owned();
        let upstream_meta = crate::util::metadata::PintoMetadata::read(manifest).ok();
        write_lr_activity_json(
            &json_path,
            lc_prefix,
            upstream_meta.as_ref().map(|_| upstream_meta_path.as_str()),
            &rows,
            &strata,
            args.json_fwer_threshold,
        )?;
        info!("Wrote {}", json_path);

        crate::util::metadata::PintoMetadata::backfill_output(manifest, |o| {
            o.lr_activity = Some(recorded(std::path::Path::new(&json_path)))
        });
    }

    Ok(())
}

/// `pinto lra --method contact-effect`: expression, coordinates and spatial
/// contacts, from the `--from` run or built from the coordinates given.
fn contact_effect(args: &SrtLrActivityArgs, input: &LraInput, out: &str) -> anyhow::Result<()> {
    use crate::lr_activity::ligand_effect::pipeline::run;
    use crate::lr_activity::ligand_effect::{RunInputs, Settings};
    let contacts = load_contacts(input)?;
    let manifest = record_manifest(input, out, &contacts)?;
    run(&RunInputs {
        settings: Settings {
            out,
            manifest: &manifest,
            lr_pairs: args.lr_pairs.as_deref(),
            targets: &args.targets,
            species: args.species,
            seed: args.input.seed,
            block_size: input.block_size,
            only: &args.only,
        },
        data: &contacts.data,
        axis: &contacts.axis,
        pairs: &contacts.pairs,
        cell_section: &contacts.cell_section,
        coords: &contacts.coords,
        cell_type: contacts
            .cell_type
            .as_ref()
            .map(|(c, n)| (c.as_slice(), n.as_slice())),
    })
}

/// `pinto lra-simulate`: the contact-effect analysis's power and
/// calibration check on this tissue.
pub fn fit_lra_simulate(args: &LraSimulateArgs) -> anyhow::Result<()> {
    use crate::lr_activity::ligand_effect::pipeline::run_simulate;
    use crate::lr_activity::ligand_effect::simulate::Study;
    use crate::lr_activity::ligand_effect::{RunInputs, Settings};
    if args.example_study {
        println!("{}", Study::example_json());
        return Ok(());
    }
    let study = match args.study.as_deref() {
        Some(path) => Study::read(path)?,
        None => Study::default(),
    };
    let input = args.input.resolve()?;
    let out = input.out(args.out.as_deref())?;
    mkdir_parent(&out)?;
    let contacts = load_contacts(&input)?;
    let manifest = record_manifest(&input, &out, &contacts)?;
    run_simulate(
        &RunInputs {
            settings: Settings {
                out: &out,
                manifest: &manifest,
                lr_pairs: None,
                targets: &[],
                species: args.species,
                seed: args.input.seed,
                block_size: input.block_size,
                only: &[],
            },
            data: &contacts.data,
            axis: &contacts.axis,
            pairs: &contacts.pairs,
            cell_section: &contacts.cell_section,
            coords: &contacts.coords,
            cell_type: contacts
                .cell_type
                .as_ref()
                .map(|(c, n)| (c.as_slice(), n.as_slice())),
        },
        &study,
    )
}

pub(super) fn stratum_entry(
    batch_label: &str,
    stratum: usize,
    strata_map: &CommunityStrata,
    edges: &[(usize, usize, u32, Option<Box<str>>)],
    cell_names: &[Box<str>],
    batch_filter: Option<&str>,
) -> StratumEntry {
    let name_of = |c: usize| {
        cell_names
            .get(c)
            .cloned()
            .unwrap_or_else(|| format!("cell_{c}").into_boxed_str())
    };
    // Each edge appears once per orientation in the stratum; keep the
    // un-flipped instances so the sidecar lists every participating pair
    // exactly once, with no direction claimed.
    let edges_named: Vec<(Box<str>, Box<str>)> = strata_map
        .oriented(stratum)
        .iter()
        .filter(|&&(_, flipped)| !flipped)
        .filter(|&&(e, _)| {
            let b = &edges[e as usize].3;
            match batch_filter {
                // Edges left unbatched (no `batch` column in coord_pairs)
                // are matched by the synthetic `BATCH_LABEL_ALL` stratum
                // — `derive_cell_batch_labels` already gave those cells
                // that label, so the per-edge filter has to agree.
                Some(bf) => match b.as_ref().map(|s| s.as_ref()) {
                    Some(eb) => eb == bf,
                    None => bf == BATCH_LABEL_ALL,
                },
                None => true,
            }
        })
        .map(|&(e, _)| {
            let (i, j, _, _) = edges[e as usize];
            (name_of(i), name_of(j))
        })
        .collect();
    StratumEntry {
        batch: Box::from(batch_label),
        community: strata_map.community(stratum) as i32,
        edges: edges_named,
    }
}
