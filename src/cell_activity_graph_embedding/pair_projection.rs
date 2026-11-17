//! Per-pair Poisson-MAP projection onto cage's frozen feature embedding.
//!
//! cage trains cells and features into one shared `D`-dim space by predicting
//! spatial adjacency. What the rest of pinto consumes, though, is a *pair*
//! latent: `lc` / `dsvd` cluster cell pairs into link communities and derive a
//! cell's propensity from the mix its incident edges carry. This module
//! produces that pair latent the same way `senna bge` / `senna gem` produce a
//! cell latent in phase 2 — freeze the feature side, re-estimate each node
//! against its own counts.
//!
//! The node here is the pair, and its observation is the POOLED count
//! `n_uv,g = x_gu + x_gv`: Poisson-exact (a sum of Poissons is Poisson), and it
//! puts a pair that straddles a boundary *between* the two programs it pools
//! rather than on either one — which is what earns interface edges their own
//! cluster without an explicit difference channel.
//!
//! ```text
//! s_uv,g = ⟨e_g, e_uv⟩ + b_g + β_uv
//! μ_uv,g = exp(s_uv,g)
//! L      = Σ_{g ∈ G} μ_uv,g  −  Σ_g n_uv,g · s_uv,g  +  (λ/2)‖e_uv‖²
//! ```
//!
//! - `e_g` is cage's trained `e_feat` row, **frozen**. That is what makes the
//!   problem identified: the basis is pinned by the dictionary, so the `O(D)`
//!   rotation freedom that sank the free-edge-embedding experiment cannot
//!   exist here, and the objective (Poisson NLL + ridge) is strictly convex in
//!   `e_uv`.
//! - `b_g` is the **empirical** log feature abundance, not cage's `b_feat`. The
//!   trained feature bias came out of a logistic NCE and is a graph-popularity
//!   term, not a log-rate; this likelihood needs a log-rate. Fixing `b_g` to
//!   data also closes the gauge freedom geu's phase 2 has to correct for
//!   (`θ ← θ − v`, `b_g ← b_g + ⟨e_g, v⟩` leaves every score identical *only*
//!   when `b_g` is free to absorb the shift) — so no gauge fix is needed here,
//!   and `θ = 0` means "composition equal to the population average" rather
//!   than an arbitrary corner of the space.
//! - `β_uv` is a free scalar per pair. It absorbs the pooled library size, so
//!   `e_uv` carries composition only — the job `b_cell` does in bge/gem phase 2
//!   ("always fitted … keeping `e_c` depth-corrected"). With `β_uv` free the
//!   Poisson MAP *is* the multinomial MAP over the pair's feature composition.
//!
//! # How it is solved
//!
//! The encoder ([`encoder`]) amortizes it: a shared trunk reads each
//! endpoint's sufficient statistic — the counts enter the objective only
//! through `Σ_g n_g e_g`, `N` and `Σ_g n_g b_g`, three sparse sums — into a
//! cell code and a small gated mixture turns two codes into the pair code,
//! trained on this same likelihood and run once over every pair on the
//! device; its placement is used as it is. Under `cage` it trains on the
//! finest pseudobulks' pairs, thinned to cell-pair depth, and only encodes
//! the cell pairs ([`TrainOn`]). Given `e_uv`, the intercept is
//! closed-form (`Σ_g μ_g = N_uv`), so it is never learned.

use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::matrix::utils::generate_minibatch_intervals;

/// Clamp on the linear predictor before `exp`. f32 overflows at 88; the same
/// bound geu puts on every Poisson fit in the workspace.
const SCORE_CLAMP: f32 = 30.0;

pub(crate) mod encoder;
mod scoring;
pub use encoder::PairEncoderSpec;
pub use scoring::PairScore;

/// What a trained encoder trains on: the finest pseudobulks (`cage`), each
/// the sum of its cells' rows by `cell_to_pb`, and the pseudobulk edges
/// `edges` between them, each pair thinned to a depth drawn from the pairs
/// being placed. The cell pairs are then only encoded.
#[derive(Debug, Clone, Copy)]
pub struct TrainOn<'a> {
    pub cell_to_pb: &'a [usize],
    pub n_pb: usize,
    pub edges: &'a [(u32, u32)],
}

/// Where the encoder that places the pairs comes from.
#[derive(Debug, Clone, Copy)]
pub enum PairSolver<'a> {
    /// Fit the encoder on `on`, place everything with it, and save it to
    /// `save_to`.
    TrainEncoder {
        spec: &'a PairEncoderSpec,
        on: TrainOn<'a>,
        dev: &'a Device,
        save_to: &'a str,
    },
    /// Place everything with an encoder saved by an earlier run.
    LoadEncoder { path: &'a str, dev: &'a Device },
}

/// Knobs for [`project_pairs`]: where the encoder comes from, plus how the
/// pair axis is walked. The objective's one parameter, the ridge, travels
/// with the encoder — set in [`PairEncoderSpec`] when it is trained, read
/// back from the file when it is loaded.
#[derive(Debug, Clone)]
pub struct PairProjectionArgs<'a> {
    pub solver: PairSolver<'a>,
    /// Seed for training the encoder: its initial weights, each epoch's
    /// shuffle of the pairs and cells, and the sample of pairs the
    /// evaluation line scores. Unused when the encoder is loaded.
    pub seed: u64,
    /// Pairs per encoder block, and cells per read block.
    pub pair_block: usize,
    /// Feature names the agreement correlations are computed over, from
    /// `--eval-features`. `None` leaves them `NaN`: correlating over the whole
    /// active axis means a sort per pair, and pairs outnumber cells by an order
    /// of magnitude, so it is opt-in rather than a silent cost.
    pub eval_features: Option<Vec<Box<str>>>,
    /// Score every pair's held-out likelihood ([`PairScore`]). Off, `scores`
    /// comes back empty: the exhaustive pass per pair is only worth paying
    /// where the numbers are written out.
    pub score_pairs: bool,
}

/// Per-endpoint batch division, applied to each cell's counts *before* they are
/// pooled — the same per-batch fold divide `senna bge` applies in phase 2, so a
/// pair's latent reflects de-batched composition. Without it a multi-batch run
/// clusters edges by batch: an edge's counts carry its endpoints' batch
/// shifts, whether it joins two cells of one section (a spatial graph) or
/// cells of any batches (an expression graph, built on the batch-corrected
/// projection).
/// A stage-2 fit's stage-1 divisor (`--offset`) divides the endpoints the
/// same way, by the same self-normalising ratio
/// ([`crate::util::batch_effects::Divisor::ratio_column`]).
pub use crate::util::batch_effects::BatchDivisor as PairBatchDivisor;
use crate::util::batch_effects::Divisor;

/// Every cell's own placement, in column order: the MAP of its doubled
/// profile `2x_c`, whose composition is `x_c`'s — the self-pair, so a cell
/// sits where the same map puts a pair of two copies of it.
pub struct CellLatent {
    /// `[n_cells × D]`.
    pub latent: Mat,
    /// Per-cell intercept, `[n_cells]`.
    pub bias: Vec<f32>,
}

/// What the projection hands back, in `edges` order.
pub struct PairLatent {
    /// `[n_pairs × D]` pair embedding `e_uv`.
    pub latent: Mat,
    /// Fitted per-pair intercept `β_uv`, `[n_pairs]`. Not consumed downstream
    /// (clustering is on the composition), but it is the pair's log pooled
    /// depth and worth keeping for diagnostics.
    pub bias: Vec<f32>,
    /// Held-out predictive score per pair, against the model's own abundance
    /// null; empty unless [`PairProjectionArgs::score_pairs`].
    pub scores: Vec<PairScore>,
    pub cells: CellLatent,
}

/// The frozen side of the projection, flattened once for the inner loop.
///
/// `e_feat` is `[G × D]` column-major (nalgebra), so a per-feature row read is
/// strided — every solve would walk the matrix against the cache. Rows are
/// copied out row-major once here instead, restricted to features that carry any
/// count at all (a feature with zero total appears in no pair's profile and
/// contributes `exp(-∞) = 0` to the partition).
pub struct PairDictionary {
    /// Row-major `[n_active × D]`.
    feat: Vec<f32>,
    /// Empirical log feature abundance, `[n_active]`.
    b: Vec<f32>,
    /// Global feature id → active-list position, `u32::MAX` when inactive.
    local_of_feature: Vec<u32>,
    d: usize,
}

impl PairDictionary {
    /// Build the frozen side from cage's `[G × D]` feature embedding and the
    /// per-feature count totals over all cells (`n_cells` turns those totals into
    /// the per-cell mean the log-rate offset needs).
    pub fn new(e_feat: &Mat, feature_totals: &[f64], n_cells: usize) -> anyhow::Result<Self> {
        let n_features = feature_totals.len();
        let d = e_feat.ncols();
        anyhow::ensure!(
            e_feat.nrows() == n_features,
            "pair projection: e_feat has {} rows, expected {n_features}",
            e_feat.nrows()
        );
        anyhow::ensure!(d > 0, "pair projection: empty embedding dimension");
        anyhow::ensure!(n_cells > 0, "pair projection: no cells");

        let active: Vec<usize> = (0..n_features)
            .filter(|&g| feature_totals[g] > 0.0)
            .collect();
        anyhow::ensure!(
            !active.is_empty(),
            "pair projection: every feature has zero total count"
        );

        let mut local_of_feature = vec![u32::MAX; n_features];
        let mut feat = Vec::with_capacity(active.len() * d);
        let mut b = Vec::with_capacity(active.len());
        for (local, &g) in active.iter().enumerate() {
            local_of_feature[g] = local as u32;
            for j in 0..d {
                feat.push(e_feat[(g, j)]);
            }
            // Mean count per cell, on the log scale the Poisson rate lives on.
            // The pooled-pair factor of two is constant across features and is
            // absorbed by `β_uv`.
            let m = feature_totals[g] / n_cells as f64;
            b.push(m.ln() as f32);
        }

        let total: f64 = active.iter().map(|&g| feature_totals[g]).sum();
        anyhow::ensure!(
            total > 0.0 && total.is_finite(),
            "pair projection: feature totals sum to {total}, expected a positive finite value"
        );
        Ok(Self {
            feat,
            b,
            local_of_feature,
            d,
        })
    }

    /// Number of features carrying counts — the axis the partition runs over.
    #[must_use]
    pub fn n_active(&self) -> usize {
        self.b.len()
    }

    /// Map a `(global feature id, count)` profile onto the active-list positions the
    /// encoder and the scorer both index by. Features with no counts anywhere are dropped:
    /// they carry no information and are not on the partition axis.
    #[must_use]
    fn to_local(&self, obs: &[(u32, f32)]) -> Vec<(u32, f32)> {
        obs.iter()
            .filter_map(|&(g, n)| {
                let l = *self.local_of_feature.get(g as usize)?;
                (l != u32::MAX && n > 0.0).then_some((l, n))
            })
            .collect()
    }

    /// The statistic a local profile enters the objective through:
    /// `(Σ_g n_g e_g, Σ_g n_g, Σ_g n_g b_g)`.
    pub(crate) fn statistic(&self, features: &[u32], counts: &[f32]) -> (Vec<f32>, f32, f32) {
        let mut sums = vec![0f32; self.d];
        let (total, offset) = self.accumulate(
            features.iter().copied().zip(counts.iter().copied()),
            &mut sums,
        );
        (sums, total, offset)
    }

    /// [`Self::statistic`] of a `(position, count)` profile, its `S` added
    /// into `sums` (`[D]`), returning `(N, c)`.
    pub(crate) fn accumulate(
        &self,
        obs: impl Iterator<Item = (u32, f32)>,
        sums: &mut [f32],
    ) -> (f32, f32) {
        let d = self.d;
        let (mut total, mut offset) = (0f32, 0f32);
        for (g, n) in obs {
            let row = &self.feat[g as usize * d..(g as usize + 1) * d];
            for (s, &e) in sums.iter_mut().zip(row) {
                *s += n * e;
            }
            total += n;
            offset += n * self.b[g as usize];
        }
        (total, offset)
    }

    /// The clamped log-rates `⟨e_g, θ⟩ + b_g` over the active axis.
    fn log_rates(&self, theta: &[f32]) -> Vec<f32> {
        let d = self.d;
        self.feat
            .chunks_exact(d)
            .zip(&self.b)
            .map(|(row, &b)| {
                let dot: f32 = row.iter().zip(theta).map(|(&e, &t)| e * t).sum();
                (dot + b).clamp(-SCORE_CLAMP, SCORE_CLAMP)
            })
            .collect()
    }
}

/// Project every cell pair — and every cell — onto cage's frozen feature embedding.
///
/// `feature_totals` is per FEATURE, already folded off the row axis — both the
/// partition and each pair's profile live there, because `e_feat` is per feature
/// and a channelized matrix's two rows are one feature's pooled count rather than
/// two categories of the multinomial. It is passed in rather than computed here
/// because it is a whole-matrix streaming pass and the caller has already made
/// it for the splice report.
///
/// `e_feat` is cage's trained `[G × D]` feature embedding, used as-is: the
/// selection gate is already expressed in its values, so re-applying `pip` here
/// would shrink the same selection twice. Features the gate drove to `‖e_g‖ ≈ 0`
/// contribute a constant `exp(b_g + β)` to the partition and therefore cannot
/// pull on `e_uv` — no special-casing needed.
///
/// Every cell is read once, in contiguous column blocks, into the corpus the
/// pairs are encoded from (the training pseudobulks' rows, [`TrainOn`], are
/// summed from it);
/// every pair is then the merge of two rows and every cell its own doubled
/// row.
pub fn project_pairs(
    data: &SparseIoVec,
    edges: &[(u32, u32)],
    e_feat: &Mat,
    batch: Option<Divisor<'_>>,
    args: &PairProjectionArgs<'_>,
    axis: &FeatureAxis,
    feature_totals: &[f64],
) -> anyhow::Result<PairLatent> {
    let n_features = axis.n_features();
    let n_cells = data.num_columns();
    let d = e_feat.ncols();
    anyhow::ensure!(
        axis.n_rows() == data.num_rows(),
        "pair projection: feature axis has {} rows, data has {}",
        axis.n_rows(),
        data.num_rows()
    );
    anyhow::ensure!(
        e_feat.nrows() == n_features,
        "pair projection: e_feat has {} rows, data has {n_features} features",
        e_feat.nrows()
    );
    anyhow::ensure!(d > 0, "pair projection: empty embedding dimension");
    anyhow::ensure!(
        feature_totals.len() == n_features,
        "pair projection: {} feature totals, expected {n_features}",
        feature_totals.len()
    );

    let dict = PairDictionary::new(e_feat, feature_totals, n_cells)?;
    let scored_positions: Option<Vec<u32>> = match args.eval_features.as_ref() {
        Some(names) => {
            let positions = dict.eval_positions(axis.feature_names(), names);
            anyhow::ensure!(
                !positions.is_empty(),
                "--eval-features matched no feature that carries counts in this sample"
            );
            info!(
                "Agreement axis: {} of {} named features carry counts here",
                positions.len(),
                names.len()
            );
            Some(positions)
        }
        None => None,
    };
    // Resolved once for the whole run: the membership bitmap and the null's
    // log-partition are the same for every pair.
    let eval_axis = dict.eval_axis(scored_positions);
    if dict.n_active() < n_features {
        info!(
            "Pair projection: {} of {n_features} features carry counts; the rest sit out the partition",
            dict.n_active(),
        );
    }
    let corpus = build_corpus(data, &dict, batch, axis, args.pair_block)?;

    let (enc, shared) = match args.solver {
        PairSolver::TrainEncoder {
            spec,
            on,
            dev,
            save_to,
        } => {
            let set = encoder::TrainingSet::pseudobulks(
                &dict,
                &corpus,
                on.cell_to_pb,
                on.n_pb,
                on.edges,
                edges,
            )?;
            // The moments come from the rows epoch 0 trains on.
            let first = set.draw(args.seed, 0);
            let moments = set.moments(d, first.as_ref(), &corpus);
            let enc = encoder::PairEncoder::build(&dict, moments, spec, args.seed, dev)?;
            // Training on the cells themselves shares their one upload with
            // the encoding; a thinned set needs only its evaluation pairs'.
            let shared = match set.thinning {
                None => Some(enc.corpus_stats(&corpus)?),
                Some(_) => None,
            };
            let placed = encoder::Placed {
                corpus: &corpus,
                edges,
                stats: shared.as_ref(),
            };
            let trained = enc.train(&set, placed, first, spec, args.seed)?;
            info!(
                "Pair encoder: {} steps; held-out NLL/count after {:.4}",
                trained.steps, trained.nll_per_count
            );
            enc.save(save_to)?;
            info!("Wrote {}", crate::util::staging::final_str(save_to));
            (enc, shared)
        }
        PairSolver::LoadEncoder { path, dev } => {
            let enc = encoder::PairEncoder::load(&dict, path, dev)?;
            info!(
                "Pair encoder: loaded {path} (L={}, K={}, λ={})",
                enc.trunk_width(),
                enc.n_experts(),
                enc.ridge()
            );
            (enc, None)
        }
    };
    let stats = match shared {
        Some(stats) => stats,
        None => enc.corpus_stats(&corpus)?,
    };
    let mut placed =
        enc.encode_all(&corpus, &stats, edges, args.pair_block, encoder::CELL_BLOCK)?;
    // The device copy is not needed by the scoring pass.
    drop(stats);

    if args.score_pairs {
        placed.scores = edges
            .par_iter()
            .enumerate()
            .map(|(i, &(u, v))| {
                let obs = corpus[u as usize].pooled(&corpus[v as usize]);
                let z: Vec<f32> = placed.latent.row(i).iter().copied().collect();
                dict.score_local(&obs, &z, &eval_axis)
            })
            .collect();
    }
    Ok(placed)
}

////////////////
// The corpus //
////////////////

/// Every cell's active-axis row, read once in contiguous column blocks.
fn build_corpus(
    data: &SparseIoVec,
    dict: &PairDictionary,
    batch: Option<Divisor<'_>>,
    axis: &FeatureAxis,
    block: usize,
) -> anyhow::Result<Vec<encoder::CellRow>> {
    let n_cells = data.num_columns();
    let mut corpus: Vec<encoder::CellRow> = Vec::with_capacity(n_cells);
    let bar = new_progress_bar(n_cells as u64).with_message("reading cells");
    for (lb, ub) in generate_minibatch_intervals(n_cells, axis.n_features(), Some(block.max(1))) {
        let slab = data.read_columns_csc(lb..ub)?;
        let (offsets, rows, vals) = (slab.col_offsets(), slab.row_indices(), slab.values());
        let block_rows: Vec<encoder::CellRow> = (lb..ub)
            .into_par_iter()
            .map(|c| {
                let col = c - lb;
                let (s, e) = (offsets[col], offsets[col + 1]);
                let counts = endpoint_counts(&rows[s..e], &vals[s..e], c as u32, batch);
                encoder::CellRow::from_profile(dict, &axis.pool_profile(counts))
            })
            .collect();
        corpus.extend(block_rows);
        bar.inc((ub - lb) as u64);
    }
    bar.finish_and_clear();
    let nnz: usize = corpus.iter().map(|r| r.features.len()).sum();
    info!(
        "Pair corpus: {n_cells} cells, {nnz} counts on the {}-feature active axis ({:.1} MB)",
        dict.n_active(),
        (nnz * 8) as f64 / 1e6
    );
    Ok(corpus)
}

/// One cell's `(row, count)` profile from its CSC column, batch-divided when
/// the run has batches — the one place a cell's counts are read.
fn endpoint_counts(
    rows: &[usize],
    vals: &[f32],
    cell: u32,
    batch: Option<Divisor<'_>>,
) -> Vec<(u32, f32)> {
    let mut vals = vals.to_vec();
    if let Some(d) = batch {
        // a δ registered on the data has no per-cell batch here: cage and
        // predict pass theirs with each cell's batch
        debug_assert!(!matches!(d, Divisor::Registered(_)));
        d.ratio_column(rows, &mut vals, cell as usize, None);
    }
    rows.iter()
        .zip(vals)
        .filter(|(_, x)| *x > 0.0)
        .map(|(&r, x)| (r as u32, x))
        .collect()
}
