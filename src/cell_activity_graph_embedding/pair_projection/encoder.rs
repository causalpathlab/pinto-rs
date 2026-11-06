//! Amortized phase 2: one encoder places every pair, and every cell, on the
//! frozen dictionary.
//!
//! The exact solver ([`super::newton_polish`]) settles the same strictly
//! convex problem once per node — the multinomial MAP of the pooled counts
//! against `e_feat` — and pairs outnumber cells by an order of magnitude.
//! The map from counts to that optimum is smooth, so a small encoder trained
//! on the same objective places every pair in one forward pass, on the
//! device, and the per-pair solver becomes the check on it (and the finisher
//! of its rare misses) rather than the workhorse.
//!
//! The MAP is a function of one statistic. Write `S = Σ_g n_g e_g` for the
//! count-weighted sum of dictionary rows, `N = Σ_g n_g` for the depth and
//! `c = Σ_g n_g b_g`; then
//!
//! ```text
//! s_g    = ⟨e_g, z⟩ + b_g
//! L(z)   = N · lse_g(s_g) − ⟨S, z⟩ − c + (λ/2)‖z‖²
//! ```
//!
//! so the counts enter the objective only through `(S, N, c)`, three exact
//! sums over a cell's nonzeros, and pool additively over a pair. Nothing here
//! is ever dense over the features on the count side: the corpus carries each
//! cell's `(S, N, c)`, a step's input is `[2B, D + 1]`, and the one `G`-wide
//! tensor is the partition `lse_g` — a GEMM output on the device, the same
//! sum the exact solver forms, never an upload.
//!
//! The encoder is a cell trunk applied twice and a [`SymmetricPairHead`]: the
//! trunk reads `[S/N ‖ ln(N + 1)]`, standardised by population moments saved
//! with the encoder, through two layer-normed ReLU layers into a cell code
//! `h`; the head — a gated mixture of linear experts over
//! `[(h_u + h_v)/2 ‖ h_u ⊙ h_v ‖ the pair's own statistic]` — turns two codes
//! into the pair code `z_uv`. Symmetric by construction, and with no
//! per-feature parameter, so the trained encoder transfers to any re-aligned
//! axis. Normalisation is per row (layer norm), never per batch: a batch
//! norm's running variance for a unit that rarely fires sits near zero, and
//! the rare input that fires it is then blown up at inference — the wild
//! placement a shared map must never produce. And the placement is kept
//! inside the ball the optimum provably lies in: at the optimum
//! `λz = S − N p̄` with `p̄` a convex combination of dictionary rows, so
//! `‖z‖ ≤ 2N·max_g‖e_g‖/λ` — binding only for a row with a handful of
//! counts, exactly where a shared map would otherwise extrapolate.
//!
//! Every placement also leaves with a certificate: the Newton decrement
//! `½ ∇ᵀH⁻¹∇` at the placement, with `∇ = N p̄ − S + λz` and
//! `H = N·Cov_p(e) + λI` — the excess likelihood the local quadratic model
//! puts on it, in nats, exact for a quadratic and the standard Newton
//! stopping statistic. A row it puts far out is a row the map extrapolated
//! on — a rare composition, a near-empty profile — and the caller finishes
//! those few exactly.
//!
//! `β_uv` is never learned: given `z` the intercept is `ln N − lse(s)` in
//! closed form. The self-pair term is what makes the cell's own placement a
//! trained quantity: `z_uu` is the MAP of the doubled profile `2x_u`, whose
//! composition is `x_u`'s, so `.cell_embedding.parquet` comes from the same
//! map as every pair. A row with no counts is put at the origin afterwards,
//! where the solver puts it.

use super::PairDictionary;
use crate::util::common::*;
use legume_numeric::candle::candle_core::{DType, Device, Tensor};
use legume_numeric::candle::candle_nn::{
    layer_norm, linear, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, VarMap,
};
use legume_numeric::candle::encoder::{SymmetricPairHead, SymmetricPairHeadArgs};
use legume_numeric::candle::fast_index::gather_rows;
use legume_numeric::candle::nn::seed_uniform_vars;

mod train;
pub(crate) use train::ExactCheck;

///////////////
// Constants //
///////////////

/// The var-name prefix the encoder is saved under.
const VAR_PREFIX: &str = "pair_enc";
/// The non-var tensors saved beside the vars: the statistic's population
/// moments and the encoder's own hyperparameters `[L, K, λ]`.
const STAT_MEAN_TENSOR: &str = "pair_enc.stat_mean";
const STAT_STD_TENSOR: &str = "pair_enc.stat_std";
const HPARAMS_TENSOR: &str = "pair_enc.hparams";
/// Cells placed per pass at inference, at most.
pub(crate) const CELL_BLOCK: usize = 4096;
/// Elements of one `[rows × G]` partition block, whatever `G` is: every
/// scoring pass — a training step's pairs and endpoints, a cell block, a pair
/// block — takes as many rows as fit this budget, so a whole-transcriptome
/// axis takes fewer rows per pass and a panel takes the caps, and the device
/// holds a fixed amount of dense work either way.
const BLOCK_ELEMENTS: usize = 1 << 25;
/// Elements of one chunk of feature outer products `[rows × D(D+1)/2]` in the
/// certificate's second moment: 64 MB of f32 whatever G and D are.
const EE_CHUNK_ELEMENTS: usize = 1 << 24;
/// The chunks are kept for a whole placement pass while all of them fit this
/// (about 280 MB of f32); beyond it they are built per block instead.
const EE_CACHE_ELEMENTS: usize = 70 << 20;

/// The encoder's shape and training budget. `cage` runs [`Default`]; the
/// closing check against the exact solver says how well it did, and a tiny
/// fixture shrinks it.
#[derive(Debug, Clone)]
pub struct PairEncoderSpec {
    /// Width `L` of the cell code.
    pub trunk_width: usize,
    /// Experts `K` in the pair head; `1` is a plain linear head.
    pub n_experts: usize,
    /// Passes over the pairs, at least: a run makes `MIN_STEPS` optimizer
    /// steps whatever its pair count. `0` does not train.
    pub epochs: usize,
    /// Pairs per optimizer step, at most: a step is also held to
    /// [`BLOCK_ELEMENTS`] on the feature axis.
    pub batch: usize,
    /// Ridge `λ` on the pair latent (never on the intercept, which must stay
    /// free to absorb depth): the objective's one parameter, saved with the
    /// encoder so a loaded one is checked and finished under its own.
    pub ridge: f32,
}

impl Default for PairEncoderSpec {
    fn default() -> Self {
        Self {
            trunk_width: 64,
            n_experts: 4,
            epochs: 3,
            batch: 4096,
            ridge: 1.0,
        }
    }
}

/// Rows of a `[rows × g]` block within [`BLOCK_ELEMENTS`], at most `cap` and
/// at least one.
fn rows_within_budget(g: usize, cap: usize) -> usize {
    (BLOCK_ELEMENTS / g.max(1)).clamp(1, cap.max(1))
}

////////////
// Corpus //
////////////

/// One cell's batch-divided counts on the active axis, sorted by position,
/// with the statistic the objective reads them through.
#[derive(Clone)]
pub(crate) struct CellRow {
    pub features: Vec<u32>,
    pub counts: Vec<f32>,
    /// `N = Σ_g n_g`.
    pub total: f32,
    /// `S = Σ_g n_g e_g`, `[D]`.
    pub sums: Vec<f32>,
    /// `c = Σ_g n_g b_g`: the data term's constant.
    pub offset: f32,
}

impl CellRow {
    /// From a `(global feature, count)` profile sorted by feature with no
    /// duplicates — what [`crate::util::feature_axis::FeatureAxis::pool_profile`]
    /// hands out — keeping the features the dictionary carries.
    pub(crate) fn from_profile(dict: &PairDictionary, profile: &[(u32, f32)]) -> Self {
        debug_assert!(
            profile.windows(2).all(|w| w[0].0 < w[1].0),
            "a cell profile must be sorted by feature with no duplicates"
        );
        let (features, counts): (Vec<u32>, Vec<f32>) = dict.to_local(profile).into_iter().unzip();
        let (sums, total, offset) = dict.statistic(&features, &counts);
        Self {
            features,
            counts,
            total,
            sums,
            offset,
        }
    }

    /// The pooled `(position, count)` profile of two rows, sorted.
    pub(crate) fn pooled(&self, other: &CellRow) -> Vec<(u32, f32)> {
        let mut out = Vec::with_capacity(self.features.len() + other.features.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.features.len() && j < other.features.len() {
            match self.features[i].cmp(&other.features[j]) {
                std::cmp::Ordering::Less => {
                    out.push((self.features[i], self.counts[i]));
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.push((other.features[j], other.counts[j]));
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    out.push((self.features[i], self.counts[i] + other.counts[j]));
                    i += 1;
                    j += 1;
                }
            }
        }
        out.extend(
            self.features[i..]
                .iter()
                .copied()
                .zip(self.counts[i..].iter().copied()),
        );
        out.extend(
            other.features[j..]
                .iter()
                .copied()
                .zip(other.counts[j..].iter().copied()),
        );
        out
    }

    /// The self-pair: the row pooled with itself.
    pub(crate) fn doubled(&self) -> Vec<(u32, f32)> {
        self.features
            .iter()
            .zip(&self.counts)
            .map(|(&g, &n)| (g, 2.0 * n))
            .collect()
    }
}

/// Every cell's `(S, N, c)` on the device, row `i` for cell `i`: the whole
/// corpus's, uploaded once, gathered per block by endpoint id.
struct CorpusStats {
    /// `[n × D]`.
    sums: Tensor,
    /// `[n]` each.
    totals: Tensor,
    offsets: Tensor,
}

impl CorpusStats {
    fn new(corpus: &[CellRow], d: usize, dev: &Device) -> anyhow::Result<Self> {
        let n = corpus.len();
        let sums: Vec<f32> = corpus.iter().flat_map(|r| r.sums.iter().copied()).collect();
        Ok(Self {
            sums: Tensor::from_vec(sums, (n, d), dev)?,
            totals: Tensor::from_vec(corpus.iter().map(|r| r.total).collect(), n, dev)?,
            offsets: Tensor::from_vec(corpus.iter().map(|r| r.offset).collect(), n, dev)?,
        })
    }

    /// The rows of `ids`.
    fn gather(&self, ids: &Tensor) -> anyhow::Result<Stats> {
        Ok(Stats {
            sums: gather_rows(&self.sums, ids)?,
            totals: gather_rows(&self.totals, ids)?,
            offsets: gather_rows(&self.offsets, ids)?,
        })
    }
}

/// Population mean and standard deviation of every cell's statistic
/// `[S/N ‖ ln(N + 1)]` over the corpus — `(mean, std)`, `[D + 1]` each, the
/// std floored so a constant coordinate is left alone.
fn stat_moments(d: usize, corpus: &[CellRow]) -> (Vec<f32>, Vec<f32>) {
    let (sum, sq) = corpus
        .par_iter()
        .fold(
            || (vec![0f64; d + 1], vec![0f64; d + 1]),
            |(mut sum, mut sq), row| {
                let n = row.total.max(1.0);
                for (j, &v) in row.sums.iter().enumerate() {
                    let v = f64::from(v / n);
                    sum[j] += v;
                    sq[j] += v * v;
                }
                let ln_n = f64::from((row.total + 1.0).ln());
                sum[d] += ln_n;
                sq[d] += ln_n * ln_n;
                (sum, sq)
            },
        )
        .reduce(
            || (vec![0f64; d + 1], vec![0f64; d + 1]),
            |(mut a, mut b), (c, e)| {
                a.iter_mut().zip(&c).for_each(|(x, y)| *x += y);
                b.iter_mut().zip(&e).for_each(|(x, y)| *x += y);
                (a, b)
            },
        );
    let n = corpus.len().max(1) as f64;
    let mean: Vec<f32> = sum.iter().map(|&v| (v / n) as f32).collect();
    let std: Vec<f32> = sum
        .iter()
        .zip(&sq)
        .map(|(&s, &q)| ((q / n - (s / n).powi(2)).max(0.0).sqrt() as f32).max(1e-3))
        .collect();
    (mean, std)
}

////////////////
// Placements //
////////////////

/// Where a set of nodes (pairs, or cells) landed: latent, intercept, and the
/// certificate — the Newton decrement at the placement, the excess in nats
/// the local quadratic model puts on it over the optimum, zero there.
pub(crate) struct Placement {
    /// `[n × D]`.
    pub latent: Mat,
    pub bias: Vec<f32>,
    pub gap: Vec<f32>,
}

impl Placement {
    pub(crate) fn len(&self) -> usize {
        self.bias.len()
    }

    pub(crate) fn set(&mut self, i: usize, theta: &[f32], beta: f32, gap: f32) {
        for (j, &t) in theta.iter().enumerate().take(self.latent.ncols()) {
            self.latent[(i, j)] = t;
        }
        self.bias[i] = beta;
        self.gap[i] = gap;
    }
}

/// Every pair's and every cell's placement, in the callers' orders.
pub(crate) struct Encoded {
    pub pairs: Placement,
    pub cells: Placement,
}

////////////
// Device //
////////////

/// The frozen side on the device: the dictionary for the composition mean,
/// its transpose for the partition, the index of the upper triangle the
/// curvature's outer products are taken on (the products themselves are built
/// on demand, never kept as a `[G × D²]` table), the log-rate offset and the
/// statistic's population moments.
struct DeviceDict {
    /// `[G × D]`.
    e_gd: Tensor,
    /// `[D × G]`.
    e_hd: Tensor,
    /// `[1 × G]`.
    b_1d: Tensor,
    /// `[1 × (D + 1)]` each.
    stat_mean: Tensor,
    stat_std: Tensor,
    /// `max_g ‖e_g‖`.
    max_row_norm: f32,
    /// The upper triangle of `D × D`, (row, column) index tensors, row-major.
    tri_rows: Tensor,
    tri_cols: Tensor,
    g: usize,
    d: usize,
}

impl DeviceDict {
    fn new(
        dict: &PairDictionary,
        stat_mean: &[f32],
        stat_std: &[f32],
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let (g, d) = (dict.n_active(), dict.d);
        anyhow::ensure!(
            stat_mean.len() == d + 1 && stat_std.len() == d + 1,
            "pair encoder: statistic moments have {} and {} entries, expected {}",
            stat_mean.len(),
            stat_std.len(),
            d + 1
        );
        let e_gd = Tensor::from_slice(&dict.feat, (g, d), dev)?;
        let (r, c): (Vec<u32>, Vec<u32>) = (0..d as u32)
            .flat_map(|r| (r..d as u32).map(move |c| (r, c)))
            .unzip();
        let tri = r.len();
        Ok(Self {
            tri_rows: Tensor::from_vec(r, tri, dev)?,
            tri_cols: Tensor::from_vec(c, tri, dev)?,
            e_hd: e_gd.t()?.contiguous()?,
            e_gd,
            b_1d: Tensor::from_slice(&dict.b, (1, g), dev)?,
            stat_mean: Tensor::from_slice(stat_mean, (1, d + 1), dev)?,
            stat_std: Tensor::from_slice(stat_std, (1, d + 1), dev)?,
            max_row_norm: dict
                .feat
                .chunks_exact(d)
                .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
                .fold(0f32, f32::max),
            g,
            d,
        })
    }

    /// The outer products `e_g e_gᵀ`, upper triangle, of features
    /// `start..start + len`: `[len × D(D+1)/2]`. A symmetric second moment
    /// needs `D(D+1)/2` of its `D²` entries.
    fn outer_chunk(&self, start: usize, len: usize) -> anyhow::Result<Tensor> {
        // Gathered as rows of the transpose: `[D × len]` → `[T × len]`.
        let e = self.e_hd.narrow(1, start, len)?.contiguous()?;
        Ok(
            (gather_rows(&e, &self.tri_rows)? * gather_rows(&e, &self.tri_cols)?)?
                .t()?
                .contiguous()?,
        )
    }

    /// Features per outer-product chunk, each within [`EE_CHUNK_ELEMENTS`].
    fn chunk_rows(&self) -> usize {
        (EE_CHUNK_ELEMENTS / (self.d * (self.d + 1) / 2)).max(1)
    }

    /// Every chunk, once for a whole placement pass, when together they fit
    /// [`EE_CACHE_ELEMENTS`] (about 280 MB at G = 33k, D = 64); `None` beyond,
    /// and [`Self::second_moment`] then builds and drops them one at a time.
    fn outer_chunks_cached(&self) -> anyhow::Result<Option<Vec<Tensor>>> {
        let tri = self.d * (self.d + 1) / 2;
        if self.g.saturating_mul(tri) > EE_CACHE_ELEMENTS {
            return Ok(None);
        }
        let rows = self.chunk_rows();
        (0..self.g)
            .step_by(rows)
            .map(|start| self.outer_chunk(start, rows.min(self.g - start)))
            .collect::<anyhow::Result<_>>()
            .map(Some)
    }

    /// `Σ_g p_bg e_g e_gᵀ`, upper triangle, for each row of `p` `[B × G]` →
    /// `[B × D(D+1)/2]`, accumulated over chunks of features: the cached ones,
    /// or each built, used and dropped in turn, so at most one chunk is held.
    /// A resident `[G × D²]` table would be 0.5 GB at G = 33k, D = 64, and
    /// several GB on a channelized axis or at D = 128.
    fn second_moment(&self, p: &Tensor, cached: Option<&[Tensor]>) -> anyhow::Result<Tensor> {
        let tri = self.d * (self.d + 1) / 2;
        let rows = self.chunk_rows();
        let mut acc = Tensor::zeros((p.dim(0)?, tri), p.dtype(), p.device())?;
        for (k, start) in (0..self.g).step_by(rows).enumerate() {
            let len = rows.min(self.g - start);
            let built;
            let ee = match cached {
                Some(c) => &c[k],
                None => {
                    built = self.outer_chunk(start, len)?;
                    &built
                }
            };
            // Contiguous: some backends' matmul refuse a strided left side.
            let block = p.narrow(1, start, len)?.contiguous()?;
            acc = (acc + block.matmul(ee)?)?;
        }
        Ok(acc)
    }

    fn device(&self) -> &Device {
        self.b_1d.device()
    }
}

/////////////////
// The encoder //
/////////////////

/// Statistic → cell code: two layer-normed ReLU layers and a linear head.
struct CellTrunk {
    /// `[D + 1] → L` and `L → L`, each followed by layer norm and ReLU.
    layers: Vec<(Linear, LayerNorm)>,
    /// `L → L`, linear: the code is signed.
    head: Linear,
}

impl CellTrunk {
    fn new(stat_dim: usize, width: usize, vb: VarBuilder) -> anyhow::Result<Self> {
        let mut layers = Vec::with_capacity(2);
        let mut in_dim = stat_dim;
        for k in 0..2 {
            let lin = linear(in_dim, width, vb.pp(format!("fc.{k}")))?;
            let norm = layer_norm(width, LayerNormConfig::default(), vb.pp(format!("ln.{k}")))?;
            layers.push((lin, norm));
            in_dim = width;
        }
        Ok(Self {
            layers,
            head: linear(width, width, vb.pp("head"))?,
        })
    }

    /// Standardised statistic `[n, D + 1]` → code `[n, L]`.
    fn forward(&self, stat: &Tensor) -> anyhow::Result<Tensor> {
        let mut h = stat.clone();
        for (lin, norm) in &self.layers {
            h = norm.forward(&lin.forward(&h)?)?.relu()?;
        }
        Ok(self.head.forward(&h)?)
    }
}

/// The trained trunk and pair head on a device dictionary.
pub(crate) struct PairEncoder {
    cell: CellTrunk,
    head: SymmetricPairHead,
    varmap: VarMap,
    dict: DeviceDict,
    /// The ridge `λ` the encoder was fitted under: the objective's, the
    /// certificate's, and the norm ball's.
    ridge: f32,
    /// `2·max_g‖e_g‖ / λ`: the optimum's norm bound per pooled count.
    cap_per_count: f64,
}

/// One node's `(S, N, c)` on the device, `[n × D]`, `[n]`, `[n]`.
struct Stats {
    sums: Tensor,
    totals: Tensor,
    offsets: Tensor,
}

impl Stats {
    /// Two endpoints pooled: every sum adds.
    fn pooled(&self, other: &Stats) -> anyhow::Result<Stats> {
        Ok(Stats {
            sums: (&self.sums + &other.sums)?,
            totals: (&self.totals + &other.totals)?,
            offsets: (&self.offsets + &other.offsets)?,
        })
    }

    /// The self-pair: doubled.
    fn doubled(&self) -> anyhow::Result<Stats> {
        self.pooled(self)
    }
}

/// A block of pairs through the trunk and the head: what training,
/// evaluation and inference all read.
struct PairForward {
    /// `[B, 2L + D + 1]` pair features.
    features: Tensor,
    /// `[B, D]` pair codes.
    z_uv: Tensor,
    /// The pairs' pooled statistics.
    pairs: Stats,
}

impl PairEncoder {
    /// An untrained encoder on `dict`, its statistic standardised by the
    /// corpus, its vars seeded from `seed`.
    pub(crate) fn build(
        dict: &PairDictionary,
        corpus: &[CellRow],
        spec: &PairEncoderSpec,
        seed: u64,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let (mean, std) = stat_moments(dict.d, corpus);
        let this = Self::construct(
            DeviceDict::new(dict, &mean, &std, dev)?,
            spec.trunk_width,
            spec.n_experts,
            spec.ridge,
            dev,
        )?;
        // candle's `VarBuilder` initialises from an unseeded stream; the
        // layer norms keep their defaults.
        seed_uniform_vars(&this.varmap, seed, |name| name.contains(".ln."))?;
        Ok(this)
    }

    fn construct(
        dict: DeviceDict,
        trunk_width: usize,
        n_experts: usize,
        ridge: f32,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            trunk_width > 0,
            "pair encoder: trunk width must be positive"
        );
        anyhow::ensure!(ridge > 0.0, "pair encoder: the ridge must be positive");
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev).pp(VAR_PREFIX);
        // The statistic and the log depth: into the trunk per cell, and
        // beside the two codes into the head per pair.
        let stat_dim = dict.d + 1;
        let cell = CellTrunk::new(stat_dim, trunk_width, vb.pp("cell"))?;
        let head = SymmetricPairHead::new(
            SymmetricPairHeadArgs {
                code_dim: trunk_width,
                out_dim: dict.d,
                n_experts,
                extra_dim: stat_dim,
            },
            vb.pp("head"),
        )?;
        let cap_per_count = 2.0 * f64::from(dict.max_row_norm) / f64::from(ridge);
        Ok(Self {
            cell,
            head,
            varmap,
            dict,
            ridge,
            cap_per_count,
        })
    }

    /// Rebuild a saved encoder on `dict`, with the widths and the ridge the
    /// file records.
    pub(crate) fn load(dict: &PairDictionary, path: &str, dev: &Device) -> anyhow::Result<Self> {
        let tensors = legume_numeric::candle::candle_core::safetensors::load(path, dev)?;
        let side = |name: &str| -> anyhow::Result<Vec<f32>> {
            Ok(tensors
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("{path}: no `{name}` tensor"))?
                .flatten_all()?
                .to_vec1()?)
        };
        let (mean, std) = (side(STAT_MEAN_TENSOR)?, side(STAT_STD_TENSOR)?);
        let hparams = side(HPARAMS_TENSOR)?;
        anyhow::ensure!(
            hparams.len() == 3,
            "{path}: `{HPARAMS_TENSOR}` has {} entries, expected [L, K, λ]",
            hparams.len()
        );
        let mut this = Self::construct(
            DeviceDict::new(dict, &mean, &std, dev)?,
            hparams[0] as usize,
            hparams[1] as usize,
            hparams[2],
            dev,
        )?;
        // Matches by name and ignores the side tensors, which are not vars.
        this.varmap.load(path).map_err(|e| {
            anyhow::anyhow!(
                "{path}: the saved pair encoder does not fit this pinto's network ({e}); the model \
                 was fitted by an earlier version — rerun `pinto cage`"
            )
        })?;
        Ok(this)
    }

    /// Write every var and the side tensors to one safetensors file.
    pub(crate) fn save(&self, path: &str) -> anyhow::Result<()> {
        let mut tensors: std::collections::HashMap<String, Tensor> = self
            .varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
            .collect();
        tensors.insert(
            STAT_MEAN_TENSOR.to_string(),
            self.dict.stat_mean.flatten_all()?,
        );
        tensors.insert(
            STAT_STD_TENSOR.to_string(),
            self.dict.stat_std.flatten_all()?,
        );
        let hparams = [
            self.trunk_width() as f32,
            self.n_experts() as f32,
            self.ridge,
        ];
        tensors.insert(
            HPARAMS_TENSOR.to_string(),
            Tensor::new(&hparams, self.dict.device())?,
        );
        legume_numeric::candle::candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    pub(crate) fn trunk_width(&self) -> usize {
        self.head.code_dim()
    }

    pub(crate) fn n_experts(&self) -> usize {
        self.head.n_experts()
    }

    pub(crate) fn ridge(&self) -> f32 {
        self.ridge
    }

    /// `[S / max(N, 1) ‖ ln(N + 1)]` standardised by the population moments
    /// → `[n, D + 1]`.
    fn stat_of(&self, stats: &Stats) -> anyhow::Result<Tensor> {
        let n1 = stats.totals.unsqueeze(1)?;
        let mean = stats.sums.broadcast_div(&n1.clamp(1.0, f64::INFINITY)?)?;
        let ln_n = (&n1 + 1.0)?.log()?;
        let raw = Tensor::cat(&[&mean, &ln_n], 1)?;
        Ok(raw
            .broadcast_sub(&self.dict.stat_mean)?
            .broadcast_div(&self.dict.stat_std)?)
    }

    /// Cell codes → `[n, L]`.
    fn cell_codes(&self, stats: &Stats) -> anyhow::Result<Tensor> {
        self.cell.forward(&self.stat_of(stats)?)
    }

    /// The pair code of two endpoints from their codes and pooled statistic:
    /// the head's output kept inside the optimum's ball for the pooled
    /// depth. Also returns the pair features, for the gate readout.
    fn pair_code(
        &self,
        h_u: &Tensor,
        h_v: &Tensor,
        pooled: &Stats,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let stat = self.stat_of(pooled)?;
        let features = self.head.features(h_u, h_v, Some(&stat))?;
        let z = self.head.forward(h_u, h_v, Some(&stat))?;
        let cap = pooled
            .totals
            .unsqueeze(1)?
            .affine(self.cap_per_count, 0.0)?; // [B, 1]
        let norm = z.sqr()?.sum_keepdim(1)?.sqrt()?; // [B, 1]
                                                     // min(1, cap / ‖z‖), with ‖z‖ = 0 left alone.
        let scale = (cap / (norm + 1e-12)?)?.clamp(0.0, 1.0)?;
        Ok((z.broadcast_mul(&scale)?, features))
    }

    /// The self-pair of every row of a block: its code with itself, at the
    /// doubled depth (`doubled` is the rows' statistics doubled, which the
    /// caller also scores against).
    fn self_pair_code(&self, h: &Tensor, doubled: &Stats) -> anyhow::Result<Tensor> {
        Ok(self.pair_code(h, h, doubled)?.0)
    }

    /// A block of pairs, from their endpoints' statistics (the `u`s then the
    /// `v`s), through the trunk and the head.
    fn pair_forward(&self, endpoints: Stats) -> anyhow::Result<PairForward> {
        let b = endpoints.totals.dim(0)? / 2;
        let h = self.cell_codes(&endpoints)?;
        let half = |t: &Tensor, k: usize| t.narrow(0, k * b, b);
        let (h_u, h_v) = (half(&h, 0)?, half(&h, 1)?);
        let side = |k: usize| -> anyhow::Result<Stats> {
            Ok(Stats {
                sums: half(&endpoints.sums, k)?,
                totals: half(&endpoints.totals, k)?,
                offsets: half(&endpoints.offsets, k)?,
            })
        };
        let pairs = side(0)?.pooled(&side(1)?)?;
        let (z_uv, features) = self.pair_code(&h_u, &h_v, &pairs)?;
        Ok(PairForward {
            features,
            z_uv,
            pairs,
        })
    }

    /// Scores `z·e_hd + b` → `[n, G]`.
    fn scores(&self, z: &Tensor) -> anyhow::Result<Tensor> {
        Ok(z.matmul(&self.dict.e_hd)?.broadcast_add(&self.dict.b_1d)?)
    }

    /// The profiled multinomial NLL of placements `z` for the nodes `stats`,
    /// per row: `N·lse(s) − ⟨S, z⟩ − c`.
    fn nll(&self, z: &Tensor, scores: &Tensor, stats: &Stats) -> anyhow::Result<Tensor> {
        let lse = scores.log_sum_exp(1)?;
        let data = (z * &stats.sums)?.sum(1)?;
        Ok((((&stats.totals * lse)? - data)? - &stats.offsets)?)
    }

    /// The Newton decrement `½ ∇ᵀH⁻¹∇` per row for placements `z` of the
    /// nodes `stats`, with `∇ = N p̄ − S + λz` and `H = N·Cov_p(e) + λI`
    /// under the composition `p` the scores fit: the composition's mean and
    /// second moment come off the device, the `D × D` solves run here.
    fn decrement(
        &self,
        z: &Tensor,
        scores: &Tensor,
        stats: &Stats,
        outer: Option<&[Tensor]>,
    ) -> anyhow::Result<Vec<f32>> {
        let d = self.dict.d;
        let tri = d * (d + 1) / 2;
        let p = legume_numeric::candle::candle_nn::ops::softmax(scores, 1)?; // [B, G]
        let pbar: Vec<f32> = p.matmul(&self.dict.e_gd)?.flatten_all()?.to_vec1()?; // [B, D]
        let m2: Vec<f32> = self
            .dict
            .second_moment(&p, outer)?
            .flatten_all()?
            .to_vec1()?;
        // m2 is [B, D(D+1)/2], the upper triangle row-major: (r, c) → its
        // place in it, either way round.
        let at = |r: usize, c: usize| {
            let (r, c) = if r <= c { (r, c) } else { (c, r) };
            r * d - r * (r + 1) / 2 + c
        };
        let z: Vec<f32> = z.flatten_all()?.to_vec1()?;
        let sums: Vec<f32> = stats.sums.flatten_all()?.to_vec1()?;
        let totals: Vec<f32> = stats.totals.to_vec1()?;
        let ridge = self.ridge;
        Ok(totals
            .par_iter()
            .enumerate()
            .map(|(i, &n)| {
                let (pb, zi, si) = (
                    &pbar[i * d..(i + 1) * d],
                    &z[i * d..(i + 1) * d],
                    &sums[i * d..(i + 1) * d],
                );
                let grad = nalgebra::DVector::<f32>::from_iterator(
                    d,
                    (0..d).map(|j| n * pb[j] - si[j] + ridge * zi[j]),
                );
                let hess = nalgebra::DMatrix::<f32>::from_fn(d, d, |r, c| {
                    let cov = m2[i * tri + at(r, c)] - pb[r] * pb[c];
                    n * cov + if r == c { ridge } else { 0.0 }
                });
                match hess.cholesky() {
                    Some(chol) => 0.5 * grad.dot(&chol.solve(&grad)),
                    None => f32::INFINITY,
                }
            })
            .collect())
    }

    /// One step's loss: a block of pairs' objective, plus the self-pair
    /// objective of a block of cells, each weighted by how many pair
    /// endpoints it is, over `2 · batch` (the endpoint rows of a full step).
    /// Each cell comes once per epoch, so an epoch's summed loss is the
    /// one that scored every pair's two endpoints in every step, at a fraction
    /// of the rows: a cell is an endpoint of about `2E/n` pairs.
    fn step_loss(
        &self,
        endpoints: Stats,
        selves: Option<(Stats, Tensor)>,
        batch: usize,
    ) -> anyhow::Result<Tensor> {
        let half_ridge = f64::from(self.ridge) / 2.0;
        let fwd = self.pair_forward(endpoints)?;
        let nll_uv = self.nll(&fwd.z_uv, &self.scores(&fwd.z_uv)?, &fwd.pairs)?;
        let ridge_uv = fwd.z_uv.sqr()?.sum(1)?.affine(half_ridge, 0.0)?;
        let pairs = (nll_uv + ridge_uv)?.mean_all()?;
        let Some((cells, weight)) = selves else {
            return Ok(pairs);
        };
        let h = self.cell_codes(&cells)?;
        let doubled = cells.doubled()?;
        let z_uu = self.self_pair_code(&h, &doubled)?;
        let nll_uu = self.nll(&z_uu, &self.scores(&z_uu)?, &doubled)?;
        let ridge_uu = z_uu.sqr()?.sum(1)?.affine(half_ridge, 0.0)?;
        // Over the endpoint rows a step of `batch` pairs has.
        let own = ((nll_uu + ridge_uu)? * weight)?
            .sum_all()?
            .affine(1.0 / (2 * batch) as f64, 0.0)?;
        Ok((pairs + own)?)
    }
}
