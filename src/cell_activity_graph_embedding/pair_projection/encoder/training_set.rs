//! What the pair encoder trains on.
//!
//! `cage-cell` trains on the pairs it places, read from their cells' own
//! rows. `cage` trains on its finest pseudobulks instead: one row per
//! pseudobulk, the sum of its cells' batch-divided rows, and the
//! pseudobulk-pseudobulk edges phase 1 trained on. The cell pairs are then
//! only encoded, a forward pass, so the cost of training follows the
//! pseudobulk count and not the cell count.
//!
//! A pseudobulk pair pools many cells' counts, and the encoder reads depth
//! (`ln(N + 1)` and the mean `S/N`, whose noise shrinks with `N`). So each
//! training pair is thinned to a cell pair's depth: every epoch draws a
//! depth from the pooled totals of the pairs being placed and keeps that
//! many of the pair's counts, sampled without replacement over its nonzeros
//! (a binomial thinning conditioned on its total). The trunk's
//! standardisation moments come from such a thinned draw too.

use super::super::PairDictionary;
use super::CellRow;
use crate::util::common::*;
use legume_numeric::matrix::rand_util::mix_seed;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::borrow::Cow;

/// Salt of the stream the moments' thinned draw comes from.
const MOMENTS_SALT: u64 = 0x004d_4f4d_454e_5453;

/// The rows the encoder trains on and the pairs between them.
pub(crate) struct TrainingSet<'a> {
    /// One row per training unit: cells, or finest pseudobulks.
    pub(crate) corpus: Cow<'a, [CellRow]>,
    /// The training pairs, indexing `corpus`.
    pub(crate) edges: &'a [(u32, u32)],
    /// Set when the pairs are thinned to the placed pairs' depth.
    pub(crate) thinning: Option<Thinning<'a>>,
}

/// The depth a training pair is thinned to, drawn from the placed pairs'.
pub(crate) struct Thinning<'a> {
    dict: &'a PairDictionary,
    /// The pooled total of every pair to be placed.
    depths: Vec<f32>,
}

impl Thinning<'_> {
    fn draw(&self, rng: &mut StdRng) -> f32 {
        self.depths[rng.random_range(0..self.depths.len())]
    }

    /// The median pooled depth, for the log.
    fn median(&self) -> f32 {
        let mut d = self.depths.clone();
        let mid = d.len() / 2;
        *d.select_nth_unstable_by(mid, f32::total_cmp).1
    }
}

/// One thinned draw of some training pairs (and, optionally, of every
/// unit's own row): what one epoch, the evaluation or the moments read.
pub(crate) struct ThinnedTable {
    pub(crate) rows: Vec<CellRow>,
    /// Per drawn pair, in the order asked for, its two rows in `rows`.
    pub(crate) pairs: Vec<(u32, u32)>,
    /// Unit `c`'s own thinned row is `rows[self_base + c]`, when drawn.
    pub(crate) self_base: u32,
}

impl<'a> TrainingSet<'a> {
    /// The cells and the pairs being placed, as they are (`cage-cell`).
    pub(crate) fn cells(corpus: &'a [CellRow], edges: &'a [(u32, u32)]) -> Self {
        Self {
            corpus: Cow::Borrowed(corpus),
            edges,
            thinning: None,
        }
    }

    /// The finest pseudobulks (`cage`): each the sum of its cells' rows by
    /// `cell_to_pb`, the pseudobulk edges `pb_edges` between them, thinned
    /// to the pooled depth of the cell pairs `cell_edges` over `cells`.
    pub(crate) fn pseudobulks(
        dict: &'a PairDictionary,
        cells: &[CellRow],
        cell_to_pb: &[usize],
        n_pb: usize,
        pb_edges: &'a [(u32, u32)],
        cell_edges: &[(u32, u32)],
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            cell_to_pb.len() == cells.len(),
            "pair encoder: {} pseudobulk labels for {} cells",
            cell_to_pb.len(),
            cells.len()
        );
        anyhow::ensure!(
            cell_to_pb.iter().all(|&p| p < n_pb),
            "pair encoder: a cell's pseudobulk is beyond the {n_pb} trained"
        );
        anyhow::ensure!(
            pb_edges.iter().all(|&(u, v)| (u.max(v) as usize) < n_pb),
            "pair encoder: a pseudobulk edge names a pseudobulk beyond the {n_pb} trained"
        );
        anyhow::ensure!(
            cell_edges
                .iter()
                .all(|&(u, v)| (u.max(v) as usize) < cells.len()),
            "pair encoder: a cell pair names a cell beyond the {}",
            cells.len()
        );
        anyhow::ensure!(
            !cell_edges.is_empty(),
            "pair encoder: no cell pairs to take the training depth from"
        );
        let depths = cell_edges
            .par_iter()
            .map(|&(u, v)| cells[u as usize].total + cells[v as usize].total)
            .collect();
        Ok(Self {
            corpus: Cow::Owned(pseudobulk_corpus(dict, cells, cell_to_pb, n_pb)),
            edges: pb_edges,
            thinning: Some(Thinning { dict, depths }),
        })
    }

    /// What the training log says it trains on.
    pub(crate) fn describe(&self) -> String {
        match &self.thinning {
            None => format!(
                "{} cell pairs over {} cells",
                self.edges.len(),
                self.corpus.len()
            ),
            Some(th) => format!(
                "{} pseudobulk pairs over {} pseudobulks, thinned each epoch to the pooled depth \
                 of the {} cell pairs (median {:.0})",
                self.edges.len(),
                self.corpus.len(),
                th.depths.len(),
                th.median()
            ),
        }
    }

    /// The rows the trunk's standardisation moments are taken over: the
    /// corpus, or the endpoint rows of one thinned draw of every pair.
    pub(crate) fn moment_rows(&self, seed: u64) -> Cow<'_, [CellRow]> {
        match &self.thinning {
            None => Cow::Borrowed(&self.corpus[..]),
            Some(th) => {
                let all: Vec<usize> = (0..self.edges.len()).collect();
                Cow::Owned(
                    self.thinned(th, &all, false, mix_seed(seed, MOMENTS_SALT))
                        .rows,
                )
            }
        }
    }

    /// The pairs `pick` (indices into `edges`) thinned, each to a depth
    /// drawn afresh, and with `selves` every unit's own row thinned to half
    /// a drawn depth (its self-pair, the row doubled, is then at one).
    /// Deterministic in `seed`: pair `i` and unit `c` each draw from their
    /// own stream.
    pub(crate) fn thinned(
        &self,
        th: &Thinning<'_>,
        pick: &[usize],
        selves: bool,
        seed: u64,
    ) -> ThinnedTable {
        let n_pick = pick.len();
        let (us, vs): (Vec<CellRow>, Vec<CellRow>) = pick
            .par_iter()
            .map(|&i| {
                let mut rng = StdRng::seed_from_u64(mix_seed(seed, i as u64));
                let (u, v) = self.edges[i];
                let target = th.draw(&mut rng).round() as u64;
                let rows = [&self.corpus[u as usize], &self.corpus[v as usize]];
                let mut parts = thin_to(&rows, target, &mut rng).into_iter();
                let mut row = || {
                    let (f, c) = parts.next().expect("one part per row");
                    CellRow::from_local(th.dict, f, c)
                };
                (row(), row())
            })
            .unzip();
        let mut rows = us;
        rows.extend(vs);
        if selves {
            let base = self.edges.len() as u64;
            let own: Vec<CellRow> = self
                .corpus
                .par_iter()
                .enumerate()
                .map(|(c, r)| {
                    let mut rng = StdRng::seed_from_u64(mix_seed(seed, base + c as u64));
                    let target = (th.draw(&mut rng) / 2.0).round() as u64;
                    let (f, n) = thin_to(&[r], target, &mut rng)
                        .pop()
                        .expect("one part per row");
                    CellRow::from_local(th.dict, f, n)
                })
                .collect();
            rows.extend(own);
        }
        ThinnedTable {
            rows,
            pairs: (0..n_pick as u32).map(|k| (k, n_pick as u32 + k)).collect(),
            self_base: 2 * n_pick as u32,
        }
    }
}

/// One row per pseudobulk: the sum of its cells' rows, sparse, on the same
/// active axis.
pub(crate) fn pseudobulk_corpus(
    dict: &PairDictionary,
    cells: &[CellRow],
    cell_to_pb: &[usize],
    n_pb: usize,
) -> Vec<CellRow> {
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_pb];
    for (c, &p) in cell_to_pb.iter().enumerate() {
        members[p].push(c);
    }
    members
        .par_iter()
        .map(|m| {
            let mut obs: Vec<(u32, f32)> = m
                .iter()
                .flat_map(|&c| {
                    let r = &cells[c];
                    r.features.iter().copied().zip(r.counts.iter().copied())
                })
                .collect();
            obs.sort_unstable_by_key(|&(g, _)| g);
            let (mut features, mut counts) = (Vec::new(), Vec::<f32>::new());
            for (g, n) in obs {
                if features.last() == Some(&g) {
                    *counts.last_mut().expect("paired with features") += n;
                } else {
                    features.push(g);
                    counts.push(n);
                }
            }
            CellRow::from_local(dict, features, counts)
        })
        .collect()
}

/// `rows` thinned together to `target` counts, as `(positions, counts)`
/// per row: `target` of their pooled units kept, sampled without
/// replacement, so a pair is thinned as its pooled profile and each
/// endpoint keeps its share. Rows already within `target` come back as they
/// are. A fractional (batch-divided) count is first rounded to whole units
/// at random, up with probability its fraction, so the kept total is
/// `target` unless those units fall short of it, and a kept count never
/// exceeds its original rounded up. Only the nonzeros are walked.
pub(crate) fn thin_to(
    rows: &[&CellRow],
    target: u64,
    rng: &mut StdRng,
) -> Vec<(Vec<u32>, Vec<f32>)> {
    let total: f64 = rows.iter().map(|r| f64::from(r.total)).sum();
    if target as f64 >= total {
        return rows
            .iter()
            .map(|r| (r.features.clone(), r.counts.clone()))
            .collect();
    }
    let units: Vec<Vec<u64>> = rows
        .iter()
        .map(|r| {
            r.counts
                .iter()
                .map(|&x| {
                    let whole = x.floor();
                    let frac = x - whole;
                    whole as u64 + u64::from(frac > 0.0 && rng.random::<f32>() < frac)
                })
                .collect()
        })
        .collect();
    let m: u64 = units.iter().flatten().sum();
    let keep = target.min(m);
    // Draw whichever of the kept and the dropped units is fewer.
    let draw_kept = keep <= m - keep;
    let picks = distinct_sorted(rng, m, if draw_kept { keep } else { m - keep });
    let (mut p, mut base) = (0usize, 0u64);
    rows.iter()
        .zip(&units)
        .map(|(r, u)| {
            let (mut features, mut counts) = (Vec::new(), Vec::new());
            for (&g, &n) in r.features.iter().zip(u) {
                let end = base + n;
                let start = p;
                while p < picks.len() && picks[p] < end {
                    p += 1;
                }
                let hit = (p - start) as u64;
                let kept = if draw_kept { hit } else { n - hit };
                if kept > 0 {
                    features.push(g);
                    counts.push(kept as f32);
                }
                base = end;
            }
            (features, counts)
        })
        .collect()
}

/// `k` distinct values of `0..m`, uniformly at random among such sets,
/// sorted. Draws with replacement and redraws the repeats, which is quick
/// when `k ≤ m / 2`, as [`thin_to`] asks.
fn distinct_sorted(rng: &mut StdRng, m: u64, k: u64) -> Vec<u64> {
    debug_assert!(k <= m);
    let k = k as usize;
    let mut out: Vec<u64> = Vec::with_capacity(k);
    while out.len() < k {
        let need = k - out.len();
        out.extend((0..need).map(|_| rng.random_range(0..m)));
        out.sort_unstable();
        out.dedup();
    }
    out
}
