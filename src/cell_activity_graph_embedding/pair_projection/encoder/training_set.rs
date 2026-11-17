//! What the pair encoder trains on.
//!
//! `cage-cell` trains on the pairs it places, read from their cells' own
//! rows. `cage` trains on its finest pseudobulks instead: one row per
//! pseudobulk, the sum of its cells' batch-divided rows, and the
//! pseudobulk-pseudobulk edges phase 1 trained on. The cell pairs are then
//! only encoded, a forward pass, so the cost of training follows the
//! pseudobulk count and not the cell count.
//!
//! A pseudobulk pools many cells' counts, and the encoder reads depth
//! (`ln(N + 1)` and the mean `S/N`, whose noise shrinks with `N`). So each
//! training row is thinned to a cell's depth: every epoch draws, per
//! pseudobulk pair, a depth from the pooled totals of the cell pairs being
//! placed, and keeps half of it from each endpoint (so a large pseudobulk
//! does not crowd out a small one, as a cell pair's two cells are about
//! even); each pseudobulk's own row is kept at half a drawn depth too. The
//! counts kept are sampled without replacement over the row's nonzeros (a
//! binomial thinning conditioned on its total). The trunk's
//! standardisation moments come from the first epoch's draw.

use super::super::PairDictionary;
use super::{CellRow, MIN_STEP_PAIRS};
use crate::util::common::*;
use legume_numeric::matrix::rand_util::{mix_seed, stream_seed};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

/// The rows the encoder trains on and the pairs between them.
pub(crate) struct TrainingSet<'a> {
    /// Training units: cells, or finest pseudobulks.
    pub(crate) n_units: usize,
    /// The training pairs, between units.
    pub(crate) edges: &'a [(u32, u32)],
    /// The units some training pair names, ascending: the only ones whose
    /// own row a training step reads.
    pub(crate) visited: Vec<u32>,
    /// Set when the rows are pseudobulks thinned to the placed pairs' depth.
    /// Without it the rows are the placed cells as they are.
    pub(crate) thinning: Option<Thinning<'a>>,
}

/// A pseudobulk's counts on the active axis, sorted by position: all the
/// thinning reads.
pub(crate) struct SparseRow {
    pub(crate) features: Vec<u32>,
    pub(crate) counts: Vec<f32>,
}

/// The pseudobulk rows, the depth each is thinned to (drawn from the
/// placed pairs'), and where a [`ThinnedTable`] keeps each row.
pub(crate) struct Thinning<'a> {
    dict: &'a PairDictionary,
    /// One row per pseudobulk.
    pub(crate) rows: Vec<SparseRow>,
    /// The pooled total of every pair to be placed.
    depths: Vec<f32>,
    /// Their median, for the log.
    median_depth: f32,
    /// Unit → its position in [`TrainingSet::visited`] (`u32::MAX` when
    /// unvisited).
    rank: Vec<u32>,
}

impl Thinning<'_> {
    fn draw(&self, rng: &mut SmallRng) -> f32 {
        self.depths[rng.random_range(0..self.depths.len())]
    }
}

/// One thinned draw of every training pair's endpoints and every visited
/// unit's own row, as the statistics the encoder reads, packed `[S ‖ N ‖
/// c]` per row: rows `[0, E)` the `u`s, `[E, 2E)` the `v`s, then the
/// visited units' own (see [`TrainingSet::pair_rows`] and
/// [`TrainingSet::self_row`]). The thinned counts themselves are never
/// kept.
pub(crate) struct ThinnedTable {
    /// Row-major `[rows × (D + 2)]`.
    pub(crate) packed: Vec<f32>,
    pub(crate) d: usize,
}

impl ThinnedTable {
    pub(crate) fn n_rows(&self) -> usize {
        self.packed.len() / (self.d + 2)
    }

    /// Row `i`'s `(S, N)`.
    pub(crate) fn row(&self, i: usize) -> (&[f32], f32) {
        let w = self.d + 2;
        let r = &self.packed[i * w..(i + 1) * w];
        (&r[..self.d], r[self.d])
    }
}

/// The units some pair of `edges` names, ascending.
fn visited_units(n_units: usize, edges: &[(u32, u32)]) -> Vec<u32> {
    let mut seen = vec![false; n_units];
    for &(u, v) in edges {
        seen[u as usize] = true;
        seen[v as usize] = true;
    }
    (0..n_units as u32).filter(|&c| seen[c as usize]).collect()
}

impl<'a> TrainingSet<'a> {
    /// The `n_cells` cells and the pairs being placed, as they are
    /// (`cage-cell`).
    pub(crate) fn cells(n_cells: usize, edges: &'a [(u32, u32)]) -> Self {
        Self {
            n_units: n_cells,
            edges,
            visited: visited_units(n_cells, edges),
            thinning: None,
        }
    }

    /// The finest pseudobulks (`cage`): each the sum of its cells' rows by
    /// `cell_to_pb`, the pseudobulk edges `pb_edges` between them, thinned
    /// to the pooled depth of the cell pairs `cell_edges` over `cells`.
    /// With too few pseudobulk pairs for one training step, the cells and
    /// their pairs instead, with a warning.
    pub(crate) fn pseudobulks(
        dict: &'a PairDictionary,
        cells: &[CellRow],
        cell_to_pb: &[usize],
        n_pb: usize,
        pb_edges: &'a [(u32, u32)],
        cell_edges: &'a [(u32, u32)],
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
        if pb_edges.len() < MIN_STEP_PAIRS {
            warn!(
                "Pair encoder: {} pseudobulk pairs over {n_pb} pseudobulks are fewer than one \
                 training step's {MIN_STEP_PAIRS}; training on the {} cell pairs instead",
                pb_edges.len(),
                cell_edges.len()
            );
            return Ok(Self::cells(cells.len(), cell_edges));
        }
        let mut depths: Vec<f32> = cell_edges
            .par_iter()
            .map(|&(u, v)| cells[u as usize].total + cells[v as usize].total)
            .collect();
        let median_depth = {
            let mid = depths.len() / 2;
            *depths.select_nth_unstable_by(mid, f32::total_cmp).1
        };
        let visited = visited_units(n_pb, pb_edges);
        let mut rank = vec![u32::MAX; n_pb];
        for (k, &c) in visited.iter().enumerate() {
            rank[c as usize] = k as u32;
        }
        Ok(Self {
            n_units: n_pb,
            edges: pb_edges,
            visited,
            thinning: Some(Thinning {
                dict,
                rows: pseudobulk_rows(cells, cell_to_pb, n_pb),
                depths,
                median_depth,
                rank,
            }),
        })
    }

    /// What the training log says it trains on.
    pub(crate) fn describe(&self) -> String {
        match &self.thinning {
            None => format!(
                "{} cell pairs over {} cells",
                self.edges.len(),
                self.n_units
            ),
            Some(th) => format!(
                "{} pseudobulk pairs over {} pseudobulks, each endpoint thinned each epoch to \
                 half the pooled depth of the {} cell pairs (median {:.0})",
                self.edges.len(),
                self.n_units,
                th.depths.len(),
                th.median_depth
            ),
        }
    }

    /// The rows of pair `i`'s two endpoints in an epoch's statistics.
    pub(crate) fn pair_rows(&self, i: usize) -> (u32, u32) {
        match self.thinning {
            None => self.edges[i],
            Some(_) => {
                let e = self.edges.len();
                (i as u32, (e + i) as u32)
            }
        }
    }

    /// The row of visited unit `c`'s own row in an epoch's statistics.
    pub(crate) fn self_row(&self, c: u32) -> u32 {
        match &self.thinning {
            None => c,
            Some(th) => {
                let r = th.rank[c as usize];
                debug_assert!(r != u32::MAX, "unit {c} is in no training pair");
                2 * self.edges.len() as u32 + r
            }
        }
    }

    /// The trunk's standardisation moments (`(mean, std)`, `[D + 1]`
    /// each), over the rows a training step reads: the placed `cells`
    /// (each both an endpoint and its own self row), or every row of the
    /// first epoch's draw `first`, the pairs' endpoints and the units' own.
    pub(crate) fn moments(
        &self,
        d: usize,
        first: Option<&ThinnedTable>,
        cells: &[CellRow],
    ) -> (Vec<f32>, Vec<f32>) {
        match first {
            None => super::stat_moments(d, cells.len(), |i| (&cells[i].sums[..], cells[i].total)),
            Some(t) => super::stat_moments(d, t.n_rows(), |i| t.row(i)),
        }
    }

    /// Epoch `epoch`'s draw (`None` without thinning): every training
    /// pair's endpoints, each thinned to half a depth drawn afresh per
    /// pair, and every visited unit's own row, thinned to half a depth of
    /// its own (its self-pair, the row doubled, is then at one).
    /// Deterministic in `(seed, epoch)`: pair `i` and unit `c` each draw
    /// from their own stream.
    pub(crate) fn draw(&self, seed: u64, epoch: usize) -> Option<ThinnedTable> {
        let th = self.thinning.as_ref()?;
        let seed = stream_seed(seed, "pair-encoder-thin", epoch, 0);
        let d = th.dict.d;
        // Per row `[S ‖ N ‖ c]`.
        let w = d + 2;
        let n_pairs = self.edges.len();
        let mut packed = vec![0f32; (2 * n_pairs + self.visited.len()) * w];
        let (us, rest) = packed.split_at_mut(n_pairs * w);
        let (vs, selves) = rest.split_at_mut(n_pairs * w);
        us.par_chunks_mut(w)
            .zip(vs.par_chunks_mut(w))
            .enumerate()
            .for_each_init(Scratch::default, |scratch, (i, (out_u, out_v))| {
                let mut rng = SmallRng::seed_from_u64(mix_seed(seed, i as u64));
                let half = (th.draw(&mut rng) / 2.0).round() as u64;
                let (u, v) = self.edges[i];
                for (unit, o) in [(u, out_u), (v, out_v)] {
                    let row = &th.rows[unit as usize];
                    thin_stats(th.dict, row, half, &mut rng, scratch, o);
                }
            });
        selves
            .par_chunks_mut(w)
            .zip(self.visited.par_iter())
            .for_each_init(Scratch::default, |scratch, (out, &c)| {
                let salt = (n_pairs + c as usize) as u64;
                let mut rng = SmallRng::seed_from_u64(mix_seed(seed, salt));
                let half = (th.draw(&mut rng) / 2.0).round() as u64;
                let row = &th.rows[c as usize];
                thin_stats(th.dict, row, half, &mut rng, scratch, out);
            });
        Some(ThinnedTable { packed, d })
    }
}

/// One sparse row per pseudobulk: the sum of its cells' rows, on the same
/// active axis.
pub(crate) fn pseudobulk_rows(
    cells: &[CellRow],
    cell_to_pb: &[usize],
    n_pb: usize,
) -> Vec<SparseRow> {
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
            obs.sort_unstable_by_key(|o| o.0);
            obs.dedup_by(|next, kept| {
                if next.0 == kept.0 {
                    kept.1 += next.1;
                    true
                } else {
                    false
                }
            });
            let (features, counts) = obs.into_iter().unzip();
            SparseRow { features, counts }
        })
        .collect()
}

/// Per-thread buffers for [`kept_units`].
#[derive(Default)]
pub(crate) struct Scratch {
    /// Per nonzero of the row: whole units, then units kept.
    units: Vec<u64>,
    picks: Vec<u64>,
}

/// `row` thinned to `target` units, leaving each nonzero's kept units in
/// `scratch.units`.
///
/// Every count is first rounded to whole units at random, up with
/// probability its fraction (a batch-divided count is fractional), whether
/// or not the row is then thinned, so every row of every epoch is whole
/// units. `target` of those units are then kept, sampled without
/// replacement, or all of them when they are fewer: the kept total is
/// `min(target, units)` and a kept count never exceeds its original
/// rounded up. Only the nonzeros are walked.
fn kept_units<R: Rng>(row: &SparseRow, target: u64, rng: &mut R, scratch: &mut Scratch) {
    let units = &mut scratch.units;
    units.clear();
    units.extend(row.counts.iter().map(|&x| {
        let whole = x.floor();
        let frac = x - whole;
        whole as u64 + u64::from(frac > 0.0 && rng.random::<f32>() < frac)
    }));
    let m: u64 = units.iter().sum();
    if target >= m {
        return;
    }
    // Draw whichever of the kept and the dropped units is fewer.
    let draw_kept = target <= m - target;
    let k = if draw_kept { target } else { m - target };
    distinct_sorted(rng, m, k, &mut scratch.picks);
    let picks = &scratch.picks;
    let (mut p, mut end) = (0usize, 0u64);
    for n in units.iter_mut() {
        end += *n;
        let start = p;
        while p < picks.len() && picks[p] < end {
            p += 1;
        }
        let hit = (p - start) as u64;
        *n = if draw_kept { hit } else { *n - hit };
    }
}

/// `row` thinned by [`kept_units`], as a sparse `(positions, counts)`:
/// what [`thin_stats`] reads the statistic of, for the tests.
#[cfg(test)]
pub(crate) fn thin_to<R: Rng>(row: &SparseRow, target: u64, rng: &mut R) -> (Vec<u32>, Vec<f32>) {
    let mut scratch = Scratch::default();
    kept_units(row, target, rng, &mut scratch);
    row.features
        .iter()
        .zip(&scratch.units)
        .filter(|&(_, &k)| k > 0)
        .map(|(&g, &k)| (g, k as f32))
        .unzip()
}

/// `row` thinned by [`kept_units`], written as its statistic `[S ‖ N ‖ c]`
/// into `out` (`D + 2` long): what the encoder reads, accumulated straight
/// from the kept units over the nonzeros that keep any.
pub(crate) fn thin_stats(
    dict: &PairDictionary,
    row: &SparseRow,
    target: u64,
    rng: &mut SmallRng,
    scratch: &mut Scratch,
    out: &mut [f32],
) {
    let d = dict.d;
    kept_units(row, target, rng, scratch);
    out.fill(0.0);
    let (sums, rest) = out.split_at_mut(d);
    let kept = row
        .features
        .iter()
        .zip(&scratch.units)
        .filter(|&(_, &k)| k > 0)
        .map(|(&g, &k)| (g, k as f32));
    let (total, offset) = dict.accumulate(kept, sums);
    rest[0] = total;
    rest[1] = offset;
}

/// `k` distinct values of `0..m`, uniformly at random among such sets,
/// sorted, into `out`. Draws with replacement and redraws the repeats,
/// which is quick when `k ≤ m / 2`, as [`kept_units`] asks.
fn distinct_sorted<R: Rng>(rng: &mut R, m: u64, k: u64, out: &mut Vec<u64>) {
    debug_assert!(k <= m);
    let k = k as usize;
    out.clear();
    while out.len() < k {
        let need = k - out.len();
        out.extend((0..need).map(|_| rng.random_range(0..m)));
        out.sort_unstable();
        out.dedup();
    }
}
