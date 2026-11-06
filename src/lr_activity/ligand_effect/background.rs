//! Matched background programmes: random programmes of like expression that a
//! real effect should exceed.

use super::options::{BACKGROUND_NEIGHBOURS, BACKGROUND_POOL};

/// Genes for the background pool: up to [`super::options::BACKGROUND_POOL`] of
/// the `n_genes` genes not in `exclude` (every ligand, receptor and programme
/// gene), and at most a fifth of them so a small panel keeps most of its genes
/// for the identity; drawn at random with `seed`, sorted.
pub fn background_pool_genes(n_genes: usize, exclude: &[usize], seed: u64) -> Vec<usize> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let ex: std::collections::HashSet<usize> = exclude.iter().copied().collect();
    let mut cand: Vec<usize> = (0..n_genes).filter(|g| !ex.contains(g)).collect();
    let k = BACKGROUND_POOL.min(cand.len() / 5);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed ^ 0x000d_ec07);
    cand.shuffle(&mut rng);
    cand.truncate(k);
    cand.sort_unstable();
    cand
}

/// Mean of `x / size factor` over all rows.
pub fn mean_expression(nz: &[(u32, f32)], sf: &[f64]) -> f64 {
    nz.iter()
        .map(|&(r, x)| x as f64 / sf[r as usize])
        .sum::<f64>()
        / sf.len().max(1) as f64
}

/// Genes outside every pair and programme, by mean expression, from which
/// each pair's background programme is matched.
#[derive(Default)]
pub struct BackgroundPool<'a> {
    /// `(log mean expression, non-zeros)`, sorted by the first.
    genes: Vec<(f64, &'a [(u32, f32)])>,
}

impl<'a> BackgroundPool<'a> {
    /// The pool from its genes' non-zero counts on the tissue's rows; genes
    /// never detected are left out.
    pub fn new(sf: &[f64], genes: Vec<&'a [(u32, f32)]>) -> Self {
        let mut genes: Vec<(f64, &'a [(u32, f32)])> = genes
            .into_iter()
            .filter(|nz| !nz.is_empty())
            .map(|nz| (mean_expression(nz, sf).ln(), nz))
            .collect();
        genes.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self { genes }
    }

    /// Each target's weight and log mean expression, by decreasing `|w|`: the
    /// order [`BackgroundPool::draw`] matches them in.
    pub fn plan(sf: &[f64], targets: &[(f64, &[(u32, f32)])]) -> Vec<(f64, f64)> {
        let mut order: Vec<usize> = (0..targets.len()).collect();
        order.sort_by(|&a, &b| targets[b].0.abs().total_cmp(&targets[a].0.abs()));
        order
            .into_iter()
            .map(|k| {
                let (w, nz) = targets[k];
                (w, mean_expression(nz, sf).max(1e-12).ln())
            })
            .collect()
    }

    /// A background programme for a programme: a random unused gene among the
    /// [`super::options::BACKGROUND_NEIGHBOURS`] nearest in log mean expression
    /// to each target of `plan`, with the target's weight. Fewer genes than
    /// targets when the pool runs out; empty when it holds fewer than half the
    /// programme.
    pub fn draw(&self, plan: &[(f64, f64)], seed: u64) -> Vec<(f64, &'a [(u32, f32)])> {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut used = vec![false; self.genes.len()];
        let mut out = Vec::with_capacity(plan.len());
        for &(w, key) in plan {
            let at = self.genes.partition_point(|g| g.0 < key);
            // The nearest unused, walking outward from the target's place.
            let (mut lo, mut hi) = (at, at);
            let mut near: Vec<usize> = Vec::with_capacity(BACKGROUND_NEIGHBOURS);
            while near.len() < BACKGROUND_NEIGHBOURS && (lo > 0 || hi < used.len()) {
                let go_low = match (lo > 0, hi < used.len()) {
                    (true, true) => key - self.genes[lo - 1].0 <= self.genes[hi].0 - key,
                    (low, _) => low,
                };
                let g = if go_low {
                    lo -= 1;
                    lo
                } else {
                    hi += 1;
                    hi - 1
                };
                if !used[g] {
                    near.push(g);
                }
            }
            if near.is_empty() {
                break;
            }
            let j = near[rng.random_range(0..near.len())];
            used[j] = true;
            out.push((w, self.genes[j].1));
        }
        if 2 * out.len() < plan.len() || out.is_empty() {
            return Vec::new();
        }
        out
    }
}
