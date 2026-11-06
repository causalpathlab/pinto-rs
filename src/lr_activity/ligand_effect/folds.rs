//! Spatial tiles as cross-fitting folds and variance clusters.
//!
//! Receivers next to each other share senders, so neither a random split
//! nor an independent-rows variance is honest. Each section is cut into
//! square tiles a fixed number of contact lengths wide; a tile is one
//! variance cluster, and whole tiles are dealt to folds, so a receiver is
//! predicted by nuisance models that never saw its neighbourhood (except
//! at tile edges).

use super::crossfit::CrossFit;
use crate::util::common::*;
use legume_numeric::matrix::rand_util::mix_seed;

/// Tile the rows. `side[b]` is the tile width in section `b`'s
/// coordinate units.
pub fn spatial_folds(
    coords: &[(f32, f32)],
    section: &[u32],
    side: &[f64],
    k: usize,
    seed: u64,
) -> CrossFit {
    let mut tile_id: HashMap<(u32, i64, i64), u32> = HashMap::default();
    let mut cluster = Vec::with_capacity(coords.len());
    for (&(x, y), &b) in coords.iter().zip(section) {
        let s = side[b as usize].max(1e-9);
        let key = (
            b,
            (x as f64 / s).floor() as i64,
            (y as f64 / s).floor() as i64,
        );
        let next = tile_id.len() as u32;
        cluster.push(*tile_id.entry(key).or_insert(next));
    }
    // Deal tiles to folds by a seeded hash of the tile key, not its
    // discovery order, so the split does not depend on row order.
    let mut tile_fold = vec![0u8; tile_id.len()];
    for (&(b, tx, ty), &id) in &tile_id {
        let h = mix_seed(mix_seed(mix_seed(seed, b as u64), tx as u64), ty as u64);
        tile_fold[id as usize] = (h % k as u64) as u8;
    }
    let fold = cluster.iter().map(|&c| tile_fold[c as usize]).collect();
    CrossFit::new(fold, k, cluster)
}
