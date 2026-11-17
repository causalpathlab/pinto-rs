//! cage-cell's pair table: its start, where it is kept, and the two tables
//! downstream reads off it once trained.

use crate::cell_activity_graph_embedding::row_adam::{residency, Residency, RowAdam};
use crate::util::common::*;
use crate::util::memory::{ensure_fits, human_bytes, Budget};
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::candle::candle_nn::ParamsAdamW;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// The spread of a pair row's start, that of every table geu starts.
const INIT_STDEV: f32 = 0.1;

/// Rows drawn per seeded stream when starting the table.
const INIT_BLOCK: usize = 4096;

/// The pair table `[n_pairs × d]`, started at `N(0, INIT_STDEV)` from
/// `seed`, trained on `dev` by a lazy Adam at `lr` (the dense optimiser's
/// other settings): kept on `dev` when it and its two moments fit half of
/// `gpu_mem_fraction` of the device memory free now, else on the host,
/// within the host budget (`PINTO_MEMORY_BUDGET`, else half the memory
/// available), or refused.
pub(super) fn start_pair_table(
    n_pairs: usize,
    d: usize,
    seed: u64,
    lr: f64,
    gpu_mem_fraction: f32,
    dev: &Device,
) -> anyhow::Result<RowAdam> {
    let bytes = 3 * n_pairs * d * std::mem::size_of::<f32>();
    let free = legume_numeric::candle::device::gpu_mem_info(dev).map(|(free, _)| free);
    let store = match residency(bytes, free, gpu_mem_fraction) {
        Residency::Device => dev.clone(),
        Residency::Host => {
            ensure_fits(
                "cage-cell's pair table on the host",
                bytes,
                &Budget::now()?,
                "lower --embedding-dim",
            )?;
            Device::Cpu
        }
    };
    let mut init = vec![0f32; n_pairs * d];
    let base = legume_numeric::matrix::rand_util::name_seed(seed, "z_pair");
    init.par_chunks_mut(INIT_BLOCK * d.max(1))
        .enumerate()
        .for_each(|(block, rows)| {
            let mut rng = SmallRng::seed_from_u64(base.wrapping_add(block as u64));
            for x in rows {
                let draw: f32 = StandardNormal.sample(&mut rng);
                *x = draw * INIT_STDEV;
            }
        });
    let z = RowAdam::new(
        init,
        (n_pairs, d),
        &store,
        ParamsAdamW {
            lr,
            ..Default::default()
        },
        dev,
    )?;
    info!(
        "cage-cell: a {d}-dim row for each of {n_pairs} cell pairs, {} with its moments, {}",
        human_bytes(bytes),
        match z.residency() {
            Residency::Device => "where the steps run",
            Residency::Host => "on the host (too big for the device; each step moves its rows)",
        }
    );
    Ok(z)
}

/// The trained table `z` as a host row-major `[rows × d]` vector.
pub(super) fn table_to_host(z: &RowAdam) -> anyhow::Result<Vec<f32>> {
    let t: &Tensor = z.table();
    Ok(t.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?)
}

/// cage-cell's latent and cell embedding from its trained pair rows `z`
/// (row-major `[pairs.len() × d]`, row `e` the pair `pairs[e]`):
///
/// * the latent of every listed pair, `[fine_edges.len() × d]` in
///   `fine_edges` order, unnormalised: listed pair `k` takes row
///   `fine_to_super[k]` (a pair listed in both orientations shares it);
/// * each cell's embedding, `[n_cells × d]`: the mean of the rows of the
///   pairs it is in, each pair once (zero for a cell in none).
///
/// A listed pair with no row (`None`: it crosses batches, so no step drew
/// it) takes the mean of its two cells' embeddings.
pub(crate) fn pair_outputs(
    z: &[f32],
    d: usize,
    pairs: &[(u32, u32)],
    fine_to_super: &[Option<usize>],
    fine_edges: &[(u32, u32)],
    n_cells: usize,
) -> (Mat, Mat) {
    debug_assert_eq!(z.len(), pairs.len() * d);
    let row = |e: usize| &z[e * d..(e + 1) * d];
    let mut cells = Mat::zeros(n_cells, d);
    let mut degree = vec![0u32; n_cells];
    for (e, &(u, v)) in pairs.iter().enumerate() {
        for c in [u as usize, v as usize] {
            degree[c] += 1;
            for (j, &x) in row(e).iter().enumerate() {
                cells[(c, j)] += x;
            }
        }
    }
    for (c, &n) in degree.iter().enumerate() {
        if n > 0 {
            cells.row_mut(c).scale_mut(1.0 / n as f32);
        }
    }
    let mut latent = Mat::zeros(fine_edges.len(), d);
    for (k, (&(u, v), e)) in fine_edges.iter().zip(fine_to_super).enumerate() {
        match *e {
            Some(e) => {
                for (j, &x) in row(e).iter().enumerate() {
                    latent[(k, j)] = x;
                }
            }
            None => {
                for j in 0..d {
                    latent[(k, j)] = 0.5 * (cells[(u as usize, j)] + cells[(v as usize, j)]);
                }
            }
        }
    }
    (latent, cells)
}

/// Every row of `m` scaled to unit L2 norm (a norm below 1e-8 divides by
/// 1e-8), in place: what the shared clustering and propensity routine
/// takes, so edge k-means clusters on composition and not on scale.
pub(crate) fn normalize_rows(m: &mut Mat) {
    let mut sq = vec![0f32; m.nrows()];
    for col in m.column_iter() {
        for (s, &x) in sq.iter_mut().zip(col.iter()) {
            *s += x * x;
        }
    }
    let inv: Vec<f32> = sq.iter().map(|&s| 1.0 / s.sqrt().max(1e-8)).collect();
    for mut col in m.column_iter_mut() {
        for (x, &w) in col.iter_mut().zip(&inv) {
            *x *= w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{normalize_rows, pair_outputs};
    use crate::util::common::Mat;

    /// Three cells, two pairs, listed as four (one pair in both orientations,
    /// one crossing batches with no row): every listed pair takes its pair's
    /// row in listed order, a cell's embedding is the mean of its pairs'
    /// rows (each once), a cell in none is zero, and the row-less pair
    /// takes its two cells' mean.
    #[test]
    fn the_latent_follows_the_listed_pairs_and_a_cell_averages_its_pairs() {
        let d = 2;
        let z = [1.0f32, 2.0, 3.0, -4.0];
        let pairs = [(0u32, 1u32), (1, 2)];
        let fine = [(0u32, 1u32), (1, 2), (1, 0), (0, 3)];
        let to_super = [Some(0), Some(1), Some(0), None];
        let (latent, cells) = pair_outputs(&z, d, &pairs, &to_super, &fine, 4);
        assert_eq!(latent.nrows(), fine.len());
        let r = |m: &Mat, i: usize| vec![m[(i, 0)], m[(i, 1)]];
        assert_eq!(r(&latent, 0), [1.0, 2.0]);
        assert_eq!(r(&latent, 1), [3.0, -4.0]);
        assert_eq!(r(&latent, 2), [1.0, 2.0], "both orientations, one row");
        assert_eq!(r(&cells, 0), [1.0, 2.0]);
        assert_eq!(r(&cells, 1), [2.0, -1.0], "the mean of its two pairs");
        assert_eq!(r(&cells, 2), [3.0, -4.0]);
        assert_eq!(r(&cells, 3), [0.0, 0.0], "a cell in no pair");
        assert_eq!(r(&latent, 3), [0.5, 1.0], "the row-less pair");

        let mut m = latent.clone();
        normalize_rows(&mut m);
        for i in 0..m.nrows() {
            let n: f32 = m.row(i).iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-6, "row {i}: norm {n}");
        }
    }
}
