//! The retrieval of [`data_beans::alg::retrieval_impute`], with each
//! reference cell's counts divided by its batch's δ before they are pooled.
//!
//! Same neighbours, same weights, same pooling: kNN over the reference
//! propensity rows that carry evidence, softmax weights
//! `w ∝ exp(-d² / 2τ²)`, and a streamed weighted average of the reference
//! columns. The divide has no hook upstream, so the core is repeated here;
//! without one the result is the upstream one (a test holds them equal).
//! This mirrors `data_beans::alg::retrieval_impute::retrieval_impute`, and
//! should go once that function takes a per-column transform.

use crate::util::common::*;
use data_beans::alg::retrieval_impute::RetrievalImputeConfig;
use legume_numeric::matrix::knn_match::{ColumnDict, SearchScratch};

/// The run's δ on the reference: `[reference rows × batches]`, and each
/// reference cell's column of it (`None`: left as is).
pub(crate) struct RefDivide {
    pub delta: Mat,
    pub batch_of_cell: Vec<Option<usize>>,
}

/// `[N_query × G_ref]`: each query row's weighted average of its nearest
/// reference cells' counts, each count divided by `divide`'s δ for its row
/// and cell (a plain `x / δ`, which keeps counts on their scale, δ having
/// geometric mean 1 over the batches).
pub(crate) fn retrieval_impute(
    query_latent: &Mat,
    ref_latent: &Mat,
    ref_data: &SparseIoVec,
    cfg: &RetrievalImputeConfig,
    divide: Option<&RefDivide>,
) -> anyhow::Result<Mat> {
    let (n_query, n_ref) = (query_latent.nrows(), ref_latent.nrows());
    anyhow::ensure!(
        query_latent.ncols() == ref_latent.ncols(),
        "latent dimension mismatch: query K={} vs reference K={}",
        query_latent.ncols(),
        ref_latent.ncols()
    );
    anyhow::ensure!(cfg.knn >= 1, "knn must be at least 1");
    anyhow::ensure!(n_ref > 0, "reference latent has no rows");
    anyhow::ensure!(
        ref_data.num_columns() == n_ref,
        "reference data has {} cells but reference latent has {n_ref}; \
         the data files don't match the latent",
        ref_data.num_columns(),
    );
    let g_ref = ref_data.num_rows();
    anyhow::ensure!(g_ref > 0, "reference data has no features");

    let live: Vec<usize> = (0..n_ref)
        .filter(|&i| ref_latent.row(i).iter().any(|&x| x != 0.0))
        .collect();
    anyhow::ensure!(!live.is_empty(), "every reference latent row is zero");
    if live.len() < n_ref {
        warn!(
            "{} of {n_ref} reference rows have a zero latent and are excluded from the index",
            n_ref - live.len()
        );
    }
    let k_dim = ref_latent.ncols();
    let dict = ColumnDict::<u32>::from_dmatrix(
        Mat::from_fn(k_dim, live.len(), |r, c| ref_latent[(live[c], r)]),
        live.iter().map(|&i| i as u32).collect(),
    );
    let knn = cfg.knn.min(live.len());
    let tau = cfg.temperature.max(1e-6);

    // Each reference cell's consumers: (query row, weight).
    let neighbours: Vec<Option<(Vec<u32>, Vec<f32>)>> = (0..n_query)
        .into_par_iter()
        .map_init(SearchScratch::default, |scratch, i| {
            let query: Vec<f32> = query_latent.row(i).iter().copied().collect();
            if query.iter().all(|&x| x == 0.0) {
                return Ok(None);
            }
            let (ids, d) = dict
                .search_by_query_data_reuse(&query, knn, scratch)
                .map_err(|e| anyhow::anyhow!("kNN search for query row {i}: {e}"))?;
            Ok(Some((ids, softmax_weights(&d, tau))))
        })
        .collect::<anyhow::Result<_>>()?;
    let n_zero = neighbours.iter().filter(|n| n.is_none()).count();
    if n_zero > 0 {
        warn!(
            "{n_zero} of {n_query} query rows have a zero latent and were skipped; \
             their imputed rows stay zero"
        );
    }
    let mut consumers: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_ref];
    for (q, nbr) in neighbours.iter().enumerate() {
        if let Some((ids, w)) = nbr {
            for (&c, &w) in ids.iter().zip(w) {
                consumers[c as usize].push((q as u32, w));
            }
        }
    }

    // Accumulated transposed, a query row per contiguous column.
    let mut imputed_t = Mat::zeros(g_ref, n_query);
    let mut groups: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_query];
    let chunk = cfg.chunk.max(64);
    for lb in (0..n_ref).step_by(chunk) {
        let ub = (lb + chunk).min(n_ref);
        if consumers[lb..ub].iter().all(Vec::is_empty) {
            continue;
        }
        let mut csc = ref_data.read_columns_csc(lb..ub)?;
        if let Some(div) = divide {
            for (j, mut col) in csc.col_iter_mut().enumerate() {
                let Some(b) = div.batch_of_cell[lb + j] else {
                    continue;
                };
                let (rows, vals) = col.rows_and_values_mut();
                for (&g, v) in rows.iter().zip(vals.iter_mut()) {
                    let d = div.delta[(g, b)];
                    if d > 0.0 {
                        *v /= d;
                    }
                }
            }
        }
        for c in 0..(ub - lb) {
            for &(q, w) in &consumers[lb + c] {
                groups[q as usize].push((c as u32, w));
            }
        }
        imputed_t
            .as_mut_slice()
            .par_chunks_mut(g_ref)
            .zip(groups.par_iter_mut())
            .for_each(|(out, grp)| {
                for &(c, w) in grp.iter() {
                    let col = csc.col(c as usize);
                    for (&g, &v) in col.row_indices().iter().zip(col.values()) {
                        out[g] += w * v;
                    }
                }
                grp.clear();
            });
    }
    Ok(imputed_t.transpose())
}

/// `w_k ∝ exp(-d_k² / 2τ²)`, summing to 1.
fn softmax_weights(distances: &[f32], tau: f32) -> Vec<f32> {
    let scale = 1.0 / (2.0 * tau * tau);
    let logits: Vec<f32> = distances.iter().map(|d| -d * d * scale).collect();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let inv = 1.0 / w.iter().sum::<f32>().max(1e-12);
    w.iter().map(|x| x * inv).collect()
}
