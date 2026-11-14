//! Reading a run's edges and expression, and collapsing cells into
//! pseudobulk samples.

use super::BATCH_LABEL_ALL;
use crate::lr_activity::io::*;
use crate::util::common::*;
use legume_numeric::param::dmatrix_gamma::GammaMatrix;
use legume_numeric::param::traits::{CalibrateTarget, Inference, TwoStatParam};
use nalgebra::DMatrix;

const GAMMA_B0: f32 = 1.0;

/// Gamma posterior hyperparameters for pseudobulk rates. `Gamma(1, 1)` is
/// the standard weak-but-proper prior used elsewhere in pinto (propensity,
/// link-community profiles, dsvd).
const GAMMA_A0: f32 = 1.0;

/// One edge as lra uses it: endpoints as data columns, link community, batch.
pub(super) type LraEdge = (usize, usize, u32, Option<Box<str>>);

/// Read a prior `pinto lc` run's pairs: every pair (`anchor`), and the
/// physically adjacent subset that is tested (`edges`).
pub(super) fn load_edges(
    lc_prefix: &str,
    cell_to_col: &HashMap<Box<str>, usize>,
) -> anyhow::Result<(Vec<LraEdge>, Vec<LraEdge>)> {
    let lc_edges_path = format!("{}.link_community.parquet", lc_prefix);
    let coord_pairs_path = format!("{}.coord_pairs.parquet", lc_prefix);
    info!("Reading edge assignments from {}", lc_edges_path);
    let mut edge_records = read_link_community(&lc_edges_path)?;
    info!("Attaching per-edge batch from {}", coord_pairs_path);
    attach_batch_from_coord_pairs(&mut edge_records, &coord_pairs_path)?;

    // Two lists with different jobs. `anchor_edges` is everything, and fixes
    // which community each cell belongs to. `edges` is the physically adjacent
    // subset, and is what actually gets tested: a directional test on a pair
    // that is merely expression-similar has no estimand, since the two cells
    // never touch. See `orientation` for why the anchor may be wider.
    let mut anchor_edges: Vec<(usize, usize, u32, Option<Box<str>>)> =
        Vec::with_capacity(edge_records.len());
    let mut edges: Vec<(usize, usize, u32, Option<Box<str>>)> =
        Vec::with_capacity(edge_records.len());
    let mut unresolved = 0usize;
    for e in edge_records {
        match (
            cell_to_col.get(&e.left_cell).copied(),
            cell_to_col.get(&e.right_cell).copied(),
        ) {
            (Some(i), Some(j)) => {
                let edge = (i, j, e.community, e.batch);
                if e.is_spatial {
                    edges.push(edge.clone());
                }
                anchor_edges.push(edge);
            }
            _ => unresolved += 1,
        }
    }
    let n_expression = anchor_edges.len() - edges.len();
    if n_expression > 0 {
        info!(
            "{} of {} pairs are expression-similar rather than adjacent: they set \
             each cell's community but are not themselves tested",
            n_expression,
            anchor_edges.len()
        );
    }
    if unresolved > 0 {
        info!(
            "Dropped {} edges whose cell names are not in the expression data",
            unresolved
        );
    }
    anyhow::ensure!(
        !edges.is_empty(),
        "no edges resolved against expression data"
    );
    Ok((anchor_edges, edges))
}

pub(crate) fn load_expr_data(
    data_files: &[Box<str>],
    preload: bool,
) -> anyhow::Result<SparseIoVec> {
    anyhow::ensure!(!data_files.is_empty(), "empty data files");
    let mut data_vec = SparseIoVec::new();
    let names = crate::util::input::stacked_data_names(data_files)?;
    for (data_file, data_name) in data_files.iter().zip(names) {
        info!("Importing data file: {}", data_file);
        let data = crate::util::input::open_sparse(data_file, preload)?;
        data_vec.push(Arc::from(data), data_name)?;
    }
    Ok(data_vec)
}

/// Modal batch among edges incident to each cell. Cells with no batched
/// edges fall back to `BATCH_LABEL_ALL` (also the default for runs where
/// `coord_pairs.parquet` carried no batch columns).
pub(crate) fn derive_cell_batch_labels(
    edges: &[(usize, usize, u32, Option<Box<str>>)],
    n_cells: usize,
) -> Vec<Box<str>> {
    let mut counts: Vec<HashMap<Box<str>, usize>> =
        (0..n_cells).map(|_| HashMap::default()).collect();
    for (i, j, _k, b_opt) in edges {
        if let Some(b) = b_opt {
            *counts[*i].entry(b.clone()).or_insert(0) += 1;
            *counts[*j].entry(b.clone()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .map(|m| {
            m.into_iter()
                .max_by_key(|(_, c)| *c)
                .map(|(b, _)| b)
                .unwrap_or_else(|| BATCH_LABEL_ALL.into())
        })
        .collect()
}

/// Group cells into pseudobulk samples = unique (batch, propensity-bin)
/// combinations. Returns (sample_id_per_cell, per-sample batch label,
/// per-sample propensity-bin).
pub(super) fn assign_samples(
    cell_batch: &[Box<str>],
    propbin: &[usize],
) -> (Vec<usize>, Vec<Box<str>>, Vec<usize>) {
    let mut key_to_id: HashMap<(Box<str>, usize), usize> = HashMap::default();
    let mut sample_batch_label: Vec<Box<str>> = Vec::new();
    let mut sample_propbin: Vec<usize> = Vec::new();
    let mut sample_id_per_cell: Vec<usize> = Vec::with_capacity(cell_batch.len());
    for i in 0..cell_batch.len() {
        let key = (cell_batch[i].clone(), propbin[i]);
        let id = if let Some(&id) = key_to_id.get(&key) {
            id
        } else {
            let id = sample_batch_label.len();
            sample_batch_label.push(key.0.clone());
            sample_propbin.push(key.1);
            key_to_id.insert(key, id);
            id
        };
        sample_id_per_cell.push(id);
    }
    (sample_id_per_cell, sample_batch_label, sample_propbin)
}

/// Per-(community, sample) role-weighted pseudobulk for the LR genes,
/// Per-stratum pseudobulk of the LR genes over (batch, propensity-bin)
/// samples, weighted by each cell's soft membership in the stratum's link
/// community, stored as Gamma posterior log-means: `log_mean[c][(g, s)] =
/// E[log λ | data] = ψ(num + a0) - log(denom + b0)`. Using the variational
/// log-mean instead of `log1p(num/(denom+1))` breaks the zero-count tie
/// pile-up: a sample with `num = 0` and small `denom` lands far above one
/// with `num = 0` and large `denom` (correctly less confident in λ ≈ 0).
///
/// One plane per stratum: with both orientations of every within-community
/// edge enumerated, the old sender/receiver roles coincide exactly.
pub(super) struct CollapseOut {
    pub(super) log_mean: Vec<DMatrix<f32>>,
    pub(super) denom: Vec<Vec<f32>>,
    /// Calibrated Gamma posteriors per community used by the per-permutation
    /// log-posterior draw inside `score_pairs_for_stratum`. Fisher weights
    /// are NOT pre-applied here.
    pub(super) gamma: Vec<GammaMatrix>,
}

pub(super) fn collapse_pseudobulk(
    x_lr: &DMatrix<f32>,
    sample_id: &[usize],
    p_member: &DMatrix<f32>,
    n_strata: usize,
    n_samples: usize,
    active: &HashSet<u32>,
) -> CollapseOut {
    let n_lr = x_lr.nrows();
    let n_cells = x_lr.ncols();

    // Communities are independent — accumulate each one in its own thread.
    type StratumCollapse = (DMatrix<f32>, Vec<f32>, GammaMatrix);
    let per_stratum: Vec<StratumCollapse> = (0..n_strata)
        .into_par_iter()
        .map(|c| {
            // A stratum the sparsity filter already rejected is never scored;
            // building its entry anyway would retain dense planes and a full
            // digamma/trigamma calibration for the whole run. Return an empty
            // entry and keep the vectors indexed by stratum id.
            if !active.contains(&(c as u32)) {
                return (
                    DMatrix::zeros(0, 0),
                    Vec::new(),
                    GammaMatrix::new((0, 0), GAMMA_A0, GAMMA_B0),
                );
            }
            let mut num = DMatrix::<f32>::zeros(n_lr, n_samples);
            let mut den = vec![0.0f32; n_samples];
            for i in 0..n_cells {
                let s = sample_id[i];
                let pm = p_member[(i, c)];
                if pm > 0.0 {
                    num.column_mut(s).axpy(pm, &x_lr.column(i), 1.0);
                    den[s] += pm;
                }
            }
            let ones_col = nalgebra::DVector::<f32>::from_element(n_lr, 1.0);
            let den_mat = &ones_col
                * nalgebra::RowDVector::<f32>::from_iterator(n_samples, den.iter().copied());

            let mut g = GammaMatrix::new((n_lr, n_samples), GAMMA_A0, GAMMA_B0);
            g.update_stat(&num, &den_mat);
            g.calibrate_with(CalibrateTarget::All);
            let log_mean = g.posterior_log_mean().clone();
            (log_mean, den, g)
        })
        .collect();

    let mut log_mean = Vec::with_capacity(n_strata);
    let mut denom = Vec::with_capacity(n_strata);
    let mut gamma = Vec::with_capacity(n_strata);
    for (lm, d, g) in per_stratum {
        log_mean.push(lm);
        denom.push(d);
        gamma.push(g);
    }
    CollapseOut {
        log_mean,
        denom,
        gamma,
    }
}

pub(super) fn stratum_present(collapse: &CollapseOut, c: usize, samples: &[usize]) -> bool {
    samples.iter().any(|&s| collapse.denom[c][s] > 0.0)
}
