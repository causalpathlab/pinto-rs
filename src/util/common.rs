pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;

pub use data_beans::sparse_io::*;
pub use data_beans::sparse_io_vector::*;

pub use indicatif::ParallelProgressIterator;
pub use legume_numeric::matrix::common_io::{basename, file_ext, read_lines};
pub use legume_numeric::matrix::dmatrix_util::*;
pub use legume_numeric::matrix::traits::*;
pub use log::{info, warn};
pub use rayon::prelude::*;

pub use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

pub use std::sync::{Arc, Mutex};

/// `name` under the folder `dir`, joined as a path: one `/` between them
/// whether or not `dir` ends in one.
pub fn in_dir(dir: &str, name: &str) -> String {
    std::path::Path::new(dir)
        .join(name)
        .to_string_lossy()
        .into_owned()
}

// Logging and progress bars come from `data_beans::aux::logging`, which draws
// through the workspace-wide `legume_numeric::matrix::progress::MULTI_PROGRESS`. A second
// `MultiProgress` here would leave its bars unbridged from the log output — see
// the `legume_numeric::matrix::progress` module docs. Bar labels are set with
// `new_progress_bar(n).with_message(..)`; the template belongs to the helper.
pub use data_beans::aux::logging::{init_logger, new_progress_bar};

// NB Fisher-info feature weighting helpers moved to
// `data_beans::alg::gene_weighting` so senna / chickpea / pinto share one
// implementation. Re-exported here for backwards compatibility within pinto.
pub use data_beans::alg::gene_weighting::{
    apply_gene_weights as apply_feature_weights, compute_nb_fisher_weights,
};

/// A clap value parser for a finite number, 0 or more.
pub fn non_negative_f64(text: &str) -> Result<f64, String> {
    let v: f64 = text.parse().map_err(|e| format!("{e}"))?;
    if v.is_finite() && v >= 0.0 {
        Ok(v)
    } else {
        Err("must be finite and not negative".to_string())
    }
}

/// Most worker threads a pinto command takes by default: a shared machine
/// is not a command's to fill, and past this the fits gain little.
pub const MAX_DEFAULT_THREADS: usize = 16;

/// The worker threads to start with: `min(MAX_DEFAULT_THREADS, cores)`,
/// unless `RAYON_NUM_THREADS` (`rayon_env`) is set, when rayon's own choice
/// stands (`None`).
pub fn default_threads(rayon_env: Option<&str>, cores: usize) -> Option<usize> {
    match rayon_env {
        Some(v) if !v.trim().is_empty() => None,
        _ => Some(cores.clamp(1, MAX_DEFAULT_THREADS)),
    }
}

/// The `q` quantile of `sorted` (ascending), the element at
/// `round(q (len − 1))`; `None` when empty.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> Option<f64> {
    let last = sorted.len().checked_sub(1)?;
    sorted.get((q * last as f64).round() as usize).copied()
}

/// The element at `len / 2` of `v` sorted (the upper median), found by
/// selection; `None` when empty. Reorders `v`.
pub fn upper_median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mid = v.len() / 2;
    Some(*v.select_nth_unstable_by(mid, f64::total_cmp).1)
}

/// What log-normalisation scales each unit's counts to before `ln(1 + x)`
/// (counts per ten thousand); a unit with no counts stays empty.
pub const LIBRARY_SCALE: f32 = 1e4;

/// The jackknife SE of replicates `v`, `sqrt((G − 1)/G Σ (v_g − v̄)²)`; NaN
/// with a non-finite replicate or fewer than two.
pub fn jackknife_se(v: &[f64]) -> f64 {
    let g = v.len() as f64;
    if v.len() < 2 || v.iter().any(|x| !x.is_finite()) {
        return f64::NAN;
    }
    let mean = v.iter().sum::<f64>() / g;
    ((g - 1.0) / g * v.iter().map(|x| (x - mean).powi(2)).sum::<f64>()).sqrt()
}
