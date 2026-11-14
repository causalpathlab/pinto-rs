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
