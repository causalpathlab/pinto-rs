//! Wall time per stage summed over calls (and threads), and counts of fits and
//! skipped pairs, for the run log.

use std::sync::atomic::{AtomicU64, Ordering};

const STAGES: [&str; 8] = [
    "frame",
    "main fits",
    "no-spill fit",
    "step fits",
    "other thresholds",
    "band",
    "(in fits: outcome ridge",
    "propensity)",
];
pub const RIDGE: usize = 6;
pub const LOGISTIC: usize = 7;
pub const FRAME: usize = 0;
pub const MAIN: usize = 1;
pub const NOSPILL: usize = 2;
pub const STEPS: usize = 3;
pub const THRESHOLDS: usize = 4;
pub const BAND: usize = 5;

static NS: [AtomicU64; STAGES.len()] = [const { AtomicU64::new(0) }; STAGES.len()];
static ALL_SKIPPED: AtomicU64 = AtomicU64::new(0);
static PROPENSITY_FITS: AtomicU64 = AtomicU64::new(0);
static HESSIANS: AtomicU64 = AtomicU64::new(0);
static UNCONVERGED: AtomicU64 = AtomicU64::new(0);

/// One logistic fit that computed `hessians` Hessians; `converged`
/// false when it stopped at the step guard, not at the optimum.
pub fn newton(hessians: usize, converged: bool) {
    PROPENSITY_FITS.fetch_add(1, Ordering::Relaxed);
    HESSIANS.fetch_add(hessians as u64, Ordering::Relaxed);
    if !converged {
        UNCONVERGED.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn time<T>(stage: usize, f: impl FnOnce() -> T) -> T {
    let t = std::time::Instant::now();
    let out = f();
    NS[stage].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    out
}

pub fn add(stage: usize, since: std::time::Instant) {
    NS[stage].fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// One pair none of whose fits ran: every subset left an arm thin.
pub fn all_skipped() {
    ALL_SKIPPED.fetch_add(1, Ordering::Relaxed);
}

/// Thread-seconds per stage and the pairs
/// with every fit skipped.
pub fn summary() -> String {
    let stages: Vec<String> = STAGES
        .iter()
        .zip(&NS)
        .map(|(s, n)| format!("{s} {:.0}s", n.load(Ordering::Relaxed) as f64 * 1e-9))
        .collect();
    let fits = PROPENSITY_FITS.load(Ordering::Relaxed);
    format!(
        "{}; {} pairs with every fit skipped (thin \
         arms); {} propensity fold fits, {:.1} Hessians each, {} not converged",
        stages.join(", "),
        ALL_SKIPPED.load(Ordering::Relaxed),
        fits,
        HESSIANS.load(Ordering::Relaxed) as f64 / fits.max(1) as f64,
        UNCONVERGED.load(Ordering::Relaxed)
    )
}
