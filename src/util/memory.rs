//! The memory budget a command plans its peak against: `PINTO_MEMORY_BUDGET`
//! bytes when set, else half of what this process could still allocate
//! (cgroup-aware). `lra` sizes the pairs it runs at once by it; the spatial
//! baselines refuse a run whose estimated peak exceeds it.

/// Memory allowed when the machine does not say what it has, and the least
/// allowed when it does.
const MIN_MEMORY_BUDGET: usize = 2 << 30;

/// The variable that sets the memory budget, in bytes: a runner starting
/// several pinto processes at once gives each its share, which half of what
/// is free, read by each process alone, would overcommit.
pub const MEMORY_BUDGET_VAR: &str = "PINTO_MEMORY_BUDGET";

/// The memory budget: `set` bytes ([`MEMORY_BUDGET_VAR`]) when given, else
/// [`half_available`]. A `set` that is not a whole number of bytes is an
/// error.
pub fn memory_budget(set: Option<&str>, available: Option<usize>) -> anyhow::Result<usize> {
    set.map_or(Ok(half_available(available)), |s| {
        s.trim()
            .parse()
            .ok()
            .filter(|&b: &usize| b > 0)
            .ok_or_else(|| {
                anyhow::anyhow!("{MEMORY_BUDGET_VAR}={s:?}: expected a positive number of bytes")
            })
    })
}

/// Half of the `available` memory, the rest left to the data and the
/// machine's other work, and never below [`MIN_MEMORY_BUDGET`].
pub fn half_available(available: Option<usize>) -> usize {
    available.map_or(MIN_MEMORY_BUDGET, |a| (a / 2).max(MIN_MEMORY_BUDGET))
}

/// Refuse a [`MEMORY_BUDGET_VAR`] that does not parse, before any work.
pub fn check_memory_budget() -> anyhow::Result<()> {
    budget_var().map(|_| ())
}

/// [`MEMORY_BUDGET_VAR`] as set, refused when it does not parse.
pub fn budget_var() -> anyhow::Result<Option<String>> {
    let set = std::env::var(MEMORY_BUDGET_VAR).ok();
    memory_budget(set.as_deref(), None)?;
    Ok(set)
}

/// Bytes this process could still allocate: the system's available memory, or
/// less when a cgroup (a batch job, a container) limits it. `None` where the
/// system does not say.
pub fn available_memory() -> Option<usize> {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let available = system.available_memory();
    let left = system
        .cgroup_limits()
        .map_or(available, |group| group.free_memory.min(available));
    (left > 0).then_some(left as usize)
}

/// `bytes` in GiB, or MiB below one GiB, for messages.
pub fn human_bytes(bytes: usize) -> String {
    let mib = bytes as f64 / f64::from(1u32 << 20);
    if mib < 1024.0 {
        format!("{mib:.1} MiB")
    } else {
        format!("{:.2} GiB", mib / 1024.0)
    }
}

/// Log `what`'s estimated peak of `estimate` bytes, and refuse it when it
/// exceeds the budget: `set` ([`MEMORY_BUDGET_VAR`], as read at the start)
/// or, unset, half of what is available now. `change` names what would
/// lower the estimate.
pub fn ensure_fits(
    what: &str,
    estimate: usize,
    set: Option<&str>,
    change: &str,
) -> anyhow::Result<()> {
    let budget = memory_budget(set, set.is_none().then(available_memory).flatten())?;
    let source = if set.is_some() {
        MEMORY_BUDGET_VAR.to_string()
    } else {
        format!("half the available memory; {MEMORY_BUDGET_VAR} sets it")
    };
    log::info!(
        "{what}: estimated peak memory {}, budget {} ({source})",
        human_bytes(estimate),
        human_bytes(budget)
    );
    anyhow::ensure!(
        estimate <= budget,
        "{what} would need an estimated {} of memory at its peak, over the budget of {} \
         ({source}); {change}, or raise {MEMORY_BUDGET_VAR} (bytes) knowingly",
        human_bytes(estimate),
        human_bytes(budget)
    );
    Ok(())
}
