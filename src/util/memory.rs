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

/// [`MEMORY_BUDGET_VAR`] as set, refused when it does not parse (a value
/// that is not UTF-8 included: it is not taken as unset).
pub fn budget_var() -> anyhow::Result<Option<String>> {
    let Some(raw) = std::env::var_os(MEMORY_BUDGET_VAR) else {
        return Ok(None);
    };
    let set = raw.into_string().map_err(|raw| {
        anyhow::anyhow!("{MEMORY_BUDGET_VAR}={raw:?}: expected a positive number of bytes")
    })?;
    memory_budget(Some(&set), None)?;
    Ok(Some(set))
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

/// The budget a run plans its peak against, taken once at its start.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    bytes: usize,
    /// Whether [`MEMORY_BUDGET_VAR`] set it.
    set: bool,
}

impl Budget {
    /// `set` bytes ([`MEMORY_BUDGET_VAR`]) when given, else half the
    /// `available` memory ([`memory_budget`]).
    pub fn new(set: Option<&str>, available: Option<usize>) -> anyhow::Result<Self> {
        Ok(Budget {
            bytes: memory_budget(set, available)?,
            set: set.is_some(),
        })
    }

    /// [`MEMORY_BUDGET_VAR`] when set, else half of what is available now:
    /// taken before the data are read, so what they take is not counted
    /// again beside an estimate that counts them.
    pub fn now() -> anyhow::Result<Self> {
        let set = budget_var()?;
        Self::new(
            set.as_deref(),
            set.is_none().then(available_memory).flatten(),
        )
    }
}

impl std::fmt::Display for Budget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = human_bytes(self.bytes);
        if self.set {
            write!(f, "{bytes} ({MEMORY_BUDGET_VAR})")
        } else {
            write!(
                f,
                "{bytes} (half the available memory; {MEMORY_BUDGET_VAR} sets it)"
            )
        }
    }
}

/// Log `what`'s estimated peak of `estimate` bytes, and refuse it when it
/// exceeds `budget`. `change` names what would lower the estimate.
pub fn ensure_fits(
    what: &str,
    estimate: usize,
    budget: &Budget,
    change: &str,
) -> anyhow::Result<()> {
    log::info!(
        "{what}: estimated peak memory {}, budget {budget}",
        human_bytes(estimate)
    );
    anyhow::ensure!(
        estimate <= budget.bytes,
        "{what} would need an estimated {} of memory at its peak, over the budget of \
         {budget}; {change}, or raise {MEMORY_BUDGET_VAR} (bytes) knowingly",
        human_bytes(estimate)
    );
    Ok(())
}
