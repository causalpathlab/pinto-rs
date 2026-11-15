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

/// A [`MEMORY_BUDGET_VAR`] value `s` as a positive number of bytes.
pub fn parse_budget(s: &str) -> anyhow::Result<usize> {
    s.trim()
        .parse()
        .ok()
        .filter(|&b: &usize| b > 0)
        .ok_or_else(|| {
            anyhow::anyhow!("{MEMORY_BUDGET_VAR}={s:?}: expected a positive number of bytes")
        })
}

/// Half of the `available` memory, the rest left to the data and the
/// machine's other work, and never below [`MIN_MEMORY_BUDGET`].
pub fn half_available(available: Option<usize>) -> usize {
    available.map_or(MIN_MEMORY_BUDGET, |a| (a / 2).max(MIN_MEMORY_BUDGET))
}

/// The bytes [`MEMORY_BUDGET_VAR`] sets, refused when it does not parse (a
/// value that is not UTF-8 included: it is not taken as unset).
pub fn budget_var() -> anyhow::Result<Option<usize>> {
    let Some(raw) = std::env::var_os(MEMORY_BUDGET_VAR) else {
        return Ok(None);
    };
    let set = raw.into_string().map_err(|raw| {
        anyhow::anyhow!("{MEMORY_BUDGET_VAR}={raw:?}: expected a positive number of bytes")
    })?;
    parse_budget(&set).map(Some)
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
    /// The budget in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// `set` bytes ([`MEMORY_BUDGET_VAR`]) when given, else half the
    /// `available` memory ([`half_available`]).
    pub fn new(set: Option<usize>, available: Option<usize>) -> Self {
        Budget {
            bytes: set.unwrap_or_else(|| half_available(available)),
            set: set.is_some(),
        }
    }

    /// [`MEMORY_BUDGET_VAR`] when set, else half of what is available now:
    /// taken before the data are read, so what they take is not counted
    /// again beside an estimate that counts them.
    pub fn now() -> anyhow::Result<Self> {
        Ok(Self::measured(budget_var()?))
    }

    /// `set` bytes when given, else half of what is available now
    /// ([`available_memory`]), read only when `set` is not given.
    pub fn measured(set: Option<usize>) -> Self {
        Self::new(set, set.is_none().then(available_memory).flatten())
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
