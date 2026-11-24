//! Where the tables are written.

/// The curves written beside the contact-effect table `table`, the full
/// one or a subset's: `{table without .parquet}.curve.parquet`.
pub fn curve_beside(table: &str) -> String {
    beside(table, "curve")
}

/// The per-type-pair table (`--marginal-baselines`) beside the
/// contact-effect table `table`: `{table without .parquet}.celltype.parquet`.
pub fn celltype_beside(table: &str) -> String {
    beside(table, "celltype")
}

/// `{table without .parquet}.{what}.parquet`.
fn beside(table: &str, what: &str) -> String {
    format!(
        "{}.{what}.parquet",
        table.strip_suffix(".parquet").unwrap_or(table)
    )
}

/// `{out}.lra-simulate.parquet`, the power and calibration check.
pub fn lra_simulate_path(out: &str) -> String {
    format!("{out}.lra-simulate.parquet")
}

/// `{out}.lra-simulate.json`, a single-scenario call's sidecar: written last,
/// so its presence marks the call complete.
pub fn lra_simulate_sidecar_path(out: &str) -> String {
    format!("{out}{SIDECAR_SUFFIX}")
}

/// `{out}.lra.merge.json`: the run key of the shards `pinto lra merge` put
/// together, beside its tables.
pub fn merge_record_path(out: &str) -> String {
    format!("{out}.lra.merge.json")
}

/// What a single-scenario call's sidecar is named by, after its prefix.
pub const SIDECAR_SUFFIX: &str = ".lra-simulate.json";
