//! Where the tables are written.

/// `{out}.lra.parquet`, the contact-effect table.
pub fn lra_path(out: &str) -> String {
    format!("{out}.lra.parquet")
}

/// `{out}.lra.curve.parquet`, each pair's threshold-response curve.
pub fn lra_curve_path(out: &str) -> String {
    curve_beside(&lra_path(out))
}

/// The curves written beside the contact-effect table `table`, the full
/// one or a subset's: `{table without .parquet}.curve.parquet`.
pub fn curve_beside(table: &str) -> String {
    format!(
        "{}.curve.parquet",
        table.strip_suffix(".parquet").unwrap_or(table)
    )
}

/// `{out}.lra-simulate.parquet`, the power and calibration check.
pub fn lra_simulate_path(out: &str) -> String {
    format!("{out}.lra-simulate.parquet")
}
