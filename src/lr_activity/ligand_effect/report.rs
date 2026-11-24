//! Across pairs: each row's p-values, co-expression's local fdr, and why a
//! row has no estimate. Nothing is judged: the statistics are reported and
//! the reader decides.
//!
//! The causal statistics (the competent effect, the effect in
//! receptor-negative receivers, their contrast) are reported as estimate,
//! SE, df and two-sided p: their SE is already calibrated by the tile
//! jackknife, and their null is the theoretical one. Co-expression, a
//! baseline that spatial autocorrelation shifts and widens alike in every
//! pair, also gets Efron's local fdr within its section, under the null
//! fitted to the bulk of the section's pairs and under `N(0, 1)`.

use super::inference::{upper_normal_p, ONE_CLUSTER_DF};
use super::options::thin_arm;
use super::rows::{EffectRow, LocalFdr};
use crate::util::local_fdr::{two_groups, NormalNull, SignalSide};

/// Fewest finite co-expression scores in a section for their local fdr: the
/// two-group model's null and mixture density are fitted to the section
/// itself, and a handful of scores fit neither. A smaller section (a short
/// `--only` list, a small community) reports co-expression's p alone.
pub const MIN_STATISTICS_FOR_LOCAL_FDR: usize = 100;

/// p-values, co-expression's local fdr and warnings for every row, within
/// each section (the pooled rows, `all`, form their own family). `min_arm`
/// is [`super::options::EffectOpts::min_arm`], to say why a pair is not
/// estimable.
pub fn report(rows: &mut [EffectRow], min_arm: usize) {
    let mut by_section: std::collections::HashMap<Box<str>, Vec<usize>> = Default::default();
    for (i, r) in rows.iter().enumerate() {
        by_section.entry(r.section.clone()).or_default().push(i);
    }
    for (section, idx) in &by_section {
        let coexpr_z: Vec<f64> = idx
            .iter()
            .map(|&i| rows[i].coexpr_log_or / rows[i].coexpr_se)
            .collect();
        let coexpr_lfdr = coexpression_local_fdr(section, &coexpr_z);
        let ligands: Vec<String> = idx
            .iter()
            .map(|&i| rows[i].ligand.to_ascii_uppercase())
            .collect();
        let mut pairs_of_ligand: std::collections::HashMap<&str, usize> = Default::default();
        for l in &ligands {
            *pairs_of_ligand.entry(l).or_default() += 1;
        }
        for (k, &i) in idx.iter().enumerate() {
            let r = &mut rows[i];
            r.n_pairs_same_ligand = pairs_of_ligand[ligands[k].as_str()];
            r.coexpr_lfdr = coexpr_lfdr[k];
            report_row(r, min_arm);
        }
    }
}

/// One row's own p-values and warnings, needing no other row: what a run of
/// `lra simulate` reports.
pub fn report_row(row: &mut EffectRow, min_arm: usize) {
    row.effect_dr_p = row.effects.competent.p();
    row.coexpr_p = upper_normal_p(row.coexpr_log_or / row.coexpr_se);
    row.warnings = warnings(row, min_arm);
}

/// Co-expression's local fdr over one section's scores `z` (one-sided,
/// positive), under the null fitted to their bulk and under `N(0, 1)`; NaN
/// in a section too small to fit.
fn coexpression_local_fdr(section: &str, z: &[f64]) -> Vec<LocalFdr> {
    let finite = z.iter().filter(|x| !x.is_nan()).count();
    let null = (finite >= MIN_STATISTICS_FOR_LOCAL_FDR)
        .then(|| NormalNull::empirical(z))
        .flatten();
    let Some(null) = null else {
        log::info!(
            "co-expression, section {section}: {finite} scores, too few for a local fdr \
             ({MIN_STATISTICS_FOR_LOCAL_FDR} needed)"
        );
        return vec![LocalFdr::NAN; z.len()];
    };
    let empirical = two_groups(z, null, SignalSide::Above);
    let theoretical = two_groups(z, NormalNull::THEORETICAL, SignalSide::Above);
    log::info!(
        "co-expression, section {section}: null N({:.2}, {:.2}²), null share {:.2}, over \
         {finite} scores",
        null.mean,
        null.sd,
        empirical.null_share
    );
    empirical
        .lfdr
        .iter()
        .zip(&theoretical.lfdr)
        .map(|(&e, &t)| LocalFdr {
            empirical_null: e,
            theoretical_null: t,
        })
        .collect()
}

/// Why the row's competent effect has no estimate.
fn not_estimable_reason(row: &EffectRow, min_arm: usize) -> String {
    let head = row.effects.competent;
    if row.n_competent == 0 {
        "not estimable: no receptor-competent receivers".into()
    } else if thin_arm(min_arm, head.n, head.n_treated) {
        format!(
            "not estimable: thin arm ({} exposed, {} unexposed competent receivers; \
             internal guard min_arm {min_arm}, see outputs.lra.settings in the manifest)",
            head.n_treated,
            head.n.saturating_sub(head.n_treated),
        )
    } else if head.df == ONE_CLUSTER_DF {
        "not estimable: the competent receivers lie in one variance cluster (spatial \
         tile), so the standard error has no degrees of freedom"
            .into()
    } else {
        "not estimable: low overlap (mean overlap weight or its effective sample size \
         under the internal guards min_overlap / min_ess, see outputs.lra.settings in the \
         manifest)"
            .into()
    }
}

/// The row's warnings: why it has no estimate, or that its SE could not be
/// widened by the tile jackknife.
fn warnings(row: &EffectRow, min_arm: usize) -> String {
    let head = row.effects.competent;
    if !(head.theta.is_finite() && head.se.is_finite()) {
        not_estimable_reason(row, min_arm)
    } else if row.jackknife.failed() {
        "the tile jackknife could not run (deleting a tile leaves an arm thin): its SE is \
         the clustered one, likely too small for a region in few tiles"
            .into()
    } else {
        String::new()
    }
}
