//! Across pairs: each row's p-values, co-expression's local fdr, and why a
//! row has no estimate. Nothing is judged: the statistics are reported and
//! the reader decides.
//!
//! The causal statistics (the competent effect, the effect in
//! receptor-negative receivers, their contrast) are reported as estimate,
//! SE, df and two-sided p against the theoretical null: the effect's and
//! the contrast's SE from the tile jackknife, the receptor-negative
//! effect's the clustered one, which no jackknife widens. Co-expression, a
//! baseline that spatial autocorrelation shifts and widens alike in every
//! pair, also gets Efron's local fdr within its section, under the null
//! fitted to the bulk of the section's pairs and under `N(0, 1)`.

use super::inference::{upper_normal_p, ONE_CLUSTER_DF};
use super::options::thin_arm;
use super::rows::EffectRow;
use crate::util::local_fdr::{two_groups, NormalNull};

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
        let (coexpr_lfdr, coexpr_lfdr_theoretical_null) =
            coexpression_local_fdr(section, &coexpr_z);
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
            r.coexpr_lfdr_theoretical_null = coexpr_lfdr_theoretical_null[k];
            r.coexpr_p = upper_normal_p(coexpr_z[k]);
            r.effect_dr_p = r.effects.competent.p();
            r.warnings = warnings(r, min_arm);
        }
    }
}

/// Co-expression's local fdr over one section's scores `z` (one-sided,
/// positive), under the null fitted to their bulk and under `N(0, 1)`; NaN
/// in a section too small to fit, and under the fitted null where the bulk
/// has no spread.
fn coexpression_local_fdr(section: &str, z: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let finite = z.iter().filter(|x| x.is_finite()).count();
    if finite < MIN_STATISTICS_FOR_LOCAL_FDR {
        log::info!(
            "co-expression, section {section}: {finite} scores, too few for a local fdr \
             ({MIN_STATISTICS_FOR_LOCAL_FDR} needed)"
        );
        return (vec![f64::NAN; z.len()], vec![f64::NAN; z.len()]);
    }
    let theoretical = two_groups(z, NormalNull::THEORETICAL);
    let empirical = match NormalNull::empirical(z) {
        Some(null) => {
            let fit = two_groups(z, null);
            log::info!(
                "co-expression, section {section}: null N({:.2}, {:.2}²), null share {:.2}, \
                 over {finite} scores",
                null.mean,
                null.sd,
                fit.null_share
            );
            fit.lfdr
        }
        None => {
            log::info!("co-expression, section {section}: no spread in the bulk, no fitted null");
            vec![f64::NAN; z.len()]
        }
    };
    (empirical, theoretical.lfdr)
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
