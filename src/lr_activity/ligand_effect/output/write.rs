//! Writing the `lra` and `lra-simulate` tables.

use super::columns::{f32s, Columns};
use crate::lr_activity::ligand_effect::baselines::Baselines;
use crate::lr_activity::ligand_effect::rows::{CurvePoint, EffectRow, Moments};
use crate::lr_activity::ligand_effect::simulate::{SimRow, COMPARED};
use legume_numeric::matrix::parquet::{write_named_table, Column};

/// The `q_scope` of a table of every pair.
pub const Q_SCOPE_ALL: &str = "all pairs";

/// `{out}.lra.parquet`: one row per pair and section (`all` for the pooled
/// row). `verdict` is a [`super::super::verdict::Verdict`] code, `verdict_note`
/// its words; the headline's [`super::super::rows::Moments`] are computed here.
/// `q_scope` says which pairs the q-values were corrected over:
/// [`Q_SCOPE_ALL`], or a `--only` subset.
pub fn write_effects(path: &str, rows: &[EffectRow], q_scope: &str) -> anyhow::Result<()> {
    let moments: Vec<(&EffectRow, Moments)> = rows
        .iter()
        .map(|r| (r, Moments::of(&r.effects.competent)))
        .collect();
    let mut c = Columns::new(&moments);
    c.s("section", |(r, _)| &r.section)
        .s("ligand", |(r, _)| &r.ligand)
        .s("receptor", |(r, _)| &r.receptor)
        .i("n_targets", |(r, _)| r.n_targets)
        .i("n_targets_removed", |(r, _)| r.n_removed)
        .i("n_receivers", |(r, _)| r.n_receivers)
        .i("n_competent", |(r, _)| r.n_competent)
        .i("n_exposed", |(r, _)| r.n_exposed)
        .f("exposure_threshold", |(r, _)| r.threshold)
        .f("coexpr_log_or", |(r, _)| r.coexpr_log_or)
        .f("coexpr_se", |(r, _)| r.coexpr_se)
        .est("marginal_cor", |(r, _)| r.marginal_cor)
        .est("unadjusted_diff", |(r, _)| r.effects.unadjusted_diff)
        .est_interval("effect_ipw", |(r, _)| r.effects.ipw)
        .f("effect_outcome", |(r, _)| r.effects.outcome.theta)
        .est("effect_competent", |(r, _)| r.effects.competent)
        .f("effect_competent_df", |(r, _)| r.effects.competent.df)
        .f("effect_p", |(r, _)| r.effect_p)
        .f("effect_q", |(r, _)| r.effect_q)
        .f("coexpr_p", |(r, _)| r.coexpr_p)
        .f("coexpr_q", |(r, _)| r.coexpr_q)
        .i("n_pairs_same_ligand", |(r, _)| r.n_pairs_same_ligand)
        .f("ligand_q", |(r, _)| r.ligand_q)
        .est("effect_competent_section", |(r, _)| r.effects.steps[0])
        .est("effect_competent_spatial", |(r, _)| r.effects.steps[1])
        .est("effect_competent_identity", |(r, _)| r.effects.steps[2])
        .est("effect_competent_nospill", |(r, _)| r.effects.nospill)
        .est("effect_incompetent", |(r, _)| r.effects.incompetent)
        .f("effect_incompetent_q", |(r, _)| r.effect_incompetent_q)
        .est("effect_contrast", |(r, _)| r.effects.effect)
        .f("effect_log10_bf", |(r, _)| r.effects.abf.log10_bf)
        .f("tau_top", |(r, _)| r.effects.abf.top_tau)
        .f("tau_top_weight", |(r, _)| r.effects.abf.top_weight)
        .f("effect_contrast_q", |(r, _)| r.effect_contrast_q)
        .f("mean_overlap", |(r, _)| r.effects.competent.mean_h)
        .f("overlap_ess", |(r, _)| r.effects.competent.ess)
        .f("near_determined_share", |(r, _)| r.effects.near_determined)
        .f("exposure_eta2", |(r, _)| r.exposure_eta2)
        .i("n_competent_bins", |(r, _)| r.n_competent_bins)
        .i("n_incompetent_bins", |(r, _)| r.n_incompetent_bins)
        .s("verdict", |(r, _)| r.verdict.code())
        .s("verdict_note", |(r, _)| r.verdict.note())
        .s("warnings", |(r, _)| &r.warnings)
        .f("effect_lo", |(_, m)| m.lo)
        .f("effect_hi", |(_, m)| m.hi)
        .f("effect_pct", |(_, m)| m.pct)
        .f("effect_pct_sd", |(_, m)| m.pct_sd)
        .f("effect_pct_lo", |(_, m)| m.pct_lo)
        .f("effect_pct_hi", |(_, m)| m.pct_hi)
        .s("q_scope", |_| q_scope);
    c.write(path)
}

/// `{out}.lra.curve.parquet`: the threshold–response curve of each pair's
/// pooled row, one row per pair and threshold: the competent effect at exposure threshold
/// `tau` (`quantile` of the positive exposures), its SE and df, the
/// pointwise 95% interval and the simultaneous 95% band over the pair's
/// thresholds.
pub fn write_curve(path: &str, rows: &[EffectRow]) -> anyhow::Result<()> {
    let pts: Vec<(&EffectRow, &CurvePoint)> = rows
        .iter()
        .flat_map(|r| r.curve.iter().map(move |p| (r, p)))
        .collect();
    let s = |f: fn(&EffectRow) -> &str| -> Vec<Box<str>> {
        pts.iter().map(|(r, _)| f(r).into()).collect()
    };
    let g = |f: fn(&CurvePoint) -> f64| f32s(&pts, |(_, p)| f(p));
    let (ligand, receptor) = (s(|r| &r.ligand), s(|r| &r.receptor));
    let cols = [
        ("tau", g(|p| p.tau)),
        ("quantile", g(|p| p.quantile)),
        ("effect", g(|p| p.effect.theta)),
        ("se", g(|p| p.effect.se)),
        ("df", g(|p| p.effect.df)),
        ("lo", g(|p| p.lo)),
        ("hi", g(|p| p.hi)),
        ("band_lo", g(|p| p.band_lo)),
        ("band_hi", g(|p| p.band_hi)),
    ];
    let mut table = vec![
        ("ligand".into(), Column::Str(&ligand)),
        ("receptor".into(), Column::Str(&receptor)),
    ];
    table.extend(
        cols.iter()
            .map(|(n, v)| (n.to_string().into(), Column::F32(v))),
    );
    let names: Vec<Box<str>> = (0..pts.len()).map(|i| i.to_string().into()).collect();
    write_named_table(path, "row", &names, &table)
}

/// Columns of `{out}.lra-simulate.parquet` that `read_sim_runs` reads
/// back, shared by the writer and the reader.
pub(super) mod sim_cols {
    pub const EFFECT_PVE: &str = "effect_pve";
    pub const LIGAND_FIELD_PVE: &str = "ligand_field_pve";
    pub const RESPONSE_FIELD_PVE: &str = "response_field_pve";
    pub const SCALE: &str = "confound_scale";
    pub const CONFOUNDER: &str = "confounder";
    pub const SHAPE: &str = "confound_shape";
    pub const HETEROGENEOUS: &str = "heterogeneous";
    pub const MARKER: &str = "marker";
    pub const REP: &str = "rep";
    pub const ESTIMABLE: &str = "estimable";
    pub const DOSE_RESPONSE: &str = "dose_response";
    pub const DOSE_MIDPOINT: &str = "dose_midpoint";
    pub const DOSE_ORDER: &str = "dose_order";

    /// `{method}_reject`.
    pub fn reject(method: &str) -> String {
        format!("{method}_reject")
    }
}

/// `{out}.lra-simulate.parquet`: one row per pair and scenario run. `truth` is
/// the contrast the estimate targets and `truth_exposed` the mean effect among
/// the truly exposed ([`super::super::simulate::SimRow`]); `estimable`
/// is 0 where the estimate was not reported. The co-expression baselines
/// ([`super::super::baselines`]) come with their p and a 0/1 `_reject` at 5%,
/// as have the other methods of [`super::super::simulate::COMPARED`].
pub fn write_sim(path: &str, rows: &[SimRow]) -> anyhow::Result<()> {
    use sim_cols::*;
    let mut c = Columns::new(rows);
    c.s("ligand", |r| &r.ligand)
        .s("receptor", |r| &r.receptor)
        .f(EFFECT_PVE, |r| r.scenario.effect_pve)
        .f(LIGAND_FIELD_PVE, |r| r.scenario.ligand_field_pve)
        .f(RESPONSE_FIELD_PVE, |r| r.scenario.response_field_pve)
        .f(SCALE, |r| r.scenario.scale)
        .s(CONFOUNDER, |r| r.scenario.confounder.label())
        .s(SHAPE, |r| r.scenario.shape.label())
        .i(HETEROGENEOUS, |r| r.scenario.heterogeneous as usize)
        .i(MARKER, |r| r.scenario.marker as usize)
        .i(REP, |r| r.scenario.rep)
        .f("truth", |r| r.truth)
        .f("truth_exposed", |r| r.truth_exposed)
        .i(ESTIMABLE, |r| r.estimable() as usize)
        .f("estimate", |r| r.estimate.theta)
        .f("se", |r| r.estimate.se)
        .f("df", |r| r.estimate.df)
        .est("marginal_cor", |r| r.marginal_cor)
        .est("unadjusted_diff", |r| r.unadjusted_diff)
        .est_interval("effect_ipw", |r| r.ipw)
        .f("effect_outcome", |r| r.outcome)
        .est("effect_incompetent", |r| r.incompetent)
        .est("estimate_section", |r| r.steps[0])
        .est("estimate_spatial", |r| r.steps[1])
        .est("estimate_identity", |r| r.steps[2])
        .est("effect_contrast", |r| r.contrast)
        .i(DOSE_RESPONSE, |r| r.scenario.dose_response as usize)
        .f(DOSE_MIDPOINT, |r| r.scenario.dose_midpoint)
        .i(DOSE_ORDER, |r| r.scenario.dose_order as usize)
        .s("verdict", |r| r.verdict.code())
        .i("survived", |r| r.verdict.survived() as usize)
        .f("mean_overlap", |r| r.estimate.mean_h)
        .f("misclassified", |r| r.misclassified)
        .i("celltype_tested", |r| r.baselines.celltype_tested);
    let b = |f: fn(&Baselines) -> f64| move |r: &SimRow| f(&r.baselines);
    c.f("celltype_score", b(|x| x.celltype_score))
        .f("celltype_p", b(|x| x.celltype_p))
        .f("moran_i", b(|x| x.moran_i))
        .f("moran_p", b(|x| x.moran_p))
        .f("contact_score", b(|x| x.contact_score))
        .f("contact_p", b(|x| x.contact_p))
        .f("coexpr_log_or", b(|x| x.coexpr_log_or))
        .f("coexpr_se", b(|x| x.coexpr_se))
        .f("coexpr_p", b(|x| x.coexpr_p))
        .f("seconds", |r| r.seconds)
        .f("baseline_seconds", |r| r.baseline_seconds);
    for (k, m) in COMPARED.iter().enumerate() {
        c.i(&reject(m), |r| r.rejections()[k] as usize);
    }
    c.write(path)
}
