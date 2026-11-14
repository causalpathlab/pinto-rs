//! Writing the `lra` and `lra-simulate` tables.

use super::columns::{f32s, Columns};
use crate::lr_activity::ligand_effect::baselines::{Baselines, CellTypeTest, TypeLabels, TypePair};
use crate::lr_activity::ligand_effect::rows::{CurvePoint, EffectRow, Moments};
use crate::lr_activity::ligand_effect::simulate::{RegionRead, SimRow, COMPARED};
use legume_numeric::matrix::parquet::{write_named_table, Column};

/// The `q_scope` of a table of every pair.
pub const Q_SCOPE_ALL: &str = "all pairs";

/// `{out}.lra.parquet`: one row per pair and section (`all` for the pooled
/// row). `verdict` is a [`super::super::verdict::Verdict`] code, `verdict_note`
/// its words; the headline's [`super::super::rows::Moments`] are computed here.
/// `q_scope` says which pairs the q-values were corrected over:
/// [`Q_SCOPE_ALL`], or a `--only` subset. With `marginal`
/// (`--marginal-baselines`) the co-expression baselines follow, before
/// `q_scope`: `celltype_score`, `celltype_p`, `celltype_tested`, `moran_i`,
/// `moran_p`, `contact_score` and `contact_p`, NaN (and `celltype_tested` 0) on every
/// row but a pair's pooled one.
pub fn write_effects(
    path: &str,
    rows: &[EffectRow],
    q_scope: &str,
    marginal: bool,
) -> anyhow::Result<()> {
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
        .est("effect_vs_rest", |(r, _)| r.effects.versus_rest)
        .f("effect_vs_rest_df", |(r, _)| r.effects.versus_rest.df)
        .f("effect_vs_rest_p", |(r, _)| r.effects.versus_rest.p())
        .f("jackknife_ratio", |(r, _)| r.jackknife.competent)
        .f("jackknife_vs_rest_ratio", |(r, _)| r.jackknife.versus_rest)
        .i("jackknife_tiles", |(r, _)| r.jackknife.replicates)
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
        .f("effect_pct_hi", |(_, m)| m.pct_hi);
    if marginal {
        let b = |f: fn(&Baselines) -> f64| {
            move |(r, _): &(&EffectRow, Moments)| r.marginal.as_ref().map_or(f64::NAN, f)
        };
        c.f("celltype_score", b(|x| x.celltype_score))
            .f("celltype_p", b(|x| x.celltype_p))
            .i("celltype_tested", |(r, _)| {
                r.marginal.map_or(0, |x| x.celltype_tested)
            })
            .f("moran_i", b(|x| x.moran_i))
            .f("moran_p", b(|x| x.moran_p))
            .f("contact_score", b(|x| x.contact_score))
            .f("contact_p", b(|x| x.contact_p));
    }
    c.s("q_scope", |_| q_scope);
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

/// One row of `{out}.lra.celltype.parquet`: a pair's tested sender type ×
/// receiver type, named.
pub struct CelltypeRow {
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    pub sender_type: Box<str>,
    pub receiver_type: Box<str>,
    pub stat: TypePair,
    pub q: f64,
    pub permutations: usize,
    pub labels: &'static str,
}

/// The rows of one pair's cell-type test, its types named by `labels`.
pub fn celltype_rows(
    ligand: &str,
    receptor: &str,
    test: &CellTypeTest,
    labels: &TypeLabels,
) -> Vec<CelltypeRow> {
    test.pairs
        .iter()
        .zip(test.q())
        .map(|(t, q)| CelltypeRow {
            ligand: ligand.into(),
            receptor: receptor.into(),
            sender_type: labels.names[t.sender as usize].clone(),
            receiver_type: labels.names[t.receiver as usize].clone(),
            stat: *t,
            q,
            permutations: test.permutations,
            labels: labels.source.name(),
        })
        .collect()
}

/// `{out}.lra.celltype.parquet` (`--marginal-baselines`): one row per pair
/// and tested sender type × receiver type. CellChat's score
/// (`mean_ligand_sender × mean_receptor_receiver`, `log1p` of the
/// depth-normalized counts), its one-sided permutation `p`, its BH `q`
/// within the pair's tested type pairs, the detection shares that let it be
/// tested, the permutations run, and where the types came from (`labels`:
/// `cell-types` or `identity-bins`).
pub fn write_celltype(path: &str, rows: &[CelltypeRow]) -> anyhow::Result<()> {
    let mut c = Columns::new(rows);
    c.s("ligand", |r| &r.ligand)
        .s("receptor", |r| &r.receptor)
        .s("sender_type", |r| &r.sender_type)
        .s("receiver_type", |r| &r.receiver_type)
        .i("n_sender", |r| r.stat.n_sender)
        .i("n_receiver", |r| r.stat.n_receiver)
        .f("mean_ligand_sender", |r| r.stat.mean_ligand)
        .f("mean_receptor_receiver", |r| r.stat.mean_receptor)
        .f("ligand_detected_sender", |r| r.stat.ligand_detected)
        .f("receptor_detected_receiver", |r| r.stat.receptor_detected)
        .f("score", |r| r.stat.score)
        .f("p", |r| r.stat.p)
        .f("q", |r| r.q)
        .i("permutations", |r| r.permutations)
        .s("labels", |r| r.labels);
    c.write(path)
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
    /// The region code the effect was planted in; NaN where it was planted
    /// everywhere. Read when present, so older tables still read.
    pub const REGION: &str = "region";
    /// The planted region's name (`--plant-region`); empty where the effect
    /// was planted everywhere.
    pub const PLANTED_REGION: &str = "planted_region";

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
/// `planted` names the region every run's effect was planted in, if one.
pub fn write_sim(path: &str, rows: &[SimRow], planted: Option<&str>) -> anyhow::Result<()> {
    sim_columns(rows, planted).write(path)
}

/// A single-scenario call's table: [`write_sim`]'s columns, then what a
/// sweep scores each row by. `planted` (0/1) is the row's label, one per
/// row; `t` is the estimate over its SE, a one-sided score since the planted
/// effect is positive, and `p` its two-sided p on the t reference
/// ([`crate::lr_activity::ligand_effect::inference::Estimate::p`]);
/// `marginal_cor_p`, `unadjusted_diff_p` and `effect_ipw_p` the same for the
/// other methods. A row not estimable has NaN scores. The scores are the
/// pooled row's: with a planted region, its row's are `region_estimate` and
/// `region_se`.
pub fn write_sim_scored(
    path: &str,
    rows: &[SimRow],
    region: Option<&str>,
    planted: &[bool],
) -> anyhow::Result<()> {
    anyhow::ensure!(planted.len() == rows.len(), "a label per row");
    let label: Vec<usize> = planted.iter().map(|&p| usize::from(p)).collect();
    let mut c = sim_columns(rows, region);
    c.v("planted", label)
        .f("t", |r| r.estimate.z())
        .f("p", |r| r.estimate.p())
        .f("marginal_cor_p", |r| r.marginal_cor.p())
        .f("unadjusted_diff_p", |r| r.unadjusted_diff.p())
        .f("effect_ipw_p", |r| r.ipw.p());
    c.write(path)
}

/// The columns of [`write_sim`].
fn sim_columns<'r>(rows: &'r [SimRow], planted: Option<&'r str>) -> Columns<'r, SimRow> {
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
    let g = |f: fn(&RegionRead) -> f64| move |r: &SimRow| r.region.as_ref().map_or(f64::NAN, f);
    c.f(REGION, |r| r.scenario.region.map_or(f64::NAN, f64::from))
        .s(PLANTED_REGION, |_| planted.unwrap_or_default())
        .f("region_truth", g(|x| x.truth))
        .f("region_truth_exposed", g(|x| x.truth_exposed))
        .f("region_estimate", g(|x| x.estimate.theta))
        .f("region_se", g(|x| x.estimate.se))
        .f("region_df", g(|x| x.estimate.df))
        .s("region_verdict", |r| {
            r.region.as_ref().map_or("", |x| x.verdict.code())
        })
        .f(
            "region_reject",
            g(|x| f64::from(u8::from(x.estimate.rejects()))),
        )
        .f("region_jackknife_ratio", g(|x| x.jackknife.competent))
        .f(
            "region_jackknife_vs_rest_ratio",
            g(|x| x.jackknife.versus_rest),
        )
        .f("region_vs_rest", g(|x| x.versus_rest.theta))
        .f("region_vs_rest_se", g(|x| x.versus_rest.se))
        .f(
            "region_vs_rest_reject",
            g(|x| f64::from(u8::from(x.versus_rest.rejects()))),
        );
    for (k, m) in COMPARED.iter().enumerate() {
        c.i(&reject(m), |r| r.rejections()[k] as usize);
    }
    c
}
