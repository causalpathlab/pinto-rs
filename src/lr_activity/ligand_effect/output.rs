//! Parquet tables, the verdict vocabulary, the run summary and the
//! benchmark summary.

use super::baselines::Baselines;
use super::estimate::{EffectRow, Moments};
use super::overlap::Estimate;
use super::simulate::{SimRow, COMPARED};
use legume_numeric::matrix::parquet::{write_named_table, Column};

// * Verdicts

/// A pair's verdict, defined once: the parquet's `verdict` code and
/// `verdict_note` words, the console summary, `pinto lra --help`, the MCP
/// instructions and the README all read it from here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Verdict {
    Supported,
    EffectWithoutCoexpression,
    ColocationOnly,
    NoEvidence,
    ControlFailed,
    CelltypeInseparable,
    InsufficientData,
}

impl Verdict {
    /// The pair's effect survived the controls.
    pub fn survived(self) -> bool {
        matches!(
            self,
            Verdict::Supported | Verdict::EffectWithoutCoexpression
        )
    }

    /// How many of `verdicts` are each verdict, in [`Verdict::ALL`]'s order.
    pub fn counts(
        verdicts: impl IntoIterator<Item = Verdict>,
    ) -> [(Verdict, usize); Verdict::ALL.len()] {
        let mut out = Verdict::ALL.map(|v| (v, 0));
        for v in verdicts {
            out[v.index()].1 += 1;
        }
        out
    }

    /// Its place in [`Verdict::ALL`] (a test checks the two agree).
    pub fn index(self) -> usize {
        match self {
            Verdict::Supported => 0,
            Verdict::EffectWithoutCoexpression => 1,
            Verdict::ColocationOnly => 2,
            Verdict::NoEvidence => 3,
            Verdict::ControlFailed => 4,
            Verdict::CelltypeInseparable => 5,
            Verdict::InsufficientData => 6,
        }
    }

    /// Every verdict, in the order the run summary lists them.
    pub const ALL: [Verdict; 7] = [
        Verdict::Supported,
        Verdict::EffectWithoutCoexpression,
        Verdict::ColocationOnly,
        Verdict::NoEvidence,
        Verdict::ControlFailed,
        Verdict::CelltypeInseparable,
        Verdict::InsufficientData,
    ];

    /// The machine code, the `verdict` column.
    pub fn code(self) -> &'static str {
        match self {
            Verdict::Supported => "supported",
            Verdict::EffectWithoutCoexpression => "effect_without_coexpression",
            Verdict::ColocationOnly => "colocation_only",
            Verdict::NoEvidence => "no_evidence",
            Verdict::ControlFailed => "control_failed",
            Verdict::CelltypeInseparable => "celltype_inseparable",
            Verdict::InsufficientData => "insufficient_data",
        }
    }

    /// The plain words, the `verdict_note` column.
    pub fn note(self) -> &'static str {
        match self {
            Verdict::Supported => {
                "receivers next to ligand-carrying cells change their response programme, \
                 ligand and receptor are co-expressed across contacts, and the effect \
                 exceeds every negative control"
            }
            Verdict::EffectWithoutCoexpression => {
                "receivers respond to neighbouring ligand, but ligand and receptor are not \
                 co-expressed across contacts beyond chance"
            }
            Verdict::ColocationOnly => {
                "ligand and receptor sit together across contacts, but receivers show no \
                 change in their response programme"
            }
            Verdict::NoEvidence => {
                "neither a response nor co-expression at this false discovery rate"
            }
            Verdict::ControlFailed => {
                "an effect appeared, but it does not exceed a negative control (receivers \
                 without the receptor, distant cells that do not touch, or a matched background \
                 programme), or one of these could not be checked: likely confounding"
            }
            Verdict::CelltypeInseparable => {
                "being exposed to the ligand is nearly the same as having a neighbour of \
                 one cell type, so the ligand cannot be told apart from that cell type"
            }
            Verdict::InsufficientData => {
                "too few exposed or unexposed receptor-carrying receivers, or too little \
                 overlap between them, to estimate an effect"
            }
        }
    }
}

impl std::str::FromStr for Verdict {
    type Err = anyhow::Error;

    /// A verdict by its code.
    fn from_str(code: &str) -> anyhow::Result<Self> {
        Verdict::ALL
            .into_iter()
            .find(|v| v.code() == code)
            .ok_or_else(|| anyhow::anyhow!("unknown verdict {code:?}"))
    }
}

/// The verdict list for `--help` and the README, one `code: note` per line.
pub fn verdict_help() -> String {
    Verdict::ALL
        .iter()
        .map(|v| format!("  {:<28} {}\n", v.code(), v.note()))
        .collect()
}

/// The verdicts in one line, `code: note` joined by `; `, for the MCP
/// server's instructions.
#[cfg(feature = "view")]
pub fn verdict_line() -> String {
    Verdict::ALL
        .iter()
        .map(|v| format!("{}: {}", v.code(), v.note()))
        .collect::<Vec<_>>()
        .join("; ")
}

fn f32s<T>(rows: &[T], f: impl Fn(&T) -> f64) -> Vec<f32> {
    rows.iter().map(|r| f(r) as f32).collect()
}

fn i32s<T>(rows: &[T], f: impl Fn(&T) -> usize) -> Vec<i32> {
    rows.iter().map(|r| f(r) as i32).collect()
}

/// Builds a table's columns in order: `est` pushes an estimate's theta
/// and `_se` beside each other, as the curve table lays its points out.
struct Columns<'r, T> {
    rows: &'r [T],
    f32s: Vec<(String, Vec<f32>)>,
    i32s: Vec<(String, Vec<i32>)>,
    strs: Vec<(String, Vec<Box<str>>)>,
    /// (kind, index into the kind's list) in table order.
    order: Vec<(u8, usize)>,
}

impl<'r, T> Columns<'r, T> {
    fn new(rows: &'r [T]) -> Self {
        Self {
            rows,
            f32s: Vec::new(),
            i32s: Vec::new(),
            strs: Vec::new(),
            order: Vec::new(),
        }
    }

    fn f(&mut self, name: &str, f: impl Fn(&T) -> f64) -> &mut Self {
        self.order.push((0, self.f32s.len()));
        self.f32s.push((name.into(), f32s(self.rows, f)));
        self
    }

    fn i(&mut self, name: &str, f: impl Fn(&T) -> usize) -> &mut Self {
        self.order.push((1, self.i32s.len()));
        self.i32s.push((name.into(), i32s(self.rows, f)));
        self
    }

    fn s(&mut self, name: &str, f: impl Fn(&T) -> &str) -> &mut Self {
        self.order.push((2, self.strs.len()));
        self.strs
            .push((name.into(), self.rows.iter().map(|r| f(r).into()).collect()));
        self
    }

    /// `name` and `name_se`.
    fn est(&mut self, name: &str, f: impl Fn(&T) -> Estimate) -> &mut Self {
        self.f(name, |r| f(r).theta)
            .f(&format!("{name}_se"), |r| f(r).se)
    }

    fn write(&self, path: &str) -> anyhow::Result<()> {
        let table: Vec<(Box<str>, Column<'_>)> = self
            .order
            .iter()
            .map(|&(kind, k)| match kind {
                0 => (self.f32s[k].0.as_str().into(), Column::F32(&self.f32s[k].1)),
                1 => (self.i32s[k].0.as_str().into(), Column::I32(&self.i32s[k].1)),
                _ => (self.strs[k].0.as_str().into(), Column::Str(&self.strs[k].1)),
            })
            .collect();
        let names: Vec<Box<str>> = (0..self.rows.len()).map(|i| i.to_string().into()).collect();
        write_named_table(path, "row", &names, &table)
    }
}

/// The `q_scope` of a table of every pair.
pub const Q_SCOPE_ALL: &str = "all pairs";

/// `{out}.lra.parquet`: one row per pair and section (`all` for the pooled
/// row). `verdict` is a [`Verdict`] code, `verdict_note` its words; the
/// headline's [`Moments`] are computed here. `q_scope` says which pairs the
/// q-values were corrected over: [`Q_SCOPE_ALL`], or a `--only` subset.
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
        .est("crude", |(r, _)| r.effects.crude)
        .est("effect_ipw", |(r, _)| r.effects.ipw)
        .est("effect_outcome", |(r, _)| r.effects.outcome)
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
        .est("ring_effect", |(r, _)| r.effects.ring)
        .f("ring_effect_q", |(r, _)| r.ring_effect_q)
        .est("effect_contrast", |(r, _)| r.effects.effect)
        .f("effect_log10_bf", |(r, _)| r.effects.abf.log10_bf)
        .f("tau_top", |(r, _)| r.effects.abf.top_tau)
        .f("tau_top_weight", |(r, _)| r.effects.abf.top_weight)
        .f("effect_contrast_q", |(r, _)| r.effect_contrast_q)
        .est("effect_over_ring", |(r, _)| r.effects.over_ring)
        .f("effect_over_ring_q", |(r, _)| r.effect_over_ring_q)
        .est("effect_over_background", |(r, _)| r.effect_over_background)
        .f("effect_over_background_q", |(r, _)| {
            r.effect_over_background_q
        })
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
        .est("background_effect", |(r, _)| r.background_effect)
        .f("background_effect_df", |(r, _)| r.background_effect.df)
        .f("background_q", |(r, _)| r.background_q)
        .i("n_background_genes", |(r, _)| r.n_background_genes)
        .s("q_scope", |_| q_scope);
    c.write(path)
}

/// `{out}.lra.curve.parquet`: the threshold–response curve of each pair's
/// pooled row, one row per pair and threshold: the competent effect at exposure threshold
/// `tau` (`quantile` of the positive exposures), its SE and df, the
/// pointwise 95% interval and the simultaneous 95% band over the pair's
/// thresholds.
pub fn write_curve(path: &str, rows: &[EffectRow]) -> anyhow::Result<()> {
    let pts: Vec<(&EffectRow, &super::estimate::CurvePoint)> = rows
        .iter()
        .flat_map(|r| r.curve.iter().map(move |p| (r, p)))
        .collect();
    let s = |f: fn(&EffectRow) -> &str| -> Vec<Box<str>> {
        pts.iter().map(|(r, _)| f(r).into()).collect()
    };
    let g = |f: fn(&super::estimate::CurvePoint) -> f64| f32s(&pts, |(_, p)| f(p));
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

/// Columns of `{out}.lra-simulate.parquet` that [`read_sim_runs`] reads
/// back, shared by the writer and the reader.
mod sim_cols {
    pub const PLANTED_EFFECT: &str = "planted_effect";
    pub const RING_EFFECT: &str = "planted_ring_effect";
    pub const CONFOUND_LIGAND: &str = "confound_ligand";
    pub const CONFOUND_RESPONSE: &str = "confound_response";
    pub const SCALE: &str = "confound_scale";
    pub const CONFOUNDER: &str = "confounder";
    pub const SHAPE: &str = "confound_shape";
    pub const HETEROGENEOUS: &str = "heterogeneous";
    pub const MARKER: &str = "marker";
    pub const SPILLOVER: &str = "spillover";
    pub const SHIFT: &str = "shift";
    pub const REP: &str = "rep";
    pub const CONFOUND_RECEPTOR: &str = "confound_receptor";
    pub const ESTIMABLE: &str = "estimable";
    pub const BACKGROUND_CONFOUNDING: &str = "background_confounding";
    pub const GENERIC_SHIFT: &str = "generic_shift";
    pub const DOSE_RESPONSE: &str = "dose_response";
    pub const DOSE_MIDPOINT: &str = "dose_midpoint";
    pub const DOSE_ORDER: &str = "dose_order";

    /// `{method}_reject`.
    pub fn reject(method: &str) -> String {
        format!("{method}_reject")
    }
}

/// `{out}.lra-simulate.parquet`: one row per pair and scenario run.
/// `truth` is the estimator-free sample average effect of the true
/// exposure, `truth_weighted` the same under the estimator's overlap
/// weights; `estimable` is 0 where the estimate was not reported. The
/// co-expression baselines ([`super::baselines`]) come with their p and a
/// 0/1 `_reject` at 5%, as have the other methods of [`COMPARED`].
pub fn write_sim(path: &str, rows: &[SimRow]) -> anyhow::Result<()> {
    use sim_cols::*;
    let mut c = Columns::new(rows);
    c.s("ligand", |r| &r.ligand)
        .s("receptor", |r| &r.receptor)
        .f(PLANTED_EFFECT, |r| r.scenario.effect)
        .f(CONFOUND_LIGAND, |r| r.scenario.confound_ligand)
        .f(CONFOUND_RESPONSE, |r| r.scenario.confound_response)
        .f(SCALE, |r| r.scenario.scale)
        .s(CONFOUNDER, |r| r.scenario.confounder.label())
        .s(SHAPE, |r| r.scenario.shape.label())
        .i(HETEROGENEOUS, |r| r.scenario.heterogeneous as usize)
        .i(MARKER, |r| r.scenario.marker as usize)
        .f(RING_EFFECT, |r| r.scenario.ring_effect)
        .f(SPILLOVER, |r| r.scenario.spillover)
        .f(SHIFT, |r| r.scenario.shift)
        .i(REP, |r| r.scenario.rep)
        .f("truth", |r| r.truth)
        .f("truth_weighted", |r| r.truth_weighted)
        .i(ESTIMABLE, |r| r.estimable() as usize)
        .f("estimate", |r| r.estimate.theta)
        .f("se", |r| r.estimate.se)
        .f("df", |r| r.estimate.df)
        .est("marginal_cor", |r| r.marginal_cor)
        .est("crude", |r| r.crude)
        .est("effect_ipw", |r| r.ipw)
        .est("effect_outcome", |r| r.outcome)
        .est("effect_incompetent", |r| r.incompetent)
        .est("ring_effect", |r| r.ring)
        .f("ring_truth", |r| r.ring_truth)
        .est("estimate_no_spill", |r| r.no_spill)
        .est("estimate_section", |r| r.steps[0])
        .est("estimate_spatial", |r| r.steps[1])
        .est("estimate_identity", |r| r.steps[2])
        .est("effect_contrast", |r| r.contrast)
        .est("effect_over_ring", |r| r.over_ring)
        .est("background_effect", |r| r.background)
        .est("effect_over_background", |r| r.over_background)
        .f(BACKGROUND_CONFOUNDING, |r| {
            r.scenario.background_confounding
        })
        .f(GENERIC_SHIFT, |r| r.scenario.generic_shift)
        .i(DOSE_RESPONSE, |r| r.scenario.dose_response as usize)
        .f(DOSE_MIDPOINT, |r| r.scenario.dose_midpoint)
        .i(DOSE_ORDER, |r| r.scenario.dose_order as usize)
        .s("verdict", |r| r.verdict.code())
        .i("survived", |r| r.verdict.survived() as usize)
        .f("mean_overlap", |r| r.estimate.mean_h)
        .f("misclassified", |r| r.misclassified)
        .i(CONFOUND_RECEPTOR, |r| r.scenario.confound_receptor as usize)
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

// * Run summary

/// Decimals to print a percent change and its interval with: under a percent,
/// whole percents would read as +0% to +0%.
pub fn percent_digits(pct: f64, lo: f64, hi: f64) -> usize {
    if pct.abs().max(lo.abs()).max(hi.abs()) < 1.0 {
        2
    } else {
        0
    }
}

/// A percent change and its 95% interval, as the summaries word them.
pub fn percent_text(pct: f64, lo: f64, hi: f64) -> String {
    let d = percent_digits(pct, lo, hi);
    format!("{pct:+.d$}% (95% CI {lo:+.d$}% to {hi:+.d$}%)")
}

/// The few lines a run ends with: verdict counts over the pooled pairs,
/// the pairs whose effect survived the controls, the table's path, and a
/// pointer to `pinto lra-simulate` when this run has no power check.
pub fn console_summary(rows: &[EffectRow], path: &str, power_check: Option<&str>) -> String {
    let pooled: Vec<PooledEffect> = rows
        .iter()
        .filter(|r| &*r.section == super::estimate::ALL_SECTIONS)
        .map(PooledEffect::from)
        .collect();
    let counts: Vec<String> = Verdict::counts(pooled.iter().map(|r| r.verdict))
        .iter()
        .map(|(v, n)| format!("{n} {}", v.code()))
        .collect();
    let mut out = format!("lra: {} pairs: {}\n", pooled.len(), counts.join(", "));
    let survived = rank_pooled(&pooled);
    let top: Vec<String> = survived
        .iter()
        .take(5)
        .map(|r| {
            format!(
                "{}-{} {} {}",
                r.ligand,
                r.receptor,
                percent_text(r.pct, r.pct_lo, r.pct_hi),
                r.verdict.code()
            )
        })
        .collect();
    if top.is_empty() {
        out += "  no pair's effect survived the controls\n";
    } else {
        out += &format!(
            "  survived the controls: {}\n",
            top[..top.len().min(3)].join("; ")
        );
        if top.len() > 3 {
            out += &format!("                         {}\n", top[3..].join("; "));
        }
        if survived.len() > top.len() {
            out += &format!(
                "                         and {} more in the table\n",
                survived.len() - top.len()
            );
        }
    }
    out += &format!("  table: {path} (verdict_note says why)");
    match power_check {
        Some(p) => out += &format!("\n  power check: {p}"),
        None => {
            out += "\n  no power check for this tissue yet: run `pinto lra-simulate` with the \
                    same input to see which effect sizes it can detect"
        }
    }
    out
}

/// A scenario in words, without its replicate: the summary's grouping key.
fn describe(s: &super::simulate::Scenario) -> String {
    let mut d = format!("θ={:.2}", s.effect);
    if s.confound_ligand != 0.0 || s.confound_response != 0.0 {
        d += &format!(
            " α={} β={} {}/{}@{}",
            s.confound_ligand,
            s.confound_response,
            s.confounder.label(),
            s.shape.label(),
            s.scale
        );
    } else {
        d += " unconfounded";
    }
    if s.dose_response {
        d += " dose";
    }
    if s.ring_effect != 0.0 {
        d += &format!(" θr={:.2}", s.ring_effect);
    }
    if s.generic_shift != 0.0 {
        d += &format!(" generic={}", s.generic_shift);
    }
    if s.spillover != 0.0 {
        d += &format!(" κ={}", s.spillover);
    }
    if s.shift != 0.0 {
        d += &format!(" shift={}", s.shift);
    }
    if s.heterogeneous {
        d += " heterogeneous";
    }
    if s.marker {
        d += " marker";
    }
    d
}

/// Indices of `items` grouped by scenario ([`describe`]), groups in the
/// order their scenario first appears; one pass with a map.
fn scenario_groups<T>(
    items: &[T],
    scenario: impl Fn(&T) -> &super::simulate::Scenario,
) -> Vec<(String, Vec<usize>)> {
    let mut at: std::collections::HashMap<String, usize> = Default::default();
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, r) in items.iter().enumerate() {
        let d = describe(scenario(r));
        let k = *at.entry(d.clone()).or_insert_with(|| {
            groups.push((d, Vec::new()));
            groups.len() - 1
        });
        groups[k].1.push(i);
    }
    groups
}

/// Per scenario: a title line (runs, misclassified exposure, ring and
/// no-spill rejection), then one line each for the estimate, the crude
/// contrast and the partial adjustments, with estimable share, bias and
/// coverage against both truths, rejection rate (type I error or power)
/// and, under a planted effect, attenuation. Then the comparison
/// ([`comparison`]).
pub fn sim_summary(rows: &[SimRow]) -> String {
    fn est(e: &Estimate) -> (f64, f64, f64) {
        (e.theta, e.se, e.crit())
    }
    let share = |v: &[bool]| -> String {
        if v.is_empty() {
            "-".into()
        } else {
            format!(
                "{:.2}",
                v.iter().filter(|&&b| b).count() as f64 / v.len() as f64
            )
        }
    };
    // (estimate, se, two-sided 5% critical value of its t reference).
    type View = fn(&SimRow) -> (f64, f64, f64);
    let views: [(&str, View); 7] = [
        ("estimate", |r| est(&r.estimate)),
        ("crude", |r| est(&r.crude)),
        ("ipw", |r| est(&r.ipw)),
        ("outcome", |r| est(&r.outcome)),
        ("section", |r| est(&r.steps[0])),
        ("spatial", |r| est(&r.steps[1])),
        ("identity", |r| est(&r.steps[2])),
    ];
    let mut out = String::from(
        "`truth` is the sample average effect of the true exposure over competent \
         receivers (estimator-free); `_w` columns use the estimator-weighted truth. \
         `estim` is the share of runs with a reported estimate; `reject` the share \
         significant at 5%: type I error where nothing is planted (or the field is \
         shifted), power elsewhere; `atten` mean estimate / mean truth under a planted \
         effect. section/spatial/identity are the partial adjustments.\n",
    );
    out += &format!(
        "  {:<9} {:>5}  {:>7} {:>7}  {:>5} {:>7}  {:<6} {:>6}  {:>6}\n",
        "", "estim", "bias", "bias_w", "cover", "cover_w", "rate", "reject", "atten"
    );
    for (key, idx) in scenario_groups(rows, |r| &r.scenario) {
        let g: Vec<&SimRow> = idx.iter().map(|&i| &rows[i]).collect();
        let s = g[0].scenario;
        let null = is_null(&s);
        let miscl = g.iter().map(|r| r.misclassified).sum::<f64>() / g.len() as f64;
        let rej_of = |e: fn(&SimRow) -> Estimate| -> String {
            let v: Vec<bool> = g
                .iter()
                .map(|r| e(r))
                .filter(|e| e.theta.is_finite())
                .map(|e| e.rejects())
                .collect();
            share(&v)
        };
        out += &format!(
            "{key}: {} runs, misclassified exposure {miscl:.2}, ring reject {}, no-spill reject {}, background reject {}, verdict survived {}\n",
            g.len(),
            rej_of(|r| r.ring),
            rej_of(|r| r.no_spill),
            rej_of(|r| r.background),
            share(&g.iter().map(|r| r.verdict.survived()).collect::<Vec<_>>()),
        );
        for (name, view) in views {
            let ok: Vec<(f64, f64, f64, f64, f64)> = g
                .iter()
                .filter_map(|r| {
                    let (th, se, crit) = view(r);
                    (th.is_finite() && se.is_finite() && crit.is_finite() && r.truth.is_finite())
                        .then_some((th, se, r.truth, r.truth_weighted, crit))
                })
                .collect();
            let estim = ok.len() as f64 / g.len() as f64;
            if ok.is_empty() {
                out += &format!("  {name:<9} {estim:>5.2}  (not estimable)\n");
                continue;
            }
            let n = ok.len() as f64;
            let bias = ok.iter().map(|o| o.0 - o.2).sum::<f64>() / n;
            let wv: Vec<f64> = ok
                .iter()
                .filter(|o| o.3.is_finite())
                .map(|o| o.0 - o.3)
                .collect();
            let bias_w = if wv.is_empty() {
                f64::NAN
            } else {
                wv.iter().sum::<f64>() / wv.len() as f64
            };
            let cover: Vec<bool> = ok.iter().map(|o| (o.0 - o.2).abs() <= o.4 * o.1).collect();
            let cover_w: Vec<bool> = ok
                .iter()
                .filter(|o| o.3.is_finite())
                .map(|o| (o.0 - o.3).abs() <= o.4 * o.1)
                .collect();
            let reject: Vec<bool> = ok.iter().map(|o| (o.0 / o.1).abs() > o.4).collect();
            let mean_truth = ok.iter().map(|o| o.2).sum::<f64>() / n;
            let atten = if null || mean_truth.abs() < 1e-12 {
                "-".to_string()
            } else {
                format!(
                    "{:.2}",
                    ok.iter().map(|o| o.0).sum::<f64>() / n / mean_truth
                )
            };
            out += &format!(
                "  {name:<9} {estim:>5.2}  {bias:>+7.3} {bias_w:>+7.3}  {:>5} {:>7}  {:<6} {:>6}  {atten:>6}\n",
                share(&cover),
                share(&cover_w),
                if null { "type I" } else { "power" },
                share(&reject),
            );
        }
    }
    out.push_str(&comparison(rows));
    out
}

/// A null scenario: nothing planted, or the seen field shifted away from
/// the one receivers respond to.
fn is_null(s: &super::simulate::Scenario) -> bool {
    s.effect == 0.0 || s.shift > 0.0
}

/// One benchmark run as the comparison reads it: its scenario, whether the
/// estimate was reported, and whether each method of [`COMPARED`] rejects
/// at 5%.
#[derive(Clone, Copy, Debug)]
pub struct SimRun {
    pub scenario: super::simulate::Scenario,
    pub estimable: bool,
    /// Per method of [`COMPARED`]: 1 when it rejects, else 0; NaN when a table
    /// from an earlier version lacks the method.
    pub rejects: [f64; COMPARED.len()],
}

impl From<&SimRow> for SimRun {
    fn from(r: &SimRow) -> Self {
        Self {
            scenario: r.scenario,
            estimable: r.estimable(),
            rejects: r.rejections().map(|hit| hit as u8 as f64),
        }
    }
}

/// A scenario's line of the comparison.
#[derive(Clone, Debug)]
pub struct ScenarioRates {
    /// The scenario in words, without its replicate.
    pub scenario: String,
    /// Nothing planted (or the field shifted): the rates are type I error.
    pub null: bool,
    #[cfg_attr(not(feature = "view"), allow(dead_code))]
    pub runs: usize,
    /// Share of runs with a reported estimate.
    pub estimable: f64,
    /// Share of runs rejecting at 5%, per method of [`COMPARED`].
    pub rates: [f64; COMPARED.len()],
}

/// The comparison's numbers: per scenario, then the means over the
/// confounded nulls and over the planted effects (label, scenarios
/// averaged, mean rate per method), where there are any.
#[derive(Clone, Debug)]
pub struct Comparison {
    pub scenarios: Vec<ScenarioRates>,
    pub means: Vec<(&'static str, usize, [f64; COMPARED.len()])>,
}

/// [`Comparison`] of `runs`, scenarios in the order they first appear. A
/// run without a statistic (an estimate without overlap) counts as not
/// rejecting.
pub fn compare_runs(runs: &[SimRun]) -> Comparison {
    let groups = scenario_groups(runs, |r| &r.scenario);
    let (mut nulls, mut planted) = (Vec::new(), Vec::new());
    let mut scenarios = Vec::with_capacity(groups.len());
    for (key, idx) in groups {
        let g: Vec<&SimRun> = idx.iter().map(|&i| &runs[i]).collect();
        let s = g[0].scenario;
        let null = is_null(&s);
        let mut rates = [0.0; COMPARED.len()];
        for r in &g {
            for (x, hit) in rates.iter_mut().zip(r.rejects) {
                *x += hit / g.len() as f64;
            }
        }
        if !null {
            planted.push(rates);
        } else if s.confound_ligand != 0.0 || s.confound_response != 0.0 {
            nulls.push(rates);
        }
        scenarios.push(ScenarioRates {
            scenario: key,
            null,
            runs: g.len(),
            estimable: g.iter().filter(|r| r.estimable).count() as f64 / g.len() as f64,
            rates,
        });
    }
    let means = [
        ("mean type I, confounded nulls", &nulls),
        ("mean power, planted effects", &planted),
    ]
    .into_iter()
    .filter(|(_, set)| !set.is_empty())
    .map(|(label, set): (_, &Vec<[f64; COMPARED.len()]>)| {
        let mut m = [0.0; COMPARED.len()];
        for (k, x) in m.iter_mut().enumerate() {
            *x = set.iter().map(|v| v[k]).sum::<f64>() / set.len() as f64;
        }
        (label, set.len(), m)
    })
    .collect();
    Comparison { scenarios, means }
}

/// The methods side by side: per scenario, each one's share of runs
/// rejecting at 5% ([`COMPARED`]; a run without the statistic, such as an
/// estimate without overlap, counts as not rejecting), the estimable share,
/// then the mean over scenarios of type I error under the confounded nulls
/// and of power under the planted effects, and the baselines' share of the
/// run time.
pub fn comparison(rows: &[SimRow]) -> String {
    let runs: Vec<SimRun> = rows.iter().map(SimRun::from).collect();
    let c = compare_runs(&runs);
    let width = c
        .scenarios
        .iter()
        .map(|s| s.scenario.chars().count())
        .max()
        .unwrap_or(0)
        .max(8);
    let mut out = format!(
        "\nComparison at 5%: share of runs rejecting (type I where nothing is planted or the \
         field is shifted, power elsewhere). celltype: cell-type label permutation \
         (CellPhoneDB/CellChat); moran: bivariate Moran's I (SpatialDM); contact: contact \
         co-expression score (NICHES/COMMOT); coexpr: lra's co-expression log odds.\n  \
         {:<width$} {:<6} {:>5} ",
        "scenario", "rate", "estim"
    );
    for m in COMPARED {
        out += &format!(" {m:>8}");
    }
    out += "\n";
    for s in &c.scenarios {
        out += &format!(
            "  {:<width$} {:<6} {:>5.2} ",
            s.scenario,
            if s.null { "type I" } else { "power" },
            s.estimable
        );
        for x in s.rates {
            out += &format!(" {x:>8.2}");
        }
        out += "\n";
    }
    for (label, n, m) in &c.means {
        let head = format!("{label} ({n})");
        out += &format!("  {head:<w$} ", w = width + 14);
        for x in m {
            out += &format!(" {x:>8.2}");
        }
        out += "\n";
    }
    let total: f64 = rows.iter().map(|r| r.seconds).sum();
    let base: f64 = rows.iter().map(|r| r.baseline_seconds).sum();
    out += &format!(
        "  baselines: {base:.1} s of {total:.1} s run time summed over runs ({:.0}%)\n",
        100.0 * base / total.max(1e-12)
    );
    out
}

// * Reading the tables back

/// A pooled row of `{out}.lra.parquet` as a summary reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct PooledEffect {
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    pub verdict: Verdict,
    pub effect: f64,
    pub effect_p: f64,
    pub effect_q: f64,
    pub pct: f64,
    pub pct_lo: f64,
    pub pct_hi: f64,
}

impl From<&EffectRow> for PooledEffect {
    fn from(r: &EffectRow) -> Self {
        let m = Moments::of(&r.effects.competent);
        Self {
            ligand: r.ligand.clone(),
            receptor: r.receptor.clone(),
            verdict: r.verdict,
            effect: r.effects.competent.theta,
            effect_p: r.effect_p,
            effect_q: r.effect_q,
            pct: m.pct,
            pct_lo: m.pct_lo,
            pct_hi: m.pct_hi,
        }
    }
}

/// A pooled row of `{out}.lra.parquet` as the viewer lists it.
#[cfg(feature = "view")]
#[derive(Clone, Debug, PartialEq)]
pub struct PairRow {
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    /// `ligand` and `receptor` in upper case: what the list searches and
    /// sorts by.
    pub key: (Box<str>, Box<str>),
    pub verdict: Verdict,
    pub note: Box<str>,
    pub warnings: Box<str>,
    /// The headline competent effect (log rate ratio), its SE, p and q.
    pub effect: f64,
    pub effect_se: f64,
    pub effect_p: f64,
    pub effect_q: f64,
    pub coexpr_q: f64,
    /// The headline as a percent change, `exp(effect) − 1`, with its 95%
    /// interval.
    pub pct: f64,
    pub pct_lo: f64,
    pub pct_hi: f64,
    /// The exposure threshold τ the pair used.
    pub threshold: f64,
    pub n_receivers: usize,
    pub n_competent: usize,
    pub n_exposed: usize,
}

#[cfg(feature = "view")]
impl PairRow {
    /// `LIGAND:RECEPTOR`.
    pub fn name(&self) -> String {
        format!("{}:{}", self.ligand, self.receptor)
    }
}

#[cfg(feature = "view")]
impl From<&PairRow> for PooledEffect {
    fn from(r: &PairRow) -> Self {
        Self {
            ligand: r.ligand.clone(),
            receptor: r.receptor.clone(),
            verdict: r.verdict,
            effect: r.effect,
            effect_p: r.effect_p,
            effect_q: r.effect_q,
            pct: r.pct,
            pct_lo: r.pct_lo,
            pct_hi: r.pct_hi,
        }
    }
}

/// The pooled rows of a contact-effect table, in table order, and which
/// pairs its q-values were corrected over (its `q_scope`: [`Q_SCOPE_ALL`],
/// or a `--only` subset's words); a table from before the column says
/// nothing.
#[cfg(feature = "view")]
pub fn read_pair_rows(path: &str) -> anyhow::Result<(Vec<PairRow>, Option<Box<str>>)> {
    use legume_numeric::matrix::parquet::read_table_columns;
    let strings = [
        "section",
        "ligand",
        "receptor",
        "verdict",
        "verdict_note",
        "warnings",
        "q_scope",
    ];
    let numbers = [
        "effect_competent",
        "effect_competent_se",
        "effect_p",
        "effect_q",
        "coexpr_q",
        "effect_pct",
        "effect_pct_lo",
        "effect_pct_hi",
        "exposure_threshold",
        "n_receivers",
        "n_competent",
        "n_exposed",
    ];
    // Tables from before `q_scope` have none.
    let (s, x) = read_table_columns(path, &strings, &numbers)
        .or_else(|_| read_table_columns(path, &strings[..strings.len() - 1], &numbers))?;
    let q_scope = s.get(6).and_then(|v| v.first().cloned());
    let count = |v: f64| {
        if v.is_finite() && v > 0. {
            v as usize
        } else {
            0
        }
    };
    let rows = (0..s[0].len())
        .filter(|&i| &*s[0][i] == super::estimate::ALL_SECTIONS)
        .map(|i| {
            Ok(PairRow {
                ligand: s[1][i].clone(),
                receptor: s[2][i].clone(),
                key: (s[1][i].to_uppercase().into(), s[2][i].to_uppercase().into()),
                verdict: s[3][i]
                    .parse()
                    .map_err(|e| anyhow::anyhow!("{path}: {e}"))?,
                note: s[4][i].clone(),
                warnings: s[5][i].clone(),
                effect: x[0][i],
                effect_se: x[1][i],
                effect_p: x[2][i],
                effect_q: x[3][i],
                coexpr_q: x[4][i],
                pct: x[5][i],
                pct_lo: x[6][i],
                pct_hi: x[7][i],
                threshold: x[8][i],
                n_receivers: count(x[9][i]),
                n_competent: count(x[10][i]),
                n_exposed: count(x[11][i]),
            })
        })
        .collect::<anyhow::Result<_>>()?;
    Ok((rows, q_scope))
}

/// The pooled rows of `{out}.lra.parquet` (section `all`) as a summary
/// reads them, in table order.
#[cfg(feature = "view")]
pub fn read_pooled_effects(path: &str) -> anyhow::Result<Vec<PooledEffect>> {
    Ok(read_pair_rows(path)?
        .0
        .iter()
        .map(PooledEffect::from)
        .collect())
}

/// One point of a pair's threshold–response curve, as
/// `{out}.lra.curve.parquet` holds it (log rate ratios).
#[cfg(feature = "view")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveRow {
    pub tau: f64,
    pub quantile: f64,
    pub effect: f64,
    pub lo: f64,
    pub hi: f64,
    pub band_lo: f64,
    pub band_hi: f64,
}

/// Each pair's curve points, by `(ligand, receptor)`.
#[cfg(feature = "view")]
pub type Curves = std::collections::HashMap<(Box<str>, Box<str>), Vec<CurveRow>>;

/// Every pair's curve, by `(ligand, receptor)`, points in table order.
#[cfg(feature = "view")]
pub fn read_curves(path: &str) -> anyhow::Result<Curves> {
    use legume_numeric::matrix::parquet::read_table_columns;
    let (s, x) = read_table_columns(
        path,
        &["ligand", "receptor"],
        &[
            "tau", "quantile", "effect", "lo", "hi", "band_lo", "band_hi",
        ],
    )?;
    let mut out = Curves::default();
    for i in 0..s[0].len() {
        out.entry((s[0][i].clone(), s[1][i].clone()))
            .or_default()
            .push(CurveRow {
                tau: x[0][i],
                quantile: x[1][i],
                effect: x[2][i],
                lo: x[3][i],
                hi: x[4][i],
                band_lo: x[5][i],
                band_hi: x[6][i],
            });
    }
    Ok(out)
}

/// The pooled rows ranked as the run summary ranks them: estimable
/// headline effects by p.
pub fn rank_pooled(rows: &[PooledEffect]) -> Vec<&PooledEffect> {
    let mut v: Vec<&PooledEffect> = rows
        .iter()
        .filter(|r| r.effect.is_finite() && r.effect_p.is_finite() && r.verdict.survived())
        .collect();
    v.sort_by(|a, b| a.effect_p.total_cmp(&b.effect_p));
    v
}

/// The runs of `{out}.lra-simulate.parquet` as [`compare_runs`] reads them,
/// each column by its name ([`sim_cols`]).
#[cfg(feature = "view")]
pub fn read_sim_runs(path: &str) -> anyhow::Result<Vec<SimRun>> {
    use super::simulate::{ConfoundShape, ConfounderKind, Scenario};
    use legume_numeric::matrix::parquet::read_table_columns;
    use sim_cols::*;
    const NUMBERS: [&str; 12] = [
        PLANTED_EFFECT,
        RING_EFFECT,
        CONFOUND_LIGAND,
        CONFOUND_RESPONSE,
        SCALE,
        HETEROGENEOUS,
        MARKER,
        SPILLOVER,
        SHIFT,
        CONFOUND_RECEPTOR,
        REP,
        ESTIMABLE,
    ];
    let (s, x) = read_table_columns(path, &[CONFOUNDER, SHAPE], &NUMBERS)?;
    let col = |name: &str| -> &[f64] {
        &x[NUMBERS
            .iter()
            .position(|&n| n == name)
            .expect("a column read")]
    };
    // The table keeps f32: back to the value as written (0.3, not
    // 0.30000001), so a scenario reads as the run's summary words it.
    let num = |name: &str, i: usize| -> f64 {
        let v = col(name)[i] as f32;
        v.to_string().parse().unwrap_or(f64::from(v))
    };
    let flag = |name: &str, i: usize| col(name)[i] != 0.0;
    // Columns a table from an earlier version lacks: the scenario's default.
    let later = |name: &str| -> Option<Vec<f64>> {
        read_table_columns(path, &[], &[name])
            .ok()
            .and_then(|(_, mut c)| c.pop())
    };
    let rejects: Vec<Option<Vec<f64>>> = COMPARED.iter().map(|m| later(&reject(m))).collect();
    let plain = Scenario::default();
    let (background_confounding, generic_shift) =
        (later(BACKGROUND_CONFOUNDING), later(GENERIC_SHIFT));
    let (dose_response, dose_midpoint, dose_order) = (
        later(DOSE_RESPONSE),
        later(DOSE_MIDPOINT),
        later(DOSE_ORDER),
    );
    let exact = |c: &Option<Vec<f64>>, i: usize, default: f64| -> f64 {
        c.as_ref().map_or(default, |c| {
            let v = c[i] as f32;
            v.to_string().parse().unwrap_or(f64::from(v))
        })
    };
    (0..s[0].len())
        .map(|i| {
            let confounder: ConfounderKind = s[0][i]
                .parse()
                .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
            let shape: ConfoundShape = s[1][i]
                .parse()
                .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
            let scenario = Scenario {
                effect: num(PLANTED_EFFECT, i),
                ring_effect: num(RING_EFFECT, i),
                confound_ligand: num(CONFOUND_LIGAND, i),
                confound_response: num(CONFOUND_RESPONSE, i),
                scale: num(SCALE, i),
                confounder,
                shape,
                heterogeneous: flag(HETEROGENEOUS, i),
                marker: flag(MARKER, i),
                spillover: num(SPILLOVER, i),
                shift: num(SHIFT, i),
                confound_receptor: flag(CONFOUND_RECEPTOR, i),
                rep: col(REP)[i] as usize,
                background_confounding: exact(
                    &background_confounding,
                    i,
                    plain.background_confounding,
                ),
                generic_shift: exact(&generic_shift, i, plain.generic_shift),
                dose_response: exact(&dose_response, i, 0.0) != 0.0,
                dose_midpoint: exact(&dose_midpoint, i, plain.dose_midpoint),
                dose_order: exact(&dose_order, i, plain.dose_order as f64) as i32,
            };
            let mut hits = [f64::NAN; COMPARED.len()];
            for (h, c) in hits.iter_mut().zip(&rejects) {
                if let Some(c) = c {
                    *h = (c[i] != 0.0) as u8 as f64;
                }
            }
            Ok(SimRun {
                scenario,
                estimable: flag(ESTIMABLE, i),
                rejects: hits,
            })
        })
        .collect()
}
