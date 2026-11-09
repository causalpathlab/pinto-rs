//! The text summaries printed after a run.

use super::compare::{comparison, is_null, scenario_groups};
use crate::lr_activity::ligand_effect::inference::Estimate;
use crate::lr_activity::ligand_effect::rows::{EffectRow, Moments};
use crate::lr_activity::ligand_effect::simulate::SimRow;
use crate::lr_activity::ligand_effect::verdict::Verdict;

/// Decimals to print a percent change and its interval with, from the
/// largest of them: two under a percent (whole percents would read +0% to
/// +0%), one under ten, none above.
fn percent_digits(pct: f64, lo: f64, hi: f64) -> usize {
    match pct.abs().max(lo.abs()).max(hi.abs()) {
        m if m < 1.0 => 2,
        m if m < 10.0 => 1,
        _ => 0,
    }
}

/// `x` percent at `d` decimals, signed; one that rounds to zero reads `+0`,
/// never `-0`.
fn signed_percent(x: f64, d: usize) -> String {
    let text = format!("{x:+.d$}");
    match text.strip_prefix('-') {
        Some(digits) if digits.chars().all(|c| c == '0' || c == '.') => format!("+{digits}%"),
        _ => format!("{text}%"),
    }
}

/// `s`, a number, without the trailing zeros of its decimals (nor a bare
/// point): `6.030` → `6.03`, `2.000` → `2`.
pub fn strip_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

/// A percent change and its 95% interval, as the summaries word them.
pub fn percent_text(pct: f64, lo: f64, hi: f64) -> String {
    let d = percent_digits(pct, lo, hi);
    let [l, h] = [lo, hi].map(|x| signed_percent(x, d));
    format!("{} (95% CI {l} to {h})", percent_short(pct, lo, hi))
}

/// A percent change alone, at the digits [`percent_text`] gives it; `—`
/// when it is not finite.
pub fn percent_short(pct: f64, lo: f64, hi: f64) -> String {
    if pct.is_finite() {
        signed_percent(pct, percent_digits(pct, lo, hi))
    } else {
        "—".into()
    }
}

/// The few lines a run ends with: verdict counts over the pooled pairs,
/// the pairs whose effect survived the controls, the table's path, and a
/// pointer to `pinto lra-simulate` when this run has no power check.
pub fn console_summary(rows: &[EffectRow], path: &str, power_check: Option<&str>) -> String {
    let pooled: Vec<PooledEffect> = rows
        .iter()
        .filter(|r| &*r.section == crate::lr_activity::ligand_effect::tissue::ALL_SECTIONS)
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

/// The share of `v` that holds, or `-` for none to take it over.
fn share(v: &[bool]) -> String {
    if v.is_empty() {
        "-".into()
    } else {
        format!(
            "{:.2}",
            v.iter().filter(|&&b| b).count() as f64 / v.len() as f64
        )
    }
}

fn est(e: &Estimate) -> (f64, f64, f64) {
    (e.theta, e.se, e.crit())
}

/// (estimate, se, two-sided 5% critical value of its t reference).
type View = fn(&SimRow) -> (f64, f64, f64);

/// The lines of a scenario, in print order.
const VIEWS: [(&str, View); 6] = [
    ("estimate", |r| est(&r.estimate)),
    ("crude", |r| est(&r.crude)),
    ("ipw", |r| est(&r.ipw)),
    ("section", |r| est(&r.steps[0])),
    ("spatial", |r| est(&r.steps[1])),
    ("identity", |r| est(&r.steps[2])),
];

/// What the columns mean, and the column headings.
fn sim_legend() -> String {
    let mut out = String::from(
        "`truth` is the realized effect contrasted between the arms the estimator \
         sees, averaged over thresholds by its weights; `_x` columns use the mean \
         effect among the truly exposed. \
         `estim` is the share of runs with a reported estimate; `reject` the share \
         significant at 5%: type I error where nothing is planted (or the field is \
         shifted), power elsewhere; `atten` mean estimate / mean effect among the \
         exposed under a planted effect. section/spatial/identity are the partial adjustments.\n",
    );
    out += &format!(
        "  {:<9} {:>5}  {:>7} {:>7}  {:>5} {:>7}  {:<6} {:>6}  {:>6}\n",
        "", "estim", "bias", "bias_x", "cover", "cover_x", "rate", "reject", "atten"
    );
    out
}

/// The share of `g` whose `e` rejects, among those with an estimate.
fn reject_share(g: &[&SimRow], e: fn(&SimRow) -> Estimate) -> String {
    let v: Vec<bool> = g
        .iter()
        .map(|r| e(r))
        .filter(|e| e.theta.is_finite())
        .map(|e| e.rejects())
        .collect();
    share(&v)
}

/// A scenario's title line: runs, misclassified exposure, and the controls'
/// rejection and the verdict.
fn scenario_title(key: &str, g: &[&SimRow]) -> String {
    let miscl = g.iter().map(|r| r.misclassified).sum::<f64>() / g.len() as f64;
    format!(
        "{key}: {} runs, misclassified exposure {miscl:.2}, ring reject {}, no-spill reject {}, background reject {}, verdict survived {}\n",
        g.len(),
        reject_share(g, |r| r.ring),
        reject_share(g, |r| r.no_spill),
        reject_share(g, |r| r.background),
        share(&g.iter().map(|r| r.verdict.survived()).collect::<Vec<_>>()),
    )
}

/// One estimator's line in a scenario: estimable share, bias and coverage
/// against both truths, rejection rate and attenuation.
fn view_line(name: &str, view: View, g: &[&SimRow], null: bool) -> String {
    let ok: Vec<(f64, f64, f64, f64, f64)> = g
        .iter()
        .filter_map(|r| {
            let (th, se, crit) = view(r);
            (th.is_finite() && se.is_finite() && crit.is_finite() && r.truth.is_finite())
                .then_some((th, se, r.truth, r.truth_exposed, crit))
        })
        .collect();
    let estim = ok.len() as f64 / g.len() as f64;
    if ok.is_empty() {
        return format!("  {name:<9} {estim:>5.2}  (not estimable)\n");
    }
    let n = ok.len() as f64;
    let bias = ok.iter().map(|o| o.0 - o.2).sum::<f64>() / n;
    let wv: Vec<f64> = ok
        .iter()
        .filter(|o| o.3.is_finite())
        .map(|o| o.0 - o.3)
        .collect();
    let bias_x = if wv.is_empty() {
        f64::NAN
    } else {
        wv.iter().sum::<f64>() / wv.len() as f64
    };
    let cover: Vec<bool> = ok.iter().map(|o| (o.0 - o.2).abs() <= o.4 * o.1).collect();
    let cover_x: Vec<bool> = ok
        .iter()
        .filter(|o| o.3.is_finite())
        .map(|o| (o.0 - o.3).abs() <= o.4 * o.1)
        .collect();
    let reject: Vec<bool> = ok.iter().map(|o| (o.0 / o.1).abs() > o.4).collect();
    let mean_exposed = ok.iter().map(|o| o.3).sum::<f64>() / n;
    let atten = if null || mean_exposed.is_nan() || mean_exposed.abs() < 1e-12 {
        "-".to_string()
    } else {
        format!(
            "{:.2}",
            ok.iter().map(|o| o.0).sum::<f64>() / n / mean_exposed
        )
    };
    format!(
        "  {name:<9} {estim:>5.2}  {bias:>+7.3} {bias_x:>+7.3}  {:>5} {:>7}  {:<6} {:>6}  {atten:>6}\n",
        share(&cover),
        share(&cover_x),
        if null { "type I" } else { "power" },
        share(&reject),
    )
}

/// Per scenario: a title line (runs, misclassified exposure, ring and
/// no-spill rejection), then one line each for the estimate, the crude
/// contrast and the partial adjustments, with estimable share, bias and
/// coverage against both truths, rejection rate (type I error or power)
/// and, under a planted effect, attenuation. Then the comparison
/// ([`super::compare::comparison`]).
pub fn sim_summary(rows: &[SimRow]) -> String {
    let mut out = sim_legend();
    for (key, idx) in scenario_groups(rows, |r| &r.scenario) {
        let g: Vec<&SimRow> = idx.iter().map(|&i| &rows[i]).collect();
        let null = is_null(&g[0].scenario);
        out += &scenario_title(&key, &g);
        for (name, view) in VIEWS {
            out += &view_line(name, view, &g, null);
        }
    }
    out.push_str(&comparison(rows));
    out
}

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
