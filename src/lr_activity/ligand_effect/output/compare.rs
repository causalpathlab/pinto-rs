//! Comparing methods across simulated runs, scenario by scenario.

use crate::lr_activity::ligand_effect::simulate::{SimRow, COMPARED};

/// One benchmark run as the comparison reads it: its scenario, whether the
/// estimate was reported, and whether each method of
/// [`super::super::simulate::COMPARED`] rejects at 5%.
#[derive(Clone, Copy, Debug)]
pub struct SimRun {
    pub scenario: crate::lr_activity::ligand_effect::simulate::Scenario,
    pub estimable: bool,
    /// Per method of [`super::super::simulate::COMPARED`]: 1 when it rejects,
    /// else 0; NaN when a table from an earlier version lacks the method.
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
    /// Nothing planted: the rates are type I error.
    pub null: bool,
    #[cfg(feature = "view")]
    pub runs: usize,
    /// Share of runs with a reported estimate.
    pub estimable: f64,
    /// Share of runs rejecting at 5%, per method of [`super::super::simulate::COMPARED`].
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
        } else if s.ligand_field_pve != 0.0 || s.response_field_pve != 0.0 {
            nulls.push(rates);
        }
        scenarios.push(ScenarioRates {
            scenario: key,
            null,
            #[cfg(feature = "view")]
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

/// The methods side by side: per scenario, each one's share of runs rejecting
/// at 5% ([`super::super::simulate::COMPARED`]; a run without the statistic,
/// such as an estimate without overlap, counts as not rejecting), the estimable
/// share, then the mean over scenarios of type I error under the confounded
/// nulls and of power under the planted effects, and the baselines' share of
/// the run time.
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
        "\nComparison at 5%: share of runs rejecting (type I where nothing is planted, \
         power elsewhere). celltype: cell-type label permutation \
         (CellPhoneDB/CellChat); moran: bivariate Moran's I (SpatialDM); contact: contact \
         co-expression score (NICHES/COMMOT); coexpr: lra's co-expression log odds.\n  \
         {:<width$} {:<6} {:>5} ",
        "scenario", "rate", "estim"
    );
    // Each method's column is as wide as its name, at least 8.
    let col: Vec<usize> = COMPARED.iter().map(|m| m.len().max(8)).collect();
    for (m, w) in COMPARED.iter().zip(&col) {
        out += &format!(" {m:>w$}");
    }
    out += "\n";
    for s in &c.scenarios {
        out += &format!(
            "  {:<width$} {:<6} {:>5.2} ",
            s.scenario,
            if s.null { "type I" } else { "power" },
            s.estimable
        );
        for (x, w) in s.rates.iter().zip(&col) {
            out += &format!(" {x:>w$.2}");
        }
        out += "\n";
    }
    for (label, n, m) in &c.means {
        let head = format!("{label} ({n})");
        out += &format!("  {head:<w$} ", w = width + 14);
        for (x, w) in m.iter().zip(&col) {
            out += &format!(" {x:>w$.2}");
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

/// A scenario in words, without its replicate: the summary's grouping key.
fn describe(s: &crate::lr_activity::ligand_effect::simulate::Scenario) -> String {
    let mut d = format!("effect PVE={}", s.effect_pve);
    if s.ligand_field_pve != 0.0 || s.response_field_pve != 0.0 {
        d += &format!(
            " field PVE ligand={} response={} {}/{}@{}",
            s.ligand_field_pve,
            s.response_field_pve,
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
pub(super) fn scenario_groups<T>(
    items: &[T],
    scenario: impl Fn(&T) -> &crate::lr_activity::ligand_effect::simulate::Scenario,
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

/// A null scenario: nothing planted.
pub(super) fn is_null(s: &crate::lr_activity::ligand_effect::simulate::Scenario) -> bool {
    s.effect_pve == 0.0
}
