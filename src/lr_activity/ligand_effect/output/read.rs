//! Reading the tables back, for the viewer and for `pinto run`.

use super::compare::SimRun;
use super::summary::PooledEffect;
use super::write::sim_cols;
use crate::lr_activity::ligand_effect::simulate::{
    ConfoundShape, ConfounderKind, Scenario, COMPARED,
};
use crate::lr_activity::ligand_effect::verdict::Verdict;

/// The effect against one negative control: the effect minus the
/// control's (log rate ratio), and the two-sided q of that difference, so a
/// small q can also mean the effect falls short of the control. Tested
/// among the pairs with an effect, so NaN for the rest, for a control that
/// could not be checked, and in a table from before these columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Control {
    pub over: f64,
    pub q: f64,
}

/// One negative control as the viewer reads it.
pub struct ControlSpec {
    /// Its name in the panel.
    pub short: &'static str,
    /// Its name for agents.
    pub key: &'static str,
    /// The columns of the effect minus it, and of that difference's q.
    pub column: &'static str,
    pub column_q: &'static str,
}

/// The negative controls of [`PairRow::controls`], in its order.
pub const CONTROLS: [ControlSpec; 1] = [ControlSpec {
    short: "receptor−",
    key: "receptor_negative",
    column: "effect_contrast",
    column_q: "effect_contrast_q",
}];

/// A pooled row of `{out}.lra.parquet` as the viewer lists it.
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
    /// The headline exposure threshold τ: the one the counts and the
    /// sensitivity fits read.
    pub threshold: f64,
    /// The threshold the average over thresholds leans on most, and its
    /// weight; NaN in a table from before the average.
    pub tau_top: f64,
    pub tau_top_weight: f64,
    /// The effect against each negative control, in [`CONTROLS`] order.
    pub controls: [Control; 1],
    pub n_receivers: usize,
    pub n_competent: usize,
    pub n_exposed: usize,
}

impl PairRow {
    /// `LIGAND:RECEPTOR`.
    pub fn name(&self) -> String {
        format!("{}:{}", self.ligand, self.receptor)
    }
}

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
/// pairs its q-values were corrected over (its `q_scope`: [`super::Q_SCOPE_ALL`],
/// or a `--only` subset's words); a table from before the column says
/// nothing.
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
    // Tables from before the average over thresholds and the controls'
    // columns have none.
    let rows = s[0].len();
    let [tau_top, tau_top_weight] = &later_columns(path, &["tau_top", "tau_top_weight"], rows)[..]
    else {
        unreachable!("two columns asked, two read")
    };
    let control_columns: Vec<&str> = CONTROLS
        .iter()
        .flat_map(|c| [c.column, c.column_q])
        .collect();
    let controls = later_columns(path, &control_columns, rows);
    // Each control's difference and its q.
    let controls: Vec<&[Vec<f64>]> = controls.chunks(2).collect();
    let count = |v: f64| {
        if v.is_finite() && v > 0. {
            v as usize
        } else {
            0
        }
    };
    let rows = (0..s[0].len())
        .filter(|&i| &*s[0][i] == crate::lr_activity::ligand_effect::tissue::ALL_SECTIONS)
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
                tau_top: tau_top[i],
                tau_top_weight: tau_top_weight[i],
                controls: std::array::from_fn(|c| Control {
                    over: controls[c][0][i],
                    q: controls[c][1][i],
                }),
            })
        })
        .collect::<anyhow::Result<_>>()?;
    Ok((rows, q_scope))
}

/// The pooled rows of `{out}.lra.parquet` (section `all`) as a summary
/// reads them, in table order.
pub fn read_pooled_effects(path: &str) -> anyhow::Result<Vec<PooledEffect>> {
    Ok(read_pair_rows(path)?
        .0
        .iter()
        .map(PooledEffect::from)
        .collect())
}

/// One point of a pair's threshold–response curve, as
/// `{out}.lra.curve.parquet` holds it (log rate ratios).
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
pub type Curves = std::collections::HashMap<(Box<str>, Box<str>), Vec<CurveRow>>;

/// Every pair's curve, by `(ligand, receptor)`, points in table order.
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

/// The number columns every `{out}.lra-simulate.parquet` has.
const SIM_NUMBERS: [&str; 8] = {
    use sim_cols::*;
    [
        EFFECT_PVE,
        LIGAND_FIELD_PVE,
        RESPONSE_FIELD_PVE,
        SCALE,
        HETEROGENEOUS,
        MARKER,
        REP,
        ESTIMABLE,
    ]
};

/// The table keeps f32: back to the value as written (0.3, not
/// 0.30000001), so a scenario reads as the run's summary words it.
fn as_written(v: f64) -> f64 {
    let v = v as f32;
    v.to_string().parse().unwrap_or(f64::from(v))
}

/// The columns `names` a table from an earlier version may lack, in one
/// read when it has them all, else one by one, each missing one NaN.
fn later_columns(path: &str, names: &[&str], rows: usize) -> Vec<Vec<f64>> {
    legume_numeric::matrix::parquet::read_table_columns(path, &[], names)
        .map(|(_, x)| x)
        .unwrap_or_else(|_| {
            names
                .iter()
                .map(|n| later_column(path, n).unwrap_or_else(|| vec![f64::NAN; rows]))
                .collect()
        })
}

/// A column a table from an earlier version lacks.
fn later_column(path: &str, name: &str) -> Option<Vec<f64>> {
    legume_numeric::matrix::parquet::read_table_columns(path, &[], &[name])
        .ok()
        .and_then(|(_, mut c)| c.pop())
}

/// A later column's value at row `i`, or the scenario's default without it.
fn later_value(c: &Option<Vec<f64>>, i: usize, default: f64) -> f64 {
    c.as_ref().map_or(default, |c| as_written(c[i]))
}

/// `{out}.lra-simulate.parquet` as read: confounder and shape, [`SIM_NUMBERS`],
/// and the columns a table from an earlier version lacks.
struct SimTable<'p> {
    path: &'p str,
    strings: Vec<Vec<Box<str>>>,
    numbers: Vec<Vec<f64>>,
    rejects: Vec<Option<Vec<f64>>>,
    dose_response: Option<Vec<f64>>,
    dose_midpoint: Option<Vec<f64>>,
    dose_order: Option<Vec<f64>>,
    /// Columns of scenarios made for a control since removed.
    retired: Vec<Vec<f64>>,
}

/// Columns an earlier version wrote for scenarios that tested a control since
/// removed. A run with one set is not the plain scenario its other columns
/// describe, so it is left out. `background_confounding` is not here, since
/// every earlier run wrote it.
const RETIRED_SCENARIO_COLUMNS: [&str; 4] =
    ["planted_ring_effect", "generic_shift", "spillover", "shift"];

impl<'p> SimTable<'p> {
    fn read(path: &'p str) -> anyhow::Result<Self> {
        use sim_cols::*;
        let later = |name: &str| later_column(path, name);
        anyhow::ensure!(
            later("planted_effect").is_none(),
            "{path} is from before effect sizes were set by PVE: run lra-simulate again"
        );
        anyhow::ensure!(
            later("crude_reject").is_none(),
            "{path} is from before `crude` was renamed `unadjusted_diff`: run lra-simulate again"
        );
        let (strings, numbers) = legume_numeric::matrix::parquet::read_table_columns(
            path,
            &[CONFOUNDER, SHAPE],
            &SIM_NUMBERS,
        )?;
        Ok(Self {
            path,
            strings,
            numbers,
            rejects: COMPARED.iter().map(|m| later(&reject(m))).collect(),
            dose_response: later(DOSE_RESPONSE),
            dose_midpoint: later(DOSE_MIDPOINT),
            dose_order: later(DOSE_ORDER),
            retired: RETIRED_SCENARIO_COLUMNS
                .iter()
                .filter_map(|c| later(c))
                .collect(),
        })
    }

    /// Whether row `i` is of a scenario made for a control since removed.
    fn retired(&self, i: usize) -> bool {
        self.retired.iter().any(|c| c[i] != 0.0)
    }

    fn col(&self, name: &str) -> &[f64] {
        &self.numbers[SIM_NUMBERS
            .iter()
            .position(|&n| n == name)
            .expect("a column read")]
    }

    fn num(&self, name: &str, i: usize) -> f64 {
        as_written(self.col(name)[i])
    }

    fn flag(&self, name: &str, i: usize) -> bool {
        self.col(name)[i] != 0.0
    }

    /// Row `i`'s scenario.
    fn scenario(&self, i: usize) -> anyhow::Result<Scenario> {
        use sim_cols::*;
        let path = self.path;
        let plain = Scenario::default();
        let confounder: ConfounderKind = self.strings[0][i]
            .parse()
            .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
        let shape: ConfoundShape = self.strings[1][i]
            .parse()
            .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
        Ok(Scenario {
            effect_pve: self.num(EFFECT_PVE, i),
            ligand_field_pve: self.num(LIGAND_FIELD_PVE, i),
            response_field_pve: self.num(RESPONSE_FIELD_PVE, i),
            scale: self.num(SCALE, i),
            confounder,
            shape,
            heterogeneous: self.flag(HETEROGENEOUS, i),
            marker: self.flag(MARKER, i),
            rep: self.col(REP)[i] as usize,
            dose_response: later_value(&self.dose_response, i, 0.0) != 0.0,
            dose_midpoint: later_value(&self.dose_midpoint, i, plain.dose_midpoint),
            dose_order: later_value(&self.dose_order, i, plain.dose_order as f64) as i32,
        })
    }

    /// Row `i` as a run.
    fn run(&self, i: usize) -> anyhow::Result<SimRun> {
        let scenario = self.scenario(i)?;
        let mut hits = [f64::NAN; COMPARED.len()];
        for (h, c) in hits.iter_mut().zip(&self.rejects) {
            if let Some(c) = c {
                *h = (c[i] != 0.0) as u8 as f64;
            }
        }
        Ok(SimRun {
            scenario,
            estimable: self.flag(sim_cols::ESTIMABLE, i),
            rejects: hits,
        })
    }
}

/// The runs of `{out}.lra-simulate.parquet` as [`super::compare_runs`] reads them,
/// each column by its name ([`super::sim_cols`]), without the runs of
/// scenarios made for a control since removed.
pub fn read_sim_runs(path: &str) -> anyhow::Result<Vec<SimRun>> {
    let table = SimTable::read(path)?;
    (0..table.strings[0].len())
        .filter(|&i| !table.retired(i))
        .map(|i| table.run(i))
        .collect()
}
