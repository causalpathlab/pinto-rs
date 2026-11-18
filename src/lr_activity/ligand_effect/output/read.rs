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
/// among the pairs with an effect, so NaN for the rest and for a control
/// that could not be checked.
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
    column: "effect_dr_contrast",
    column_q: "effect_dr_contrast_q",
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
    pub effect_dr_se: f64,
    pub effect_dr_p: f64,
    pub effect_dr_q: f64,
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
    /// weight.
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
            effect_dr_p: r.effect_dr_p,
            effect_dr_q: r.effect_dr_q,
            pct: r.pct,
            pct_lo: r.pct_lo,
            pct_hi: r.pct_hi,
        }
    }
}

/// The pooled rows of a contact-effect table, in table order, and which
/// pairs its q-values were corrected over (its `q_scope`: [`super::Q_SCOPE_ALL`],
/// or a `--only` subset's words).
pub fn read_pair_rows(path: &str) -> anyhow::Result<(Vec<PairRow>, Box<str>)> {
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
        "effect_dr",
        "effect_dr_se",
        "effect_dr_p",
        "effect_dr_q",
        "coexpr_q",
        "effect_dr_pct",
        "effect_dr_pct_lo",
        "effect_dr_pct_hi",
        "exposure_threshold",
        "n_receivers",
        "n_competent",
        "n_exposed",
        "tau_top",
        "tau_top_weight",
    ];
    // the controls' columns follow the named ones, tau_top's two last
    let n = numbers.len();
    let numbers: Vec<&str> = numbers
        .into_iter()
        .chain(CONTROLS.iter().flat_map(|c| [c.column, c.column_q]))
        .collect();
    let (s, mut x) = read_table_columns(path, &strings, &numbers)?;
    let q_scope = s[6]
        .first()
        .cloned()
        .unwrap_or_else(|| super::Q_SCOPE_ALL.into());
    // Each control's difference and its q, after tau_top and its weight.
    let controls = x.split_off(n);
    let controls: Vec<&[Vec<f64>]> = controls.chunks(2).collect();
    let (tau_top, tau_top_weight) = (&x[n - 2], &x[n - 1]);
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
                effect_dr_se: x[1][i],
                effect_dr_p: x[2][i],
                effect_dr_q: x[3][i],
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

/// The number columns of `{out}.lra-simulate.parquet` read back, besides
/// the rejections.
const SIM_NUMBERS: [&str; 11] = {
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
        DOSE_RESPONSE,
        DOSE_MIDPOINT,
        DOSE_ORDER,
    ]
};

/// The table keeps f32: back to the value as written (0.3, not
/// 0.30000001), so a scenario reads as the run's summary words it.
fn as_written(v: f64) -> f64 {
    let v = v as f32;
    v.to_string().parse().unwrap_or(f64::from(v))
}

/// `{out}.lra-simulate.parquet` as read: confounder and shape,
/// [`SIM_NUMBERS`] and each compared method's rejections.
struct SimTable<'p> {
    path: &'p str,
    strings: Vec<Vec<Box<str>>>,
    numbers: Vec<Vec<f64>>,
    rejects: Vec<Vec<f64>>,
    /// The `region` column.
    region: Vec<f64>,
}

impl<'p> SimTable<'p> {
    /// The table at `path`, in one read. A table from an earlier version
    /// lacks columns read here and is refused.
    fn read(path: &'p str) -> anyhow::Result<Self> {
        use anyhow::Context;
        use sim_cols::*;
        let reject_names = COMPARED.map(reject);
        let names: Vec<&str> = SIM_NUMBERS
            .into_iter()
            .chain(reject_names.iter().map(String::as_str))
            .chain([REGION])
            .collect();
        let (strings, mut numbers) =
            legume_numeric::matrix::parquet::read_table_columns(path, &[CONFOUNDER, SHAPE], &names)
                .with_context(|| {
                    format!("{path} is not a table this version of lra-simulate wrote")
                })?;
        let region = numbers.pop().expect("the region column read last");
        let rejects = numbers.split_off(SIM_NUMBERS.len());
        Ok(Self {
            path,
            strings,
            numbers,
            rejects,
            region,
        })
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
            dose_response: self.flag(DOSE_RESPONSE, i),
            dose_midpoint: self.num(DOSE_MIDPOINT, i),
            dose_order: self.col(DOSE_ORDER)[i] as i32,
            region: self.region[i].is_finite().then(|| self.region[i] as u32),
        })
    }

    /// Row `i` as a run.
    fn run(&self, i: usize) -> anyhow::Result<SimRun> {
        let scenario = self.scenario(i)?;
        Ok(SimRun {
            scenario,
            estimable: self.flag(sim_cols::ESTIMABLE, i),
            rejects: std::array::from_fn(|k| self.rejects[k][i] != 0.0),
        })
    }
}

/// The region the runs of `{out}.lra-simulate.parquet` were planted in, by
/// name; `None` where they were planted everywhere.
pub fn read_planted_region(path: &str) -> Option<Box<str>> {
    let (mut s, _) =
        legume_numeric::matrix::parquet::read_table_columns(path, &[sim_cols::PLANTED_REGION], &[])
            .ok()?;
    s.pop()?.into_iter().find(|n| !n.is_empty())
}

/// The runs of `{out}.lra-simulate.parquet` as [`super::compare_runs`] reads them,
/// each column by its name ([`super::sim_cols`]).
pub fn read_sim_runs(path: &str) -> anyhow::Result<Vec<SimRun>> {
    let table = SimTable::read(path)?;
    (0..table.strings[0].len()).map(|i| table.run(i)).collect()
}
