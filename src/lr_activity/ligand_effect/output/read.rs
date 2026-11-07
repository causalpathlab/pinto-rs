//! Reading the tables back, for the viewer and for `pinto run`.

use super::compare::SimRun;
use super::summary::PooledEffect;
use super::write::sim_cols;
use crate::lr_activity::ligand_effect::simulate::{
    ConfoundShape, ConfounderKind, Scenario, COMPARED,
};
use crate::lr_activity::ligand_effect::verdict::Verdict;

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
    /// The exposure threshold τ the pair used.
    pub threshold: f64,
    pub n_receivers: usize,
    pub n_competent: usize,
    pub n_exposed: usize,
    /// The negative controls the effect has to exceed; `None` for a table
    /// from before these columns.
    pub controls: Option<Controls>,
}

/// A negative control: its effect (log rate ratio) and SE, and the q of the
/// headline exceeding it, empty unless the headline is significant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Control {
    pub effect: f64,
    pub se: f64,
    pub exceeded_q: f64,
}

/// The controls of a pooled row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Controls {
    pub receptor_negative: Control,
    pub distant: Control,
    pub background: Control,
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
    // Each control's effect, SE and q of being exceeded; older tables have none.
    let controls = read_table_columns(
        path,
        &[],
        &[
            "effect_incompetent",
            "effect_incompetent_se",
            "effect_contrast_q",
            "ring_effect",
            "ring_effect_se",
            "effect_over_ring_q",
            "background_effect",
            "background_effect_se",
            "effect_over_background_q",
        ],
    )
    .ok()
    .map(|(_, c)| c);
    let control = |c: &[Vec<f64>], i: usize, first: usize| Control {
        effect: c[first][i],
        se: c[first + 1][i],
        exceeded_q: c[first + 2][i],
    };
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
                controls: controls.as_deref().map(|c| Controls {
                    receptor_negative: control(c, i, 0),
                    distant: control(c, i, 3),
                    background: control(c, i, 6),
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
const SIM_NUMBERS: [&str; 12] = {
    use sim_cols::*;
    [
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
    ]
};

/// The table keeps f32: back to the value as written (0.3, not
/// 0.30000001), so a scenario reads as the run's summary words it.
fn as_written(v: f64) -> f64 {
    let v = v as f32;
    v.to_string().parse().unwrap_or(f64::from(v))
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
    background_confounding: Option<Vec<f64>>,
    generic_shift: Option<Vec<f64>>,
    dose_response: Option<Vec<f64>>,
    dose_midpoint: Option<Vec<f64>>,
    dose_order: Option<Vec<f64>>,
}

impl<'p> SimTable<'p> {
    fn read(path: &'p str) -> anyhow::Result<Self> {
        use sim_cols::*;
        let (strings, numbers) = legume_numeric::matrix::parquet::read_table_columns(
            path,
            &[CONFOUNDER, SHAPE],
            &SIM_NUMBERS,
        )?;
        let later = |name: &str| later_column(path, name);
        Ok(Self {
            path,
            strings,
            numbers,
            rejects: COMPARED.iter().map(|m| later(&reject(m))).collect(),
            background_confounding: later(BACKGROUND_CONFOUNDING),
            generic_shift: later(GENERIC_SHIFT),
            dose_response: later(DOSE_RESPONSE),
            dose_midpoint: later(DOSE_MIDPOINT),
            dose_order: later(DOSE_ORDER),
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
        let plain = Scenario::default();
        let confounder: ConfounderKind = self.strings[0][i]
            .parse()
            .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
        let shape: ConfoundShape = self.strings[1][i]
            .parse()
            .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
        Ok(Scenario {
            effect: self.num(PLANTED_EFFECT, i),
            ring_effect: self.num(RING_EFFECT, i),
            confound_ligand: self.num(CONFOUND_LIGAND, i),
            confound_response: self.num(CONFOUND_RESPONSE, i),
            scale: self.num(SCALE, i),
            confounder,
            shape,
            heterogeneous: self.flag(HETEROGENEOUS, i),
            marker: self.flag(MARKER, i),
            spillover: self.num(SPILLOVER, i),
            shift: self.num(SHIFT, i),
            confound_receptor: self.flag(CONFOUND_RECEPTOR, i),
            rep: self.col(REP)[i] as usize,
            background_confounding: later_value(
                &self.background_confounding,
                i,
                plain.background_confounding,
            ),
            generic_shift: later_value(&self.generic_shift, i, plain.generic_shift),
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
/// each column by its name ([`super::sim_cols`]).
pub fn read_sim_runs(path: &str) -> anyhow::Result<Vec<SimRun>> {
    let table = SimTable::read(path)?;
    (0..table.strings[0].len()).map(|i| table.run(i)).collect()
}
