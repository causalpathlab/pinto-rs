//! What to simulate: the scenarios, the built-in grid, and the single
//! scenario a sweep asks for by its keys.

use serde::Deserialize as _;

/// The hidden confounder's spatial form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "&'static str")]
pub enum ConfounderKind {
    /// Random Fourier features, the adjustment smooth's own basis
    Fourier,
    /// Random Gaussian bumps of width `scale` contact lengths, random signs
    Bumps,
    /// A real gene outside the benchmarked pairs, smoothed by a Gaussian of
    /// width `scale` contact lengths
    RealGene,
}

impl ConfounderKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fourier => "fourier",
            Self::Bumps => "bumps",
            Self::RealGene => "realgene",
        }
    }
}

impl std::str::FromStr for ConfounderKind {
    type Err = anyhow::Error;

    /// A kind by its [`label`](Self::label).
    fn from_str(s: &str) -> anyhow::Result<Self> {
        [Self::Fourier, Self::Bumps, Self::RealGene]
            .into_iter()
            .find(|k| k.label() == s)
            .ok_or_else(|| anyhow::anyhow!("unknown confounder {s:?}"))
    }
}

impl TryFrom<String> for ConfounderKind {
    type Error = anyhow::Error;

    fn try_from(s: String) -> anyhow::Result<Self> {
        s.parse()
    }
}

impl From<ConfounderKind> for &'static str {
    fn from(k: ConfounderKind) -> Self {
        k.label()
    }
}

/// How the hidden field enters ligand production and response.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "&'static str")]
pub enum ConfoundShape {
    /// Z
    Linear,
    /// Z² − 1
    Quadratic,
}

impl ConfoundShape {
    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Quadratic => "quadratic",
        }
    }
}

impl std::str::FromStr for ConfoundShape {
    type Err = anyhow::Error;

    /// A shape by its [`label`](Self::label).
    fn from_str(s: &str) -> anyhow::Result<Self> {
        [Self::Linear, Self::Quadratic]
            .into_iter()
            .find(|k| k.label() == s)
            .ok_or_else(|| anyhow::anyhow!("unknown confound shape {s:?}"))
    }
}

impl TryFrom<String> for ConfoundShape {
    type Error = anyhow::Error;

    fn try_from(s: String) -> anyhow::Result<Self> {
        s.parse()
    }
}

impl From<ConfoundShape> for &'static str {
    fn from(k: ConfoundShape) -> Self {
        k.label()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Scenario {
    /// Share of each programme gene's log-rate variance (beyond the receiver's
    /// cell type) that the contact effect explains: the effect is
    /// `√pve / sd(x)` times the exposure term `x` (true exposure × the bin's
    /// share × the dose share), its variance taken over responding receivers
    /// within their identity bins. Effect, field and log-rate noise share a
    /// unit variance, so the PVE also sets the noise and hence power.
    pub effect_pve: f64,
    /// Share of the producer log odds' variance (beyond the cell type) that
    /// the hidden field explains; the rest is noise.
    pub ligand_field_pve: f64,
    /// Share of each programme gene's log-rate variance (beyond the cell
    /// type) that the hidden field explains, signed: negative pushes the
    /// response against the ligand. Effect, field and noise share the
    /// variance, so `effect_pve + |response_field_pve| ≤ 1`.
    pub response_field_pve: f64,
    /// Hidden field's length scale, in contact lengths.
    pub scale: f64,
    pub confounder: ConfounderKind,
    pub shape: ConfoundShape,
    /// The effect varies by identity bin, scaled by `u_b ~ U[0.5, 1.5]`.
    pub heterogeneous: bool,
    /// Ligand, response and programme baseline confined to a marker type.
    pub marker: bool,
    /// The planted effect grows with the true dose (a Hill curve) rather than
    /// being a step at any exposure.
    pub dose_response: bool,
    /// Where the dose response is half way, as a quantile of the positive true
    /// doses, and how sharply it rises (the Hill order).
    pub dose_midpoint: f64,
    pub dose_order: i32,
    /// The effect is planted only on receivers of this region code
    /// ([`crate::lr_activity::ligand_effect::tissue::Tissue::region`]), and read on that
    /// region's row beside the pooled one; `None` plants it everywhere.
    pub region: Option<u32>,
    pub rep: usize,
}

impl Scenario {
    /// Effect, field and noise share a unit variance: each PVE at least 0,
    /// the response's two summing to at most 1, the ligand's below 1. The
    /// field's scale is positive, the dose response's midpoint a quantile and
    /// its order at least 1.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.effect_pve >= 0.0
                && self.effect_pve + self.response_field_pve.abs() <= 1.0
                && (0.0..1.0).contains(&self.ligand_field_pve),
            "PVE out of range: effect {}, response field {}, ligand field {} (each a share \
             of variance; effect and |response field| sum to at most 1, ligand field below 1)",
            self.effect_pve,
            self.response_field_pve,
            self.ligand_field_pve
        );
        anyhow::ensure!(
            self.scale.is_finite() && self.scale > 0.0,
            "confound_scale {} is not a positive length",
            self.scale
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.dose_midpoint) && self.dose_order >= 1,
            "dose response out of range: midpoint {} (a quantile, 0 to 1), order {} (at least 1)",
            self.dose_midpoint,
            self.dose_order
        );
        Ok(())
    }
}

/// A scenario as `lra-simulate --set` names it: each key the
/// `{out}.lra-simulate.parquet` column that records it, so a sweep's axis is
/// a column of the table. Unset keys keep [`Scenario::default`]; the planted
/// region is `--plant-region`'s.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScenarioKeys {
    pub effect_pve: f64,
    pub ligand_field_pve: f64,
    pub response_field_pve: f64,
    pub confound_scale: f64,
    pub confounder: ConfounderKind,
    pub confound_shape: ConfoundShape,
    #[serde(deserialize_with = "flag")]
    pub heterogeneous: bool,
    #[serde(deserialize_with = "flag")]
    pub marker: bool,
    #[serde(deserialize_with = "flag")]
    pub dose_response: bool,
    pub dose_midpoint: f64,
    pub dose_order: i32,
    pub rep: usize,
}

/// A flag as given or as its 0/1 column records it.
fn flag<'de, D: serde::Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    use serde::de::Error;
    match serde_json::Value::deserialize(d)? {
        serde_json::Value::Bool(b) => Ok(b),
        serde_json::Value::Number(n) if n.as_u64() == Some(0) => Ok(false),
        serde_json::Value::Number(n) if n.as_u64() == Some(1) => Ok(true),
        v => Err(D::Error::custom(format!(
            "expected true, false, 0 or 1, got {v}"
        ))),
    }
}

impl Default for ScenarioKeys {
    fn default() -> Self {
        Self::from(&Scenario::default())
    }
}

impl From<&Scenario> for ScenarioKeys {
    fn from(s: &Scenario) -> Self {
        Self {
            effect_pve: s.effect_pve,
            ligand_field_pve: s.ligand_field_pve,
            response_field_pve: s.response_field_pve,
            confound_scale: s.scale,
            confounder: s.confounder,
            confound_shape: s.shape,
            heterogeneous: s.heterogeneous,
            marker: s.marker,
            dose_response: s.dose_response,
            dose_midpoint: s.dose_midpoint,
            dose_order: s.dose_order,
            rep: s.rep,
        }
    }
}

impl ScenarioKeys {
    /// Set keys by `key=value`, as `srt-simulate --set` sets its parameters
    /// ([`crate::simulate::set_fields`]).
    pub fn with_overrides(self, sets: &[String]) -> anyhow::Result<Self> {
        crate::simulate::set_fields(&self, sets, "scenario key")
    }

    /// The scenario, planted everywhere.
    pub fn scenario(&self) -> Scenario {
        Scenario {
            effect_pve: self.effect_pve,
            ligand_field_pve: self.ligand_field_pve,
            response_field_pve: self.response_field_pve,
            scale: self.confound_scale,
            confounder: self.confounder,
            shape: self.confound_shape,
            heterogeneous: self.heterogeneous,
            marker: self.marker,
            dose_response: self.dose_response,
            dose_midpoint: self.dose_midpoint,
            dose_order: self.dose_order,
            region: None,
            rep: self.rep,
        }
    }
}

/// `lra-simulate`'s single-scenario mode: one scenario, its effect planted on
/// `planted` pairs and withheld (effect 0, all else alike) on `null` further
/// pairs, so one call can be scored by AUROC and AUPRC.
#[derive(Clone, Debug)]
pub struct SingleScenario {
    pub scenario: Scenario,
    pub planted: usize,
    pub null: usize,
}

impl SingleScenario {
    /// Planted and null pairs a call asks for when not told.
    pub const PAIRS: usize = 10;

    /// The scenario `sets` makes of the defaults, checked
    /// ([`Scenario::validate`]), on `planted` and `null` pairs, at least
    /// one each.
    pub fn new(sets: &[String], planted: usize, null: usize) -> anyhow::Result<Self> {
        let scenario = ScenarioKeys::default().with_overrides(sets)?.scenario();
        scenario.validate()?;
        anyhow::ensure!(
            planted >= 1 && null >= 1,
            "a single-scenario call needs a planted and a null pair at least (asked {planted} \
             and {null})"
        );
        Ok(Self {
            scenario,
            planted,
            null,
        })
    }
}

impl Default for Scenario {
    /// No effect, no confounding, no departure. The confounder kind only
    /// matters once a field PVE is set: Fourier, the original field, unless
    /// a scenario names another (the grids name theirs).
    fn default() -> Self {
        Self {
            effect_pve: 0.0,
            ligand_field_pve: 0.0,
            response_field_pve: 0.0,
            scale: 32.0,
            confounder: ConfounderKind::Fourier,
            shape: ConfoundShape::Linear,
            heterogeneous: false,
            marker: false,
            dose_response: false,
            dose_midpoint: 0.5,
            dose_order: 2,
            region: None,
            rep: 0,
        }
    }
}

/// The built-in benchmark: the choices its grid is built from, its
/// replicates, and how many pairs it runs on.
#[derive(Debug)]
pub struct GridOpts {
    /// Planted effects as PVE of the programme genes' log rates
    /// ([`Scenario::effect_pve`]); each runs beside no effect.
    pub effect_pve: Vec<f64>,
    /// PVE of the hidden field on ligand production and on the response
    /// in confounded scenarios ([`Scenario::response_field_pve`]).
    pub field_pve: f64,
    /// Replicates of every scenario.
    pub reps: usize,
    /// Pairs benchmarked: the most abundant distinct ligands.
    pub pairs: usize,
    pub confounders: Vec<ConfounderKind>,
    /// Confounder length scales, in contact lengths; the first is the one
    /// the nonlinear shapes use.
    pub scales: Vec<f64>,
    pub shapes: Vec<ConfoundShape>,
    /// Ligand, response and programme confined to one cell type.
    pub marker: bool,
    /// An effect varying by identity bin (planted effects only).
    pub heterogeneous: bool,
    /// Planted effects grow with the true dose.
    pub dose_response: bool,
    /// Where it is half way, as a quantile of the positive true doses.
    pub dose_midpoint: f64,
    /// How sharply it rises (the Hill order).
    pub dose_order: i32,
}

impl Default for GridOpts {
    fn default() -> Self {
        Self {
            effect_pve: vec![0.02],
            field_pve: 0.25,
            reps: 3,
            pairs: 3,
            confounders: vec![ConfounderKind::Bumps, ConfounderKind::RealGene],
            scales: vec![32.0, 4.0],
            shapes: vec![ConfoundShape::Linear, ConfoundShape::Quadratic],
            marker: true,
            heterogeneous: true,
            dose_response: false,
            dose_midpoint: 0.5,
            dose_order: 2,
        }
    }
}

/// The grid: no effect and each planted one, each with
///
/// - no confounding;
/// - per confounder kind: linear confounding at every scale, same-sign and
///   opposite-sign (field PVE `field_pve`, response sign ±), and each nonlinear shape at the
///   first scale, same-sign;
/// - a heterogeneous effect (planted effects only);
/// - the marker type, unconfounded.
pub fn default_grid(g: &GridOpts) -> Vec<Scenario> {
    let mut out = Vec::new();
    let thetas = std::iter::once(0.0).chain(g.effect_pve.iter().copied().filter(|&t| t != 0.0));
    for theta in thetas {
        for rep in 0..g.reps {
            let plain = Scenario {
                effect_pve: theta,
                dose_response: g.dose_response && theta != 0.0,
                dose_midpoint: g.dose_midpoint,
                dose_order: g.dose_order,
                rep,
                ..Scenario::default()
            };
            out.push(plain);
            for &confounder in &g.confounders {
                let conf = Scenario {
                    confounder,
                    ligand_field_pve: g.field_pve,
                    response_field_pve: g.field_pve,
                    ..plain
                };
                for (k, &scale) in g.scales.iter().enumerate() {
                    for &shape in &g.shapes {
                        if shape == ConfoundShape::Linear {
                            out.push(Scenario { scale, ..conf });
                            out.push(Scenario {
                                scale,
                                response_field_pve: -g.field_pve,
                                ..conf
                            });
                        } else if k == 0 {
                            out.push(Scenario {
                                scale,
                                shape,
                                ..conf
                            });
                        }
                    }
                }
            }
            if theta != 0.0 && g.heterogeneous {
                out.push(Scenario {
                    heterogeneous: true,
                    ..plain
                });
            }
            if g.marker {
                out.push(Scenario {
                    marker: true,
                    ..plain
                });
            }
        }
    }
    out
}
