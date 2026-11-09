//! What to simulate: the scenarios, the default grid, and the study file.

/// The hidden confounder's spatial form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
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

/// How the hidden field enters ligand production and response.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
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

#[derive(Clone, Copy, Debug)]
pub struct Scenario {
    /// Share of each programme gene's log-rate variance (beyond the receiver's
    /// cell type) that the contact effect explains: the effect is
    /// `√pve / sd(x)` times the exposure term `x` (true exposure × the bin's
    /// share × the dose share), its variance taken over responding receivers
    /// within their identity bins. Effect, field and log-rate noise share a
    /// unit variance, so the PVE also sets the noise and hence power.
    pub effect_pve: f64,
    /// θ_ring: log-rate change in responding receivers whose ring partners
    /// make ligand; 0 keeps the effect at contact range. A log-scale size
    /// outside the PVE budget, as are `generic_shift` and the marker lift.
    pub ring_effect: f64,
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
    /// κ: share of each contact neighbour's counts measured in the receiver.
    pub spillover: f64,
    /// Displacement of the ligand field the estimator sees, in contact
    /// lengths; 0 for none.
    pub shift: f64,
    /// How much of the field's push on the response reaches the background
    /// programme: 1 the same push, 0 none (its field PVE is this squared).
    pub background_confounding: f64,
    /// The planted effect grows with the true dose (a Hill curve) rather than
    /// being a step at any exposure.
    pub dose_response: bool,
    /// Where the dose response is half way, as a quantile of the positive true
    /// doses, and how sharply it rises (the Hill order).
    pub dose_midpoint: f64,
    pub dose_order: i32,
    /// γ: each cell has a hidden trait that raises its ligand and, by γ per
    /// unit of its contacts' mean trait, every gene of its competent contacts,
    /// background programmes included. It acts at contact range on every
    /// programme, so neither the receptor-negative receivers nor distant cells
    /// can show it; 0 for none.
    pub generic_shift: f64,
    pub rep: usize,
}

impl Scenario {
    /// The field's share of a background programme's log-rate variance:
    /// the response field's, scaled by `background_confounding²`.
    pub fn background_field_pve(&self) -> f64 {
        self.background_confounding.powi(2) * self.response_field_pve
    }

    /// Effect, field and noise share a unit variance: each PVE at least 0,
    /// the response's two summing to at most 1, the ligand's below 1.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.effect_pve >= 0.0
                && self.effect_pve + self.response_field_pve.abs() <= 1.0
                && (0.0..1.0).contains(&self.ligand_field_pve)
                && self.background_field_pve().abs() <= 1.0,
            "PVE out of range: effect {}, response field {}, ligand field {}, background \
             confounding {} (each a share of variance; effect and |response field| sum to at \
             most 1, ligand field below 1, background confounding² × |response field| at most 1)",
            self.effect_pve,
            self.response_field_pve,
            self.ligand_field_pve,
            self.background_confounding
        );
        Ok(())
    }
}

impl Default for Scenario {
    /// No effect, no confounding, no departure. The confounder kind only
    /// matters once a field PVE is set: Fourier, the original field, unless
    /// a scenario names another (the grids name theirs).
    fn default() -> Self {
        Self {
            effect_pve: 0.0,
            ring_effect: 0.0,
            ligand_field_pve: 0.0,
            response_field_pve: 0.0,
            scale: 32.0,
            confounder: ConfounderKind::Fourier,
            shape: ConfoundShape::Linear,
            heterogeneous: false,
            marker: false,
            spillover: 0.0,
            shift: 0.0,
            background_confounding: 1.0,
            dose_response: false,
            dose_midpoint: 0.5,
            dose_order: 2,
            generic_shift: 0.0,
            rep: 0,
        }
    }
}

/// The choices a benchmark grid is built from: the `scenarios` section of
/// a study. Fields left out of the JSON keep their defaults; unknown ones
/// are an error, so a misspelt field does not silently fall back.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GridOpts {
    #[serde(rename = "_doc", skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    /// Planted effects as PVE of the programme genes' log rates
    /// ([`Scenario::effect_pve`]); each runs beside no effect.
    pub effect_pve: Vec<f64>,
    /// PVE of the hidden field on ligand production and on the response
    /// in confounded scenarios ([`Scenario::response_field_pve`]).
    pub field_pve: f64,
    /// Replicates of every scenario (set from the study's `reps`).
    #[serde(skip)]
    pub reps: usize,
    pub confounders: Vec<ConfounderKind>,
    /// Confounder length scales, in contact lengths; the first is the one
    /// the nonlinear shapes and the shifted field use.
    pub scales: Vec<f64>,
    pub shapes: Vec<ConfoundShape>,
    /// Segmentation spillover shares κ, each its own unconfounded scenario.
    pub spillover: Vec<f64>,
    /// Displacements of the ligand field the estimator sees, in contact
    /// lengths, each with the first confounder at the first scale.
    pub shift: Vec<f64>,
    /// Ligand, response and programme confined to one cell type.
    pub marker: bool,
    /// An effect varying by identity bin (planted effects only).
    pub heterogeneous: bool,
    /// Planted effects that also reach ring range (θ_ring), each its own
    /// unconfounded scenario.
    pub ring_effect: Vec<f64>,
    /// How much of the field's push on the response reaches the background
    /// programme: 1 the same push, 0 none (its field PVE is this squared).
    pub background_confounding: f64,
    /// Planted effects grow with the true dose.
    pub dose_response: bool,
    /// Where it is half way, as a quantile of the positive true doses.
    pub dose_midpoint: f64,
    /// How sharply it rises (the Hill order).
    pub dose_order: i32,
    /// Each a scenario with a generic contact-range shift of this size.
    pub generic_shift: Vec<f64>,
}

impl Default for GridOpts {
    fn default() -> Self {
        Self {
            doc: None,
            effect_pve: vec![0.02],
            field_pve: 0.25,
            reps: 5,
            confounders: vec![ConfounderKind::Bumps, ConfounderKind::RealGene],
            scales: vec![32.0, 4.0],
            shapes: vec![ConfoundShape::Linear, ConfoundShape::Quadratic],
            spillover: vec![0.1],
            shift: vec![4.0],
            marker: true,
            heterogeneous: true,
            ring_effect: vec![0.3],
            background_confounding: 1.0,
            dose_response: false,
            dose_midpoint: 0.5,
            dose_order: 2,
            generic_shift: Vec::new(),
        }
    }
}

/// Scenario keys of studies from before sizes were shares of variance.
const RETIRED_SCENARIO_KEYS: [&str; 2] = ["effect_sizes", "confound_receptor"];

/// The grid: no effect and each planted one, each with
///
/// - no confounding;
/// - per confounder kind: linear confounding at every scale, same-sign and
///   opposite-sign (field PVE `field_pve`, response sign ±), and each nonlinear shape at the
///   first scale, same-sign;
/// - a heterogeneous effect (planted effects only);
/// - unconfounded segmentation spillover at each κ, and an effect that also
///   reaches ring range at each θ_ring;
/// - the first confounder at the first scale, seen shifted by each shift;
/// - the marker type, unconfounded.
pub fn default_grid(g: &GridOpts) -> Vec<Scenario> {
    let mut out = Vec::new();
    let first_scale = g.scales.first().copied().unwrap_or(32.0);
    let thetas = std::iter::once(0.0).chain(g.effect_pve.iter().copied().filter(|&t| t != 0.0));
    for theta in thetas {
        for rep in 0..g.reps {
            let plain = Scenario {
                effect_pve: theta,
                background_confounding: g.background_confounding,
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
            for &spillover in &g.spillover {
                out.push(Scenario { spillover, ..plain });
            }
            for &generic_shift in &g.generic_shift {
                out.push(Scenario {
                    generic_shift,
                    ..plain
                });
            }
            for &ring_effect in &g.ring_effect {
                out.push(Scenario {
                    ring_effect,
                    ..plain
                });
            }
            if let Some(&confounder) = g.confounders.first() {
                for &shift in &g.shift {
                    out.push(Scenario {
                        confounder,
                        ligand_field_pve: g.field_pve,
                        response_field_pve: g.field_pve,
                        scale: first_scale,
                        shift,
                        ..plain
                    });
                }
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

/// A `pinto lra-simulate` study: the scenario grid, its replicates, and how
/// many pairs (the most abundant distinct ligands) it runs on.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Study {
    #[serde(rename = "_doc", skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    /// Replicates of every scenario.
    pub reps: usize,
    /// Pairs benchmarked: the most abundant distinct ligands.
    pub pairs: usize,
    pub scenarios: GridOpts,
}

impl Default for Study {
    fn default() -> Self {
        Self {
            doc: None,
            reps: 3,
            pairs: 3,
            scenarios: GridOpts::default(),
        }
    }
}

impl Study {
    /// The grid options with the study's replicates.
    pub fn grid(&self) -> GridOpts {
        GridOpts {
            reps: self.reps,
            ..self.scenarios.clone()
        }
    }

    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let value: serde_json::Value = serde_json::from_str(text)?;
        if let Some(k) = RETIRED_SCENARIO_KEYS
            .iter()
            .find(|k| value["scenarios"].get(**k).is_some())
        {
            anyhow::bail!(
                "scenarios.{k} is retired: effect and field sizes are now shares of variance \
                 explained (effect_pve, field_pve), not log-rate sizes; see the example study"
            );
        }
        let s: Study = serde_json::from_value(value)?;
        anyhow::ensure!(s.reps > 0, "reps must be at least 1");
        anyhow::ensure!(s.pairs > 0, "pairs must be at least 1");
        anyhow::ensure!(
            s.scenarios.scales.iter().all(|&x| x > 0.0),
            "scenarios.scales must be positive contact lengths"
        );
        for scen in default_grid(&s.grid()) {
            scen.validate()
                .map_err(|e| anyhow::anyhow!("scenarios: {e}"))?;
        }
        Ok(s)
    }

    pub fn read(path: &str) -> anyhow::Result<Self> {
        let text =
            std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("--study {path}: {e}"))?;
        Self::from_json(&text).map_err(|e| anyhow::anyhow!("--study {path}: {e}"))
    }

    /// The built-in study as JSON, each section explained in its `_doc`
    /// (JSON has no comments). It parses back to [`Study::default`].
    pub fn example_json() -> String {
        let study = Study {
            doc: Some(
                "pinto lra-simulate study. Plants ligand effects of known size on this \
                 tissue, with and without hidden spatial confounders, and checks whether \
                 lra finds them (power) and stays quiet when there is nothing (type I \
                 error). reps: replicates of every scenario. pairs: how many ligand-receptor \
                 pairs to plant on (the most abundant distinct ligands). Fields left out \
                 keep these values; _doc fields are ignored."
                    .into(),
            ),
            scenarios: GridOpts {
                doc: Some(
                    "Strengths are PVE, shares of a programme gene's log-rate variance beyond \
                     its cell type (generalized linear model): effect, field and noise sum to 1. \
                     Each effect runs beside no effect. effect_pve: the contact effect's share. \
                     field_pve: the hidden field's share, on ligand production (log odds) and \
                     on the response, in confounded scenarios. ring_effect and generic_shift \
                     are log-scale sizes outside that budget. confounders: hidden \
                     field kinds (bumps, realgene, fourier). scales: their length scales in \
                     contact lengths. shapes: how the field acts (linear, quadratic). \
                     spillover: shares of a neighbour's counts bleeding into \
                     the receiver. shift: displacement of the ligand field, in contact \
                     lengths (a null). marker: effect confined to one cell type. \
                     heterogeneous: effect varying by cell type. ring_effect: effect sizes \
                     reaching distant, non-touching cells. background_confounding: how \
                     much of the field's push on the response reaches the background programme \
                     (1 the same push, 0 none). dose_response: planted effects grow with the \
                     true dose instead of being a step at any exposure, half way at the \
                     quantile dose_midpoint of the positive doses and rising with the Hill \
                     order dose_order. generic_shift: sizes of a contact-range shift of every gene \
                     in competent receivers, tied to the ligand by a hidden trait of the \
                     neighbours."
                        .into(),
                ),
                ..GridOpts::default()
            },
            ..Study::default()
        };
        serde_json::to_string_pretty(&study).expect("a study serializes")
    }
}
