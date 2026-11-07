//! What to simulate: the scenarios, the default grid, and the study file.

/// The hidden confounder's spatial form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConfounderKind {
    /// Random Fourier features, the adjustment smooth's own basis
    Fourier,
    /// Random Gaussian bumps of width `scale` contact lengths, random signs
    Bumps,
    /// A real gene outside the benchmarked pairs, smoothed over contacts
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
    /// θ: log-rate change of the programme in exposed responding receivers.
    pub effect: f64,
    /// α: hidden field's push on ligand production (log odds).
    pub confound_ligand: f64,
    /// β: hidden field's push on the response.
    pub confound_response: f64,
    /// Hidden field's length scale, in contact lengths.
    pub scale: f64,
    pub confounder: ConfounderKind,
    pub shape: ConfoundShape,
    /// θ varies by identity bin, `θ u_b` with `u_b ~ U[0.5, 1.5]`.
    pub heterogeneous: bool,
    /// Ligand, response and programme baseline confined to a marker type.
    pub marker: bool,
    /// κ: share of each contact neighbour's counts measured in the receiver.
    pub spillover: f64,
    /// Displacement of the ligand field the estimator sees, in contact
    /// lengths; 0 for none.
    pub shift: f64,
    /// Simulate the receptor, driven by the hidden field at strength β.
    pub confound_receptor: bool,
    /// How much of the field's push on the response (β) reaches the background programme
    /// programme: 1 the same push, 0 none.
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
    /// programme, so the receptor-negative receivers cannot show it; 0 for none.
    pub generic_shift: f64,
    pub rep: usize,
}

impl Default for Scenario {
    /// No effect, no confounding, no departure. The confounder kind only
    /// matters once `α` or `β` is set: Fourier, the original field, unless
    /// a scenario names another (the grids name theirs).
    fn default() -> Self {
        Self {
            effect: 0.0,
            confound_ligand: 0.0,
            confound_response: 0.0,
            scale: 32.0,
            confounder: ConfounderKind::Fourier,
            shape: ConfoundShape::Linear,
            heterogeneous: false,
            marker: false,
            spillover: 0.0,
            shift: 0.0,
            confound_receptor: false,
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
    /// Planted effects θ, as log-rate changes of the programme; each runs
    /// beside θ = 0.
    pub effect_sizes: Vec<f64>,
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
    /// Simulate the receptor in every scenario, driven by the hidden field
    /// at the response's strength; false keeps the real receptor.
    pub confound_receptor: bool,
    /// How much of the field's push on the response reaches the background programme
    /// programme: 1 the same push, 0 none.
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
            effect_sizes: vec![0.3],
            reps: 5,
            confounders: vec![ConfounderKind::Bumps, ConfounderKind::RealGene],
            scales: vec![32.0, 4.0],
            shapes: vec![ConfoundShape::Linear, ConfoundShape::Quadratic],
            spillover: vec![0.1],
            shift: vec![4.0],
            marker: true,
            heterogeneous: true,
            confound_receptor: true,
            background_confounding: 1.0,
            dose_response: false,
            dose_midpoint: 0.5,
            dose_order: 2,
            generic_shift: Vec::new(),
        }
    }
}

/// The grid: no effect and each planted one, each with
///
/// - no confounding;
/// - per confounder kind: linear confounding at every scale, same-sign and
///   opposite-sign (`α = 1, β = ±1`), and each nonlinear shape at the
///   first scale, same-sign;
/// - a heterogeneous effect (planted effects only);
/// - unconfounded segmentation spillover at each κ;
/// - the first confounder at the first scale, seen shifted by each shift;
/// - the marker type, unconfounded.
pub fn default_grid(g: &GridOpts) -> Vec<Scenario> {
    let mut out = Vec::new();
    let first_scale = g.scales.first().copied().unwrap_or(32.0);
    let thetas = std::iter::once(0.0).chain(g.effect_sizes.iter().copied().filter(|&t| t != 0.0));
    for theta in thetas {
        for rep in 0..g.reps {
            let plain = Scenario {
                effect: theta,
                confound_receptor: g.confound_receptor,
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
                    confound_ligand: 1.0,
                    confound_response: 1.0,
                    ..plain
                };
                for (k, &scale) in g.scales.iter().enumerate() {
                    for &shape in &g.shapes {
                        if shape == ConfoundShape::Linear {
                            out.push(Scenario { scale, ..conf });
                            out.push(Scenario {
                                scale,
                                confound_response: -1.0,
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
            if let Some(&confounder) = g.confounders.first() {
                for &shift in &g.shift {
                    out.push(Scenario {
                        confounder,
                        confound_ligand: 1.0,
                        confound_response: 1.0,
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
        let s: Study = serde_json::from_str(text)?;
        anyhow::ensure!(s.reps > 0, "reps must be at least 1");
        anyhow::ensure!(s.pairs > 0, "pairs must be at least 1");
        anyhow::ensure!(
            s.scenarios.scales.iter().all(|&x| x > 0.0),
            "scenarios.scales must be positive contact lengths"
        );
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
                    "Each effect size runs beside no effect. effect_sizes: log-rate change \
                     of the response programme (0.3 is about +35%). confounders: hidden \
                     field kinds (bumps, realgene, fourier). scales: their length scales in \
                     contact lengths. shapes: how the field acts (linear, quadratic). \
                     spillover: shares of a neighbour's counts bleeding into \
                     the receiver. shift: displacement of the ligand field, in contact \
                     lengths (a null). marker: effect confined to one cell type. \
                     heterogeneous: effect varying by cell type. confound_receptor: simulate \
                     the receptor too, driven by the same hidden field (false keeps the \
                     real receptor, which the field cannot touch). background_confounding: how \
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
