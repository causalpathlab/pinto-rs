//! The verdict vocabulary and the rule that assigns it.

/// What one row's verdict rests on.
#[derive(Clone, Copy, Debug, Default)]
pub struct Evidence {
    /// The headline (the competent effect) has an estimate.
    pub estimable: bool,
    /// The headline's q is below the FDR level.
    pub effect: bool,
    /// The co-expression q is below the FDR level.
    pub coexpr: bool,
    /// A control beside a significant effect did not hold: the effect does not
    /// exceed the incompetent, ring or background effect, or one could not be checked.
    pub control_failed: bool,
    /// `exposure_eta2` reaches the internal guard `max_exposure_eta2`
    /// (`outputs.lra.settings` in the manifest).
    pub confounded: bool,
}

/// The verdict. In order: no headline estimate; exposure that cannot be
/// told from neighbour identity (then a significant effect is not a
/// finding); a significant effect beside a failed control; else the 2 × 2
/// of co-expression and effect.
pub fn verdict(e: Evidence) -> Verdict {
    if !e.estimable {
        Verdict::InsufficientData
    } else if e.confounded {
        Verdict::CelltypeInseparable
    } else if e.control_failed && e.effect {
        Verdict::ControlFailed
    } else {
        match (e.coexpr, e.effect) {
            (true, true) => Verdict::Supported,
            (true, false) => Verdict::ColocationOnly,
            (false, true) => Verdict::EffectWithoutCoexpression,
            (false, false) => Verdict::NoEvidence,
        }
    }
}

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
