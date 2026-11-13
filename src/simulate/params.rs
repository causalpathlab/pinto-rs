//! Every dial of the simulator in one flat struct, recorded whole in the
//! manifest. A scenario is the defaults with a few fields changed, and
//! `--set key=value` changes any field on top.

use serde::{Deserialize, Serialize};

/// What the donor-specific domain is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PatchKind {
    /// No donor-specific domain.
    None,
    /// Dominated by a cell state no other donor has.
    NewState,
    /// The first two shared states, evenly: a mix no shared domain has.
    NewMix,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SimParams {
    pub seed: u64,

    // panel
    pub n_donors: usize,
    pub sections_per_donor: usize,
    /// Spots per side of the square grid used when no coordinates are given.
    pub grid_side: usize,
    pub grid_spacing_um: f64,

    // genes
    pub n_genes: usize,
    /// Genes in each cell state's programme.
    pub programme_genes: usize,
    /// Mean and SD over genes of the baseline log rate per unit depth.
    pub log_mean: f64,
    pub log_mean_sd: f64,
    /// Median NB dispersion (variance = mean + dispersion · mean²) and its
    /// log-scale SD over genes.
    pub dispersion: f64,
    pub dispersion_sd: f64,
    /// Log-scale SD of spot depth.
    pub depth_sd: f64,

    // planted truth
    /// Shared domains, each dominated by its own shared cell state.
    pub n_domains: usize,
    /// Share of a domain's dominant state in its composition.
    pub dominant: f64,
    /// Dirichlet concentration of spot mixtures around the composition.
    pub kappa: f64,
    pub domain_length_um: f64,
    /// Spread of domain proportions between donors (log scale).
    pub jit_b: f64,
    pub patch_kind: PatchKind,
    /// Donor that owns the donor-specific domain (0-based).
    pub patch_owner: usize,
    /// Share of the owner's spots in the donor-specific domain.
    pub patch_share: f64,
    pub patch_length_um: f64,
    /// For a new state: cosine of its programme to the first shared state's.
    pub patch_cosine: f64,

    // The log rate, as proportions of variance explained (PVE).
    //
    // The four shares below and spot noise split `log_rate_var` between
    // them, and noise is what they leave: `1 - type - donor - inter -
    // section`. Raising one share with the others fixed therefore shrinks
    // the noise, not the other shares. A sweep over the nuisance shares has
    // to choose what gives way: with `pve_type` fixed, the noise shrinks and
    // the cell-type signal grows against it; with `pve_type = t (1 - N)`, N
    // the nuisance total, type and noise shrink together and keep their
    // ratio; with the noise fixed, the cell-type share shrinks.
    // [`SimParams::shares_summary`] spells out what a setting implies.
    /// Total variance of the log rate around its gene baselines.
    pub log_rate_var: f64,
    /// Share of the cell states (the signal that defines domains).
    pub pve_type: f64,
    /// Share of donor-wide effects, the same in every cell state.
    pub pve_donor: f64,
    /// Share of donor-by-cell-state effects.
    pub pve_inter: f64,
    /// Share of per-section effects.
    pub pve_section: f64,
    /// Donors and sections (0-based, in panel order) that carry no nuisance.
    pub clean_donors: Vec<usize>,
    pub clean_sections: Vec<usize>,

    // Gaussian copula across genes; strength 0 turns it off
    pub copula_rank: usize,
    /// Share of each gene's latent variance carried by the copula factors.
    pub copula_strength: f64,

    /// Draw the same spots with every nuisance term removed: the exact
    /// counterfactual of the run with the same seed.
    pub nuisance_off: bool,
}

/// The log rate's terms, in the order every share array uses.
pub const TERMS: [&str; 5] = ["type", "donor", "inter", "section", "noise"];

/// Of groups `0..n`, those not in `clean`.
fn on_count(n: usize, clean: &[usize]) -> usize {
    (0..n).filter(|i| !clean.contains(i)).count()
}

/// Why a term that varies across `n` groups, `on` of them not clean, has
/// nothing to vary across, if so.
fn why_alone(n: usize, on: usize, one: &'static str, all: &'static str) -> Option<&'static str> {
    match (n, on) {
        (1, _) => Some(one),
        (_, 0) => Some(all),
        _ => None,
    }
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            seed: 1,
            n_donors: 3,
            sections_per_donor: 2,
            grid_side: 40,
            grid_spacing_um: 100.0,
            n_genes: 1000,
            programme_genes: 50,
            log_mean: -1.0,
            log_mean_sd: 1.5,
            dispersion: 0.3,
            dispersion_sd: 0.5,
            depth_sd: 0.3,
            n_domains: 4,
            dominant: 0.95,
            kappa: 200.0,
            domain_length_um: 800.0,
            jit_b: 0.0,
            patch_kind: PatchKind::None,
            patch_owner: 2,
            patch_share: 0.15,
            patch_length_um: 400.0,
            patch_cosine: 0.3,
            log_rate_var: 1.0,
            pve_type: 0.3,
            pve_donor: 0.0,
            pve_inter: 0.0,
            pve_section: 0.0,
            clean_donors: Vec::new(),
            clean_sections: Vec::new(),
            copula_rank: 10,
            copula_strength: 0.1,
            nuisance_off: false,
        }
    }
}

fn l1() -> SimParams {
    SimParams {
        pve_donor: 0.15,
        pve_inter: 0.15,
        pve_section: 0.02,
        ..SimParams::default()
    }
}

fn l2() -> SimParams {
    SimParams {
        patch_kind: PatchKind::NewState,
        ..l1()
    }
}

fn l2_mix() -> SimParams {
    SimParams {
        patch_kind: PatchKind::NewMix,
        ..l1()
    }
}

fn clean_one() -> SimParams {
    SimParams {
        clean_donors: vec![0],
        ..l2()
    }
}

fn null_shift() -> SimParams {
    SimParams {
        jit_b: 0.5,
        patch_kind: PatchKind::NewState,
        ..SimParams::default()
    }
}

/// A named scenario: what it is, and its settings.
pub struct Scenario {
    pub name: &'static str,
    pub about: &'static str,
    pub params: fn() -> SimParams,
}

/// The named scenarios.
pub const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "L0",
        about: "shared domains only, no donor or section effect",
        params: SimParams::default,
    },
    Scenario {
        name: "L1",
        about: "L0 + donor-wide and donor-by-type nuisance",
        params: l1,
    },
    Scenario {
        name: "L2",
        about: "L1 + a domain one donor alone has (a new state)",
        params: l2,
    },
    Scenario {
        name: "L2-mix",
        about: "L1 + a domain one donor alone has (a new mix of shared states)",
        params: l2_mix,
    },
    Scenario {
        name: "clean-one",
        about: "L2 with one donor clean of nuisance",
        params: clean_one,
    },
    Scenario {
        name: "null-shift",
        about: "no nuisance, but donor proportions and a donor-specific domain differ",
        params: null_shift,
    },
];

impl SimParams {
    pub fn scenario(name: &str) -> anyhow::Result<Self> {
        SCENARIOS
            .iter()
            .find(|s| s.name == name)
            .map(|s| (s.params)())
            .ok_or_else(|| {
                let names: Vec<&str> = SCENARIOS.iter().map(|s| s.name).collect();
                anyhow::anyhow!("unknown scenario {name:?}; one of {}", names.join(", "))
            })
    }

    /// Set fields by `key=value`, the value read as JSON when it parses
    /// (numbers, booleans, lists) and as a string otherwise.
    pub fn with_overrides(self, sets: &[String]) -> anyhow::Result<Self> {
        let mut v = serde_json::to_value(&self)?;
        let obj = v.as_object_mut().expect("params serialize to an object");
        for s in sets {
            let (k, raw) = s
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("--set {s:?}: expected key=value"))?;
            anyhow::ensure!(obj.contains_key(k), "--set {k}: no such parameter");
            let val = serde_json::from_str(raw).unwrap_or(serde_json::Value::String(raw.into()));
            obj.insert(k.to_string(), val);
        }
        Ok(serde_json::from_value(v)?)
    }

    /// The share of spot noise in the log rate: what the PVE terms leave.
    pub fn pve_noise(&self) -> f64 {
        1.0 - self.pve_type - self.pve_donor - self.pve_inter - self.pve_section
    }

    /// The nuisance terms (donor, donor x type, section): each one's
    /// parameter name, its share as set, and why it is off, if it is:
    /// `nuisance_off` (as [`crate::simulate::expression::active`] decides),
    /// or nothing for it to vary across (one donor or section, or every one
    /// clean), where `expression::scaling` gives it no scale. A donor-by-type
    /// term still varies across cell states with one donor. A test holds the
    /// two in step.
    pub fn nuisance_terms(&self) -> [(&'static str, f64, Option<&'static str>); 3] {
        let off = if crate::simulate::expression::active(self)[1] {
            let donors = on_count(self.n_donors, &self.clean_donors);
            let sections = on_count(self.n_sections(), &self.clean_sections);
            [
                why_alone(self.n_donors, donors, "one donor", "every donor clean"),
                (donors == 0).then_some("every donor clean"),
                why_alone(
                    self.n_sections(),
                    sections,
                    "one section",
                    "every section clean",
                ),
            ]
        } else {
            [Some("nuisance_off"); 3]
        };
        let set = [self.pve_donor, self.pve_inter, self.pve_section];
        let names = ["pve_donor", "pve_inter", "pve_section"];
        [0, 1, 2].map(|k| (names[k], set[k], off[k]))
    }

    /// The shares of the log rate's variance the run is set to have, in
    /// [`TERMS`] order and summing to 1: a nuisance term that is off carries
    /// none, and its share does not pass to the noise. A target, before
    /// drawing; `Sim::realised_pve` measures what the draws hold.
    pub fn effective_shares(&self) -> [f64; 5] {
        let mut s = [self.pve_type, 0.0, 0.0, 0.0, self.pve_noise()];
        for (k, (_, set, off)) in self.nuisance_terms().into_iter().enumerate() {
            if off.is_none() {
                s[k + 1] = set;
            }
        }
        let total: f64 = s.iter().sum();
        s.map(|x| x / total)
    }

    /// The variance shares the run has, noise as the remainder, and what
    /// they imply: a share set but off, and with nuisance on, that it came
    /// out of the noise, with the type-to-noise ratio here and with the
    /// nuisance shares at 0. It describes; it suggests no setting.
    pub fn shares_summary(&self) -> String {
        let [t, d, i, sec, noise] = self.effective_shares();
        let mut s = format!(
            "variance shares of the log rate: type {t:.3}, donor {d:.3}, donor x type {i:.3}, \
             section {sec:.3}, noise {noise:.3} (the remainder)"
        );
        for (name, set, off) in self.nuisance_terms() {
            if let (true, Some(why)) = (set > 0.0, off) {
                s.push_str(&format!(". {name} is set but off ({why})"));
            }
        }
        let nuisance = d + i + sec;
        if nuisance > 0.0 {
            let here = if noise > 0.0 {
                format!("{:.3}", t / noise)
            } else {
                "unbounded (no noise)".into()
            };
            s.push_str(&format!(
                ". The nuisance ({nuisance:.3}) comes out of the noise, not the type share: \
                 type to noise is {here} here, {:.3} with the nuisance shares at 0",
                self.pve_type / (1.0 - self.pve_type)
            ));
        } else if noise == 0.0 {
            s.push_str(". There is no noise");
        }
        s
    }

    pub fn n_sections(&self) -> usize {
        self.n_donors * self.sections_per_donor
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let pves = [
            self.pve_type,
            self.pve_donor,
            self.pve_inter,
            self.pve_section,
        ];
        anyhow::ensure!(pves.iter().all(|&p| p >= 0.0), "a PVE is negative");
        anyhow::ensure!(
            self.pve_noise() >= 0.0,
            "the PVE terms sum to {} > 1",
            1.0 - self.pve_noise()
        );
        anyhow::ensure!(
            self.pve_type > 0.0,
            "pve_type must be positive: no cell-type effect, no domains to find"
        );
        anyhow::ensure!(
            self.n_donors >= 1 && self.sections_per_donor >= 1,
            "empty panel"
        );
        anyhow::ensure!(self.n_domains >= 2, "need at least two domains");
        anyhow::ensure!(self.n_genes >= 1, "n_genes must be at least 1");
        anyhow::ensure!(
            self.programme_genes >= 1 && self.programme_genes <= self.n_genes,
            "programme_genes must be in 1..=n_genes"
        );
        anyhow::ensure!(
            self.grid_side >= 1 && self.grid_spacing_um > 0.0,
            "the grid needs a spot and a positive spacing"
        );
        anyhow::ensure!(
            self.domain_length_um > 0.0 && self.patch_length_um > 0.0,
            "length scales must be positive"
        );
        anyhow::ensure!(
            self.dominant > 0.0 && self.dominant <= 1.0,
            "dominant must be in (0, 1]"
        );
        anyhow::ensure!(self.kappa > 0.0, "kappa must be positive");
        anyhow::ensure!(
            self.dispersion >= 0.0
                && self.dispersion_sd >= 0.0
                && self.depth_sd >= 0.0
                && self.log_mean_sd >= 0.0,
            "dispersion and the spreads must be non-negative"
        );
        anyhow::ensure!(self.log_rate_var > 0.0, "log_rate_var must be positive");
        anyhow::ensure!(
            self.copula_rank >= 1,
            "copula_rank must be at least 1 (copula_strength=0 turns it off)"
        );
        anyhow::ensure!(
            (0.0..1.0).contains(&self.copula_strength),
            "copula_strength must be in [0, 1)"
        );
        if self.patch_kind != PatchKind::None {
            anyhow::ensure!(
                self.patch_owner < self.n_donors,
                "patch_owner is not a donor"
            );
            anyhow::ensure!(
                self.patch_share > 0.0 && self.patch_share < 1.0,
                "patch_share must be in (0, 1)"
            );
            anyhow::ensure!(
                (-1.0..=1.0).contains(&self.patch_cosine),
                "patch_cosine must be in [-1, 1]"
            );
        }
        anyhow::ensure!(
            self.clean_donors.iter().all(|&d| d < self.n_donors),
            "clean_donors names a donor that does not exist"
        );
        anyhow::ensure!(
            self.clean_sections.iter().all(|&s| s < self.n_sections()),
            "clean_sections names a section that does not exist"
        );
        Ok(())
    }
}
