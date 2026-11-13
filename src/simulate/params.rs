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

    // log rate, as proportions of variance explained (PVE) of `log_rate_var`
    pub log_rate_var: f64,
    pub pve_type: f64,
    pub pve_donor: f64,
    pub pve_inter: f64,
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
