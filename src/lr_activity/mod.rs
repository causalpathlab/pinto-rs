//! Ligand-receptor analysis, `pinto lra`: the contact effect
//! ([`ligand_effect`], the default method) and, below, the co-expression
//! test per link community (`--method coexpression-test`).
//!
//! The co-expression test reads a `pinto fit lc` run's edge→community
//! assignments and a directional ligand→receptor list, and tests whether
//! each pair's ligand and receptor are active together along the contacts
//! of each link community: pseudobulk samples of (batch × propensity bin),
//! a weighted covariance per (batch, community, pair), and a null that
//! permutes the ligand's samples within propensity-stratified buckets.
//! [`fit`] has the design in full, [`orientation`] why the test is
//! symmetric in the pair.
//!
//! ## Statistical justification
//!
//! Pseudobulk-then-test follows the SC-DE consensus that cell-level
//! inference inflates type-I error (Squair 2021 *Nat Commun* 12:5692;
//! Crowell 2020 *Nat Commun* 11:6077; muscat). Stratified permutation is
//! a textbook restricted-exchangeability null (Good 2005 *Permutation
//! Tests*; Pesarin & Salmaso 2010 *Permutation Tests for Complex Data*).
//! Stratifying samples by a coarse expression-propensity binary code is
//! structurally similar to propensity-score stratification in observational
//! causal inference (Rosenbaum & Rubin 1983 *Biometrika* 70:41–55; with
//! the over-conditioning caveat in Rosenbaum 1984 *J R Stat Soc A*
//! 147:656–666 — mitigated here by random projection over all genes vs
//! the few-hundred LR genes in the test, plus masking to broad-stroke top
//! bits). The propensity codes themselves are sign-random-projection LSH
//! on SVD axes (Charikar 2002 *STOC '02*). The motivation is that
//! cell-label permutation nulls in the standard CCC stack (CellChat,
//! CellPhoneDB, Squidpy, NicheNet, LIANA, COMMOT) pick up
//! cell-type-marginal correlations as significant L→R coupling — see
//! Dimitrov 2022 *Nat Commun* 13:3224 for the field benchmark.

pub mod args;
pub mod edge_scores;
pub mod fit;
pub mod input;
pub mod io;
pub mod ligand_effect;
pub mod orientation;
pub mod outputs;

#[cfg(test)]
mod tests;

pub use args::{LraSimulateArgs, SrtLrActivityArgs};
pub use fit::{fit_lra_simulate, fit_srt_lr_activity};

/// What `pinto lra --method {method} -o {out}` writes (with `--only`
/// `only`, when not empty), its table first: the names its writers write
/// and the MCP reply reads.
pub fn lra_outputs(
    method: args::LraMethod,
    out: &str,
    only: &[Box<str>],
) -> Vec<std::path::PathBuf> {
    lra_outputs_of_link_community(method, out, only, None)
}

/// [`lra_outputs`] of a `--link-community` run (`None` for the marginal run).
pub fn lra_outputs_of_link_community(
    method: args::LraMethod,
    out: &str,
    only: &[Box<str>],
    link_community: Option<i64>,
) -> Vec<std::path::PathBuf> {
    use args::LraMethod::*;
    match method {
        ContactEffect => {
            let (table, curve) = ligand_effect::link_community_scope::table_and_curve_paths(
                out,
                only,
                link_community,
            );
            vec![table.into(), curve.into()]
        }
        CoexpressionTest => vec![
            format!("{out}.lr_activity.parquet").into(),
            format!("{out}.lr_activity.json").into(),
        ],
        CoexpressionScores => vec![format!("{out}.lr_scores.parquet").into()],
    }
}

/// `pinto lra --from {from}` with `method` and `options` (`--out` when
/// given, else beside the run), each choice that is not lra's default,
/// then `--only only` when not empty: the one command line `run_lra` and
/// the viewer start.
#[cfg(feature = "view")]
impl crate::util::metadata::LraOptions {
    pub fn argv(
        &self,
        from: Option<&str>,
        out: Option<&str>,
        method: args::LraMethod,
        only: &[String],
    ) -> Vec<String> {
        use crate::tui::flag_value;
        let mut argv: Vec<String> = vec!["lra".into()];
        if let Some(from) = from {
            argv.push(flag_value("--from", from));
        }
        if let Some(o) = out {
            argv.push(flag_value("--out", o));
        }
        if self.marginal_baselines.is_some() {
            argv.push("--marginal-baselines".into());
            if let Some(n) = self.marginal_permutations {
                if n != crate::lr_activity::ligand_effect::PERMUTATIONS {
                    argv.push(flag_value("--marginal-permutations", n));
                }
            }
        }
        // each value in its flag's word, so a negative one reaches lra's check
        let mut flag = |name: &str, value: String| argv.push(flag_value(name, value));
        if method != args::LraMethod::default() {
            flag("--method", crate::util::cli::value_name(&method));
        }
        if let Some(p) = &self.lr_pairs {
            flag("--lr-pairs", p.clone());
        }
        if !self.targets.is_empty() {
            flag("--targets", self.targets.join(","));
        }
        if let Some(c) = &self.cell_types {
            flag("--cell-types", c.clone());
        }
        if let Some(r) = &self.region {
            flag("--region", r.clone());
        }
        let d = Self::default();
        if self.region_jackknife != d.region_jackknife {
            flag("--region-jackknife", self.region_jackknife.clone());
        }
        if self.abf_prior_scale != d.abf_prior_scale {
            flag("--abf-prior-scale", self.abf_prior_scale.to_string());
        }
        if self.species != d.species {
            flag("--species", self.species.clone());
        }
        if self.seed != d.seed {
            flag("--seed", self.seed.to_string());
        }
        if !only.is_empty() {
            flag("--only", only.join(","));
        }
        argv
    }
}

/// `path` as every lra method records it in the `--from` run's manifest:
/// absolute, so it reads the same from the manifest's directory and from
/// wherever lra ran, and the final path of one staged under
/// `--atomic-outputs`, which is what the manifest holds once the run is in
/// place.
pub fn recorded(path: &std::path::Path) -> String {
    crate::util::paths::shown_path(&crate::util::staging::final_path(path))
}

/// `pinto lra --help`: the three methods, the reference, what is reported.
pub fn lra_long_about() -> String {
    "Ligand-receptor analysis on spatial contacts.\n\
         \n\
         Reads a prior run (--from run.pinto.json: lc, cage, dsvd, ...) or data\n\
         files with coordinates (--coord), and asks per ligand-receptor pair:\n\
         \n\
         \x20 --method contact-effect (default)\n\
         \x20   Do cells touching ligand-carrying neighbours change their response\n\
         \x20   program, compared with similar cells that do not? Adjusted for\n\
         \x20   region, section, cell identity, neighbour make-up, depth and\n\
         \x20   segmentation bleed, with a negative control. Writes {out}.lra.parquet.\n\
         \x20 --method coexpression-test\n\
         \x20   Permutation test of ligand-receptor co-expression within each link\n\
         \x20   community (needs an lc run and --lr-pairs).\n\
         \x20   Writes {out}.lr_activity.parquet and .json.\n\
         \x20 --method coexpression-scores\n\
         \x20   Descriptive co-expression log odds per batch and link community, no\n\
         \x20   test (needs an lc run and --lr-pairs). Writes {out}.lr_scores.parquet.\n\
         \n\
         Reference: contact-effect uses LIANA+ consensus pairs and receptor ->\n\
         signalling -> TF -> target programs (OmniPath, CollecTRI), downloaded\n\
         once into a cache and reused. --lr-pairs and --targets add to it.\n\
         PINTO_REFERENCE_DIR=<dir> moves the cache, PINTO_OFFLINE=1 never\n\
         downloads, PINTO_REFERENCE_REFRESH=1 rebuilds it.\n\
         \n\
         Nothing is judged: each pair's effect, its effect in receptor-negative\n\
         receivers and their contrast are reported with SE, df and p (the effect's\n\
         and the contrast's SE from a tile jackknife, the receptor-negative\n\
         effect's clustered), and co-expression with its local fdr.\n\
         `warnings` says why a pair has no estimate.\n\
         Check what this tissue can detect with `pinto lra simulate`.\n\
         \n\
         Example:\n\
         \x20 pinto lra --from out/run1.pinto.json\n\
         \x20 pinto lra data.zarr --coord cells.csv -o out/lra"
        .to_string()
}

/// `pinto lra simulate --help`.
pub fn lra_simulate_long_about() -> String {
    "Power and calibration check: can this tissue detect ligand effects of a\n\
     given size, and does the analysis stay honest under confounding?\n\
     \n\
     Keeps the real tissue (coordinates, contacts, cell identity, depth) and\n\
     plants ligand effects of known size on a few abundant pairs, with and\n\
     without hidden spatial confounders. Then runs the contact-effect\n\
     analysis and reports, per scenario, how often it finds a planted effect\n\
     (power), how often it calls one where none was planted (type I error),\n\
     its bias and its interval coverage, beside the unadjusted difference.\n\
     \n\
     Writes {out}.lra-simulate.parquet and prints the summary; `pinto lra`\n\
     on the same run then notes the check.\n\
     \n\
     A sweep runs one scenario per call instead (--set KEY=VALUE, --pairs,\n\
     --null-pairs or --one): planted and null pairs in one table, scored\n\
     by AUROC and AUPRC, and a sidecar {out}.lra-simulate.json written last.\n\
     \n\
     Examples:\n\
     \x20 pinto lra simulate --from out/run1.pinto.json\n\
     \x20 pinto lra simulate --from out/run1.pinto.json -o sweep/p1 --set effect_pve=0.02"
        .to_string()
}
