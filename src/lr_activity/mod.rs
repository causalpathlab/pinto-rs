//! Ligand-receptor analysis, `pinto lra`: the contact effect
//! ([`ligand_effect`], the default method) and, below, the co-expression
//! test per link community (`--method coexpression-test`).
//!
//! Consumes the output of `pinto lc` (edge→community assignments) and a
//! user-supplied directional ligand→receptor list, and tests whether each
//! LR pair shows coherent activity within each community.
//!
//! ## Design
//!
//! 1. **Pseudobulk samples** = `(batch × propensity-bin)`. The propensity
//!    bin is the sign-LSH binary code of an SVD'd random projection of the
//!    full expression matrix (`data_beans::alg::binary_sort_columns`); cells
//!    with similar broad expression land in the same bin.
//! 2. **Soft community membership per role.** From `lc` edges, each cell
//!    `i` gets `p_send[i, c]` (fraction of `i`'s incident edges in
//!    community `c` on which `i` is sender) and `p_recv[i, c]` (receiver
//!    analogue), row-normalised per cell.
//! 3. **Role-weighted pseudobulk per (community, sample).** For each LR
//!    gene we accumulate `Σ_i x[g, i] · p_role[i, c] · 1{sample(i)=s}`
//!    plus the matching weight `w_role[c, s]`. Sender and receiver
//!    pseudobulks are computed separately and matched by sample id.
//! 4. **Statistic.** Per `(batch, community, LR pair)` a weighted Spearman
//!    rank correlation between `L_send_s` and `R_recv_s` across the
//!    stratum's samples, sample-weighted by `√(w_send · w_recv)`.
//! 5. **Null.** Sample-level permutation of `L` within propensity-stratified
//!    buckets (top `shuffle_stratify_dim` bits of the propensity code).
//!    Stratification preserves the cell-population marginal — without it,
//!    free shuffles across populations pick up cell-type-marginal
//!    correlations and become anti-conservative.
//! 6. **p-values.** Both empirical (`1/(n+1)`-floored) and parametric
//!    (one-sided Gaussian tail of `z`); BH within batch is applied to the
//!    parametric p so q-values aren't pinned at the resolution floor.
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
    use args::LraMethod::*;
    match method {
        ContactEffect if !only.is_empty() => vec![
            ligand_effect::only::table_path(out, only).into(),
            ligand_effect::only::curve_path(out, only).into(),
        ],
        ContactEffect => vec![
            ligand_effect::output::lra_path(out).into(),
            ligand_effect::output::lra_curve_path(out).into(),
        ],
        CoexpressionTest => vec![
            format!("{out}.lr_activity.parquet").into(),
            format!("{out}.lr_activity.json").into(),
        ],
        CoexpressionScores => vec![format!("{out}.lr_scores.parquet").into()],
    }
}

/// `pinto lra --from {from}` with `method` and `options` (`-o out` when
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
        use clap::ValueEnum;
        let mut argv: Vec<String> = vec!["lra".into()];
        if let Some(from) = from {
            argv.extend(["--from".into(), from.into()]);
        }
        if let Some(o) = out {
            argv.extend(["-o".into(), o.into()]);
        }
        let mut flag = |name: &str, value: String| argv.extend([name.into(), value]);
        if method != args::LraMethod::default() {
            if let Some(m) = method.to_possible_value() {
                flag("--method", m.get_name().into());
            }
        }
        if let Some(p) = &self.lr_pairs {
            flag("--lr-pairs", p.clone());
        }
        if !self.targets.is_empty() {
            flag("--targets", self.targets.join(","));
        }
        let d = Self::default();
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
/// wherever lra ran.
pub fn recorded(path: &std::path::Path) -> String {
    std::path::absolute(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// `pinto lra --help`: the three methods, the reference, the verdicts.
pub fn lra_long_about() -> String {
    format!(
        "Ligand-receptor analysis on spatial contacts.\n\
         \n\
         Reads a prior run (--from run.pinto.json: lc, cage, dsvd, ...) or data\n\
         files with coordinates (--coord), and asks per ligand-receptor pair:\n\
         \n\
         \x20 --method contact-effect (default)\n\
         \x20   Do cells touching ligand-carrying neighbours change their response\n\
         \x20   programme, compared with similar cells that do not? Adjusted for\n\
         \x20   region, section, cell identity, neighbour make-up, depth and\n\
         \x20   segmentation bleed, with a negative control. Writes {{out}}.lra.parquet.\n\
         \x20 --method coexpression-test\n\
         \x20   Permutation test of ligand-receptor co-expression within each link\n\
         \x20   community (needs an lc run and --lr-pairs).\n\
         \x20   Writes {{out}}.lr_activity.parquet and .json.\n\
         \x20 --method coexpression-scores\n\
         \x20   Descriptive co-expression log odds per batch and link community, no\n\
         \x20   test (needs an lc run and --lr-pairs). Writes {{out}}.lr_scores.parquet.\n\
         \n\
         Reference: contact-effect uses LIANA+ consensus pairs and receptor ->\n\
         signalling -> TF -> target programmes (OmniPath, CollecTRI), downloaded\n\
         once into a cache and reused. --lr-pairs and --targets add to it.\n\
         PINTO_REFERENCE_DIR=<dir> moves the cache, PINTO_OFFLINE=1 never\n\
         downloads, PINTO_REFERENCE_REFRESH=1 rebuilds it.\n\
         \n\
         Verdicts (column `verdict`, explained in `verdict_note`):\n\
         {}\n\
         Check what this tissue can detect with `pinto lra-simulate`.\n\
         \n\
         Example:\n\
         \x20 pinto lra --from out/run1.pinto.json\n\
         \x20 pinto lra data.zarr --coord cells.csv -o out/lra",
        ligand_effect::verdict::verdict_help()
    )
}

/// `pinto lra-simulate --help`.
pub fn lra_simulate_long_about() -> String {
    "Power and calibration check: can this tissue detect ligand effects of a\n\
     given size, and does the analysis stay honest under confounding?\n\
     \n\
     Keeps the real tissue (coordinates, contacts, cell identity, depth) and\n\
     plants ligand effects of known size on a few abundant pairs, with and\n\
     without hidden spatial confounders. Then runs the contact-effect\n\
     analysis and reports, per scenario, how often it finds a planted effect\n\
     (power), how often it calls one where none was planted (type I error),\n\
     its bias and its interval coverage, beside the naive contrast.\n\
     \n\
     The study (scenarios, replicates, pairs) is JSON: --example-study prints\n\
     the built-in one to start from. Writes {out}.lra-simulate.parquet and\n\
     prints the summary; `pinto lra` on the same run then notes the check.\n\
     \n\
     Example:\n\
     \x20 pinto lra-simulate --from out/run1.pinto.json\n\
     \x20 pinto lra-simulate --example-study > study.json\n\
     \x20 pinto lra-simulate --from out/run1.pinto.json --study study.json"
        .to_string()
}
