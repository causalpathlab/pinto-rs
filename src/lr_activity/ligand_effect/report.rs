//! A    w.extend(flags.iter().filter(|f| f.0).map(|f| f.1.to_string()));ross pairs: multiplicity, and each row judged into a verdict.

use super::inference::{normal_score_of_two_sided_p, upper_normal_p, Estimate, ONE_CLUSTER_DF};
use super::options::thin_arm;
use super::rows::{EffectRow, LocalFdr};
use super::verdict::{verdict, Evidence};
use super::INTERNALS;
use crate::util::local_fdr::{two_groups, NormalNull, SignalSide};

/// What [`report`] needs beyond the rows.
#[derive(Clone, Copy, Debug)]
pub struct ReportOpts {
    /// The empirical-null local fdr below which a statistic counts in the
    /// verdict, in a family large enough to fit one.
    pub local_fdr_level: f64,
    /// The family-wise level a statistic's p is tested at, Bonferroni over
    /// its family, in a family too small for a local fdr; a statistic tested
    /// alone ([`judge_alone`]) is tested at it.
    pub small_family_level: f64,
    /// `exposure_eta2` at or above which a pair is "cell-type confounded".
    pub max_exposure_eta2: f64,
    /// [`super::options::EffectOpts::min_arm`], to say why a pair is not estimable.
    pub min_arm: usize,
}

impl Default for ReportOpts {
    /// What [`report`] reads of [`INTERNALS`].
    fn default() -> Self {
        Self {
            local_fdr_level: INTERNALS.local_fdr_level,
            small_family_level: INTERNALS.small_family_level,
            max_exposure_eta2: INTERNALS.max_exposure_eta2,
            min_arm: INTERNALS.min_arm,
        }
    }
}

/// Fewest finite statistics in a family for its local fdr: the two-group
/// model's null and mixture density are fitted to the family itself, and a
/// handful of statistics fit neither. A smaller family (a short `--only`
/// list, a small community) has no local fdr, and the verdict tests its
/// p-values by Bonferroni instead.
pub const MIN_STATISTICS_FOR_LOCAL_FDR: usize = 100;

/// One statistic over a family of rows: its normal scores and p-values.
struct Family {
    z: Vec<f64>,
    p: Vec<f64>,
    side: SignalSide,
}

impl Family {
    /// The two-sided test of each estimate, read on its normal score: the
    /// score with the same p against the estimate's t reference, so an
    /// estimate on few clusters weighs as little as its p says.
    fn of_estimates(estimates: impl Iterator<Item = Estimate>) -> Self {
        let (z, p) = estimates
            .map(|e| {
                let p = e.p();
                (normal_score_of_two_sided_p(p) * e.theta.signum(), p)
            })
            .unzip();
        Self {
            z,
            p,
            side: SignalSide::Both,
        }
    }

    /// The one-sided (positive) test of each z.
    fn of_upper_scores(z: Vec<f64>) -> Self {
        Self {
            p: z.iter().map(|&v| upper_normal_p(v)).collect(),
            z,
            side: SignalSide::Above,
        }
    }

    /// The local fdr of each statistic under both nulls, and the level its
    /// rule reads: the empirical-null lfdr where the family is large enough,
    /// else the p at the Bonferroni level over its finite statistics.
    fn fit(&self, opts: &ReportOpts) -> FamilyFit {
        let finite = self.z.iter().filter(|x| !x.is_nan()).count();
        let null = (finite >= MIN_STATISTICS_FOR_LOCAL_FDR)
            .then(|| NormalNull::empirical(&self.z))
            .flatten();
        let Some(null) = null else {
            return FamilyFit {
                lfdr: vec![LocalFdr::NAN; self.z.len()],
                rule: Rule::PBelow(opts.small_family_level / finite.max(1) as f64),
                finite,
            };
        };
        let empirical = two_groups(&self.z, null, self.side);
        let theoretical = two_groups(&self.z, NormalNull::THEORETICAL, self.side);
        let lfdr = empirical
            .lfdr
            .iter()
            .zip(&theoretical.lfdr)
            .map(|(&e, &t)| LocalFdr {
                empirical_null: e,
                theoretical_null: t,
            })
            .collect();
        FamilyFit {
            lfdr,
            rule: Rule::LocalFdrBelow {
                level: opts.local_fdr_level,
                null,
                null_share: empirical.null_share,
            },
            finite,
        }
    }
}

/// How a family's statistics count in the verdict.
#[derive(Clone, Copy, Debug)]
enum Rule {
    /// The empirical-null local fdr below `level`, with the null fitted.
    LocalFdrBelow {
        level: f64,
        null: NormalNull,
        null_share: f64,
    },
    /// The p below the Bonferroni level, which assumes nothing about the
    /// statistics' dependence.
    PBelow(f64),
}

/// A family's local fdrs and rule, and its finite statistics.
struct FamilyFit {
    lfdr: Vec<LocalFdr>,
    rule: Rule,
    finite: usize,
}

impl FamilyFit {
    /// Row `k`'s statistic as the verdict reads it.
    fn statistic(&self, family: &Family, k: usize) -> Statistic {
        match self.rule {
            Rule::LocalFdrBelow { level, .. } => Statistic {
                value: self.lfdr[k].empirical_null,
                level,
            },
            Rule::PBelow(level) => Statistic {
                value: family.p[k],
                level,
            },
        }
    }
}

/// p-values, local fdrs and verdicts for every row, within each section (the
/// pooled rows, `all`, form their own family).
///
/// Each statistic's family (the effect, co-expression, the contrast with
/// the receptor-negative receivers, the effect in them) is fitted on its own
/// by Efron's two-group model ([`crate::util::local_fdr`]) on its normal
/// scores, under the null fitted to the family's bulk and under the
/// theoretical one; both are kept. Pairs sharing a ligand share its
/// exposure, and pairs sharing senders share neighbourhoods, so the
/// statistics are correlated: that widens or narrows the empirical null
/// rather than invalidating it (Efron 2007), and no dependence condition is
/// assumed.
pub fn report(rows: &mut [EffectRow], opts: &ReportOpts) {
    let mut by_section: std::collections::HashMap<Box<str>, Vec<usize>> = Default::default();
    for (i, r) in rows.iter().enumerate() {
        by_section.entry(r.section.clone()).or_default().push(i);
    }
    for (section, idx) in &by_section {
        let estimates =
            |e: fn(&EffectRow) -> Estimate| Family::of_estimates(idx.iter().map(|&i| e(&rows[i])));
        let families = [
            estimates(|r| r.effects.competent),
            Family::of_upper_scores(
                idx.iter()
                    .map(|&i| rows[i].coexpr_log_or / rows[i].coexpr_se)
                    .collect(),
            ),
            estimates(|r| r.effects.effect),
            estimates(|r| r.effects.incompetent),
        ];
        let fits = families.each_ref().map(|f| f.fit(opts));
        log_fits(section, &fits);
        let ligands: Vec<String> = idx
            .iter()
            .map(|&i| rows[i].ligand.to_ascii_uppercase())
            .collect();
        let mut pairs_of_ligand: std::collections::HashMap<&str, usize> = Default::default();
        for l in &ligands {
            *pairs_of_ligand.entry(l).or_default() += 1;
        }
        for (k, &i) in idx.iter().enumerate() {
            let [effect, coexpr, contrast, incompetent] = &fits;
            let r = &mut rows[i];
            r.n_pairs_same_ligand = pairs_of_ligand[ligands[k].as_str()];
            r.effect_dr_p = families[0].p[k];
            r.coexpr_p = families[1].p[k];
            r.effect_dr_lfdr = effect.lfdr[k];
            r.coexpr_lfdr = coexpr.lfdr[k];
            r.effect_dr_contrast_lfdr = contrast.lfdr[k];
            r.effect_dr_incompetent_lfdr = incompetent.lfdr[k];
            let statistics = [0, 1, 2, 3].map(|s| fits[s].statistic(&families[s], k));
            let small_family = match effect.rule {
                Rule::PBelow(level) => Some(format!(
                    "too few estimable pairs for a local fdr ({} of the {} needed): p-values \
                     tested by Bonferroni at {level:.1e}",
                    effect.finite, MIN_STATISTICS_FOR_LOCAL_FDR
                )),
                Rule::LocalFdrBelow { .. } => None,
            };
            judge(r, statistics, small_family, opts);
        }
    }
}

/// Each family's empirical null as fitted, or that it was too small.
fn log_fits(section: &str, fits: &[FamilyFit; 4]) {
    let names = [
        "effect",
        "co-expression",
        "contrast",
        "receptor-negative effect",
    ];
    let said: Vec<String> = names
        .iter()
        .zip(fits)
        .map(|(name, f)| match f.rule {
            Rule::LocalFdrBelow {
                null, null_share, ..
            } => format!(
                "{name} N({:.2}, {:.2}²) null share {null_share:.2} over {}",
                null.mean, null.sd, f.finite
            ),
            Rule::PBelow(level) => {
                format!("{name} Bonferroni at {level:.1e} over {}", f.finite)
            }
        })
        .collect();
    log::info!("local fdr, section {section}: {}", said.join("; "));
}

/// The verdict and warnings of one row tested alone, a family of one: each
/// statistic's p at the small-family level. What a run of `lra simulate` is
/// judged by.
pub fn judge_alone(row: &mut EffectRow, opts: &ReportOpts) {
    let level = opts.small_family_level;
    let e = &row.effects;
    row.effect_dr_p = e.competent.p();
    row.coexpr_p = upper_normal_p(row.coexpr_log_or / row.coexpr_se);
    let statistics = [
        row.effect_dr_p,
        row.coexpr_p,
        e.effect.p(),
        e.incompetent.p(),
    ]
    .map(|value| Statistic { value, level });
    judge(row, statistics, None, opts);
}

/// One statistic of a row as the verdict reads it: the value its family's
/// rule reads (a local fdr or a p) and the level it counts below.
#[derive(Clone, Copy)]
struct Statistic {
    value: f64,
    level: f64,
}

impl Statistic {
    /// Whether it counts in the verdict; NaN never counts.
    fn counts(self) -> bool {
        self.value < self.level
    }

    /// Whether it has a value: a comparison that could not run has none.
    fn checked(self) -> bool {
        !self.value.is_nan()
    }
}

/// Verdict and warnings of one row from its statistics (effect,
/// co-expression, contrast with the receptor-negative receivers, effect in
/// them), the note of a family too small for a local fdr, and diagnostics.
fn judge(
    row: &mut EffectRow,
    [effect, coexpr, contrast, incompetent]: [Statistic; 4],
    small_family: Option<String>,
    opts: &ReportOpts,
) {
    let head = row.effects.competent;
    let ev = Evidence {
        estimable: head.theta.is_finite() && head.se.is_finite(),
        effect: effect.counts(),
        coexpr: coexpr.counts(),
        control_failed: false,
        confounded: row.exposure_eta2 >= opts.max_exposure_eta2,
    };
    // Beside a significant effect, the effect must exceed the negative
    // control, the same way, and a comparison that could not run has not
    // held. A hidden field moves the control too, so its being significant
    // does not rule the effect out: it is reported, deciding nothing, and
    // what decides is whether the effect is larger.
    let exceeds = contrast.counts() && row.effects.effect.theta * row.effects.competent.theta > 0.0;
    let flags: [Flag; 3] = [
        (
            ev.effect && !contrast.checked(),
            "receptor-negative receivers could not be checked",
        ),
        (
            ev.effect && contrast.checked() && !exceeds,
            "the effect does not exceed the one in receptor-negative receivers",
        ),
        (
            ev.effect && incompetent.counts(),
            "receptor-negative receivers respond",
        ),
    ];
    let ev = Evidence {
        control_failed: flags[..2].iter().any(|f| f.0),
        ..ev
    };
    row.verdict = verdict(ev);
    row.warnings = warnings(row, ev, &flags, small_family, opts);
}

/// A condition of a row and the warning it raises when it holds.
type Flag = (bool, &'static str);

/// Why the row's competent effect has no estimate.
fn not_estimable_reason(row: &EffectRow, opts: &ReportOpts) -> String {
    let head = row.effects.competent;
    if row.n_competent == 0 {
        "not estimable: no receptor-competent receivers".into()
    } else if thin_arm(opts.min_arm, head.n, head.n_treated) {
        format!(
            "not estimable: thin arm ({} exposed, {} unexposed competent receivers; \
             internal guard min_arm {}, see outputs.lra.settings in the manifest)",
            head.n_treated,
            head.n.saturating_sub(head.n_treated),
            opts.min_arm
        )
    } else if head.df == ONE_CLUSTER_DF {
        "not estimable: the competent receivers lie in one variance cluster (spatial \
         tile), so the standard error has no degrees of freedom"
            .into()
    } else {
        "not estimable: low overlap (mean overlap weight or its effective sample size \
         under the internal guards min_overlap / min_ess, see outputs.lra.settings in the \
         manifest)"
            .into()
    }
}

/// The row's warnings, joined: why it has no estimate (or that its family was
/// too small for a local fdr), what confounds it, the control it failed, then
/// the control responding.
fn warnings(
    row: &EffectRow,
    ev: Evidence,
    flags: &[Flag],
    small_family: Option<String>,
    opts: &ReportOpts,
) -> String {
    let mut w: Vec<String> = Vec::new();
    if ev.estimable {
        w.extend(small_family);
    }
    if !ev.estimable {
        w.push(not_estimable_reason(row, opts));
    }
    if ev.estimable && row.jackknife.failed() {
        w.push(
            "the tile jackknife could not run (deleting a tile leaves an arm thin): its SE \
             is the clustered one, likely too small for a region in few tiles"
                .into(),
        );
    }
    if ev.confounded {
        w.push(format!(
            "exposure {:.0}% explained by the dominant neighbour's identity: the effect \
             cannot be told from neighbour cell type",
            100.0 * row.exposure_eta2
        ));
        if ev.coexpr {
            w.push("co-expression is likely a cell-type artefact".into());
        }
    }
    w.extend(flags.iter().filter(|f| f.0).map(|f| f.1.to_string()));
    w.join("; ")
}
