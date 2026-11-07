//! Across pairs: multiplicity, and each row judged into a verdict.

use super::inference::{upper_normal_p, Estimate, ONE_CLUSTER_DF};
use super::options::thin_arm;
use super::rows::EffectRow;
use super::verdict::{verdict, Evidence};
use super::INTERNALS;

/// What [`report`] needs beyond the rows.
#[derive(Clone, Copy, Debug)]
pub struct ReportOpts {
    /// BH level for every q-value the verdict reads.
    pub fdr: f64,
    /// `exposure_eta2` at or above which a pair is "cell-type confounded".
    pub max_exposure_eta2: f64,
    /// [`super::options::EffectOpts::min_arm`], to say why a pair is not estimable.
    pub min_arm: usize,
}

impl Default for ReportOpts {
    /// What [`report`] reads of [`INTERNALS`].
    fn default() -> Self {
        Self {
            fdr: INTERNALS.fdr,
            max_exposure_eta2: INTERNALS.max_exposure_eta2,
            min_arm: INTERNALS.min_arm,
        }
    }
}

/// Benjamini–Hochberg q-values. Non-finite p are left out of the count and
/// get NaN.
pub fn bh(p: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..p.len()).filter(|&i| p[i].is_finite()).collect();
    idx.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let m = idx.len() as f64;
    let mut q = vec![f64::NAN; p.len()];
    let mut run = 1.0f64;
    for (r, &i) in idx.iter().enumerate().rev() {
        run = run.min(p[i] * m / (r + 1) as f64);
        q[i] = run;
    }
    q
}

/// Simes p of one family, `min_r p_(r) k / r` over its `k` finite p; NaN
/// when there are none.
pub fn simes(p: &[f64]) -> f64 {
    let mut v: Vec<f64> = p.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let k = v.len() as f64;
    v.iter()
        .enumerate()
        .map(|(r, &x)| x * k / (r + 1) as f64)
        .fold(1.0, f64::min)
}

/// p-values, BH q-values and verdicts for every row, within each section
/// (the pooled rows, `all`, form their own family).
///
/// Pairs sharing a ligand share its exposure (the same `T`), and pairs
/// sharing senders share neighbourhoods, so their tests are positively
/// correlated. BH stays valid under positive regression dependence (PRDS;
/// Benjamini & Yekutieli 2001, Ann Stat 29:1165), which one-sided normal
/// tests with non-negative correlations satisfy; for the two-sided effect
/// test it is an approximation. `ligand_q` asks the coarser question with
/// one test per ligand, so the strongest correlation, within a ligand, stays
/// inside a family: each ligand's headline p are combined by Simes
/// (`min_r p_(r) k / r` over its `k` pairs, valid under PRDS) and BH runs
/// over ligands. `n_pairs_same_ligand` is that `k`.
///
/// The controls (receptor-negative receivers, the background, and
/// the contrast with the first) are corrected only among the pairs
/// whose effect is significant, and are NaN for the rest: a control decides a
/// verdict only beside a significant effect, and over every pair its q would
/// rest on what unrelated pairs in the table happen to show.
pub fn report(rows: &mut [EffectRow], opts: &ReportOpts) {
    let mut by_section: std::collections::HashMap<Box<str>, Vec<usize>> = Default::default();
    for (i, r) in rows.iter().enumerate() {
        by_section.entry(r.section.clone()).or_default().push(i);
    }
    for idx in by_section.values() {
        let col = |g: &dyn Fn(&EffectRow) -> f64| -> Vec<f64> {
            idx.iter().map(|&i| g(&rows[i])).collect()
        };
        let pe = col(&|r| r.effects.competent.p());
        let pc = col(&|r| upper_normal_p(r.coexpr_log_or / r.coexpr_se));
        let qe = bh(&pe);
        let qc = bh(&pc);
        let control = |g: &dyn Fn(&EffectRow) -> f64| -> Vec<f64> {
            let p: Vec<f64> = col(g)
                .into_iter()
                .zip(&qe)
                .map(|(p, &q)| if q < opts.fdr { p } else { f64::NAN })
                .collect();
            bh(&p)
        };
        let q1 = control(&|r| r.effects.effect.p());
        let q3 = control(&|r| r.effects.incompetent.p());
        let qd = control(&|r| r.background_effect.p());
        let qo = control(&|r| r.effect_over_background.p());
        let mut by_ligand: std::collections::HashMap<String, Vec<usize>> = Default::default();
        for (k, &i) in idx.iter().enumerate() {
            by_ligand
                .entry(rows[i].ligand.to_ascii_uppercase())
                .or_default()
                .push(k);
        }
        let families: Vec<&Vec<usize>> = by_ligand.values().collect();
        let simes_p: Vec<f64> = families
            .iter()
            .map(|ks| simes(&ks.iter().map(|&k| pe[k]).collect::<Vec<_>>()))
            .collect();
        let ql = bh(&simes_p);
        for (g, ks) in families.iter().enumerate() {
            for &k in ks.iter() {
                let r = &mut rows[idx[k]];
                r.ligand_q = ql[g];
                r.n_pairs_same_ligand = ks.len();
            }
        }
        for (k, &i) in idx.iter().enumerate() {
            let r = &mut rows[i];
            r.effect_p = pe[k];
            r.effect_q = qe[k];
            r.coexpr_p = pc[k];
            r.coexpr_q = qc[k];
            r.effect_contrast_q = q1[k];
            r.effect_incompetent_q = q3[k];
            r.background_q = qd[k];
            r.effect_over_background_q = qo[k];
        }
    }
    for r in rows.iter_mut() {
        judge(r, opts);
    }
}

/// Verdict and warnings of one row from its q-values and diagnostics.
fn judge(row: &mut EffectRow, opts: &ReportOpts) {
    let head = row.effects.competent;
    let controls = failed_controls(row, opts);
    let ev = Evidence {
        estimable: head.theta.is_finite() && head.se.is_finite(),
        effect: row.effect_q < opts.fdr,
        coexpr: row.coexpr_q < opts.fdr,
        control_failed: controls.iter().any(|c| c.0),
        confounded: row.exposure_eta2 >= opts.max_exposure_eta2,
    };
    row.verdict = verdict(ev);
    row.warnings = warnings(row, ev, &controls, opts);
}

/// A condition of a row and the warning it raises when it holds.
type Flag = (bool, &'static str);

/// Beside a significant effect, the effect must exceed each negative
/// control, the same way, and a comparison that could not run has not held,
/// but for a background programme where none exists (no match in the pool,
/// or a section's row). A hidden field moves the controls too, so one being
/// significant does not rule the effect out: it is reported ([`control_notes`]),
/// and what decides is whether the effect is larger.
fn failed_controls(row: &EffectRow, opts: &ReportOpts) -> [Flag; 4] {
    let sig = |q: f64| q < opts.fdr;
    let head = row.effects.competent;
    let effect = sig(row.effect_q);
    let unchecked = |q: f64| effect && q.is_nan();
    let exceeds = |q: f64, over: Estimate| sig(q) && over.theta * head.theta > 0.0;
    let beaten = |q: f64, over: Estimate| effect && !q.is_nan() && !exceeds(q, over);
    let over_negative_q = row.effect_contrast_q;
    let over_background_q = row.effect_over_background_q;
    let has_background = row.n_background_genes > 0;
    [
        (
            unchecked(over_negative_q),
            "receptor-negative receivers could not be checked",
        ),
        (
            beaten(over_negative_q, row.effects.effect),
            "the effect does not exceed the one in receptor-negative receivers",
        ),
        (
            has_background && unchecked(over_background_q),
            "the matched background programme could not be checked",
        ),
        (
            has_background && beaten(over_background_q, row.effect_over_background),
            "the effect does not exceed the one on a matched background programme",
        ),
    ]
}

/// The negative controls significant on their own: reported, deciding nothing.
fn control_notes(row: &EffectRow, opts: &ReportOpts) -> [Flag; 2] {
    let sig = |q: f64| q < opts.fdr;
    [
        (
            sig(row.effect_incompetent_q),
            "receptor-negative receivers respond",
        ),
        (
            sig(row.background_q),
            "a matched background programme responds too",
        ),
    ]
}

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

/// The row's warnings, joined: why it has no estimate, what confounds it, the
/// controls it failed, then the controls that respond.
fn warnings(row: &EffectRow, ev: Evidence, controls: &[Flag], opts: &ReportOpts) -> String {
    let mut w: Vec<String> = Vec::new();
    if !ev.estimable {
        w.push(not_estimable_reason(row, opts));
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
    w.extend(
        controls
            .iter()
            .chain(&control_notes(row, opts))
            .filter(|c| c.0)
            .map(|c| c.1.to_string()),
    );
    w.join("; ")
}
