//! `view_lra`: a run's contact-effect tables and their pairs, listed or
//! one pair on the map.

use super::super::lra::{self as lra_layer, LraView, PairShow, Sort};
use super::{value_enum, Open};
use crate::lr_activity::ligand_effect::output::{CONTROLS, Q_SCOPE_ALL};
use crate::lr_activity::ligand_effect::verdict::Verdict;
use crate::lr_activity::outputs::opt_finite;
use crate::mcp::shown_path;
use legume_mcp::{Call, Content};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;

/// Pairs `view_lra` lists without a limit.
const LIST: usize = 20;

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(in crate::view) struct LraArgs {
    table: Option<String>,
    verdict: Option<String>,
    search: Option<String>,
    sort: Option<String>,
    limit: Option<usize>,
    pair: Option<String>,
    show: Option<String>,
}

impl Open {
    /// `view_lra`: the run's contact-effect tables and their pairs, listed
    /// as asked; with `pair`, that pair on the map from now on. What it is
    /// given stays set, as `view_map`'s arguments do.
    pub(super) fn lra(&mut self, a: LraArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
        let reply = self.lra_with(a, &|msg| cx.progress(msg))?;
        Ok(vec![Content::Json(reply)])
    }

    /// [`Open::lra`]'s work and reply, telling `progress` what it reads.
    pub(in crate::view) fn lra_with(
        &mut self,
        a: LraArgs,
        progress: &dyn Fn(&str),
    ) -> anyhow::Result<Value> {
        // As the manifest says now: a run_lra may have ended since it was opened.
        let run = &self.base.run;
        let tables = super::super::data::Run::open(&run.prefix)
            .map_or_else(|_| run.tables(), |now| now.tables());
        match self.lra.as_mut() {
            Some(v) => v.add_tables(tables),
            None => {
                let view = LraView::new(tables)?.ok_or_else(|| {
                    anyhow::anyhow!(
                        "{} has no lra table: run_lra on {} first",
                        run.name(),
                        run.manifest
                            .as_deref()
                            .map_or_else(|| run.prefix.clone(), shown_path)
                    )
                })?;
                self.lra = Some(view);
            }
        }
        // What changes, checked before any of it is kept.
        let v = self.lra.as_mut().expect("read above");
        let mut filter = v.filter().clone();
        if let Some(code) = a.verdict.as_deref() {
            filter.verdict = match code {
                "" | "all" => None,
                code => Some(code.parse().map_err(|_| {
                    let codes: Vec<&str> = Verdict::ALL.iter().map(|v| v.code()).collect();
                    anyhow::anyhow!(
                        "verdict {code:?}: one of {}, or \"\" for all",
                        codes.join(", ")
                    )
                })?),
            };
        }
        if let Some(search) = &a.search {
            filter.search = search.trim().to_string();
        }
        if let Some(sort) = a.sort.as_deref() {
            filter.sort = value_enum::<Sort>(sort, "sort")?;
        }
        let show = a
            .show
            .as_deref()
            .map(|s| value_enum::<PairShow>(s, "show"))
            .transpose()?;
        let staged = match a.table.as_deref() {
            Some(t) => Some(match find_table(&v.tables, t) {
                Some(at) => v.stage(lra_layer::Which::At(at))?,
                None => v.stage(lra_layer::Which::Path(Box::new(lra_layer::TableRef::at(
                    Path::new(t),
                    &self.base.run,
                ))))?,
            }),
            None => None,
        };
        // The pair selected from now on, in the table shown from now on.
        let table = staged.as_ref().map_or_else(|| v.table(), |s| v.staged(s));
        let selected = match a.pair.as_deref() {
            Some(name) => LraView::resolve(table, name)?,
            None => v.selection_in(table),
        };
        // A pair asked for is drawn, and one drawn stays drawn in the
        // table switched to: drawn before anything is kept, as it is the
        // one thing left that can fail.
        let draw = selected.is_some() && (a.pair.is_some() || self.pair_drawn);
        if let (true, Some(i)) = (draw, selected) {
            let (row, options) = (table.rows[i].clone(), table.r.options.clone());
            if v.needs_receivers_for(&options) {
                progress("reading the run's receivers as lra saw them");
            }
            v.draw(&self.base, &options, &row, self.at)?;
        }
        if let Some(s) = staged {
            v.commit(s);
        }
        v.selected = selected;
        self.pair_drawn = draw;
        v.set_filter(filter);
        if let Some(show) = show {
            v.show = show;
        }
        Ok(self.lra_reply(a.limit.unwrap_or(LIST)))
    }

    /// The lra layer as `view_lra` says it: the table, the verdict counts,
    /// the pairs listed, and the pair selected.
    fn lra_reply(&self, limit: usize) -> Value {
        let v = self.lra.as_ref().expect("the layer is read");
        let table = v.table();
        let rows = &table.rows;
        let counts: serde_json::Map<String, Value> =
            Verdict::counts(rows.iter().map(|r| r.verdict))
                .iter()
                .map(|(v, n)| (v.code().into(), (*n).into()))
                .collect();
        let shown = v.visible();
        let list: Vec<Value> = shown
            .iter()
            .take(limit)
            .map(|&i| {
                let r = &rows[i];
                json!({
                    "pair": r.name(),
                    "verdict": r.verdict.code(),
                    "change": lra_layer::change(r),
                    "q": opt_finite(r.effect_q),
                    "marked": v.marked.contains(&r.name()),
                })
            })
            .collect();
        let q_scope: String = match &table.q_scope {
            _ if !table.subset_corrected() => Q_SCOPE_ALL.into(),
            Some(scope) => scope.to_string(),
            None => "q corrected within the --only subset".into(),
        };
        let selected = v.selected_row().map(|r| {
            let mut out = json!({
                "pair": r.name(),
                "verdict": r.verdict.code(),
                "note": r.note,
                "warnings": r.warnings,
                "change": lra_layer::change(r),
                "effect": opt_finite(r.effect),
                "effect_se": opt_finite(r.effect_se),
                "q": opt_finite(r.effect_q),
                "coexpr_q": opt_finite(r.coexpr_q),
                "subset_corrected": table.subset_corrected(),
                "threshold": opt_finite(r.threshold),
                "tau_top": opt_finite(r.tau_top),
                "tau_top_weight": opt_finite(r.tau_top_weight),
                "controls": CONTROLS.iter().zip(&r.controls).map(|(spec, c)| json!({
                    "control": spec.key,
                    "effect_minus_control": opt_finite(c.over),
                    "q": opt_finite(c.q),
                })).collect::<Vec<_>>(),
                "receivers": r.n_receivers,
                "competent": r.n_competent,
                "exposed": r.n_exposed,
                "curve_points": table.curve(v.selected.expect("selected")).len(),
            });
            if let Some(on) = v.shown() {
                let classes: serde_json::Map<_, _> = on
                    .legend()
                    .iter()
                    .map(|(name, _, n)| ((*name).to_string(), Value::from(*n)))
                    .collect();
                let count = |f: &[bool]| f.iter().filter(|&&x| x).count();
                out["map"] = json!({
                    "classes": classes,
                    "exposed": count(&on.map.exposed),
                    "competent": count(&on.map.competent),
                    "threshold": on.map.threshold,
                    "competence": on.map.competence,
                });
                out["drawn"] = self.pair_drawn.into();
            }
            out
        });
        let f = v.filter();
        json!({
            "table": table.r.label(),
            "path": shown_path(&table.r.path),
            "q_scope": q_scope,
            "tables": v.tables.iter().map(|t| json!({"table": t.label(), "path": shown_path(&t.path)})).collect::<Vec<_>>(),
            "pairs": rows.len(),
            "verdicts": counts,
            "filter": {
                "verdict": f.verdict.map(|v| v.code()),
                "search": f.search,
                "sort": f.sort.name(),
            },
            "shown": shown.len(),
            "list": list,
            "selected": selected,
            "show": super::super::value_name(&v.show),
            "reading": "change: the headline effect as a percent change (exp(effect) - 1) with \
                        its 95% CI, averaged over exposure thresholds by their evidence \
                        (one threshold when lra was given --threshold); \
                        tau_top: the threshold the average weighs most, at weight \
                        tau_top_weight; threshold: the one the counts and the map read; \
                        controls: the effect minus the negative control \
                        (receptor-negative receivers) as a log rate ratio, with its two-sided q \
                        (so a small q with a negative difference falls short of the \
                        control; verdict and warnings give the call), tested among the \
                        pairs with an effect: null for the rest, for a control that could \
                        not be checked, and in a table from before them; \
                        q: Benjamini-Hochberg over the table's pairs (q_scope). With \
                        a pair, view_map draws it: senders (ligand+), receivers exposed or not \
                        (contact exposure above threshold) and receptor+ or not; show: exposure \
                        draws the exposure itself. view_export curve: true writes its \
                        threshold-response curve.",
        })
    }
}

/// The table `name` names among `tables`: `full`, or a subset by its
/// list (`L1,L2`, order and case aside), or by its path.
fn find_table(tables: &[lra_layer::TableRef], name: &str) -> Option<usize> {
    let list: Vec<Box<str>> = name.split(',').map(|s| s.trim().into()).collect();
    let canonical = crate::lr_activity::ligand_effect::only::canonical(&list);
    let path = shown_path(Path::new(name));
    tables.iter().position(|t| {
        shown_path(&t.path) == path
            || match &t.subset {
                None => name.eq_ignore_ascii_case("full"),
                Some(s) => s.only == canonical,
            }
    })
}
