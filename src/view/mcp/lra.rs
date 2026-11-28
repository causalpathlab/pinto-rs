//! `view_lra`: a run's contact-effect tables and their pairs, listed or
//! one pair on the map.

use super::super::lra::{self as lra_layer, LraView, PairShow, Sort};
use super::{value_enum, Open};
use crate::lr_activity::ligand_effect::output::{has_estimate, CONTROLS};
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
    search: Option<String>,
    sort: Option<String>,
    limit: Option<usize>,
    pair: Option<String>,
    show: Option<String>,
    /// Pairs to mark and unmark, by name, in the table shown.
    mark: Vec<String>,
    unmark: Vec<String>,
    /// Mark every pair the list shows (`M` in the terminal).
    mark_shown: bool,
    /// Unmark every pair.
    clear_marks: bool,
    /// Start `pinto lra --only` on the marked pairs, or the one selected
    /// when none is (ctrl+r in the terminal), as a background job.
    start: bool,
}

impl Open {
    /// `view_lra`: the run's contact-effect tables and their pairs, listed
    /// as asked; with `pair`, that pair on the map from now on. What it is
    /// given stays set, as `view_map`'s arguments do.
    pub(super) fn lra(&mut self, a: LraArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
        let start = a.start;
        let reply = self.lra_with(a, &|msg| cx.progress(msg))?;
        let mut out = vec![Content::Json(reply)];
        if start {
            out.extend(self.start_marked(cx)?);
        }
        Ok(out)
    }

    /// `start`: `pinto lra --only` on the marked pairs, or the one selected,
    /// with the options of the table shown, as the terminal's ctrl+r starts
    /// it, run as `run_lra` runs it (a finished subset is read back).
    fn start_marked(&self, cx: &Call) -> anyhow::Result<Vec<Content>> {
        let v = self.lra.as_ref().expect("the layer is read");
        let only = v.pairs_to_run();
        anyhow::ensure!(
            !only.is_empty(),
            "nothing to run: mark pairs (mark, mark_shown) or select one (pair) first"
        );
        let run = &self.base.run;
        crate::run::lra_subset::start_subset(&run.manifest, &run.prefix, &v.options(), &only, cx)
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
                        shown_path(&run.manifest)
                    )
                })?;
                self.lra = Some(view);
            }
        }
        // What changes, checked before any of it is kept.
        let v = self.lra.as_mut().expect("read above");
        let mut filter = v.filter().clone();
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
        // Marks, by the names the table shown from now on gives its pairs.
        let named = |name: &String| -> anyhow::Result<String> {
            let i = LraView::resolve(table, name)?
                .ok_or_else(|| anyhow::anyhow!("mark and unmark take pair names, not \"\""))?;
            Ok(table.rows[i].name())
        };
        let mark: Vec<String> = a.mark.iter().map(named).collect::<anyhow::Result<_>>()?;
        let unmark: Vec<String> = a.unmark.iter().map(named).collect::<anyhow::Result<_>>()?;
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
        if a.clear_marks {
            v.marked.clear();
        }
        if a.mark_shown {
            let shown = v.shown_names();
            v.marked.extend(shown);
        }
        v.marked.extend(mark);
        for name in &unmark {
            v.marked.remove(name);
        }
        if let Some(show) = show {
            v.show = show;
        }
        Ok(self.lra_reply(a.limit.unwrap_or(LIST)))
    }

    /// The lra layer as `view_lra` says it: the table, how many pairs,
    /// the pairs listed, and the pair selected.
    fn lra_reply(&self, limit: usize) -> Value {
        let v = self.lra.as_ref().expect("the layer is read");
        let table = v.table();
        let rows = &table.rows;
        let estimable = rows
            .iter()
            .filter(|r| has_estimate(r.effect, r.effect_dr_p))
            .count();
        let shown = v.visible();
        let list: Vec<Value> = shown
            .iter()
            .take(limit)
            .map(|&i| {
                let r = &rows[i];
                json!({
                    "pair": r.name(),
                    "change": lra_layer::change(r),
                    "p": opt_finite(r.effect_dr_p),
                    "marked": v.marked.contains(&r.name()),
                })
            })
            .collect();
        let q_scope = table.q_scope.to_string();
        let selected = v.selected_row().map(|r| {
            let curve = table.curve(v.selected.expect("selected"));
            let mut out = json!({
                "pair": r.name(),
                "warnings": r.warnings,
                "change": lra_layer::change(r),
                "effect_dr": opt_finite(r.effect),
                "effect_dr_se": opt_finite(r.effect_dr_se),
                "effect_dr_p": opt_finite(r.effect_dr_p),
                "coexpr_lfdr": opt_finite(r.coexpr_lfdr),
                "subset_fitted": table.subset_fitted(),
                "threshold": opt_finite(r.threshold),
                "tau_top": opt_finite(r.tau_top),
                "tau_top_weight": opt_finite(r.tau_top_weight),
                "controls": CONTROLS.iter().zip(&r.controls).map(|(spec, c)| json!({
                    "control": spec.key,
                    "effect_minus_control": opt_finite(c.over),
                    "p": opt_finite(c.p),
                })).collect::<Vec<_>>(),
                "receivers": r.n_receivers,
                "competent": r.n_competent,
                "exposed": r.n_exposed,
                "curve_points": curve.len(),
                "curve": curve.iter().map(|p| json!({
                    "tau": opt_finite(p.tau),
                    "quantile": opt_finite(p.quantile),
                    "effect": opt_finite(p.effect),
                    "lo": opt_finite(p.lo),
                    "hi": opt_finite(p.hi),
                    "band_lo": opt_finite(p.band_lo),
                    "band_hi": opt_finite(p.band_hi),
                })).collect::<Vec<_>>(),
            });
            if let Some(on) = v.shown() {
                let classes: serde_json::Map<_, _> = on
                    .legend()
                    .iter()
                    .map(|(name, _, n)| ((*name).to_string(), Value::from(*n)))
                    .collect();
                let (in_scope, active) = on.link_counts().unzip();
                out["map"] = json!({
                    "classes": classes,
                    "link_community": v.options().link_community,
                    "exposed": on.map.count_in_scope(&on.map.exposed),
                    "competent": on.map.count_in_scope(&on.map.competent),
                    "links_in_scope": in_scope,
                    "links_active": active,
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
            "estimable": estimable,
            "filter": {
                "search": f.search,
                "sort": f.sort.name(),
            },
            "shown": shown.len(),
            "marked": v.marked.iter().collect::<Vec<_>>(),
            "list": list,
            "selected": selected,
            "show": super::super::value_name(&v.show),
            "reading": "change: the headline effect as a percent change (exp(effect) - 1) with \
                        its 95% CI, averaged over exposure thresholds by their evidence \
                        (one threshold when lra was given --threshold); \
                        tau_top: the threshold the average weighs most, at weight \
                        tau_top_weight; threshold: the one the counts and the map read; \
                        controls: the effect minus the negative control \
                        (receptor-negative receivers) as a log rate ratio, with its \
                        two-sided p (a small p with a negative difference falls short of \
                        the control): null for a control that could not be checked; \
                        p: the two-sided p of the effect, its SE widened by a tile \
                        jackknife; coexpr_lfdr: co-expression's local false discovery \
                        rate under the null fitted to the table's pairs (q_scope), null \
                        when too few pairs to fit it; warnings: why a pair has no \
                        estimate. lra judges nothing; the reader decides. With \
                        a pair, view_map draws it: senders (ligand+), receivers exposed or not \
                        (contact exposure above threshold) and receptor+ or not; show: exposure \
                        draws the exposure itself. A --link-community table's exposure runs \
                        through its community's links alone: cells without one are no \
                        receivers (link_community, and drawn as none). With edges on, zoomed in, \
                        the links show: links_in_scope (the contacts the exposure sums over) \
                        faint, links_active (a sender's ligand reaching an exposed \
                        receptor+ receiver) from the sender's colour to the receiver's, \
                        more opaque the larger the share of the receiver's exposure (a ligand+ \
                        cell is drawn as a sender, though it can receive too). \
                        curve: the threshold-response points (effect with its CI lo, hi \
                        and simultaneous band at each threshold tau); view_map curve: \
                        true draws them under the map, view_export curve: true writes them.",
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
