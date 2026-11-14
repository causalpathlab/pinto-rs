//! Relabelling in the viewer: `R` stages labels and merges in a [`Draft`]
//! beside the round, and `S` sends them to `lupin relabel` after a
//! confirmation; lupin writes the next round and the viewer opens it. `P`
//! asks lupin what the draft would change without writing anything.

use super::super::draft::{Draft, Mark, Merge, Verdict};
use super::super::lupin;
use super::super::markers;
use super::annotate::{Modal, Prompt};
use super::{rgb, App, Pick, Show};
use crate::tui::style::{self, short};
use legume_tui::save::Field;
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::style::Style as TStyle;
use ratatui::text::{Line, Span};
use ratatui::DefaultTerminal;
use std::collections::BTreeSet;

pub const RELABEL_HELP: &[&str] = &[
    " → ← or ] [  next/prev cluster",
    " ↑ ↓ or click  choose a gene",
    " y marker of the working type",
    " n drop it  space clear  Tab type",
    " L label  K keep  M merge  u undo",
    " P preview  S apply (writes a round)",
    " z/Z zoom  s save  ? all keys",
    " R or Esc leave (the draft is kept)",
];

/// Relabelling a round's clusters.
pub struct Relabel {
    pub draft: Draft,
    /// Cluster ids in visiting order: those needing a look first.
    order: Vec<i64>,
    at: usize,
    /// The cell type `+`/`-` edit.
    target: Option<String>,
    /// Clusters chosen to merge, while choosing.
    merge: Option<BTreeSet<i64>>,
    /// While choosing a merge, the cluster `[ ]` point at and space adds.
    cursor: usize,
    /// lupin's last `--preview` of the draft.
    pub(super) preview: Option<serde_json::Value>,
}

impl Relabel {
    fn id(&self) -> i64 {
        self.order[self.at]
    }
}

pub(super) enum Staging {
    Label(i64),
    Keep(i64),
    Merge(Vec<i64>),
}

impl App<'_> {
    /// `R`: start or stop relabelling the round's clusters.
    pub(super) fn toggle_relabel(
        &mut self,
        terminal: Option<&mut DefaultTerminal>,
    ) -> anyhow::Result<()> {
        if self.relabel.is_some() {
            self.leave_relabel();
            return Ok(());
        }
        if self.job.is_some() {
            self.status = "lupin is still running".into();
            return Ok(());
        }
        if self.round.is_none() {
            self.step_show(1, terminal)?;
            if self.round.is_none() {
                return Ok(());
            }
        }
        if let Err(e) = lupin::check() {
            self.fail(e);
            return Ok(());
        }
        self.set_show(Show::Clusters);
        let round = self.round.as_ref().expect("loaded");
        let draft = Draft::load(&round.path);
        // Unlabelled and disputed clusters first, then the largest.
        let mut order: Vec<i64> = round
            .ids
            .iter()
            .copied()
            .filter(|id| round.reviewed.contains_key(id))
            .collect();
        order.sort_by_key(|id| {
            let d = &round.reviewed[id].digest;
            let settled =
                d.label.is_some() && d.evidence.as_ref().and_then(|e| e.agrees) != Some(false);
            (settled, std::cmp::Reverse(d.size))
        });
        if order.is_empty() {
            self.status = "this round has no clusters to review".into();
            return Ok(());
        }
        let staged = draft.len();
        self.relabel = Some(Relabel {
            draft,
            order,
            at: 0,
            target: None,
            merge: None,
            cursor: 0,
            preview: None,
        });
        self.visit(0);
        self.status = if staged > 0 {
            format!("relabelling; {staged} decisions staged earlier")
        } else {
            "relabelling: → ← clusters, ↑ ↓ genes, y/n marks, L label".into()
        };
        Ok(())
    }

    /// Stop relabelling; the draft stays on disk.
    pub(super) fn leave_relabel(&mut self) {
        if let Some(r) = self.relabel.take() {
            if let Err(e) = r.draft.save() {
                self.fail(format!("draft not saved: {e}"));
            } else if !r.draft.is_empty() {
                self.status = format!("{} decisions kept for later (R)", r.draft.len());
            }
            self.focus.fill(false);
            self.shown = None;
            self.markers.clear();
            self.gene = None;
            self.need_map = true;
        }
    }

    /// Show cluster `order[at]`: it (and any merge being chosen) in colour,
    /// its markers listed.
    fn visit(&mut self, at: usize) {
        let (Some(r), Some(round)) = (self.relabel.as_mut(), self.round.as_ref()) else {
            return;
        };
        r.at = at.min(r.order.len() - 1);
        let id = r.id();
        r.target = default_target(r, round, id);
        let mut focus = vec![false; round.ids.len()];
        let pointed = r.merge.as_ref().map(|_| r.order[r.cursor]);
        for c in std::iter::once(id)
            .chain(pointed)
            .chain(r.merge.iter().flatten().copied())
        {
            if let Some(g) = round.group(c) {
                focus[g] = true;
            }
        }
        let g = round.group(id);
        self.gene = None;
        self.set_focus(focus, g);
    }

    /// A click on group `g` of the cluster map or the list: visit it, or
    /// while choosing a merge, add or remove it.
    pub(super) fn visit_group(&mut self, g: usize) {
        let (Some(r), Some(round)) = (self.relabel.as_mut(), self.round.as_ref()) else {
            return;
        };
        let Some(&id) = round.ids.get(g) else {
            return;
        };
        if let Some(set) = r.merge.as_mut() {
            if id != r.order[r.at] && !set.remove(&id) {
                set.insert(id);
            }
            let at = r.at;
            self.visit(at);
            return;
        }
        if let Some(at) = r.order.iter().position(|&c| c == id) {
            self.visit(at);
        }
    }

    /// Keys while relabelling; `false` passes a key on to the map (zoom,
    /// save, help, quit and the looks).
    pub(super) fn relabel_key(&mut self, key: KeyEvent) -> anyhow::Result<bool> {
        let r = self.relabel.as_ref().expect("relabelling");
        if r.merge.is_some() {
            return Ok(self.merge_key(key));
        }
        let (at, n) = (r.at, r.order.len());
        match key.code {
            KeyCode::Esc | KeyCode::Char('R') => self.leave_relabel(),
            KeyCode::Right | KeyCode::Char(']') => self.visit((at + 1) % n),
            KeyCode::Left | KeyCode::Char('[') => self.visit((at + n - 1) % n),
            KeyCode::Down => self.step_gene(1),
            KeyCode::Up => self.step_gene(-1),
            KeyCode::Tab => self.step_target(1),
            KeyCode::BackTab => self.step_target(-1),
            KeyCode::Char('y') => self.mark(true),
            KeyCode::Char('n') => self.mark(false),
            KeyCode::Char(' ') => self.clear_mark(),
            KeyCode::Char('L') => self.ask(false),
            KeyCode::Char('K') => self.ask(true),
            KeyCode::Char('M') => self.begin_merge(),
            KeyCode::Char('u') => self.unstage(),
            KeyCode::Char('P') => self.send_draft(true),
            KeyCode::Char('S') => self.confirm_draft(),
            KeyCode::Char('x') => {
                self.gene = None;
                self.need_map = true;
            }
            KeyCode::Char('l' | 'k' | 'm') => {
                self.status = "decisions are uppercase: L labels, K keeps, M merges".into();
            }
            KeyCode::Char('c' | 'C' | ',' | '.' | 'H' | 'A' | 'r' | '4') | KeyCode::Home => {
                self.status = "leave relabel mode first (R)".into();
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Keys while choosing clusters to merge: ↑ ↓ move, space chooses,
    /// Enter names, Esc or `M` cancels; `q` and `?` pass, the rest is refused.
    fn merge_key(&mut self, key: KeyEvent) -> bool {
        let r = self.relabel.as_mut().expect("relabelling");
        let (at, n) = (r.at, r.order.len());
        match key.code {
            KeyCode::Esc | KeyCode::Char('M') => {
                r.merge = None;
                self.visit(at);
                self.status = "merge cancelled".into();
            }
            KeyCode::Down | KeyCode::Up => {
                let by = if key.code == KeyCode::Down { 1 } else { n - 1 };
                r.cursor = (r.cursor + by) % n;
                self.visit(at);
            }
            KeyCode::Char(' ') => {
                let id = r.order[r.cursor];
                if let Some(g) = self.round.as_ref().and_then(|round| round.group(id)) {
                    self.visit_group(g);
                }
            }
            KeyCode::Enter => self.ask_merge(),
            KeyCode::Char('q' | '?') => return false,
            _ => {
                self.status =
                    "in merge mode: ↑ ↓ move, space chooses, enter names, esc cancels".into();
            }
        }
        true
    }

    /// Candidate types for cluster `id`: staged, current, called, panel.
    fn candidates(&self, id: i64) -> Vec<String> {
        let (Some(r), Some(round)) = (self.relabel.as_ref(), self.round.as_ref()) else {
            return Vec::new();
        };
        let staged = r.draft.verdicts.get(&id).map(|v| match v {
            Verdict::Label { label, .. } | Verdict::Keep { label, .. } => label.clone(),
        });
        let merged = r
            .draft
            .merge_of(id)
            .map(|m| r.draft.merges[m].label.clone());
        let digest = round.reviewed.get(&id).map(|d| &d.digest);
        let current = digest.and_then(|d| d.label.clone());
        let calls = digest
            .into_iter()
            .flat_map(|d| d.calls.iter().map(|c| c.label.clone()));
        let mut out: Vec<String> = Vec::new();
        for l in staged
            .into_iter()
            .chain(merged)
            .chain(current)
            .chain(calls)
            .chain(round.known_labels())
        {
            if !out
                .iter()
                .any(|o| lupin::label_key(o) == lupin::label_key(&l))
            {
                out.push(l);
            }
        }
        out
    }

    fn step_target(&mut self, by: isize) {
        let Some(r) = self.relabel.as_ref() else {
            return;
        };
        let options = self.candidates(r.id());
        if options.is_empty() {
            return;
        }
        let at = r
            .target
            .as_ref()
            .and_then(|t| options.iter().position(|o| o == t));
        let next = match at {
            Some(i) => (i as isize + by).rem_euclid(options.len() as isize) as usize,
            None => 0,
        };
        let target = options[next].clone();
        self.status = format!("working type: {target} (y adds its markers)");
        self.relabel.as_mut().expect("relabelling").target = Some(target);
    }

    /// `y`: the gene on the map becomes a marker of the working type.
    /// `n`: it is dropped from the type that lists it (the working type
    /// when that one does). Either moves on to the next gene.
    fn mark(&mut self, add: bool) {
        let Some(feature) = self.gene.as_ref().map(|g| g.feature.clone()) else {
            self.status = "choose a gene first: ↑ ↓ or a click".into();
            return;
        };
        let gene = markers::symbol(&feature).to_string();
        let (Some(r), Some(round)) = (self.relabel.as_mut(), self.round.as_ref()) else {
            return;
        };
        let Some(working) = r.target.clone() else {
            self.status = "no working type: Tab picks one".into();
            return;
        };
        let lists = |t: &str| {
            round
                .panel
                .as_ref()
                .and_then(|p| p.genes(t))
                .is_some_and(|g| g.iter().any(|x| x.eq_ignore_ascii_case(&gene)))
        };
        // The type the edit is about: the working type to add to; for a
        // drop, the working type if it lists the gene, else the one that does.
        let label = if add || lists(&working) {
            working
        } else {
            let others = round
                .panel
                .as_ref()
                .map(|p| p.types_of(&gene))
                .unwrap_or_default();
            match others.first() {
                Some(t) => t.to_string(),
                None => {
                    self.status = format!("{gene} is no type's marker");
                    return;
                }
            }
        };
        let staged = r.draft.mark_of(&gene).is_some_and(|m| {
            lupin::label_key(&m.label) == lupin::label_key(&label) && m.add == add
        });
        if add && lists(&label) && !staged {
            self.status = format!("{gene} is already one of {label}'s markers");
            return;
        }
        let now = r.draft.toggle_mark(Mark {
            label: label.clone(),
            feature: gene.clone(),
            add,
        });
        let verb = if add { "add to" } else { "drop from" };
        self.status = if now {
            format!("{gene}: {verb} {label}'s markers")
        } else {
            format!("{gene}: {label}'s markers left as they were")
        };
        self.save_draft();
        self.step_gene(1);
    }

    /// Space: take back the mark on the gene on the map, and move on.
    fn clear_mark(&mut self) {
        let Some(feature) = self.gene.as_ref().map(|g| g.feature.clone()) else {
            self.status = "choose a gene first: ↑ ↓ or a click".into();
            return;
        };
        let gene = markers::symbol(&feature).to_string();
        let Some(r) = self.relabel.as_mut() else {
            return;
        };
        self.status = if r.draft.clear_mark(&gene) {
            format!("{gene}: no mark")
        } else {
            format!("{gene} has no mark")
        };
        self.save_draft();
        self.step_gene(1);
    }

    /// `L` (or `K` to keep the label): ask for a label and a rationale.
    fn ask(&mut self, keep: bool) {
        let (Some(r), Some(round)) = (self.relabel.as_ref(), self.round.as_ref()) else {
            return;
        };
        let id = r.id();
        if r.draft.merge_of(id).is_some() {
            self.status = format!("K{id} is in a staged merge; u takes it back");
            return;
        }
        let digest = &round.reviewed[&id].digest;
        let q_of = |l: &str| {
            digest
                .calls
                .iter()
                .find(|c| lupin::label_key(&c.label) == lupin::label_key(l))
                .and_then(|c| c.q)
        };
        let (what, label, field) = if keep {
            let Some(label) = digest.label.clone() else {
                self.status = format!("K{id} has no label to keep; L labels it");
                return;
            };
            (Staging::Keep(id), label, 1)
        } else {
            let label = r.target.clone().unwrap_or_default();
            (Staging::Label(id), label, 0)
        };
        let rationale = match q_of(&label) {
            Some(q) if q < round.alpha => format!("{label} called at q={q:.3}"),
            _ if keep => "kept after review in pinto view".into(),
            _ => String::new(),
        };
        let options = self.candidates(id);
        self.modal = Some(Modal::Prompt(Prompt {
            what,
            field,
            label: Field::new(label),
            rationale: Field::new(rationale),
            options,
        }));
        self.need_map = true;
    }

    fn begin_merge(&mut self) {
        let r = self.relabel.as_mut().expect("relabelling");
        let id = r.id();
        if r.draft.merge_of(id).is_some() {
            self.status = format!("K{id} is already in a staged merge; u takes it back");
            return;
        }
        r.merge = Some(BTreeSet::new());
        r.cursor = r.at;
        self.status = format!("merge K{id} with: ↑ ↓ and space (or clicks), enter names");
    }

    fn ask_merge(&mut self) {
        let (Some(r), Some(round)) = (self.relabel.as_ref(), self.round.as_ref()) else {
            return;
        };
        let mut ids: Vec<i64> = r.merge.iter().flatten().copied().collect();
        ids.push(r.id());
        ids.sort_unstable();
        ids.dedup();
        if ids.len() < 2 {
            self.status = "click another cluster to merge with".into();
            return;
        }
        if let Some(&c) = ids.iter().find(|&&c| r.draft.merge_of(c).is_some()) {
            self.status = format!("K{c} is already in a staged merge");
            return;
        }
        // The largest cluster's label, as a start.
        let largest = ids
            .iter()
            .max_by_key(|c| round.reviewed.get(c).map_or(0, |d| d.digest.size))
            .copied()
            .unwrap_or(ids[0]);
        let label = r
            .target
            .clone()
            .or_else(|| round.label(largest).map(String::from))
            .unwrap_or_default();
        let names: Vec<String> = ids.iter().map(|c| format!("K{c}")).collect();
        let rationale = format!("{} share one program", names.join(" "));
        let options = self.candidates(r.id());
        self.modal = Some(Modal::Prompt(Prompt {
            what: Staging::Merge(ids),
            field: 0,
            label: Field::new(label),
            rationale: Field::new(rationale),
            options,
        }));
        self.need_map = true;
    }

    pub(super) fn stage(&mut self, p: Prompt) {
        let Some(r) = self.relabel.as_mut() else {
            return;
        };
        let label = p.label.text.trim().to_string();
        let rationale = p.rationale.text.trim().to_string();
        match p.what {
            Staging::Label(id) => {
                r.draft.verdicts.insert(
                    id,
                    Verdict::Label {
                        label: label.clone(),
                        rationale,
                    },
                );
                self.status = format!("K{id} → {label} staged");
            }
            Staging::Keep(id) => {
                r.draft.verdicts.insert(
                    id,
                    Verdict::Keep {
                        label: label.clone(),
                        rationale,
                    },
                );
                self.status = format!("K{id} keeps {label}");
            }
            Staging::Merge(ids) => {
                for c in &ids {
                    r.draft.verdicts.remove(c);
                }
                let names: Vec<String> = ids.iter().map(|c| format!("K{c}")).collect();
                self.status = format!("merge {} → {label} staged", names.join("+"));
                r.draft.merges.push(Merge {
                    clusters: ids,
                    label,
                    rationale,
                });
                r.merge = None;
            }
        }
        r.preview = None;
        self.save_draft();
        self.next_undecided();
    }

    /// Visit the next cluster with nothing staged.
    fn next_undecided(&mut self) {
        let Some(r) = self.relabel.as_ref() else {
            return;
        };
        let n = r.order.len();
        let open = |c: &i64| !r.draft.verdicts.contains_key(c) && r.draft.merge_of(*c).is_none();
        let next = (1..=n)
            .map(|k| (r.at + k) % n)
            .find(|&i| open(&r.order[i]))
            .unwrap_or(r.at);
        self.visit(next);
    }

    fn unstage(&mut self) {
        let Some(r) = self.relabel.as_mut() else {
            return;
        };
        let id = r.id();
        if r.draft.unstage(id) {
            r.preview = None;
            self.status = format!("K{id}: nothing staged");
            self.save_draft();
        } else {
            self.status = format!("nothing staged on K{id}");
        }
    }

    fn save_draft(&mut self) {
        if let Some(r) = self.relabel.as_ref() {
            if let Err(e) = r.draft.save() {
                self.fail(format!("draft not saved: {e}"));
            }
        }
    }

    fn confirm_draft(&mut self) {
        let Some(r) = self.relabel.as_ref() else {
            return;
        };
        if r.draft.is_empty() {
            self.status = "nothing staged: L, K, M or y/n first".into();
            return;
        }
        if self.job.is_some() {
            self.status = "lupin is still running".into();
            return;
        }
        self.modal = Some(Modal::Confirm(r.draft.summary()));
        self.need_map = true;
    }

    /// Send the draft to `lupin relabel`: a preview, or the next round.
    pub(super) fn send_draft(&mut self, preview: bool) {
        let (Some(r), Some(round)) = (self.relabel.as_ref(), self.round.as_ref()) else {
            return;
        };
        if r.draft.is_empty() {
            self.status = "nothing staged: L, K, M or y/n first".into();
            return;
        }
        if self.job.is_some() {
            self.status = "lupin is still running".into();
            return;
        }
        self.job = Some(lupin::relabel(&round.path, r.draft.decisions(), preview));
        self.status = if preview {
            "asking lupin for a preview ...".into()
        } else {
            "lupin is writing the next round ...".into()
        };
    }

    /// The relabelling panel: the visited cluster, its markers, the list.
    pub(super) fn relabel_lines(&self, room: usize) -> Vec<(Line<'static>, Option<Pick>)> {
        // Labels and names take what the panel has beyond its usual width.
        let more = self.more();
        let (Some(r), Some(round)) = (self.relabel.as_ref(), self.round.as_ref()) else {
            return Vec::new();
        };
        let bold = style::bold();
        let dim = style::dim();
        let level = self.level();
        let id = r.id();
        let g = round.group(id);
        let digest = &round.reviewed[&id].digest;
        let swatch = |g: Option<usize>| {
            Span::styled(
                "██",
                TStyle::default().fg(rgb(g.map_or([128; 3], |g| level.palette[g]))),
            )
        };
        let row = |k: &str, v: String| {
            (
                Line::from(vec![Span::styled(format!("   {k:<8}"), dim), Span::raw(v)]),
                None,
            )
        };
        let mut out: Vec<(Line<'static>, Option<Pick>)> = vec![
            (Line::raw(""), None),
            (
                Line::from(vec![
                    Span::raw(" "),
                    swatch(g),
                    Span::styled(
                        format!(" K{id}  {} cells", super::thousands(digest.size)),
                        bold,
                    ),
                    Span::styled(format!("  {}/{}", r.at + 1, r.order.len()), dim),
                ]),
                None,
            ),
            row("label", digest.label.clone().unwrap_or_else(|| "–".into())),
        ];
        let calls: Vec<String> = digest
            .calls
            .iter()
            .filter(|c| c.q.is_some_and(|q| q < round.alpha))
            .take(3)
            .map(|c| {
                format!(
                    "{} {:.3}",
                    short(&c.label, 14 + more / 3),
                    c.q.unwrap_or(1.)
                )
            })
            .collect();
        out.push(row(
            "calls q",
            if calls.is_empty() {
                format!("none under {}", round.alpha)
            } else {
                calls.join("  ")
            },
        ));
        let staged = match (r.draft.verdicts.get(&id), r.draft.merge_of(id)) {
            (Some(Verdict::Label { label, .. }), _) => Some(format!("→ {label}")),
            (Some(Verdict::Keep { label, .. }), _) => Some(format!("keep {label}")),
            (None, Some(m)) => {
                let m = &r.draft.merges[m];
                let ids: Vec<String> = m.clusters.iter().map(|c| format!("K{c}")).collect();
                Some(format!("merge {} → {}", ids.join("+"), m.label))
            }
            (None, None) => None,
        };
        if let Some(s) = staged {
            out.push((
                Line::from(vec![
                    Span::styled("   staged  ", dim),
                    Span::styled(s, bold),
                ]),
                None,
            ));
        }
        if let Some(h) = round.reviewed[&id].history.last() {
            out.push(row("before", short(&history_line(h), 34 + more)));
        }
        if let Some(p) = preview_of(r.preview.as_ref(), id, more) {
            out.push(row("preview", p));
        }
        if let Some(set) = &r.merge {
            let mut ids: Vec<String> = set.iter().map(|c| format!("K{c}")).collect();
            ids.insert(0, format!("K{id}"));
            out.push((
                Line::styled(format!("   merging {}  Enter", ids.join("+")), bold),
                None,
            ));
        }
        let target = r.target.clone().unwrap_or_else(|| "–".into());
        out.push(row(
            "working",
            format!("{}  Tab", short(&target, 26 + more)),
        ));

        // Markers: • listed for the target, +/- staged edits.
        let panel_genes: Vec<String> = round
            .panel
            .as_ref()
            .and_then(|p| p.genes(&target))
            .map(|g| g.iter().map(|x| x.to_uppercase()).collect())
            .unwrap_or_default();
        let marker_rows = room.saturating_sub(out.len() + 8).clamp(3, 12);
        if !self.markers.is_empty() {
            out.push((
                Line::from(vec![
                    Span::styled("   markers", bold),
                    Span::styled("  fold  • listed  ↑↓ y n", dim),
                ]),
                None,
            ));
        }
        let drawn = self.gene.as_ref().map(|g| g.feature.as_ref());
        for (i, (feature, symbol, fold)) in self.markers.iter().take(marker_rows).enumerate() {
            let on = drawn
                .is_some_and(|d| d == feature.as_ref() || markers::symbol(d) == symbol.as_str());
            let listed = panel_genes.contains(&symbol.to_uppercase());
            let edit = r.draft.mark_of(symbol).map(|m| {
                let sign = if m.add { '+' } else { '-' };
                format!("{sign}{}", short(&m.label, 12))
            });
            let line = Line::from(vec![
                Span::raw(if on { " ▸ " } else { "   " }),
                Span::styled(
                    format!("{:<14} ×{fold:<6.1}", short(symbol, 14)),
                    if on { bold } else { TStyle::default() },
                ),
                Span::raw(if listed { "• " } else { "  " }),
                Span::styled(edit.unwrap_or_default(), bold),
            ]);
            out.push((line, Some(Pick::Gene(i))));
        }

        // Every cluster, in visiting order.
        out.push((Line::raw(""), None));
        out.push((
            Line::from(vec![
                Span::styled(" clusters", bold),
                Span::styled(
                    format!("  {} staged  P preview  S apply", r.draft.len()),
                    dim,
                ),
            ]),
            None,
        ));
        let rows = room.saturating_sub(out.len()).max(1);
        let first = style::first_row(r.at, rows, r.order.len());
        for (i, &c) in r.order.iter().enumerate().skip(first).take(rows) {
            let d = &round.reviewed[&c].digest;
            let g = round.group(c);
            let chosen = r.merge.as_ref().is_some_and(|s| s.contains(&c));
            let pointed = r.merge.is_some() && i == r.cursor;
            let mark = if i == r.at {
                "▸"
            } else if pointed {
                "›"
            } else if chosen {
                "+"
            } else {
                " "
            };
            let disputed = d.evidence.as_ref().and_then(|e| e.agrees) == Some(false);
            let state = match (r.draft.verdicts.get(&c), r.draft.merge_of(c)) {
                (Some(Verdict::Label { label, .. }), _) => {
                    format!("→{}", short(label, 10 + more / 2))
                }
                (Some(Verdict::Keep { .. }), _) => "✓".into(),
                (None, Some(_)) => "merge".into(),
                (None, None) if d.label.is_none() => "?".into(),
                (None, None) if disputed => "!".into(),
                (None, None) => String::new(),
            };
            let label = d.label.as_deref().unwrap_or("–");
            let w = 14 + more / 2;
            let line = Line::from(vec![
                Span::raw(mark),
                swatch(g),
                Span::styled(
                    format!(
                        " K{c:<3} {:<w$} {:>8} {state}",
                        short(label, w),
                        super::thousands(d.size)
                    ),
                    if i == r.at { bold } else { TStyle::default() },
                ),
            ]);
            out.push((line, g.map(Pick::Community)));
        }
        out
    }
}

/// The type `+`/`-` edit by default: staged, else current, else a call
/// that passes the FDR level; none otherwise (Tab picks one).
fn default_target(r: &Relabel, round: &super::Round, id: i64) -> Option<String> {
    let staged = r.draft.verdicts.get(&id).map(|v| match v {
        Verdict::Label { label, .. } | Verdict::Keep { label, .. } => label.clone(),
    });
    let digest = round.reviewed.get(&id).map(|d| &d.digest);
    let called = digest.and_then(|d| {
        d.calls
            .first()
            .filter(|c| c.q.is_some_and(|q| q < round.alpha))
            .map(|c| c.label.clone())
    });
    staged
        .or_else(|| digest.and_then(|d| d.label.clone()))
        .or(called)
}

/// A past decision on a cluster: `merged K2+K6: rationale`.
fn history_line(h: &serde_json::Value) -> String {
    let get = |k: &str| h.get(k).and_then(|v| v.as_str()).unwrap_or("");
    let why = get("rationale");
    let from = h.get("merged_from").and_then(|v| v.as_array()).map(|ids| {
        let ids: Vec<String> = ids
            .iter()
            .filter_map(|i| i.as_i64())
            .map(|i| format!("K{i}"))
            .collect();
        ids.join("+")
    });
    match (get("action"), from) {
        ("merge", Some(from)) => format!("merged {from}: {why}"),
        (action, _) => format!("{action} {}: {why}", get("label")),
    }
}

/// What the preview says about cluster `id`: its label after, and its
/// top call once marker edits are rescored; labels take `more` columns
/// beyond their usual width.
fn preview_of(v: Option<&serde_json::Value>, id: i64, more: usize) -> Option<String> {
    let c = v?.get("clusters")?.get(id.to_string())?;
    let after = c.get("label_after").and_then(|l| l.as_str()).unwrap_or("–");
    let top = c
        .get("calls")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|call| {
            let label = call.get("label")?.as_str()?;
            let q = call.get("q")?.as_f64()?;
            Some(format!("  top {} {q:.3}", short(label, 12 + more / 2)))
        })
        .unwrap_or_default();
    Some(format!("→ {}{top}", short(after, 14 + more / 2)))
}
