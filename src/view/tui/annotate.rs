//! Annotation in the viewer, through lupin: `A` annotates the level on
//! screen with a marker panel, `a` shows the round's cell types or
//! clusters, and `R` relabels and merges the round's clusters.
//!
//! Decisions are staged in a [`Draft`] beside the round and sent to
//! `lupin relabel` only on `S`, after a confirmation; lupin writes the next
//! round and the viewer opens it. `P` asks lupin what the draft would
//! change without writing anything.

use super::super::draft::{Draft, Mark, Merge, Verdict};
use super::super::lupin::{self, JobKind};
use super::super::markers;
use super::super::round::round_name;
use super::browse::Panels;
use super::{rgb, App, Pick, Show};
use crate::tui::browse::{Browser, Outcome};
use crate::tui::field::Field;
use crate::tui::style::{self, short, tail};
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::style::Style as TStyle;
use ratatui::text::{Line, Span};
use ratatui::DefaultTerminal;
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};

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
    preview: Option<serde_json::Value>,
}

impl Relabel {
    fn id(&self) -> i64 {
        self.order[self.at]
    }
}

/// A dialog that takes the keys.
pub enum Modal {
    /// Choosing a marker panel for `lupin annotate`.
    Browse(Browser<Panels>),
    Prompt(Prompt),
    /// The staged decisions, before they are sent.
    Confirm(Vec<String>),
    /// Naming an export.
    Save(super::save::SaveAs),
}

/// Typing a label, then a rationale.
pub struct Prompt {
    what: Staging,
    /// 0 the label, 1 the rationale.
    field: usize,
    label: Field,
    rationale: Field,
    /// Labels Tab completes to.
    options: Vec<String>,
}

enum Staging {
    Label(i64),
    Keep(i64),
    Merge(Vec<i64>),
}

impl App<'_> {
    // ── rounds ──────────────────────────────────────────────────────────

    /// Find the run's rounds; open `explicit` (from `--round`) or, when
    /// `--show` asks for a round, the newest.
    pub(super) fn open_rounds(&mut self, explicit: Option<PathBuf>) {
        if let Some(manifest) = &self.base.run.manifest {
            self.rounds = lupin::latest_rounds(manifest);
        }
        if let Some(p) = explicit {
            self.rounds.retain(|r| !same_file(r, &p));
            self.rounds.insert(0, p);
        }
        if self.args.show != Show::Communities || self.args.round.is_some() {
            if let Some(path) = self.rounds.first().cloned() {
                match self.load_round(&path) {
                    Ok(()) if self.args.show != Show::Communities => self.set_show(self.args.show),
                    Ok(()) => {}
                    Err(e) => self.fail(format!("{e}")),
                }
            }
        } else if let Some(r) = self.rounds.first() {
            self.status = format!("lupin round {}: c shows it", round_name(r));
        }
    }

    fn load_round(&mut self, path: &Path) -> anyhow::Result<()> {
        let round = self.base.round(path)?;
        self.round = Some(round);
        Ok(())
    }

    /// The panel's `show` row.
    pub(super) fn show_line(&self) -> String {
        match (&self.round, self.show) {
            (Some(_), Show::Types) => "cell types  c".into(),
            (Some(_), Show::Clusters) => "clusters  c  R: relabel".into(),
            (_, _) if self.job.is_some() => "communities  (lupin running)".into(),
            (Some(_), _) => "communities  c: round".into(),
            (None, _) if self.rounds.is_empty() => "communities  A: annotate".into(),
            (None, _) => "communities  c: round".into(),
        }
    }

    /// The panel's `round` row: the round shown and the level it annotated.
    pub(super) fn round_line(&self) -> Option<String> {
        let r = self.round.as_ref()?;
        let name: String = r
            .name()
            .chars()
            .rev()
            .take(18)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        Some(format!(
            "{name} ({})  , .",
            self.base.run.levels[r.level].tag
        ))
    }

    /// Draw `show`, with nothing selected.
    fn set_show(&mut self, show: Show) {
        self.show = show;
        self.bar_focus = None;
        let k = self.level().comm.k;
        self.focus = vec![false; k];
        self.shown = None;
        self.markers.clear();
        self.gene = None;
        self.picked = None;
        self.community = self.community.min(k.saturating_sub(1));
        if let super::Layer::Community(_) = self.layer {
            self.layer = super::Layer::Community(self.community);
        }
        if self.layer == super::Layer::Entropy && self.level().comm.entropy.is_none() {
            self.layer = super::Layer::Argmax;
        }
        self.need_map = true;
    }

    /// `c`/`C`: the next (`by` = 1) or previous grouping, communities →
    /// cell types → clusters, loading the newest round when none is.
    pub(super) fn step_show(
        &mut self,
        by: isize,
        terminal: &mut DefaultTerminal,
    ) -> anyhow::Result<()> {
        if self.round.is_none() {
            let Some(path) = self.rounds.first().cloned() else {
                self.status = "no lupin round yet: A annotates this level".into();
                return Ok(());
            };
            self.busy(terminal, format!("loading {} ...", round_name(&path)))?;
            if let Err(e) = self.load_round(&path) {
                self.fail(format!("{e}"));
                return Ok(());
            }
        }
        const ORDER: [Show; 3] = [Show::Communities, Show::Types, Show::Clusters];
        let at = ORDER.iter().position(|&s| s == self.show).unwrap_or(0) as isize;
        self.set_show(ORDER[(at + by).rem_euclid(3) as usize]);
        Ok(())
    }

    /// `.`/`,`: the next (`by` = 1) or previous chain's newest round.
    pub(super) fn next_round(
        &mut self,
        by: isize,
        terminal: &mut DefaultTerminal,
    ) -> anyhow::Result<()> {
        if self.rounds.len() < 2 {
            self.status = match self.rounds.len() {
                0 => "no lupin round yet: A annotates this level".into(),
                _ => "this run has one round".into(),
            };
            return Ok(());
        }
        let at = self
            .round
            .as_ref()
            .and_then(|r| self.rounds.iter().position(|p| same_file(p, &r.path)));
        let n = self.rounds.len() as isize;
        let next = at.map_or(0, |i| (i as isize + by).rem_euclid(n) as usize);
        let next = self.rounds[next].clone();
        self.busy(terminal, format!("loading {} ...", round_name(&next)))?;
        match self.load_round(&next) {
            Ok(()) => {
                let show = if self.show == Show::Communities {
                    Show::Types
                } else {
                    self.show
                };
                self.set_show(show);
                self.status = format!("round {}", round_name(&next));
            }
            Err(e) => self.fail(format!("{e}")),
        }
        Ok(())
    }

    // ── annotate ────────────────────────────────────────────────────────

    /// `A`: choose a marker panel, then annotate the level on screen.
    pub(super) fn ask_markers(&mut self) {
        if self.job.is_some() {
            self.status = "lupin is still running".into();
            return;
        }
        let Some(manifest) = self.base.run.manifest.clone() else {
            self.status = "lupin needs the run's .pinto.json".into();
            return;
        };
        if let Err(e) = lupin::check() {
            self.fail(e);
            return;
        }
        // The run's genes, to count each panel's matches.
        let base = self.base;
        let level = self.levels[self.cur]
            .as_mut()
            .expect("current level is loaded");
        let known: HashSet<String> = match base.load_features(level) {
            Ok(()) => level
                .features
                .as_ref()
                .expect("just loaded")
                .names
                .iter()
                .map(|n| markers::symbol(n).to_uppercase())
                .collect(),
            Err(_) => HashSet::new(),
        };
        // Start where the last round's panel is, else beside the run.
        let dir = self
            .round
            .as_ref()
            .and_then(|r| {
                let m = lupin::read_json(&r.path)?;
                let p = lupin::resolve(&r.path, m.pointer("/annotate/markers")?.as_str()?);
                p.parent().map(Path::to_path_buf)
            })
            .or_else(|| manifest.parent().map(Path::to_path_buf))
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or_else(|| PathBuf::from("."));
        self.open_browser(dir, known);
    }

    fn open_browser(&mut self, dir: PathBuf, known: HashSet<String>) {
        let tag = self.base.run.levels[self.cur].tag.clone();
        let want = Panels { known, tag };
        let dir = lupin::canonical(&dir);
        self.modal = Some(Modal::Browse(Browser::open(dir, want, None)));
        self.need_map = true;
    }

    fn start_annotate(&mut self, panel: &Path) {
        let Some(manifest) = self.base.run.manifest.clone() else {
            return;
        };
        let tag = self.base.run.levels[self.cur].tag.clone();
        let out = lupin::annotate_out(&self.base.run.prefix, &tag);
        self.job = Some(lupin::annotate(&manifest, &tag, panel, &out));
        self.status = format!("annotating level {tag} → {out}");
    }

    /// Take a finished lupin job's result.
    pub(super) fn poll_job(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        let Some(job) = self.job.as_mut() else {
            return Ok(());
        };
        let Some(result) = job.poll() else {
            let progress = job.progress();
            if progress != self.status {
                self.status = progress;
                self.need_panel = true;
            }
            return Ok(());
        };
        let job = self.job.take().expect("polled");
        self.need_panel = true;
        match (job.what, result) {
            (JobKind::Annotate(path), Ok(_)) => {
                self.rounds.retain(|r| !same_file(r, &path));
                self.rounds.insert(0, path.clone());
                self.open_new_round(terminal, &path, "annotated")?;
            }
            (JobKind::Annotate(_), Err(e)) => self.fail(format!("lupin annotate: {e}")),
            (JobKind::Next(from), Ok(stdout)) => {
                let Some(next) = stdout.lines().rev().find(|l| !l.trim().is_empty()) else {
                    self.status = "lupin relabel printed no round".into();
                    return Ok(());
                };
                let next = PathBuf::from(next.trim());
                let applied = self.relabel.as_ref().map_or(0, |r| r.draft.len());
                if let Some(r) = self.relabel.take() {
                    r.draft.discard();
                }
                for r in self.rounds.iter_mut().filter(|r| same_file(r, &from)) {
                    *r = next.clone();
                }
                if !self.rounds.iter().any(|r| same_file(r, &next)) {
                    self.rounds.insert(0, next.clone());
                }
                let what = format!("{applied} decisions applied");
                self.open_new_round(terminal, &next, &what)?;
            }
            (JobKind::Next(_), Err(e)) => {
                self.fail(if e.contains("not the latest round") {
                    format!("{e}; , . step to it (this draft stays with its round)")
                } else {
                    format!("lupin relabel: {e}")
                });
            }
            (JobKind::Preview, Ok(stdout)) => match serde_json::from_str(&stdout) {
                Ok(v) => {
                    let changed = preview_changed(&v);
                    if let Some(r) = self.relabel.as_mut() {
                        r.preview = Some(v);
                    }
                    self.status = changed;
                }
                Err(e) => self.fail(format!("lupin preview: {e}")),
            },
            (JobKind::Preview, Err(e)) => self.fail(format!("lupin preview: {e}")),
        }
        Ok(())
    }

    fn open_new_round(
        &mut self,
        terminal: &mut DefaultTerminal,
        path: &Path,
        what: &str,
    ) -> anyhow::Result<()> {
        self.busy(terminal, format!("loading {} ...", round_name(path)))?;
        match self.load_round(path) {
            Ok(()) => {
                self.set_show(Show::Types);
                let k = self.level().comm.by_size.len();
                self.status = format!("{}: {what}, {k} cell types", round_name(path));
            }
            Err(e) => self.fail(format!("{e}")),
        }
        Ok(())
    }

    // ── dialogs ─────────────────────────────────────────────────────────

    pub(super) fn modal_key(
        &mut self,
        key: KeyEvent,
        _terminal: &mut DefaultTerminal,
    ) -> anyhow::Result<()> {
        let Some(modal) = self.modal.take() else {
            return Ok(());
        };
        // Typing a file name leaves the map as it is.
        if let Modal::Save(s) = modal {
            self.modal = self.save_key(s, key).map(Modal::Save);
            return Ok(());
        }
        self.need_map = true;
        match modal {
            Modal::Browse(mut b) => match b.key(key) {
                Outcome::Cancelled => self.status = "annotation cancelled".into(),
                Outcome::Chosen(c) => self.start_annotate(&c.file()),
                Outcome::Ignored | Outcome::Moved => self.modal = Some(Modal::Browse(b)),
            },
            Modal::Prompt(mut p) => {
                match key.code {
                    KeyCode::Esc => {
                        self.status = "nothing staged".into();
                        return Ok(());
                    }
                    KeyCode::Enter if p.field == 0 => {
                        if p.label.text.trim().is_empty() {
                            self.status = "type a label (Tab completes)".into();
                        } else {
                            p.field = 1;
                        }
                    }
                    KeyCode::Enter => {
                        if p.rationale.text.trim().is_empty() {
                            self.status = "lupin needs a rationale".into();
                        } else {
                            self.stage(p);
                            return Ok(());
                        }
                    }
                    KeyCode::Tab if p.field == 0 => complete(&mut p),
                    _ => {
                        let field = if p.field == 0 {
                            &mut p.label
                        } else {
                            &mut p.rationale
                        };
                        field.key(key);
                    }
                }
                self.modal = Some(Modal::Prompt(p));
            }
            Modal::Save(_) => unreachable!("handled above"),
            Modal::Confirm(lines) => match key.code {
                KeyCode::Char('y') | KeyCode::Enter => self.send_draft(false),
                KeyCode::Char('n') | KeyCode::Esc | KeyCode::Char('q') => {
                    self.status = "not applied; the draft is kept".into();
                }
                _ => self.modal = Some(Modal::Confirm(lines)),
            },
        }
        Ok(())
    }

    pub(super) fn modal_lines(&self, modal: &Modal, room: usize) -> Vec<Line<'static>> {
        let more = self.more();
        let bold = style::bold();
        let dim = style::dim();
        let mut out = vec![Line::raw("")];
        match modal {
            Modal::Browse(b) => {
                out.extend(b.lines(room.saturating_sub(1), 45 + more));
            }
            Modal::Prompt(p) => {
                let title = match &p.what {
                    Staging::Label(id) => format!(" label K{id}"),
                    Staging::Keep(id) => format!(" keep K{id}'s label"),
                    Staging::Merge(ids) => format!(
                        " merge {}",
                        ids.iter()
                            .map(|c| format!("K{c}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    ),
                };
                out.push(Line::styled(title, bold));
                // Name on the first line, the text wrapped beside it.
                let mut field = |name: &str, f: &Field, width: usize, on: bool| {
                    let style = if on { bold } else { TStyle::default() };
                    for (i, piece) in f.lines(width, on, style).into_iter().enumerate() {
                        let name = if i == 0 { name } else { "" };
                        out.push(Line::from(vec![
                            Span::styled(format!(" {name:<10}"), dim),
                            piece,
                        ]));
                    }
                };
                field("label", &p.label, usize::MAX, p.field == 0);
                field("rationale", &p.rationale, 32 + more, p.field == 1);
                out.push(Line::raw(""));
                if p.field == 0 {
                    let known: Vec<&str> = p.options.iter().take(12).map(String::as_str).collect();
                    for chunk in wrap(&known.join("  "), 42 + more) {
                        out.push(Line::styled(format!(" {chunk}"), dim));
                    }
                    out.push(Line::styled(" Tab complete  Enter next  Esc cancel", dim));
                } else {
                    out.push(Line::styled(" Enter stage  Esc cancel", dim));
                }
            }
            Modal::Save(s) => out.extend(self.save_lines(s)),
            Modal::Confirm(lines) => {
                out.push(Line::styled(
                    format!(" apply {} decisions?", lines.len()),
                    bold,
                ));
                out.push(Line::styled(
                    " lupin writes the next round".to_string(),
                    dim,
                ));
                out.push(Line::raw(""));
                let rows = room.saturating_sub(out.len() + 3);
                for l in lines.iter().take(rows) {
                    out.push(Line::raw(format!("  {}", tail(l, 43 + more))));
                }
                if lines.len() > rows {
                    out.push(Line::raw(format!("  … {} more", lines.len() - rows)));
                }
                out.push(Line::raw(""));
                out.push(Line::styled(" y apply   n not now", bold));
            }
        }
        out
    }

    // ── relabel ─────────────────────────────────────────────────────────

    /// `R`: start or stop relabelling the round's clusters.
    pub(super) fn toggle_relabel(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
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
        self.focus = focus;
        self.shown = g;
        self.gene = None;
        self.list_markers();
        self.need_map = true;
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
    pub(super) fn relabel_key(
        &mut self,
        key: KeyEvent,
        _terminal: &mut DefaultTerminal,
    ) -> anyhow::Result<bool> {
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

    fn stage(&mut self, p: Prompt) {
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
    fn send_draft(&mut self, preview: bool) {
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

/// Tab: the next known label starting with what is typed.
fn complete(p: &mut Prompt) {
    let typed = p.label.text.to_lowercase();
    let matching: Vec<&String> = p
        .options
        .iter()
        .filter(|o| o.to_lowercase().starts_with(&typed) || typed.is_empty())
        .collect();
    if matching.is_empty() {
        return;
    }
    // Typed exactly one already: move on to the next match.
    let at = matching
        .iter()
        .position(|o| o.eq_ignore_ascii_case(&p.label.text));
    let next = at.map_or(0, |i| (i + 1) % matching.len());
    p.label.set(matching[next].clone());
}

/// A one-line account of a `--preview` reply.
fn preview_changed(v: &serde_json::Value) -> String {
    let cells = v.get("cells_changed").and_then(|c| c.as_u64()).unwrap_or(0);
    let relabelled = v
        .get("clusters")
        .and_then(|c| c.as_object())
        .map_or(0, |m| {
            m.values()
                .filter(|c| c.get("label_before") != c.get("label_after"))
                .count()
        });
    format!(
        "preview: {} cells change, {relabelled} clusters relabelled",
        super::thousands(cells as usize)
    )
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

fn same_file(a: &Path, b: &Path) -> bool {
    lupin::canonical(a) == lupin::canonical(b)
}

/// `s` in lines of at most `n` characters.
fn wrap(s: &str, n: usize) -> Vec<String> {
    style::wrap(s, n, n)
}
