//! Annotation in the viewer, through lupin: `A` annotates the level on
//! screen with a marker panel, and `a` shows the round's cell types or
//! clusters. Relabelling them is in [`relabel`](super::relabel).

use super::super::lupin::{self, same_file, JobKind};
use super::super::markers;
use super::super::round::round_name;
use super::browse::Panels;
use super::relabel::Staging;
use super::{App, Show};
use crate::tui::browse::{Browser, Outcome};
use crate::tui::style::{self, tail};
use legume_tui::save::Field;
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::style::Style as TStyle;
use ratatui::text::{Line, Span};
use ratatui::DefaultTerminal;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

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
    pub(super) what: Staging,
    /// 0 the label, 1 the rationale.
    pub(super) field: usize,
    pub(super) label: Field,
    pub(super) rationale: Field,
    /// Labels Tab completes to.
    pub(super) options: Vec<String>,
}

impl App<'_> {
    // ── rounds ──────────────────────────────────────────────────────────

    /// Find the run's rounds; open `explicit` (from `--round`) or, when
    /// `--show` asks for a round, the newest.
    pub(super) fn open_rounds(&mut self, explicit: Option<PathBuf>) {
        self.rounds = self.base.rounds();
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

    pub(super) fn load_round(&mut self, path: &Path) -> anyhow::Result<()> {
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
    pub(super) fn set_show(&mut self, show: Show) {
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
                let p = lupin::resolve_beside(&r.path, m.pointer("/annotate/markers")?.as_str()?)?;
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
                    for (i, piece) in f
                        .lines(width, on, style, style::selected())
                        .into_iter()
                        .enumerate()
                    {
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
    // Typed exactly one already: move on to the next match. Exactly, since
    // labels that differ only in case are two options.
    let at = matching.iter().position(|o| **o == p.label.text);
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

/// `s` in lines of at most `n` characters.
fn wrap(s: &str, n: usize) -> Vec<String> {
    style::wrap(s, n, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tab_steps_past_labels_that_differ_only_in_case() {
        // Case is kept, so both spellings are options; Tab must walk on.
        let mut p = Prompt {
            what: Staging::Label(0),
            field: 0,
            label: Field::new("ct1"),
            rationale: Field::new(""),
            options: ["ct1 a", "CT1 a", "CT1 a bright"]
                .map(String::from)
                .to_vec(),
        };
        let mut seen = Vec::new();
        for _ in 0..3 {
            complete(&mut p);
            seen.push(p.label.text.clone());
        }
        assert_eq!(seen, ["ct1 a", "CT1 a", "CT1 a bright"]);
    }
}
