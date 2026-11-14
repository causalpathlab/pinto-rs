//! The lra layer in the terminal (`I`): the run's ligand-receptor pairs
//! (`pinto lra`) listed in the panel, filtered by verdict and gene and
//! sorted, the selected pair on the map (senders, and receivers exposed or
//! not and receptor-competent or not, or the exposure itself), its
//! threshold–response curve under the map, and `pinto lra --only` on the
//! pairs marked or the one selected, run in the background as `pinto run`
//! runs a fit, its bar in the panel; its
//! table opens in the layer when it is done. A run without a table is
//! offered a full `pinto lra`.

use super::super::gene::GeneMap;
use super::super::lra::{
    change, lra_command, lra_table, next_verdict, q_text, Filter, LraView, PairShow, TableRef,
};
use super::super::render::Style;
use super::super::{curve, Level, StillPair};
use super::{rgb, App, Pick};
use crate::lr_activity::ligand_effect::output::{percent_short, strip_zeros, CONTROLS};
use crate::lr_activity::ligand_effect::verdict::Verdict;
use crate::run::tui::{JobQueue, JobState, QueuedJob};
use crate::tui::style;
use legume_tui::save::Field;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::Style as TStyle;
use ratatui::text::{Line, Span};
use ratatui::DefaultTerminal;
use std::path::{Path, PathBuf};

pub const LRA_HELP: &[&str] = &[
    " ↑ ↓ pair  v V verdict  o sort  / genes",
    " space/m mark  M mark all shown",
    " L run lra --only: marked or this pair",
    " T next table  e exposure  c curve  X stop",
    " s save  z/Z zoom  ? all keys",
    " I or Esc leave the pairs",
];

/// A `pinto lra` the layer started, as `pinto run` runs a fit: its
/// script beside the run, its log followed.
pub struct LraJob {
    queue: JobQueue,
    /// The table it writes.
    table: PathBuf,
    /// What it is, for the status line.
    what: String,
}

impl LraJob {
    /// Stop it, and wait until it has ended.
    pub fn stop(self) {
        let _ = self.queue.finish();
    }
}

/// Columns of the list's verdict column: the longest [`short`] name (each
/// is ASCII, so its bytes are its columns).
pub(super) const VERDICT_W: usize = {
    let (mut k, mut widest) = (0, 0);
    while k < Verdict::ALL.len() {
        let w = short(Verdict::ALL[k]).len();
        if w > widest {
            widest = w;
        }
        k += 1;
    }
    widest
};

/// A verdict in a few letters, for the list.
pub(super) const fn short(v: Verdict) -> &'static str {
    match v {
        Verdict::Supported => "supported",
        Verdict::EffectWithoutCoexpression => "effect",
        Verdict::ColocationOnly => "coloc",
        Verdict::NoEvidence => "none",
        Verdict::ControlFailed => "ctrl fail",
        Verdict::CelltypeInseparable => "celltype",
        Verdict::InsufficientData => "thin",
    }
}

/// Lines of a pair's reasons the panel gives, the last marked when cut.
const WHY_LINES: usize = 5;

/// An exposure threshold at three decimals, trailing zeros dropped: `0`,
/// `0.004`, `6.03`.
fn tau_text(t: f64) -> String {
    strip_zeros(&format!("{t:.3}")).to_string()
}

/// Columns of a table's label in a status line or the panel's header.
const LABEL_WIDTH: usize = 32;

impl App<'_> {
    // * Opening

    /// `I`: open or leave the layer. Opened the first time, the run's
    /// tables are read.
    pub(super) fn toggle_lra(
        &mut self,
        terminal: Option<&mut DefaultTerminal>,
    ) -> anyhow::Result<()> {
        if self.lra_on {
            self.leave_lra();
            return Ok(());
        }
        self.leave_relabel();
        if self.lra.is_none() {
            let tables = self.base.run.tables();
            if !tables.is_empty() {
                self.busy(terminal, "reading the lra table ...".into())?;
            }
            match LraView::new(tables) {
                Ok(v) => self.lra = v,
                Err(e) => {
                    self.fail(format!("{e:#}"));
                    return Ok(());
                }
            }
        }
        self.lra_on = true;
        self.status = match &self.lra {
            Some(v) => format!(
                "{}: ↑ ↓ choose a pair",
                style::middle(&v.tables[v.at].label(), LABEL_WIDTH)
            ),
            None => "no lra table for this run: L runs pinto lra on it".into(),
        };
        self.pair_pending = self.lra.as_ref().is_some_and(|v| v.selected.is_some());
        self.need_map = true;
        Ok(())
    }

    fn leave_lra(&mut self) {
        self.lra_on = false;
        self.lra_search = None;
        self.need_map = true;
        self.status = "pairs left: I opens them again".into();
    }

    // * Keys

    /// A key while the layer is open; false for the viewer's own, and
    /// for Ctrl or Alt with a key (Ctrl+C quits, as everywhere), but in
    /// the search field, which edits with them.
    pub(super) fn lra_key(
        &mut self,
        key: KeyEvent,
        terminal: Option<&mut DefaultTerminal>,
    ) -> anyhow::Result<bool> {
        let chord = key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
        if chord && key.code == KeyCode::Char('c') {
            return Ok(false);
        }
        if let Some(mut field) = self.lra_search.take() {
            match key.code {
                KeyCode::Esc => self.set_search(String::new()),
                KeyCode::Enter => self.set_search(field.text.clone()),
                _ => {
                    if field.key(key) {
                        let text = field.text.clone();
                        self.lra_search = Some(field);
                        self.set_search(text);
                    } else {
                        self.lra_search = Some(field);
                    }
                }
            }
            return Ok(true);
        }
        if chord {
            return Ok(false);
        }
        match key.code {
            KeyCode::Char('I') | KeyCode::Esc => self.leave_lra(),
            KeyCode::Char('L') => self.start_lra(),
            KeyCode::Char('X') if self.lra_job.is_some() => self.stop_lra(),
            _ if self.lra.is_none() => return Ok(false),
            KeyCode::Up => self.move_cursor(-1),
            KeyCode::Down => self.move_cursor(1),
            KeyCode::PageUp => self.move_cursor(-10),
            KeyCode::PageDown => self.move_cursor(10),
            KeyCode::Home => self.move_cursor(isize::MIN / 2),
            KeyCode::End => self.move_cursor(isize::MAX / 2),
            KeyCode::Char(' ' | 'm') => self.mark_pair(),
            KeyCode::Char('M') => self.mark_shown(),
            KeyCode::Char('v') => self.step_verdict(1),
            KeyCode::Char('V') => self.step_verdict(-1),
            KeyCode::Char('o') => {
                let sort = self.change_filter(|f| f.sort = f.sort.next()).sort;
                self.status = format!("sorted by {}", sort.name());
            }
            KeyCode::Char('/') => {
                let now = self.lra.as_ref().map(|v| v.filter().search.clone());
                self.lra_search = Some(Field::new(now.unwrap_or_default()));
            }
            KeyCode::Char('T') => self.next_table(terminal)?,
            KeyCode::Char('e') => {
                let v = self.lra.as_mut().expect("checked");
                v.show = v.show.other();
                self.status = match v.show {
                    PairShow::Classes => "senders, exposed and competent receivers".into(),
                    PairShow::Exposure => "contact exposure to the ligand".into(),
                };
                self.need_map = true;
            }
            KeyCode::Char('c') => {
                self.lra_curve = !self.lra_curve;
                self.status = if self.lra_curve {
                    "threshold-response curve under the map: c hides it".into()
                } else {
                    String::new()
                };
                self.need_map = true;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Change the list's filter with `change`, the cursor kept; the
    /// filter now.
    fn change_filter(&mut self, change: impl FnOnce(&mut Filter)) -> Filter {
        let Some(v) = self.lra.as_mut() else {
            return Filter::default();
        };
        let mut f = v.filter().clone();
        change(&mut f);
        v.set_filter(f.clone());
        self.keep_cursor();
        f
    }

    fn set_search(&mut self, text: String) {
        self.change_filter(|f| f.search = text.trim().to_string());
        if let Some(v) = self.lra.as_ref() {
            self.status = format!("{} pairs", v.visible().len());
        }
    }

    fn step_verdict(&mut self, by: isize) {
        let verdict = self
            .change_filter(|f| f.verdict = next_verdict(f.verdict, by))
            .verdict;
        self.status = match verdict {
            Some(verdict) => format!("{}: {}", verdict.code(), verdict.note()),
            None => "every verdict".into(),
        };
    }

    /// After the list changed: the cursor on the selected pair when it is
    /// still shown, else within the list.
    fn keep_cursor(&mut self) {
        let Some(v) = self.lra.as_ref() else { return };
        let shown = v.visible();
        self.lra_cursor = v
            .selected
            .and_then(|s| shown.iter().position(|&i| i == s))
            .unwrap_or_else(|| self.lra_cursor.min(shown.len().saturating_sub(1)));
        self.need_panel = true;
    }

    fn move_cursor(&mut self, by: isize) {
        let Some(v) = self.lra.as_ref() else { return };
        let n = v.visible().len();
        if n == 0 {
            self.status = "no pair matches: v V verdict, / genes".into();
            return;
        }
        let at = if v.selected.is_some() {
            (self.lra_cursor as isize).saturating_add(by)
        } else {
            0
        };
        self.pick_pair(at.clamp(0, n as isize - 1) as usize);
    }

    /// Select the pair at `at` among those shown; it is drawn with the map
    /// ([`App::draw_pending_pair`]).
    pub(super) fn pick_pair(&mut self, at: usize) {
        let Some(v) = self.lra.as_mut() else { return };
        let Some(&i) = v.visible().get(at) else {
            return;
        };
        self.lra_cursor = at;
        v.selected = Some(i);
        self.pair_pending = true;
        self.need_map = true;
    }

    /// Draw the pair selected since the map was last drawn, once.
    pub(super) fn draw_pending_pair(&mut self, terminal: Option<&mut DefaultTerminal>) {
        if std::mem::take(&mut self.pair_pending) && self.lra_on {
            self.draw_pair(terminal);
        }
    }

    /// Draw the selected pair, reading the run's receivers first when they
    /// are not read yet.
    pub(super) fn draw_pair(&mut self, terminal: Option<&mut DefaultTerminal>) {
        let first = self.lra.as_ref().is_some_and(LraView::needs_receivers);
        if first {
            let _ = self.busy(
                terminal,
                "reading the run's receivers as lra saw them ...".into(),
            );
        }
        let (base, level) = (self.base, self.cur);
        let Some(v) = self.lra.as_mut() else { return };
        if let Err(e) = v.on_map(base, level) {
            self.fail(format!("cannot draw the pair: {e:#}"));
        }
        self.need_map = true;
    }

    /// Space or `m`: mark or unmark the pair under the cursor; marked,
    /// the next pair is selected, so presses mark successive pairs.
    pub(super) fn mark_pair(&mut self) {
        let Some(v) = self.lra.as_mut() else { return };
        let at = self.lra_cursor;
        let Some(&i) = v.visible().get(at) else {
            return;
        };
        let name = v.rows()[i].name();
        let on = v.toggle_mark(name.clone());
        let (n, shown) = (v.marked.len(), v.visible().len());
        if on {
            self.pick_pair((at + 1).min(shown - 1));
        }
        self.status = format!(
            "{name} {}; {n} marked: L runs pinto lra --only on them",
            if on { "marked" } else { "unmarked" }
        );
        self.need_panel = true;
    }

    /// `M`: mark every pair the filter shows, or, all marked already, unmark
    /// them: a verdict or a gene's pairs in one key.
    fn mark_shown(&mut self) {
        let Some(v) = self.lra.as_mut() else { return };
        let names: Vec<String> = v.visible().iter().map(|&i| v.rows()[i].name()).collect();
        let on = names.iter().any(|name| !v.marked.contains(name));
        for name in &names {
            if on {
                v.marked.insert(name.clone());
            } else {
                v.marked.remove(name);
            }
        }
        let n = v.marked.len();
        self.status = format!(
            "{} {} pairs; {n} marked: L runs pinto lra --only on them",
            if on { "marked" } else { "unmarked" },
            names.len()
        );
        self.need_panel = true;
    }

    /// `T`: the next table, the full one and each subset in turn.
    fn next_table(&mut self, terminal: Option<&mut DefaultTerminal>) -> anyhow::Result<()> {
        let Some(v) = self.lra.as_mut() else {
            return Ok(());
        };
        if v.tables.len() < 2 {
            self.status = "this run has one lra table: L runs a subset of the marked pairs".into();
            return Ok(());
        }
        let next = (v.at + 1) % v.tables.len();
        let label = style::middle(&v.tables[next].label(), LABEL_WIDTH);
        self.busy(terminal, format!("reading {label} ..."))?;
        let v = self.lra.as_mut().expect("checked");
        match v.load(next) {
            Ok(()) => self.status = label,
            Err(e) => self.fail(format!("{e:#}")),
        }
        // The pair kept is drawn as the new table has it.
        self.pair_pending = self.lra.as_ref().is_some_and(|v| v.selected.is_some());
        self.keep_cursor();
        self.need_map = true;
        Ok(())
    }

    // * Drawing

    /// The lra pair as the map draws it, when the layer is open and its
    /// pair drawn.
    pub(super) fn lra_drawn(&self) -> Option<(&Level, Style<'static>, Option<&GeneMap>)> {
        let v = self.lra.as_ref().filter(|_| self.lra_on)?;
        Some(v.shown()?.drawn(self.base, v.show, self.point))
    }

    /// Whether the curve is under the map.
    pub(super) fn curve_shows(&self) -> bool {
        self.lra_on && self.lra_curve && self.lra.as_ref().is_some_and(|v| v.selected.is_some())
    }

    /// `map` with the selected pair's curve under it, as one frame.
    pub(super) fn with_curve(&mut self, map: super::Frame) -> super::Frame {
        let Some(c) = self.lra.as_ref().and_then(LraView::curve) else {
            return map;
        };
        let room = (map.w, self.room_below(&map));
        let below = curve::render(&c, room, self.base.theme(), self.px_per_cell.1);
        super::plots::stack_below(map, &below)
    }

    /// The curve, saved beside the map's export (`s`): a PNG of
    /// [`curve::STILL_WIDTH`] times the export scale, whatever room the
    /// screen gives it, and a PDF figure.
    pub(super) fn export_curve(&mut self, stem: &str) -> anyhow::Result<String> {
        let Some(c) = self.lra.as_ref().and_then(LraView::curve) else {
            return Ok(String::new());
        };
        let w = curve::STILL_WIDTH * self.args.export_scale.max(1);
        let theme = self.base.theme();
        let frame = curve::still(&c, (w, w * 3 / 5), theme);
        let png = format!("{stem}.png");
        frame.write_png(Path::new(&png))?;
        let pdf = match curve::page(&c, theme) {
            Ok(page) => {
                super::super::pdf::write_chart(page, Path::new(&format!("{stem}.pdf")))?;
                " .pdf"
            }
            Err(_) => "",
        };
        let listed = self.remember(&png, &format!("curve · {}", c.title), &frame);
        Ok(format!(", {png}{pdf}{listed}"))
    }

    /// The pair the map draws, as the export's redraw command names it.
    pub(super) fn still_pair(&self, curve: bool) -> Option<StillPair<'_>> {
        let v = self.lra.as_ref().filter(|_| self.lra_on)?;
        v.shown()?;
        v.still_pair(curve)
    }

    // * The panel

    /// The layer's panel: the table, the filter, the pairs (each
    /// clickable), the selected pair's row, the map's key, and the job.
    pub(super) fn lra_lines(&self, room: usize) -> Vec<(Line<'static>, Option<Pick>)> {
        let (bold, dim) = (style::bold(), style::dim());
        let width = usize::from(self.side.width).saturating_sub(2).max(20);
        let mut out: Vec<(Line<'static>, Option<Pick>)> = vec![(Line::raw(""), None)];
        let job: Vec<(Line<'static>, Option<Pick>)> = self
            .lra_job
            .iter()
            .flat_map(|j| {
                [
                    (
                        Line::styled(
                            format!(
                                " {}  X stops",
                                style::middle(&j.what, width.saturating_sub(10))
                            ),
                            bold,
                        ),
                        None,
                    ),
                    (Line::raw(format!(" {}", j.queue.row(0, width))), None),
                ]
            })
            .collect();
        let Some(v) = self.lra.as_ref() else {
            out.push((Line::styled(" ligand-receptor pairs", bold), None));
            out.push((Line::raw(" no lra table for this run"), None));
            out.push((
                Line::styled(" L runs pinto lra on it (contact effect)", dim),
                None,
            ));
            out.extend(job);
            return out;
        };
        let table = v.table();
        let rows = &table.rows;
        let subset = table.subset_corrected();
        out.push((
            Line::from(vec![
                Span::styled(" pairs ", bold),
                Span::raw(format!(
                    "{} · {}",
                    style::middle(&v.tables[v.at].label(), LABEL_WIDTH),
                    rows.len()
                )),
                Span::styled(
                    if v.tables.len() > 1 {
                        format!("  T {}/{}", v.at + 1, v.tables.len())
                    } else {
                        String::new()
                    },
                    dim,
                ),
            ]),
            None,
        ));
        if subset {
            out.push((
                Line::styled(" subset-corrected: q within this subset", dim),
                None,
            ));
        }
        if !v.marked.is_empty() {
            out.push((
                Line::styled(format!(" {} marked  L runs them", v.marked.len()), dim),
                None,
            ));
        }
        let filter = v.filter();
        let verdict = match filter.verdict {
            Some(x) => format!(
                "{} ({})",
                x.code(),
                Verdict::counts(rows.iter().map(|r| r.verdict))[x.index()].1
            ),
            None => "all".into(),
        };
        out.push((
            Line::from(vec![
                Span::styled(" verdict ", dim),
                Span::raw(verdict),
                Span::styled("  v V", dim),
            ]),
            None,
        ));
        let search = match &self.lra_search {
            Some(f) => format!("{}▏", f.text),
            None if filter.search.is_empty() => "—".into(),
            None => filter.search.clone(),
        };
        out.push((
            Line::from(vec![
                Span::styled(" sort ", dim),
                Span::raw(filter.sort.name()),
                Span::styled("  o   genes ", dim),
                Span::raw(search),
                Span::styled("  /", dim),
            ]),
            None,
        ));

        // The selected pair's lines, the key and the job: what the list
        // leaves room for.
        let mut tail: Vec<(Line<'static>, Option<Pick>)> = Vec::new();
        if let Some(row) = v.selected_row() {
            tail.push((Line::raw(""), None));
            tail.push((
                Line::from(vec![
                    Span::styled(format!(" {} → {}", row.ligand, row.receptor), bold),
                    Span::raw(format!("  {}", row.verdict.code())),
                ]),
                None,
            ));
            // Prose, wrapped between words to the panel, at most `most`
            // lines, a cut marked.
            let mut prose = |text: &str, style: TStyle, most: usize| {
                let wrapped = legume_tui::cli::wrap(text, width);
                let lines: Vec<&str> = wrapped.lines().collect();
                for (k, l) in lines.iter().take(most).enumerate() {
                    let cut = k + 1 == most && lines.len() > most;
                    // A word longer than the panel is cut too.
                    let l = if cut {
                        style::short(&format!("{l}…"), width)
                    } else {
                        style::short(l, width)
                    };
                    tail.push((Line::styled(format!(" {l}"), style), None));
                }
            };
            let plain = TStyle::default();
            prose(&change(row), plain, 2);
            let corrected = if subset { " (subset-corrected)" } else { "" };
            prose(
                &format!(
                    "q {}{corrected} · coexpr q {}",
                    q_text(row.effect_q),
                    q_text(row.coexpr_q)
                ),
                plain,
                2,
            );
            if row.tau_top.is_finite() {
                prose(
                    &format!(
                        "weighs τ {} most ({:.0}%)",
                        tau_text(row.tau_top),
                        100.0 * row.tau_top_weight
                    ),
                    plain,
                    2,
                );
            }
            prose(
                &format!(
                    "{} receivers · {} competent · {} exposed (τ {})",
                    row.n_receivers,
                    row.n_competent,
                    row.n_exposed,
                    tau_text(row.threshold)
                ),
                plain,
                2,
            );
            // The effect against the control, signed, with its two-sided
            // q: the verdict and the reasons say whether it passed.
            if row.controls.iter().any(|c| c.q.is_finite()) {
                let each: Vec<String> = CONTROLS
                    .iter()
                    .zip(&row.controls)
                    .map(|(spec, c)| {
                        let name = spec.short;
                        if c.over.is_finite() && c.q.is_finite() {
                            let pct = super::super::curve::pct(c.over);
                            let change = percent_short(pct, pct, pct);
                            format!("{name} {change} q {}", q_text(c.q))
                        } else {
                            format!("{name} —")
                        }
                    })
                    .collect();
                prose(
                    &format!("effect − control{corrected}: {}", each.join(" · ")),
                    plain,
                    3,
                );
            }
            // The pair's own reasons in full; the verdict's meaning when
            // it has none.
            let why = if row.warnings.is_empty() {
                &row.note
            } else {
                &row.warnings
            };
            prose(why, dim, WHY_LINES);
            match (v.shown(), v.show) {
                (Some(on), PairShow::Classes) => {
                    tail.push((Line::styled(" map  e: exposure", dim), None));
                    for (name, colour, n) in on.legend() {
                        tail.push((
                            Line::from(vec![
                                Span::raw(" "),
                                Span::styled("██", TStyle::default().fg(rgb(colour))),
                                Span::raw(format!(" {name:<21} {n:>7}")),
                            ]),
                            None,
                        ));
                    }
                }
                (Some(on), PairShow::Exposure) => {
                    let ramp = super::super::render::Layer::Gene.ramp(self.base.theme());
                    let exposure = on.exposure(self.base);
                    let title = format!("{}  e: classes", exposure.title());
                    for l in super::ramp_legend(ramp, &title, &exposure.top_label()) {
                        tail.push((l, None));
                    }
                }
                (None, _) => {}
            }
        }
        tail.extend(job);

        let shown = v.visible();
        let fit = room.saturating_sub(out.len() + tail.len()).max(3);
        // The window of the list that holds the cursor.
        let first = style::first_row(self.lra_cursor, fit, shown.len());
        for (k, &i) in shown.iter().enumerate().skip(first).take(fit) {
            let r = &rows[i];
            let at = v.selected == Some(i);
            let mark = if v.marked.contains(&r.name()) {
                "*"
            } else {
                " "
            };
            // The columns first, the name in what they leave of the line
            // (`width + 1`, as the lines starting with a space).
            let right = format!(
                " {:>7} {:>7} {:<VERDICT_W$}",
                percent_short(r.pct, r.pct_lo, r.pct_hi),
                q_text(r.effect_q),
                short(r.verdict),
            );
            let name_w = (width + 1).saturating_sub(2 + right.chars().count()).max(8);
            let name = style::middle(&r.name(), name_w);
            let text = format!("{}{mark}{name:<name_w$}{right}", if at { "▸" } else { " " },);
            let style = if at {
                bold
            } else if r.verdict == Verdict::Supported {
                TStyle::default().fg(ratatui::style::Color::Green)
            } else {
                TStyle::default()
            };
            out.push((Line::styled(text, style), Some(Pick::Pair(k))));
        }
        if shown.is_empty() {
            out.push((Line::styled(" no pair matches", dim), None));
        } else if shown.len() > fit {
            out.push((
                Line::styled(
                    format!(
                        " {}–{} of {}",
                        first + 1,
                        (first + fit).min(shown.len()),
                        shown.len()
                    ),
                    dim,
                ),
                None,
            ));
        }
        out.extend(tail);
        out
    }

    // * Running lra

    /// `X`: stop the `pinto lra` running; [`App::poll_lra`] sees it end.
    fn stop_lra(&mut self) {
        if let Some(job) = &self.lra_job {
            job.queue.stop();
            self.status = format!("stopping {}", job.what);
        }
    }

    /// The pairs `L` runs: those marked, else the one under the cursor.
    pub(super) fn pairs_to_run(&self) -> Vec<String> {
        let Some(v) = self.lra.as_ref() else {
            return Vec::new();
        };
        if v.marked.is_empty() {
            v.selected_row().map(|r| r.name()).into_iter().collect()
        } else {
            v.marked.iter().cloned().collect()
        }
    }

    /// `L`: `pinto lra --only` on the marked pairs or the one selected, or a
    /// full `pinto lra` on a run without a table, in the background. A subset
    /// whose table is there already is opened instead.
    fn start_lra(&mut self) {
        if self.lra_job.is_some() {
            self.status = "pinto lra is still running".into();
            return;
        }
        let Some(manifest) = self.base.run.manifest.clone() else {
            self.status = "pinto lra needs the run's .pinto.json".into();
            return;
        };
        let only = self.pairs_to_run();
        if self.lra.is_some() && only.is_empty() {
            self.status = "choose or mark a pair first (space or m), then L runs it".into();
            return;
        }
        let options = self
            .lra
            .as_ref()
            .map_or_else(|| self.base.run.lra_options(), LraView::options);
        let prefix = &self.base.run.prefix;
        let table = lra_table(prefix, &only, options.clone());
        if table.path.is_file() {
            self.finish_lra(&table.path, "already there");
            return;
        }
        let dir = manifest
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        let file = |p: &Path| {
            p.file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
        };
        let done = dir.join(file(&table.path));
        let script = dir.join(format!(
            "{}.cmd.sh",
            file(&table.path).trim_end_matches(".parquet")
        ));
        let argv = lra_command(Path::new(&file(&manifest)), &options, &only);
        let job = QueuedJob {
            // A script an unfinished run left goes.
            clear: true,
            ..QueuedJob::reading(
                "lra",
                &dir,
                &file(Path::new(prefix)),
                argv,
                [done, script, manifest],
            )
        };
        let what = match only.as_slice() {
            [] => "pinto lra".to_string(),
            [one] => format!("pinto lra --only {one}"),
            many => format!("pinto lra --only ({} pairs)", many.len()),
        };
        match std::env::current_exe() {
            Ok(exe) => {
                self.status = format!("{what} started; the table opens here when it is done");
                self.lra_job = Some(LraJob {
                    queue: JobQueue::start(vec![job], exe),
                    table: table.path,
                    what,
                });
            }
            Err(e) => self.fail(format!("cannot find pinto: {e}")),
        }
    }

    /// Follow the `pinto lra` the layer started; once it is done, open its
    /// table in the layer.
    pub(super) fn poll_lra(&mut self) {
        let Some(job) = self.lra_job.as_ref() else {
            return;
        };
        if !job.queue.finished() {
            self.need_panel |= self.lra_on;
            return;
        }
        let job = self.lra_job.take().expect("checked");
        let state = job.queue.states().into_iter().next();
        let last = job.queue.last_line(0);
        let table = job.table.clone();
        let what = job.what.clone();
        job.stop();
        match state {
            Some(JobState::Done) => self.finish_lra(&table, &format!("{what} done")),
            Some(JobState::Failed(why)) => self.fail(format!("{what}: {why}\n{last}")),
            _ => self.status = format!("{what} stopped"),
        }
        self.need_panel = true;
    }

    /// Open `table` in the layer: as the run's manifest now records it.
    fn finish_lra(&mut self, table: &Path, what: &str) {
        let run = super::super::data::Run::open(&self.base.run.prefix);
        let r = match &run {
            Ok(run) => TableRef::at(table, run),
            Err(_) => TableRef::at(table, &self.base.run),
        };
        let label = style::middle(&r.label(), LABEL_WIDTH);
        let loaded = match self.lra.as_mut() {
            Some(v) => v.load_path(r),
            None => LraView::new(vec![r]).map(|v| self.lra = v),
        };
        match loaded {
            Ok(()) => {
                self.lra_on = true;
                self.status = format!("{what}: {label}");
                self.keep_cursor();
                self.pair_pending = self.lra.as_ref().is_some_and(|v| v.selected.is_some());
            }
            Err(e) => self.fail(format!("{e:#}")),
        }
        self.need_map = true;
    }
}
