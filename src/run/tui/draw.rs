//! Drawing `pinto run`: a tab line, the screen, a message and the keys.
//! Same look as `pinto view`.

use super::batch;
use super::jobs::State;
use super::{App, Kind, Screen, Target};
use crate::tui::shown;
use crate::tui::style::{bold, dim, first_row, popup, selected, short as fit};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

impl App {
    pub(super) fn draw(&self, f: &mut ratatui::Frame) {
        let area = f.area();
        let status_lines = self.status(usize::from(area.width));
        let [top, body, status] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(3),
            Constraint::Length(status_lines.len() as u16),
        ])
        .areas(area);
        f.render_widget(Paragraph::new(self.tabs()), top);
        let inner = Rect {
            x: body.x + 1,
            width: body.width.saturating_sub(2),
            ..body
        };
        match self.screen {
            Screen::Data => self.draw_data(f, inner),
            Screen::Methods => self.draw_methods(f, inner),
            Screen::Params => self.draw_params(f, inner),
            Screen::Run => self.draw_run(f, inner),
        }
        f.render_widget(Paragraph::new(status_lines), status);
        if let Some(b) = &self.browser {
            // Inside the popup's border.
            let width = usize::from(area.width.min(110)).saturating_sub(2);
            let lines = b.lines(usize::from(area.height).saturating_sub(2), width);
            popup(f, area, lines, 110);
        } else if self.confirm.is_some() {
            let rows = usize::from(area.height).saturating_sub(4);
            let width = usize::from(area.width.min(120)).saturating_sub(2);
            popup(f, area, self.confirm_lines(rows, width), 120);
        } else if self.labels.is_some() {
            let rows = usize::from(area.height).saturating_sub(10).max(3);
            popup(f, area, self.label_lines(rows), 80);
        }
        if let Some(e) = &self.editor {
            let mut help = Vec::new();
            let what = match &e.target {
                Target::Header => {
                    help = vec![
                        " exp1 → exp1_lc, exp1_cage, …   results/ → results/lc, …",
                        " O on Methods changes it later; o names one --out by hand",
                    ];
                    " Output header: what every result of this run is named after".to_string()
                }
                Target::Out(i) => format!(
                    " --out for {} (empty: back under the output header)",
                    self.rows[*i].form.name
                ),
                Target::Field(m, i) => format!(" --{}", self.rows[*m].form.fields[*i].long),
                Target::Filter => " flags containing".to_string(),
                Target::Name(i) => format!(
                    " one batch for every cell of {} (empty: the file is its own batch)",
                    crate::tui::name(&self.pairs[*i].data)
                ),
                Target::Rename(_, label) => {
                    format!(" new name for label “{label}” (empty: keep it)")
                }
            };
            // At the start there is no header yet: esc goes on without one.
            let esc = if e.target == Target::Header && self.header.is_empty() {
                "esc no header"
            } else {
                "esc cancel"
            };
            let mut lines = vec![Line::from(Span::styled(what, bold()))];
            lines.extend(help.into_iter().map(|h| Line::from(Span::styled(h, dim()))));
            lines.push(Line::from(format!(" {}▏", e.text)));
            lines.push(Line::from(Span::styled(
                format!(" enter keep   {esc}"),
                dim(),
            )));
            popup(f, area, lines, 90);
        }
    }

    fn tabs(&self) -> Vec<Line<'static>> {
        let mut spans = vec![Span::styled(" pinto run   ", bold())];
        for (i, s) in self.screens().into_iter().enumerate() {
            let label = format!(" {} {} ", i + 1, s.title());
            spans.push(if s == self.screen {
                Span::styled(label, selected())
            } else {
                Span::styled(label, dim())
            });
            spans.push(Span::raw(" "));
        }
        let queued: Vec<&str> = self
            .rows
            .iter()
            .filter(|r| r.on)
            .map(|r| r.form.name.as_str())
            .collect();
        let summary = format!(
            "  {} data · {}",
            self.pairs.len(),
            if queued.is_empty() {
                "no method".to_string()
            } else {
                queued.join(", ")
            }
        );
        spans.push(Span::styled(summary, dim()));
        vec![Line::from(spans), Line::from("")]
    }

    /// The message and the keys, the keys wrapped at whole hints to
    /// `width` columns so none is cut off.
    pub(super) fn status(&self, width: usize) -> Vec<Line<'static>> {
        let keys: &[&str] = match self.screen {
            Screen::Data => &[
                "a add data",
                "c coordinates",
                "b batch label file (several: paired by name)",
                "n name its batch",
                "e rename its labels",
                "x clear",
                "X clear all",
                "d remove",
                "J K reorder",
                "enter methods",
            ],
            Screen::Methods => &[
                "space queue",
                "enter / → queue and show flags",
                "o name one --out by hand",
                "O output header",
            ],
            Screen::Params => &[
                "space / enter change",
                "← → choices",
                "r reset",
                "R reset all",
                "a advanced",
                "/ filter (esc clears it)",
                "[ ] method",
            ],
            Screen::Run => &[
                "↑ ↓ choose",
                "s stop",
                "v open the one chosen in pinto view",
            ],
        };
        let mut lines = vec![Line::from(Span::styled(
            format!(" {}", self.message.clone().unwrap_or_default()),
            bold(),
        ))];
        let all = [
            "tab / shift-tab / 1-4 screens",
            "g review and run",
            "q quit",
        ];
        for group in [keys, &all[..]] {
            let mut row = String::new();
            for k in group {
                if !row.is_empty() && row.chars().count() + 3 + k.chars().count() > width {
                    lines.push(Line::from(Span::styled(std::mem::take(&mut row), dim())));
                }
                row.push_str(if row.is_empty() { " " } else { "   " });
                row.push_str(k);
            }
            lines.push(Line::from(Span::styled(row, dim())));
        }
        lines
    }

    fn draw_data(&self, f: &mut ratatui::Frame, area: Rect) {
        let mut lines = vec![Line::from(Span::styled(
            "Data files, with the coordinates and batch labels of each",
            bold(),
        ))];
        lines.push(Line::from(""));
        if self.pairs.is_empty() {
            lines.push(Line::from(Span::styled(
                "  none yet: a opens the file browser",
                dim(),
            )));
        }
        let w = usize::from(area.width);
        let name_w = self
            .pairs
            .iter()
            .map(|p| shown(&p.data).chars().count())
            .max()
            .unwrap_or(0)
            .min(w / 2);
        for (i, p) in self.pairs.iter().enumerate() {
            let text = format!(" {:<name_w$}  {}", fit(&shown(&p.data), name_w), p.info);
            let coords = p
                .coord
                .as_ref()
                .map_or_else(|| "none".to_string(), |f| shown(f));
            let batch = match batch::kind(p) {
                batch::Kind::File => "its own".to_string(),
                batch::Kind::Named(n) => format!("“{n}” for every cell"),
                batch::Kind::Labels(f) if p.renames.is_empty() => shown(f),
                batch::Kind::Labels(f) => {
                    format!("{} ({} renamed)", shown(f), p.renames.len())
                }
            };
            let more = format!("   coordinates {coords}  ·  batch {batch}");
            let style = if i == self.pair_row {
                selected()
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(fit(&text, w), style)));
            lines.push(Line::from(Span::styled(fit(&more, w), dim())));
        }
        if let Some(why) = super::data::coord_problem(&self.pairs) {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(why, bold())));
        }
        if !self.pairs.is_empty() {
            lines.push(Line::from(""));
            lines.extend(self.batch_lines(w));
        }
        if let Some(note) = self.coord_warning() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(note, dim())));
        }
        f.render_widget(Paragraph::new(lines), area);
    }

    /// The batches the data make, each with its files and cells.
    fn batch_lines(&self, w: usize) -> Vec<Line<'static>> {
        let (batches, notes) = batch::summary(&self.pairs);
        let own = batch::all_own(&self.pairs);
        let mut lines = vec![Line::from(Span::styled(
            if own {
                format!("Batches: each file its own ({} in all)", batches.len())
            } else {
                format!("Batches: {}", batches.len())
            },
            bold(),
        ))];
        let name_w = batches
            .iter()
            .map(|b| b.name.chars().count())
            .max()
            .unwrap_or(0)
            .min(32);
        for b in &batches {
            let cells = b.cells.map_or_else(
                || "cells not known yet".to_string(),
                |n| format!("{n} cells"),
            );
            let files = if b.files == 1 {
                "1 file".to_string()
            } else {
                format!("{} files", b.files)
            };
            let text = format!("  {:<name_w$}  {files:<8}  {cells}", fit(&b.name, name_w));
            lines.push(Line::from(Span::styled(fit(&text, w), Style::default())));
        }
        for n in notes {
            lines.push(Line::from(Span::styled(fit(&format!("  {n}"), w), dim())));
        }
        lines
    }

    /// The label list of a data row's batch file.
    fn label_lines(&self, rows: usize) -> Vec<Line<'static>> {
        let Some(l) = &self.labels else {
            return Vec::new();
        };
        let p = &self.pairs[l.row];
        let empty = std::collections::BTreeMap::new();
        let counts = p.label_counts().unwrap_or(&empty);
        let mut out = vec![
            Line::from(Span::styled(
                format!(
                    " Labels of {}",
                    p.batch.as_deref().map(crate::tui::name).unwrap_or_default()
                ),
                bold(),
            )),
            Line::from(""),
        ];
        let name_w = counts
            .keys()
            .map(|k| k.chars().count())
            .max()
            .unwrap_or(0)
            .min(24);
        for (i, (label, n)) in counts
            .iter()
            .enumerate()
            .skip(first_row(l.cursor, rows, counts.len()))
            .take(rows)
        {
            let to = p
                .renames
                .get(label)
                .map_or_else(String::new, |t| format!("→ {t}"));
            let text = format!(" {:<name_w$}  {n:>9} cells  {to}", fit(label, name_w));
            let style = if i == l.cursor {
                selected()
            } else {
                Style::default()
            };
            out.push(Line::from(Span::styled(text, style)));
        }
        out.push(Line::from(""));
        out.push(Line::from(Span::styled(
            " ↑ ↓ choose   enter rename   esc back",
            dim(),
        )));
        out
    }

    fn draw_methods(&self, f: &mut ratatui::Frame, area: Rect) {
        let mut lines = vec![
            Line::from(Span::styled(
                "Methods to fit on these data, each to its own --out",
                bold(),
            )),
            Line::from(vec![
                Span::styled(" output header ", dim()),
                Span::raw(if self.header.is_empty() {
                    "none".to_string()
                } else {
                    self.header.clone()
                }),
            ]),
            Line::from(""),
        ];
        let w = usize::from(area.width);
        // A hand-typed --out is marked ✎.
        let outs: Vec<String> = self
            .rows
            .iter()
            .map(|r| {
                if r.typed {
                    format!("{} ✎", r.out)
                } else {
                    r.out.clone()
                }
            })
            .collect();
        let out_w = outs
            .iter()
            .map(|o| o.chars().count())
            .max()
            .unwrap_or(0)
            .min(32);
        for (i, r) in self.rows.iter().enumerate() {
            let changed = r.form.changed();
            let text = format!(
                " [{}] {:<13} --out {:<out_w$}  {:<11} {}",
                if r.on { "x" } else { " " },
                r.form.name,
                fit(&outs[i], out_w),
                if changed == 0 {
                    "defaults".to_string()
                } else {
                    format!("{changed} changed")
                },
                r.form.about.lines().next().unwrap_or_default()
            );
            let style = if i == self.method_row {
                selected()
            } else if r.on {
                bold()
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(fit(&text, w), style)));
        }
        f.render_widget(Paragraph::new(lines), area);
    }

    fn draw_params(&self, f: &mut ratatui::Frame, area: Rect) {
        let [head, list, help] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(3),
            Constraint::Length(10),
        ])
        .areas(area);
        let mut tabs = Vec::new();
        for m in self.param_methods() {
            let name = format!(" {} ", self.rows[m].form.name);
            tabs.push(if m == self.param_method {
                Span::styled(name, selected())
            } else {
                Span::styled(name, dim())
            });
        }
        let mut sub = format!(
            "{}{}",
            if self.advanced {
                "all flags"
            } else {
                "flags (a shows advanced ones)"
            },
            if self.filter.is_empty() {
                String::new()
            } else {
                format!(" containing “{}”", self.filter)
            }
        );
        sub.insert_str(0, "  ");
        tabs.push(Span::styled(sub, dim()));
        f.render_widget(Paragraph::new(vec![Line::from(tabs), Line::from("")]), head);

        let form = &self.rows[self.param_method].form;
        let blamed = self.blamed_here();
        let visible = self.visible();
        let rows = usize::from(list.height);
        let start = first_row(self.field_row, rows, visible.len());
        let long_w = form
            .fields
            .iter()
            .map(|x| x.long.chars().count() + 2)
            .max()
            .unwrap_or(0)
            .min(36);
        let w = usize::from(list.width);
        let mut lines = Vec::new();
        for (row, &i) in visible.iter().enumerate().skip(start).take(rows) {
            let x = &form.fields[i];
            let mark = if blamed.as_deref() == Some(x.long.as_str()) {
                "!"
            } else if x.required {
                "*"
            } else if x.advanced {
                "·"
            } else {
                " "
            };
            let flag = format!("{mark}--{:<width$}", x.long, width = long_w);
            let value = fit(&x.shown(), 28);
            let help = x.help.lines().next().unwrap_or_default().to_string();
            if row == self.field_row {
                let text = format!("{flag} {value:<28}  {help}");
                lines.push(Line::from(Span::styled(fit(&text, w), selected())));
            } else {
                let vstyle = if x.is_default() { dim() } else { bold() };
                let rest = w.saturating_sub(flag.chars().count() + 31);
                lines.push(Line::from(vec![
                    Span::raw(flag),
                    Span::raw(" "),
                    Span::styled(format!("{value:<28}"), vstyle),
                    Span::raw("  "),
                    Span::styled(fit(&help, rest), dim()),
                ]));
            }
        }
        if visible.is_empty() {
            lines.push(Line::from(Span::styled("  no flag matches", dim())));
        }
        f.render_widget(Paragraph::new(lines), list);

        let mut text = Vec::new();
        if let Some(&i) = visible.get(self.field_row) {
            let x = &form.fields[i];
            let kind = match &x.kind {
                Kind::Flag { .. } => "switch".to_string(),
                Kind::Choice(v) => v
                    .iter()
                    .map(|s| if s.is_empty() { "(unset)" } else { s.as_str() })
                    .collect::<Vec<_>>()
                    .join(" | "),
                Kind::Text => "value".to_string(),
            };
            let default = if x.default.is_empty() {
                "(unset)".to_string()
            } else {
                x.default.clone()
            };
            text.push(Line::from(vec![
                Span::styled(format!("--{}", x.long), bold()),
                Span::styled(format!("   {kind}   default {default}"), dim()),
            ]));
            text.extend(x.long_help.lines().map(|l| Line::from(l.to_string())));
        }
        f.render_widget(
            Paragraph::new(text)
                .wrap(Wrap { trim: false })
                .block(Block::new().borders(Borders::TOP).border_style(dim())),
            help,
        );
    }

    /// The flag clap complained about on the method shown, if it did.
    /// Checked again only when the command line changes.
    fn blamed_here(&self) -> Option<String> {
        let m = self.param_method;
        let form = &self.rows[m].form;
        let argv = form.argv(&["x".to_string()], &[], &[], "x");
        if let Some((cm, ca, blamed)) = self.blame.borrow().as_ref() {
            if *cm == m && *ca == argv {
                return blamed.clone();
            }
        }
        let blamed = super::form::check(&self.cli, &argv)
            .err()
            .and_then(|why| super::form::blamed(&why, &form.fields).map(str::to_string));
        *self.blame.borrow_mut() = Some((m, argv, blamed.clone()));
        blamed
    }

    fn draw_run(&self, f: &mut ratatui::Frame, area: Rect) {
        let Some(q) = &self.queue else { return };
        let rows = usize::from(area.height).saturating_sub(q.jobs.len() + 3);
        // Copied out, so the worker writing the log is not held up.
        let (states, last, finished, log) = {
            let Ok(s) = q.shared.lock() else { return };
            let skip = s.log.len().saturating_sub(rows);
            let log: Vec<String> = s.log.iter().skip(skip).cloned().collect();
            (s.states.clone(), s.last.clone(), s.finished, log)
        };
        let w = usize::from(area.width);
        let mut lines = vec![
            Line::from(Span::styled(
                if finished {
                    "Finished"
                } else {
                    "Running, one after another"
                },
                bold(),
            )),
            Line::from(""),
        ];
        for (i, ((j, state), last)) in q.jobs.iter().zip(&states).zip(&last).enumerate() {
            let (said, style) = match state {
                State::Waiting => ("waiting".to_string(), dim()),
                State::Running => (format!("running  {last}"), bold()),
                State::Done => ("done".to_string(), Style::default()),
                State::Failed(why) => (format!("failed: {why}"), bold()),
                State::Stopped => ("stopped".to_string(), dim()),
            };
            let text = format!(
                " {:<13} {:<24} {said}",
                j.method,
                fit(&shown(&j.manifest()), 24)
            );
            let style = if i == self.job_row { selected() } else { style };
            lines.push(Line::from(Span::styled(fit(&text, w), style)));
        }
        lines.push(Line::from(""));
        lines.extend(
            log.iter()
                .map(|l| Line::from(Span::styled(fit(l, w), dim()))),
        );
        f.render_widget(Paragraph::new(lines), area);
    }

    /// The confirm popup's lines, at most `rows` tall and each at most
    /// `width` wide: long command lines are wrapped here, so the scroll and
    /// the keys below count every row on screen.
    pub(super) fn confirm_lines(&self, rows: usize, width: usize) -> Vec<Line<'static>> {
        let Some(planned) = &self.confirm else {
            return Vec::new();
        };
        let wrap = |text: String, style: Style| wrap(&text, style, width);
        let mut body: Vec<Line<'static>> = Vec::new();
        for p in planned {
            body.extend(wrap(
                format!(" {}   recorded in {}", p.job.method, shown(&p.job.script())),
                bold(),
            ));
            let lines = super::script::command_lines(&p.job.argv);
            let n = lines.len();
            for (k, l) in lines.into_iter().enumerate() {
                let indent = if k == 0 { "   " } else { "     " };
                let more = if k + 1 < n { " \\" } else { "" };
                body.extend(wrap(format!("{indent}{l}{more}"), Style::default()));
            }
            if let Some(why) = &p.problem {
                body.extend(wrap(format!("   ✗ {why}"), bold()));
            }
            body.push(Line::from(""));
        }
        let blocked = planned.iter().any(|p| p.problem.is_some());
        let mut out = wrap(
            format!(
                " Run {} fit{} in turn",
                planned.len(),
                if planned.len() == 1 { "" } else { "s" }
            ),
            bold(),
        );
        out.extend(wrap(
            " each command is saved as its {out}.cmd.sh, which will not run over a result".into(),
            dim(),
        ));
        if let Some(note) = self.coord_warning() {
            out.extend(wrap(format!(" note: {note}"), dim()));
        }
        out.push(Line::from(""));
        let mut foot = wrap(
            if blocked {
                " enter go to the problem   c copy   ↑ ↓ scroll   esc back"
            } else {
                " enter run   c copy the commands   ↑ ↓ scroll   esc back"
            }
            .into(),
            dim(),
        );
        if let Some(m) = &self.message {
            foot.extend(wrap(format!(" {m}"), bold()));
        }
        let room = rows.saturating_sub(out.len() + foot.len());
        let max = body.len().saturating_sub(room);
        self.confirm_max.set(max);
        let skip = self.confirm_scroll.min(max);
        out.extend(body.into_iter().skip(skip).take(room));
        out.extend(foot);
        out
    }
}

/// `text` in rows at most `width` wide, the later ones indented under the
/// first.
fn wrap(text: &str, style: Style, width: usize) -> Vec<Line<'static>> {
    const INDENT: &str = "       ";
    crate::tui::style::wrap(text, width, width.saturating_sub(INDENT.len()))
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            let row = if i == 0 {
                row
            } else {
                format!("{INDENT}{row}")
            };
            Line::from(Span::styled(row, style))
        })
        .collect()
}
