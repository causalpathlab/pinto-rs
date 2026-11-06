//! The method, parameter and confirm screens, and the plan they build.

use super::super::{form, script};
use super::jobs::{Job, Queue};
use super::{
    batch, copy, data, free_out, jobs, App, Planned, Review, Screen, Target, CONFIRM_SETTLE,
};
use crate::tui::shown;
use form::{Field, Kind, Method};
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use script::under;
use std::path::{Path, PathBuf};

impl App {
    pub(super) fn methods_key(&mut self, k: KeyEvent) {
        let last = self.rows.len() - 1;
        match k.code {
            KeyCode::Up => self.method_row = self.method_row.saturating_sub(1),
            KeyCode::Down => self.method_row = (self.method_row + 1).min(last),
            KeyCode::Char(' ') => {
                let r = &mut self.rows[self.method_row];
                r.on = !r.on;
                self.keep_param_method();
            }
            KeyCode::Char('o') => {
                let out = self.rows[self.method_row].out.clone();
                self.edit(Target::Out(self.method_row), out);
            }
            KeyCode::Char('O') => self.edit(Target::Header, self.header.clone()),
            KeyCode::Enter | KeyCode::Right => {
                self.rows[self.method_row].on = true;
                self.show_params(self.method_row);
            }
            _ => {}
        }
    }

    fn show_params(&mut self, m: usize) {
        if self.param_method != m {
            self.field_row = 0;
        }
        self.param_method = m;
        self.screen = Screen::Params;
    }

    /// Keep the parameters screen on a method its strip lists, after the
    /// methods queued change.
    fn keep_param_method(&mut self) {
        let ms = self.param_methods();
        if !ms.contains(&self.param_method) {
            self.param_method = ms[0];
            self.field_row = 0;
        }
    }

    /// Methods the parameters screen steps through: the queued ones, or
    /// all when none is.
    pub(super) fn param_methods(&self) -> Vec<usize> {
        let on: Vec<usize> = (0..self.rows.len()).filter(|&i| self.rows[i].on).collect();
        if on.is_empty() {
            (0..self.rows.len()).collect()
        } else {
            on
        }
    }

    // ───────────── parameters ─────────────

    /// Field indices of the method shown, as listed: advanced ones only
    /// when asked for or changed, narrowed by the filter.
    pub(super) fn visible(&self) -> Vec<usize> {
        let f = self.filter.to_lowercase();
        self.rows[self.param_method]
            .form
            .fields
            .iter()
            .enumerate()
            .filter(|(_, x)| self.advanced || !x.advanced || !x.is_default())
            .filter(|(_, x)| f.is_empty() || x.long.contains(&f))
            .map(|(i, _)| i)
            .collect()
    }

    fn field(&mut self) -> Option<(usize, &mut Field)> {
        let i = *self.visible().get(self.field_row)?;
        Some((i, &mut self.rows[self.param_method].form.fields[i]))
    }

    pub(super) fn params_key(&mut self, k: KeyEvent) {
        let n = self.visible().len();
        let last = n.saturating_sub(1);
        match k.code {
            KeyCode::Up => self.field_row = self.field_row.saturating_sub(1),
            KeyCode::Down => self.field_row = (self.field_row + 1).min(last),
            KeyCode::PageUp => self.field_row = self.field_row.saturating_sub(10),
            KeyCode::PageDown => self.field_row = (self.field_row + 10).min(last),
            KeyCode::Home => self.field_row = 0,
            KeyCode::End => self.field_row = last,
            KeyCode::Char('[' | ']') => {
                let ms = self.param_methods();
                let at = ms.iter().position(|&m| m == self.param_method).unwrap_or(0);
                let d = if k.code == KeyCode::Char(']') {
                    1
                } else {
                    ms.len() - 1
                };
                self.show_params(ms[(at + d) % ms.len()]);
            }
            KeyCode::Char('a') => {
                let on = self.field().map(|(i, _)| i);
                self.advanced = !self.advanced;
                // Stay on the same flag when it is still listed.
                self.field_row = on.and_then(|i| self.row_of(i)).unwrap_or(0);
            }
            KeyCode::Char('/') => {
                let text = self.filter.clone();
                self.edit(Target::Filter, text);
            }
            KeyCode::Esc if !self.filter.is_empty() => {
                self.filter.clear();
                self.field_row = 0;
            }
            KeyCode::Char('r') => {
                if let Some((_, f)) = self.field() {
                    f.reset();
                }
            }
            KeyCode::Char('R') => {
                for f in &mut self.rows[self.param_method].form.fields {
                    f.reset();
                }
                self.message = Some("every flag back to its default".into());
            }
            KeyCode::Char(' ') | KeyCode::Left | KeyCode::Right | KeyCode::Enter => {
                let m = self.param_method;
                let Some((i, f)) = self.field() else { return };
                match (&f.kind, k.code) {
                    (Kind::Flag { .. }, KeyCode::Char(' ') | KeyCode::Enter) => f.toggle(),
                    (Kind::Choice(_), KeyCode::Left) => f.cycle(-1),
                    (Kind::Choice(_), _) => f.cycle(1),
                    (Kind::Text, KeyCode::Enter) => {
                        let text = f.value.clone();
                        self.edit(Target::Field(m, i), text);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    // ───────────── confirm ─────────────

    /// Where row `i` writes: its folder and the prefix in it.
    pub(super) fn out_of(&self, i: usize) -> (PathBuf, String) {
        let abs = self.here.join(&self.rows[i].out);
        let dir = abs
            .parent()
            .map_or_else(|| self.here.clone(), script::normalize);
        (dir, crate::tui::name(&abs))
    }

    /// The queued fits, each checked.
    pub(super) fn plan(&self) -> Vec<Planned> {
        let coord_problem = data::coord_problem(&self.pairs);
        let batch_args = batch::args(&self.pairs);
        let mut outs: Vec<PathBuf> = Vec::new();
        let mut planned = Vec::new();
        for (i, r) in self.rows.iter().enumerate().filter(|(_, r)| r.on) {
            let (dir, out) = self.out_of(i);
            let rel = |p: &Path| script::relative(p, &dir).to_string_lossy().into_owned();
            let data: Vec<String> = self.pairs.iter().map(|p| rel(&p.data)).collect();
            let coords: Vec<String> = if r.form.takes_coords() && coord_problem.is_none() {
                self.pairs
                    .iter()
                    .filter_map(|p| p.coord.as_deref().map(rel))
                    .collect()
            } else {
                Vec::new()
            };
            // Label files given are passed as they are; the others are
            // written for the run in `{out}.batches/`.
            let mut made = Vec::new();
            let mut batches = Vec::new();
            if let (true, Ok(Some(args))) = (r.form.takes_batches(), &batch_args) {
                for a in args {
                    match a {
                        batch::Arg::Given(f) => batches.push(rel(f)),
                        batch::Arg::Made(name, what) => {
                            let at = format!("{out}.batches/{name}");
                            made.push((dir.join(&at), what.clone()));
                            batches.push(at);
                        }
                    }
                }
            }
            let mut argv = r.form.argv(&data, &coords, &batches, &out);
            // Files typed where pinto run started, read where it runs.
            relative_values(&r.form, &mut argv, &self.here, &dir);
            let on_run = r.form.from_run;
            let mut job = if on_run {
                on_run_job(&r.form.name, i, &dir, &out, argv)
            } else {
                Job {
                    made,
                    ..Job::fit(&r.form.name, i, &dir, &out, argv)
                }
            };
            job.clear = self.clears(i);
            // What an unfinished run left is cleared, so only what says it
            // is done is in the way.
            let in_the_way = if job.clear {
                vec![job.done.clone()]
            } else {
                job.in_the_way()
            };
            let mut blamed = None;
            // A line clap refuses says so: on a run, the job could not
            // even tell which manifest it reads.
            let refused = || {
                form::check(&self.cli, &job.argv).err().map(|why| {
                    let who = form::blamed(&why, &r.form.fields).map(str::to_string);
                    (why, who)
                })
            };
            let problem = if on_run && r.form.from().is_none() {
                blamed = Some("from".to_string());
                Some("--from is unset: the run's .pinto.json (Parameters screen)".to_string())
            } else if let Some((why, who)) = on_run.then(refused).flatten() {
                blamed = who;
                Some(why)
            } else if on_run && !job.opens.is_file() {
                blamed = Some("from".to_string());
                Some(format!(
                    "--from {}: no such run manifest",
                    r.form.from().unwrap_or_default()
                ))
            } else if !on_run && self.pairs.is_empty() {
                Some("no data files: add some on the Data screen".to_string())
            } else if let (Some(p), false) = (&coord_problem, on_run) {
                Some(p.clone())
            } else if let (Err(why), false) = (&batch_args, on_run) {
                Some(why.clone())
            } else if let Some(there) = in_the_way.into_iter().find(|p| p.exists()) {
                Some(format!(
                    "{} exists: change --out (o on Methods)",
                    shown(&there)
                ))
            } else if outs.contains(&dir.join(&out)) {
                Some("another queued method writes the same --out".to_string())
            } else if std::fs::metadata(&dir).is_ok_and(|m| !m.is_dir()) {
                // A folder not there yet is made when the fit starts.
                Some(format!("{} is not a folder", shown(&dir)))
            } else if let Some((why, who)) = refused() {
                blamed = who;
                Some(why)
            } else {
                None
            };
            outs.push(dir.join(&out));
            planned.push(Planned {
                job,
                problem,
                blamed,
            });
        }
        planned
    }

    /// Worth saying when no data file has coordinates: the cell graph then
    /// comes from expression.
    pub(super) fn coord_warning(&self) -> Option<String> {
        (!self.pairs.is_empty() && self.pairs.iter().all(|p| p.coord.is_none())).then(|| {
            "no coordinate files: the cell graph comes from expression (c adds them)".into()
        })
    }

    pub(super) fn open_confirm(&mut self) {
        if !self.rows.iter().any(|r| r.on) {
            self.message = Some(if self.rows.iter().any(|r| r.done) {
                "every method queued finished: space on the Methods screen queues one again".into()
            } else {
                "no method queued: space on the Methods screen picks some".into()
            });
            self.screen = Screen::Methods;
            return;
        }
        if self.running() {
            self.message = Some("fits are still running".into());
            return;
        }
        self.confirm = Some(Review::new(self.plan()));
        self.confirm_scroll = 0;
    }

    pub(super) fn confirm_key(&mut self, k: KeyEvent) {
        match k.code {
            KeyCode::Esc | KeyCode::Char('q') => self.confirm = None,
            KeyCode::Up => self.confirm_scroll = self.confirm_scroll.saturating_sub(1),
            KeyCode::Down => {
                self.confirm_scroll = (self.confirm_scroll + 1).min(self.confirm_max.get());
            }
            KeyCode::Char('c') => {
                let text = self.copy_text();
                self.message = Some(match copy(&text, self.attached) {
                    Ok(how) => format!("commands copied ({how})"),
                    Err(e) => format!("cannot copy: {e}"),
                });
            }
            // The run key pressed twice at once, or held, stops here: the
            // review it opened is read before it runs. Each repeat starts
            // the wait again, so a held key runs nothing until let go.
            _ if crate::tui::is_run_key(&k)
                && self
                    .confirm
                    .as_ref()
                    .is_some_and(|r| r.at.elapsed() < CONFIRM_SETTLE) =>
            {
                if let Some(r) = self.confirm.as_mut() {
                    r.at = std::time::Instant::now();
                }
            }
            _ if k.code == KeyCode::Enter || crate::tui::is_run_key(&k) => {
                let Some(Review { planned, .. }) = self.confirm.take() else {
                    return;
                };
                if let Some(p) = planned.iter().find(|p| p.problem.is_some()) {
                    if let Some(flag) = &p.blamed {
                        self.go_to_flag(&p.job.method, flag);
                    }
                    self.message = Some(format!(
                        "{}: {}",
                        p.job.method,
                        p.problem.clone().unwrap_or_default()
                    ));
                    return;
                }
                let jobs = planned.into_iter().map(|p| p.job).collect();
                match std::env::current_exe() {
                    Ok(exe) => {
                        self.queue = Some(Queue::start(jobs, exe));
                        self.job_row = 0;
                        self.screen = Screen::Run;
                    }
                    Err(e) => self.message = Some(format!("cannot find pinto: {e}")),
                }
            }
            _ => {}
        }
    }

    /// Once the queue has ended, unqueue the methods it finished and move
    /// their `--out` on, so ctrl+r again runs only those stopped, failed or
    /// not reached, each at the `--out` it had.
    pub(super) fn settle(&mut self) {
        let Some(q) = &self.queue else { return };
        if !q.take_end() {
            return;
        }
        let states = q.states();
        let mut done = Vec::new();
        for (job, state) in q.jobs.iter().zip(&states) {
            let r = &mut self.rows[job.row];
            r.done = *state == jobs::State::Done;
            if r.done {
                r.on = false;
                if !r.typed {
                    r.out = free_out(&self.here, &under(&self.header, &r.form.name));
                }
                done.push(job.method.clone());
            }
        }
        self.keep_param_method();
        if !done.is_empty() && done.len() < states.len() {
            self.message = Some(format!(
                "{} done and unqueued: {} runs the rest (space queues one again)",
                done.join(", "),
                crate::tui::RUN_KEY
            ));
        }
    }

    /// Whether a run of row `i` clears first what the last queue's
    /// unfinished run of it left at the same `--out`.
    fn clears(&self, i: usize) -> bool {
        let Some(q) = self.queue.as_ref().filter(|q| q.finished()) else {
            return false;
        };
        let (dir, out) = self.out_of(i);
        q.jobs.iter().zip(q.states()).any(|(j, s)| {
            j.row == i
                && s != jobs::State::Done
                && j.dir == dir
                && j.out == out
                && !j.manifest().exists()
        })
    }

    /// Name every `--out` not typed by hand after the header, each the
    /// first free one.
    pub(super) fn refresh_outs(&mut self) {
        for r in self.rows.iter_mut().filter(|r| !r.typed) {
            r.out = free_out(&self.here, &under(&self.header, &r.form.name));
        }
    }

    /// Show `--flag` of `method` on the parameters screen, the advanced
    /// rows too when it is one of them.
    fn go_to_flag(&mut self, method: &str, flag: &str) {
        let Some(m) = self.rows.iter().position(|r| r.form.name == method) else {
            return;
        };
        let Some(at) = self.rows[m].form.fields.iter().position(|f| f.long == flag) else {
            return;
        };
        self.show_params(m);
        self.advanced |= self.rows[m].form.fields[at].advanced;
        self.filter.clear();
        self.field_row = self.row_of(at).unwrap_or(0);
    }

    /// Where field `i` is listed on the parameters screen.
    fn row_of(&self, i: usize) -> Option<usize> {
        self.visible().iter().position(|&j| j == i)
    }
}

/// The job of a method that reads a prior run (`--from`), `argv` as
/// pinto parses it: what says it is done (lra's table, a subset's with
/// `--only`, lra-simulate's), and the run it read, which it opens. It owns
/// only its script, `{out}.cmd.sh`: never the run's own files.
fn on_run_job(method: &str, row: usize, dir: &Path, out: &str, argv: Vec<String>) -> Job {
    use crate::lr_activity::{ligand_effect::output::lra_simulate_path, lra_outputs};
    let cli = <crate::Cli as clap::Parser>::try_parse_from(
        std::iter::once("pinto").chain(argv.iter().map(String::as_str)),
    );
    let (done, from) = match cli.map(|c| c.commands) {
        Ok(crate::Commands::LrActivity(a)) => (
            lra_outputs(a.method, out, &a.only).swap_remove(0),
            a.input.from,
        ),
        Ok(crate::Commands::LraSimulate(a)) => (lra_simulate_path(out).into(), a.input.from),
        // An argv pinto does not take: the plan's check says why, and the
        // job does not run.
        _ => (PathBuf::from(out), None),
    };
    let opens = from.map_or_else(PathBuf::new, |f| script::lexical(&dir.join(&*f)));
    let paths = [dir.join(done), dir.join(format!("{out}.cmd.sh")), opens];
    Job {
        row,
        ..Job::reading(method, dir, out, argv, paths)
    }
}

/// Values of `form`'s flags that name files ([`form::Field::path`]), as
/// typed in `here`, rewritten for a run started in `dir`; `--from` names
/// the run's manifest, given its prefix.
fn relative_values(form: &Method, argv: &mut [String], here: &Path, dir: &Path) {
    let flags: Vec<String> = form
        .fields
        .iter()
        .filter(|f| f.path)
        .map(|f| format!("--{}", f.long))
        .collect();
    for k in 1..argv.len() {
        if !flags.contains(&argv[k - 1]) {
            continue;
        }
        let files: Vec<String> = argv[k]
            .split(',')
            .map(|f| {
                let mut p = here.join(f);
                if argv[k - 1] == "--from" {
                    p = PathBuf::from(crate::util::metadata::run_manifest(&p.to_string_lossy()));
                }
                script::relative(&p, dir).to_string_lossy().into_owned()
            })
            .collect();
        argv[k] = files.join(",");
    }
}
