//! The method, parameter and confirm screens, and the plan they build.

use super::super::{form, script};
use super::jobs::{Job, Queue};
use super::{
    batch, copy, data, free_out, jobs, App, Planned, Review, Role, Screen, Target, CONFIRM_SETTLE,
};
use crate::residual::{Plan, ResidualArgs};
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
            KeyCode::Char(' ') => self.queue_row(self.method_row, !self.rows[self.method_row].on),
            KeyCode::Char('m') => self.next_stage_two(self.method_row),
            KeyCode::Char('o') => {
                let out = self.rows[self.method_row].out.clone();
                self.edit(Target::Out(self.method_row), out);
            }
            KeyCode::Char('O') => self.edit(Target::Header, self.header.clone()),
            KeyCode::Enter | KeyCode::Right => {
                if !self.rows[self.method_row].on {
                    self.queue_row(self.method_row, true);
                }
                self.show_params(self.method_row);
            }
            _ => {}
        }
    }

    /// Queue row `i`, or take it off. The residual takes its stage-2 fit
    /// with it, and off, lra on that fit too. lra on the stage-2 fit is offered only with lra queued on
    /// the residual's run, and comes with that lra's flags.
    fn queue_row(&mut self, i: usize, on: bool) {
        match self.rows[i].role {
            Role::Residual => {
                for r in &mut self.rows {
                    match r.role {
                        Role::Stage2 => r.on = on,
                        Role::Stage2Lra if !on => r.on = false,
                        _ => {}
                    }
                }
            }
            Role::Stage2Lra if on => {
                let Some(lra) = self.lra_on_residual_run() else {
                    self.message = Some(NO_LRA.into());
                    return;
                };
                let flags: Vec<(String, String)> = self.rows[lra]
                    .form
                    .fields
                    .iter()
                    .filter(|f| f.long != "from")
                    .map(|f| (f.long.clone(), f.value.clone()))
                    .collect();
                for f in &mut self.rows[i].form.fields {
                    if let Some((_, v)) = flags.iter().find(|(l, _)| *l == f.long) {
                        f.value.clone_from(v);
                    }
                }
            }
            _ => {}
        }
        self.rows[i].on = on;
        self.keep_param_method();
    }

    /// `m` on the stage-2 row: the next method it fits the residual with,
    /// each keeping its flags, its `--out` renamed unless typed.
    fn next_stage_two(&mut self, i: usize) {
        let r = &mut self.rows[i];
        if r.role != Role::Stage2 || r.alts.is_empty() {
            return;
        }
        let next = r.alts.remove(0);
        let was = std::mem::replace(&mut r.form, next);
        r.alts.push(was);
        if !r.typed {
            r.out = free_out(&self.here, &under(&self.header, &r.stem()));
        }
        if self.param_method == i {
            self.field_row = 0;
        }
    }

    /// The run `form` reads (`--from`), as its manifest.
    fn run_of(&self, form: &Method) -> Option<PathBuf> {
        let f = form.from()?;
        let manifest = crate::util::metadata::run_manifest(&self.here.join(f).to_string_lossy());
        Some(script::normalize(Path::new(&manifest)))
    }

    /// The lra row queued (or finished in the last run) on the run the
    /// residual queued (or finished) reads.
    fn lra_on_residual_run(&self) -> Option<usize> {
        let res = self
            .rows
            .iter()
            .find(|r| r.role == Role::Residual && (r.on || r.done))?;
        let run = self.run_of(&res.form)?;
        self.rows.iter().position(|r| {
            r.role == Role::Plain
                && (r.on || r.done)
                && r.form.name == "lra"
                && self.run_of(&r.form).as_ref() == Some(&run)
        })
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
        let mut planned: Vec<Planned> = Vec::new();
        // The queued residual's place in the plan and what it writes, and
        // the stage-2 fit's place.
        // Finished in the last run, not queued again and still there, each
        // is read where it is.
        let mut residual: Option<(Option<usize>, Result<Plan, String>)> = self
            .finished(Role::Residual)
            .and_then(|i| self.residual_of(i).ok())
            .filter(|p| Path::new(&p.summary).is_file())
            .map(|p| (None, Ok(p)));
        let mut stage2: Option<(Option<usize>, PathBuf)> = self
            .finished(Role::Stage2)
            .map(|i| self.fit_manifest(i))
            .filter(|m| m.is_file())
            .map(|m| (None, m));
        for (i, r) in self.rows.iter().enumerate().filter(|(_, r)| r.on) {
            let (dir, out) = self.out_of(i);
            let rel = |p: &Path| script::relative(p, &dir).to_string_lossy().into_owned();
            let mut form = r.form.clone();
            // The stage-2 rows read what the earlier ones write: the job
            // they wait on, or why they cannot be set up.
            let (staged, after) =
                self.staged(r.role, &mut form, residual.as_ref(), stage2.as_ref());
            // The picked data are the fits' alone.
            let picked = r.role == Role::Plain && !form.from_run;
            let (data, coords): (Vec<String>, Vec<String>) = match (r.role, &residual) {
                (Role::Stage2, Some((_, Ok(p)))) => (
                    p.zarrs.iter().map(|z| rel(Path::new(z))).collect(),
                    p.coords.iter().map(|c| rel(Path::new(c))).collect(),
                ),
                (Role::Plain, _) => (
                    self.pairs.iter().map(|p| rel(&p.data)).collect(),
                    if form.takes_coords() && coord_problem.is_none() {
                        self.pairs
                            .iter()
                            .filter_map(|p| p.coord.as_deref().map(rel))
                            .collect()
                    } else {
                        Vec::new()
                    },
                ),
                _ => (Vec::new(), Vec::new()),
            };
            // Label files given are passed as they are; the others are
            // written for the run in `{out}.batches/`.
            let mut made = Vec::new();
            let mut batches = Vec::new();
            if let (true, true, Ok(Some(args))) = (picked, form.takes_batches(), &batch_args) {
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
            let mut argv = form.argv(&data, &coords, &batches, &out);
            // Files typed where pinto run started, read where it runs.
            relative_values(&form, &mut argv, &self.here, &dir);
            let on_run = form.from_run;
            let label = r.label();
            let mut job = if on_run {
                on_run_job(&label, i, &dir, &out, argv)
            } else {
                Job {
                    made,
                    ..Job::fit(&label, i, &dir, &out, argv)
                }
            };
            job.after = after;
            job.clear = self.clears(i);
            // What an unfinished run left is cleared, so only what says it
            // is done is in the way.
            let mut in_the_way = if job.clear {
                vec![job.done.clone()]
            } else {
                job.in_the_way()
            };
            let mut unplanned = None;
            if r.role == Role::Residual {
                let p = residual_args(&job.argv)
                    .and_then(|a| residual_plan(&dir, &a))
                    .map_err(|e| format!("{e:#}"));
                match &p {
                    // Nothing it writes is written over.
                    Ok(p) => in_the_way.extend(p.made()),
                    Err(why) => unplanned = Some(why.clone()),
                }
                residual = Some((Some(planned.len()), p));
            }
            if r.role == Role::Stage2 {
                stage2 = Some((Some(planned.len()), job.manifest().to_path_buf()));
            }
            let mut blamed = None;
            // A line clap refuses says so: on a run, the job could not
            // even tell which manifest it reads.
            let refused = || {
                form::check(&self.cli, &job.argv).err().map(|why| {
                    let who = form::blamed(&why, &form.fields).map(str::to_string);
                    (why, who)
                })
            };
            // The stage-2 fit lra reads is made in this queue.
            let reads_made = r.role == Role::Stage2Lra;
            let problem = if staged.is_some() {
                staged
            } else if on_run && form.from().is_none() {
                blamed = Some("from".to_string());
                Some("--from is unset: the run's .pinto.json (Parameters screen)".to_string())
            } else if let Some((why, who)) = on_run.then(refused).flatten() {
                blamed = who;
                Some(why)
            } else if on_run && !reads_made && !job.opens.is_file() {
                blamed = Some("from".to_string());
                Some(format!(
                    "--from {}: no such run manifest",
                    form.from().unwrap_or_default()
                ))
            } else if unplanned.is_some() {
                unplanned
            } else if picked && self.pairs.is_empty() {
                Some("no data files: add some on the Data screen".to_string())
            } else if let (Some(p), true) = (&coord_problem, picked) {
                Some(p.clone())
            } else if let (Err(why), true) = (&batch_args, picked) {
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

    /// What a stage-2 row (`role`) reads, set in its `form`: the stage-2
    /// fit, the residual's files with the run's coordinate columns; lra on
    /// it, the stage-2 fit's manifest. With the job of the plan it waits on
    /// (none when what it reads finished in the last run), or why it cannot
    /// be set up. Other rows read nothing made here.
    fn staged(
        &self,
        role: Role,
        form: &mut Method,
        residual: Option<&(Option<usize>, Result<Plan, String>)>,
        stage2: Option<&(Option<usize>, PathBuf)>,
    ) -> (Option<String>, Option<usize>) {
        match (role, residual) {
            (Role::Stage2, None) => (
                Some("stage 2 fits the residual's files: queue residual too".into()),
                None,
            ),
            (Role::Stage2, Some((at, Err(_)))) => (
                Some("the residual it fits is not set up (see residual)".into()),
                *at,
            ),
            (Role::Stage2, Some((at, Ok(p)))) => {
                form.set("coord-column-names", &p.coord_names.join(","));
                (None, *at)
            }
            (Role::Stage2Lra, _) => match stage2 {
                None => (
                    Some("lra on stage 2 reads the stage-2 fit: queue it".into()),
                    None,
                ),
                Some((at, _)) if self.lra_on_residual_run().is_none() => (Some(NO_LRA.into()), *at),
                Some((at, manifest)) => {
                    form.set("from", &manifest.to_string_lossy());
                    (None, *at)
                }
            },
            _ => (None, None),
        }
    }

    /// The row of `role` that finished in the last run and is not queued
    /// again.
    fn finished(&self, role: Role) -> Option<usize> {
        self.rows
            .iter()
            .position(|r| r.role == role && r.done && !r.on)
    }

    /// What the residual of row `i` writes, at its `--out`.
    fn residual_of(&self, i: usize) -> anyhow::Result<Plan> {
        let (dir, out) = self.out_of(i);
        let form = &self.rows[i].form;
        let mut argv = form.argv(&[], &[], &[], &out);
        relative_values(form, &mut argv, &self.here, &dir);
        residual_args(&argv).and_then(|a| residual_plan(&dir, &a))
    }

    /// The manifest the fit of row `i` writes, at its `--out`.
    fn fit_manifest(&self, i: usize) -> PathBuf {
        let (dir, out) = self.out_of(i);
        script::outputs(&dir, &out)[0].clone()
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
                        self.go_to_flag(p.job.row, flag);
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
                done.push(job.method.clone());
            }
        }
        // A finished residual or stage-2 fit keeps its --out while what
        // reads it is still to run.
        let left = |role| self.rows.iter().any(|r| r.role == role && r.on);
        let (stage2_left, lra2_left) = (left(Role::Stage2), left(Role::Stage2Lra));
        for job in &q.jobs {
            let r = &mut self.rows[job.row];
            let read = match r.role {
                Role::Residual => stage2_left || lra2_left,
                Role::Stage2 => lra2_left,
                _ => false,
            };
            if r.done && !r.typed && !read {
                r.out = free_out(&self.here, &under(&self.header, &r.stem()));
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
            r.out = free_out(&self.here, &under(&self.header, &r.stem()));
        }
    }

    /// Show `--flag` of row `m` on the parameters screen, the advanced
    /// rows too when it is one of them.
    fn go_to_flag(&mut self, m: usize, flag: &str) {
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
        // What the residual's plan says marks it done.
        Ok(crate::Commands::Residual(a)) => (
            residual_plan(dir, &a).map_or_else(|_| PathBuf::from(out), |p| p.summary.into()),
            Some(a.run),
        ),
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

/// What lra on the stage-2 fit needs, said when it is not there.
const NO_LRA: &str =
    "lra on the stage-2 fit comes with lra queued on the residual's run: the same --from";

/// `argv` as `pinto residual` parses it.
fn residual_args(argv: &[String]) -> anyhow::Result<ResidualArgs> {
    let cli = <crate::Cli as clap::Parser>::try_parse_from(
        std::iter::once("pinto").chain(argv.iter().map(String::as_str)),
    )?;
    match cli.commands {
        crate::Commands::Residual(a) => Ok(*a),
        _ => anyhow::bail!("not a residual"),
    }
}

/// What the residual `a`, run in `dir`, writes ([`Plan`]): its paths
/// from `dir`, checked as the command checks them before reading a count.
fn residual_plan(dir: &Path, a: &ResidualArgs) -> anyhow::Result<Plan> {
    let from = |p: &str| -> Box<str> { dir.join(p).to_string_lossy().into() };
    let a = ResidualArgs {
        run: from(&a.run),
        out: from(&a.out),
        data_files: a
            .data_files
            .as_ref()
            .map(|f| f.iter().map(|p| from(p)).collect()),
        ..*a
    };
    let run = crate::util::metadata::Run::open(&a.run)?;
    Plan::new(&run, &a)
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
