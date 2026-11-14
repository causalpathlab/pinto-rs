//! Terminal front end of `pinto run`: data files with their coordinates
//! and batch labels, methods, their flags, a confirm popup with the exact
//! commands, then the queue running with its log.

mod batch;
mod data;
mod draw;
mod jobs;
mod pick;
mod plan;

use super::{form, script};
use crate::tui::browse::Browser;
use crate::tui::shown;
use data::{Pair, Pick};
use form::{Kind, Method};
use jobs::{Job, Queue};
use legume_tui::save::Field as TextField;
use legume_tui::Round;
use ratatui::crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

pub use super::METHODS;
pub(crate) use jobs::{Job as QueuedJob, Queue as JobQueue, State as JobState};

/// How long after the review opens the run key is not yet taken as "run":
/// a double tap or a held key would otherwise skip the review.
const CONFIRM_SETTLE: std::time::Duration = std::time::Duration::from_millis(500);

impl Row {
    fn new(form: Method, role: Role, here: &Path) -> Self {
        let mut r = Row {
            form,
            role,
            alts: Vec::new(),
            on: false,
            out: String::new(),
            typed: false,
            done: false,
        };
        r.out = free_out(here, &r.stem());
        r
    }

    /// The name the row is listed and its jobs are shown by.
    fn label(&self) -> String {
        match self.role {
            Role::Plain | Role::Residual => self.form.name.clone(),
            Role::Stage2 | Role::Stage2Lra => format!("stage-2 {}", self.form.name),
        }
    }

    /// What its `--out` is named after, under the output header.
    fn stem(&self) -> String {
        match self.role {
            Role::Plain | Role::Residual => self.form.name.clone(),
            Role::Stage2 | Role::Stage2Lra => format!("residual-{}", self.form.name),
        }
    }
}

/// The commands up for review, and when the review opened.
struct Review {
    planned: Vec<Planned>,
    at: std::time::Instant,
}

impl Review {
    fn new(planned: Vec<Planned>) -> Self {
        Review {
            planned,
            at: std::time::Instant::now(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Screen {
    Data,
    Methods,
    Params,
    Run,
}

impl Screen {
    const ALL: [Screen; 4] = [Screen::Data, Screen::Methods, Screen::Params, Screen::Run];

    fn title(self) -> &'static str {
        match self {
            Screen::Data => "Data",
            Screen::Methods => "Methods",
            Screen::Params => "Parameters",
            Screen::Run => "Run",
        }
    }
}

/// What a row of the Methods screen sets up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// A fit on the picked data, or a method reading a run (`--from`).
    Plain,
    /// `pinto residual` on a run.
    Residual,
    /// A fit on the residual's files.
    Stage2,
    /// lra on the stage-2 fit, offered with lra on the residual's run.
    Stage2Lra,
}

/// A method, whether it is queued, and where it writes.
struct Row {
    form: Method,
    role: Role,
    /// The methods a stage-2 row can switch to (`m`), each keeping its
    /// flags: the one shown is `form`.
    alts: Vec<Method>,
    on: bool,
    /// The `--out` prefix: relative to where pinto run started, or
    /// absolute.
    out: String,
    /// Whether `out` was typed by hand, so the output header leaves it.
    typed: bool,
    /// Whether it finished in the last run it was in.
    done: bool,
}

/// What a line being typed will become.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Target {
    /// What every result of the run is named after.
    Header,
    Out(usize),
    /// Method row, field index.
    Field(usize, usize),
    Filter,
    /// The one batch of a data row's cells.
    Name(usize),
    /// A new name for a label of a data row's batch file.
    Rename(usize, String),
}

/// The labels of a data row's batch file, listed to rename them.
struct Labels {
    row: usize,
    cursor: usize,
}

struct Editor {
    target: Target,
    text: TextField,
}

/// One queued fit as the confirm popup shows it.
struct Planned {
    job: Job,
    /// What blocks it from running.
    problem: Option<String>,
    /// The flag clap blamed, to mark its row.
    blamed: Option<String>,
}

pub(crate) struct App {
    cli: clap::Command,
    here: PathBuf,
    screen: Screen,
    pairs: Vec<Pair>,
    pair_row: usize,
    browser: Option<Browser<Pick>>,
    /// Where the browser opens next.
    browse_dir: PathBuf,
    rows: Vec<Row>,
    /// What every `--out` not typed by hand is named after: see [`super::script::under`].
    header: String,
    method_row: usize,
    /// The method whose flags the parameters screen shows.
    param_method: usize,
    field_row: usize,
    advanced: bool,
    filter: String,
    editor: Option<Editor>,
    confirm: Option<Review>,
    confirm_scroll: usize,
    /// How far the confirm popup can scroll, as last drawn.
    confirm_max: std::cell::Cell<usize>,
    /// The method, command line and flag clap blamed, as last checked.
    blame: std::cell::RefCell<Option<Blame>>,
    /// Files for the workers to read.
    tasks: Sender<Task>,
    /// What the workers found.
    found: Receiver<Found>,
    /// Files still being read.
    describing: usize,
    /// Label files being read.
    reading: std::collections::BTreeSet<PathBuf>,
    /// The label list open over the data screen.
    labels: Option<Labels>,
    queue: Option<Queue>,
    /// Whether the terminal tells shift-enter from enter.
    shift_enter: bool,
    /// Whether a terminal is there: nothing is written to it otherwise.
    attached: bool,
    /// The fit under the cursor on the run screen.
    job_row: usize,
    message: Option<String>,
    quit: bool,
    /// The manifest to open in `pinto view` once the terminal is given back.
    view: Option<PathBuf>,
}

/// A file for a worker to read.
enum Task {
    Data(PathBuf),
    Labels(PathBuf),
}

/// Workers reading files at once: enough to hide a slow one, few enough
/// that taking hundreds of files does not open them all together.
const WORKERS: usize = 4;

/// Start the workers; they end when the returned sender is dropped.
fn workers() -> (Sender<Task>, Receiver<Found>) {
    let (tasks, todo) = std::sync::mpsc::channel::<Task>();
    let (tell, found) = std::sync::mpsc::channel();
    let todo = std::sync::Arc::new(std::sync::Mutex::new(todo));
    for _ in 0..WORKERS {
        let (todo, tell) = (todo.clone(), tell.clone());
        std::thread::spawn(move || loop {
            let next = todo.lock().ok().and_then(|t| t.recv().ok());
            let Some(task) = next else { break };
            let out = match task {
                Task::Data(path) => {
                    let (info, cells) = data::describe(&path);
                    Found::Data(path, info, cells)
                }
                Task::Labels(path) => {
                    let counts = batch::label_counts(&path);
                    Found::Labels(path, counts)
                }
            };
            if tell.send(out).is_err() {
                break;
            }
        });
    }
    (tasks, found)
}

/// What a worker found out about a file.
enum Found {
    /// A data file: what it holds and its cell count.
    Data(PathBuf, String, Option<usize>),
    /// A label file's labels and their cell counts.
    Labels(
        PathBuf,
        Result<std::collections::BTreeMap<String, usize>, String>,
    ),
}

/// A method row, its command line, and the flag clap blamed in it.
type Blame = (usize, Vec<String>, Option<String>);

/// Where the queue stands, in a line: `fits running: 1 of 3 done`, with
/// failures and stops when there are any.
fn fits_status(states: &[jobs::State], finished: bool) -> String {
    use jobs::State;
    let count = |f: fn(&State) -> bool| states.iter().filter(|s| f(s)).count();
    let done = count(|s| *s == State::Done);
    let failed = count(|s| matches!(s, State::Failed(_)));
    let stopped = count(|s| *s == State::Stopped);
    let mut line = format!(
        "fits {}: {done} of {} done",
        if finished { "finished" } else { "running" },
        states.len()
    );
    if failed > 0 {
        line += &format!(", {failed} failed");
    }
    if stopped > 0 {
        line += &format!(", {stopped} stopped");
    }
    line
}

/// `pinto view` on `manifest`, as `v` opens it once the screens are left;
/// [`legume_tui::Session::run_child`] runs it in this session as this one
/// runs (plain too, when it was asked to be).
fn view_command(exe: PathBuf, manifest: &Path) -> std::process::Command {
    let mut view = std::process::Command::new(exe);
    view.arg("view").arg(shown(manifest));
    view
}

/// Run the screens; `cli` is pinto's built command, the source of every
/// method's flags and the check of every command line.
pub fn run(cli: clap::Command, args: &crate::run::RunArgs) -> anyhow::Result<()> {
    // First: hosted, this process only relays the terminal from here on,
    // and the program the host starts does the rest.
    let session = crate::tui::TOOL.session("run", &args.session())?;
    let mut app = App::new(cli, args.start())?;
    let modes = crate::tui::modes(legume_tui::Mouse::Off);
    let ended = crate::tui::with_terminal(&session, modes, |terminal| {
        let result = app.screens_loop(&session, terminal);
        // Wait for the worker, so no fit it was starting outlives us.
        let ended = app.queue.take().map(Queue::finish);
        result.map(|()| ended)
    })?;
    // Said after the screen, or, ended detached (a `pinto kill` with no
    // terminal attached), by whoever looks next; each fit's script stays
    // to run it again.
    // Plain, on stdout as before.
    if let Some((jobs, states)) = ended {
        let lines = summary(&jobs, &states);
        if session.is_hosted() {
            session.leave(&lines.join("\n"));
        } else {
            for line in lines {
                println!("{line}");
            }
        }
    }
    // Not after a `pinto kill`: it asked for the session to end.
    if let (Some(manifest), false) = (&app.view, session.stop_requested()) {
        let mut view = view_command(std::env::current_exe()?, manifest);
        // In the same session: a `pinto kill` reaches it through us.
        session.run_child(&mut view)?;
    }
    Ok(())
}

impl App {
    /// Draw and take keys until the user quits, or `pinto kill` asks the
    /// screens to stop. Detached, the fits are followed and nothing is
    /// drawn.
    fn screens_loop(
        &mut self,
        session: &legume_tui::Session,
        terminal: &mut ratatui::DefaultTerminal,
    ) -> anyhow::Result<()> {
        while !self.quit {
            self.poll();
            self.attached = session.attached();
            session.draw(terminal, |f| self.draw(f))?;
            session.status(&self.session_status());
            // Redraw often only while something changes on its own.
            // Within legume-tui's grace, so a bar its fit cleared leaves
            // the screen on time.
            let wait = if self.running() || self.describing > 0 {
                Duration::from_millis(200).min(legume_tui::child::GRACE)
            } else {
                Duration::from_secs(60)
            };
            match session.poll(terminal, wait)? {
                Round::Event(Event::Key(k)) if k.kind != KeyEventKind::Release => self.key(k),
                // The terminal was asked as the screen came up, and again
                // after every attach.
                Round::Repainted => self.shift_enter = session.keyboard_enhanced(),
                // `pinto kill` quits as ctrl+c does: the fits running are
                // stopped as the screens are left.
                Round::Stopped => self.quit = true,
                _ => {}
            }
        }
        Ok(())
    }

    /// One line for `pinto sessions` and the resume picker.
    fn session_status(&self) -> String {
        match &self.queue {
            None => "setting up fits".into(),
            Some(q) => match q.shared.lock() {
                Ok(s) => fits_status(&s.states, s.finished),
                Err(_) => fits_status(&[], true),
            },
        }
    }

    fn new(cli: clap::Command, start: PathBuf) -> anyhow::Result<Self> {
        let here = std::env::current_dir()?;
        // `..` resolved, symlinks kept: the folders as the user knows them.
        let start = script::lexical(&here.join(start));
        // The methods on a fitted run follow the fits, where pinto has them.
        let on_runs = super::ON_RUNS
            .iter()
            .filter(|m| cli.find_subcommand(m).is_some());
        let mut rows = METHODS
            .iter()
            .chain(on_runs)
            .map(|m| Ok(Row::new(Method::new(&cli, m)?, Role::Plain, &here)))
            .collect::<anyhow::Result<Vec<_>>>()?;
        rows.extend(two_stage_rows(&cli, &here)?);
        let (tasks, found) = workers();
        Ok(App {
            cli,
            here,
            screen: Screen::Data,
            pairs: Vec::new(),
            pair_row: 0,
            browser: Some(Browser::open(start.clone(), Pick::Data, None)),
            browse_dir: start,
            rows,
            header: String::new(),
            method_row: 0,
            param_method: 0,
            field_row: 0,
            advanced: false,
            filter: String::new(),
            // Asked first, over the file browser.
            editor: Some(Editor {
                target: Target::Header,
                text: TextField::new(""),
            }),
            confirm: None,
            confirm_scroll: 0,
            confirm_max: std::cell::Cell::new(0),
            blame: std::cell::RefCell::new(None),
            tasks,
            found,
            describing: 0,
            reading: std::collections::BTreeSet::new(),
            labels: None,
            queue: None,
            shift_enter: false,
            attached: true,
            job_row: 0,
            message: None,
            quit: false,
            view: None,
        })
    }

    fn running(&self) -> bool {
        self.queue.as_ref().is_some_and(|q| !q.finished())
    }

    /// Screens reachable now: Run once a queue has started.
    fn screens(&self) -> Vec<Screen> {
        Screen::ALL
            .into_iter()
            .filter(|s| *s != Screen::Run || self.queue.is_some())
            .collect()
    }

    fn key(&mut self, k: KeyEvent) {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl && k.code == KeyCode::Char('c') {
            self.quit = true;
            return;
        }
        self.message = None;
        if self.editor.is_some() {
            self.editor_key(k);
        } else if self.labels.is_some() {
            self.labels_key(k);
        } else if self.browser.is_some() {
            self.browser_key(k);
        } else if self.confirm.is_some() {
            self.confirm_key(k);
        } else if !self.global_key(k) {
            match self.screen {
                Screen::Data => self.data_key(k),
                Screen::Methods => self.methods_key(k),
                Screen::Params => self.params_key(k),
                Screen::Run => self.run_key(k),
            }
        }
        // A reset, an edit or the filter can drop the row under the cursor
        // from the flags listed.
        self.field_row = self.field_row.min(self.visible().len().saturating_sub(1));
    }

    /// Keys every screen shares. Returns whether `k` was one.
    fn global_key(&mut self, k: KeyEvent) -> bool {
        let screens = self.screens();
        let at = screens.iter().position(|s| *s == self.screen).unwrap_or(0);
        let shift_enter = crate::tui::shift_enter(&k);
        match k.code {
            KeyCode::Tab => self.screen = screens[(at + 1) % screens.len()],
            _ if shift_enter => self.screen = screens[(at + 1) % screens.len()],
            KeyCode::BackTab => self.screen = screens[(at + screens.len() - 1) % screens.len()],
            KeyCode::Char(c @ '1'..='4') => {
                if let Some(s) = screens.get(c as usize - '1' as usize) {
                    self.screen = *s;
                }
            }
            _ if crate::tui::is_run_key(&k) => self.open_confirm(),
            KeyCode::Char('q') => {
                if self.running() {
                    self.message =
                        Some("fits are running: s stops them, ctrl-c stops and quits".into());
                } else {
                    self.quit = true;
                }
            }
            _ => return false,
        }
        true
    }

    // ───────────── typing a line ─────────────

    /// Take in what the workers found out, and start reading the label
    /// files not read yet.
    fn poll(&mut self) {
        self.settle();
        while let Ok(found) = self.found.try_recv() {
            self.describing = self.describing.saturating_sub(1);
            match found {
                Found::Data(path, info, cells) => {
                    if let Some(p) = self.pairs.iter_mut().find(|p| p.data == path) {
                        p.info = info;
                        p.cells = cells;
                    }
                }
                Found::Labels(path, counts) => {
                    self.reading.remove(&path);
                    // One map for every row sharing the file.
                    let counts = counts.map(std::sync::Arc::new);
                    for p in &mut self.pairs {
                        if p.batch.as_ref() == Some(&path) && p.labels.is_none() {
                            p.labels = Some(counts.clone());
                        }
                    }
                }
            }
        }
        let unread: std::collections::BTreeSet<PathBuf> = self
            .pairs
            .iter()
            .filter(|p| p.labels.is_none())
            .filter_map(|p| p.batch.clone())
            .filter(|f| !self.reading.contains(f))
            .collect();
        for path in unread {
            self.reading.insert(path.clone());
            self.read(Task::Labels(path));
        }
    }

    /// Hand `task` to the workers.
    fn read(&mut self, task: Task) {
        if self.tasks.send(task).is_ok() {
            self.describing += 1;
        }
    }

    // ───────────── batch labels ─────────────

    fn labels_key(&mut self, k: KeyEvent) {
        let Some(l) = self.labels.as_mut() else {
            return;
        };
        let n = self.pairs[l.row]
            .label_counts()
            .map_or(0, std::collections::BTreeMap::len);
        match k.code {
            KeyCode::Esc | KeyCode::Char('q' | 'e') => self.labels = None,
            KeyCode::Up => l.cursor = l.cursor.saturating_sub(1),
            KeyCode::Down => l.cursor = (l.cursor + 1).min(n.saturating_sub(1)),
            KeyCode::PageUp => l.cursor = l.cursor.saturating_sub(10),
            KeyCode::PageDown => l.cursor = (l.cursor + 10).min(n.saturating_sub(1)),
            KeyCode::Enter => {
                let (row, cursor) = (l.row, l.cursor);
                if let Some(label) = label_at(&self.pairs[row], cursor) {
                    let now = self.pairs[row]
                        .renames
                        .get(&label)
                        .unwrap_or(&label)
                        .clone();
                    self.edit(Target::Rename(row, label), now);
                }
            }
            _ => {}
        }
    }

    /// `e`: the labels of the row's batch file, to rename them.
    fn open_labels(&mut self) {
        let Some(p) = self.pairs.get(self.pair_row) else {
            return;
        };
        self.message = Some(match (&p.batch, &p.name, &p.labels) {
            (_, Some(_), _) | (None, None, _) => {
                "e renames the labels of a batch file: b chooses one".into()
            }
            (Some(_), None, None) => "its labels are still being read".into(),
            (Some(_), None, Some(Err(e))) => format!("its labels do not read: {e}"),
            (Some(_), None, Some(Ok(_))) => {
                self.labels = Some(Labels {
                    row: self.pair_row,
                    cursor: 0,
                });
                return;
            }
        });
    }

    // ───────────── data ─────────────

    fn data_key(&mut self, k: KeyEvent) {
        let last = self.pairs.len().saturating_sub(1);
        match k.code {
            KeyCode::Up => self.pair_row = self.pair_row.saturating_sub(1),
            KeyCode::Down => self.pair_row = (self.pair_row + 1).min(last),
            KeyCode::Char('a') => self.browse(Pick::Data),
            KeyCode::Char('c' | 'b' | 'n' | 'e' | 'x' | 'X' | 'd') if self.pairs.is_empty() => {
                self.message = Some("no data files yet: a adds some".into());
            }
            KeyCode::Char('c') => self.browse(Pick::Coord),
            KeyCode::Char('b') => self.browse(Pick::Batch),
            KeyCode::Char('n') => {
                let now = self.pairs[self.pair_row].name.clone().unwrap_or_default();
                self.edit(Target::Name(self.pair_row), now);
            }
            KeyCode::Char('e') => self.open_labels(),
            KeyCode::Char('x') => {
                if let Some(p) = self.pairs.get_mut(self.pair_row) {
                    p.coord = None;
                    p.clear_batch();
                }
            }
            KeyCode::Char('X') => self.pairs.iter_mut().for_each(|p| {
                p.coord = None;
                p.clear_batch();
            }),
            KeyCode::Char('d') | KeyCode::Delete if !self.pairs.is_empty() => {
                self.pairs.remove(self.pair_row);
                self.pair_row = self.pair_row.min(self.pairs.len().saturating_sub(1));
            }
            KeyCode::Char('K') if self.pair_row > 0 => {
                self.pairs.swap(self.pair_row, self.pair_row - 1);
                self.pair_row -= 1;
            }
            KeyCode::Char('J') if self.pair_row < last => {
                self.pairs.swap(self.pair_row, self.pair_row + 1);
                self.pair_row += 1;
            }
            _ => {}
        }
    }

    // ───────────── methods ─────────────

    /// The queued commands as one would type them where pinto run started.
    fn copy_text(&self) -> String {
        self.confirm
            .iter()
            .flat_map(|r| &r.planned)
            .map(|p| {
                let cmd = format!("pinto {}", script::line(&p.job.argv));
                if p.job.dir == self.here {
                    cmd
                } else {
                    let to =
                        script::quote(&script::relative(&p.job.dir, &self.here).to_string_lossy());
                    format!("(mkdir -p {to} && cd {to} && {cmd})")
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    // ───────────── run ─────────────

    fn run_key(&mut self, k: KeyEvent) {
        let Some(q) = &self.queue else { return };
        let last = q.jobs.len().saturating_sub(1);
        match k.code {
            KeyCode::Up => self.job_row = self.job_row.saturating_sub(1),
            KeyCode::Down => self.job_row = (self.job_row + 1).min(last),
            KeyCode::Char('s') => {
                q.stop();
                self.message = Some("stopping".into());
            }
            KeyCode::Char('r') => self.open_confirm(),
            KeyCode::Char('v') => {
                let job = &q.jobs[self.job_row.min(last)];
                let state = q.states().get(self.job_row.min(last)).cloned();
                if self.running() {
                    self.message = Some("still running: s stops, or wait for the fits".into());
                } else if state == Some(jobs::State::Done) && job.view().exists() {
                    self.view = Some(job.view().to_path_buf());
                    self.quit = true;
                } else {
                    self.message = Some(format!("{} did not finish", job.method));
                }
            }
            _ => {}
        }
    }
}

/// The residual of a run and the second stage fitted on it, where pinto
/// has `residual`: the residual row, the stage-2 fit (dsvd first, `m`
/// switches it), and lra on that fit where pinto has lra.
fn two_stage_rows(cli: &clap::Command, here: &Path) -> anyhow::Result<Vec<Row>> {
    if cli.find_subcommand(super::RESIDUAL).is_none() {
        return Ok(Vec::new());
    }
    let mut rows = vec![Row::new(
        Method::new(cli, super::RESIDUAL)?,
        Role::Residual,
        here,
    )];
    let mut forms = super::STAGE_TWO
        .iter()
        .map(|m| Method::new(cli, m))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let first = forms.remove(0);
    rows.push(Row {
        alts: forms,
        ..Row::new(first, Role::Stage2, here)
    });
    if cli.find_subcommand("lra").is_some() {
        rows.push(Row::new(Method::new(cli, "lra")?, Role::Stage2Lra, here));
    }
    Ok(rows)
}

/// What is said once the terminal is given back: how each run ended, and
/// its script.
fn summary(jobs: &[Job], states: &[jobs::State]) -> Vec<String> {
    jobs.iter()
        .zip(states)
        .map(|(j, s)| {
            let script = j.script();
            let what = match s {
                jobs::State::Done => "done".to_string(),
                jobs::State::Failed(why) => format!("failed: {why}"),
                jobs::State::Stopped => "stopped".to_string(),
                jobs::State::Waiting | jobs::State::Running => "not finished".to_string(),
            };
            if script.exists() {
                format!("{} {what}; again with: bash {}", j.method, shown(script))
            } else {
                format!("{} {what}", j.method)
            }
        })
        .collect()
}

/// `{base}`, or `{base}-2`, … : the first prefix in `dir` with no
/// manifest or script yet.
fn free_out(dir: &Path, base: &str) -> String {
    first_free(base, |p| jobs::outputs(dir, p).iter().any(|f| f.exists()))
}

/// `base`, else `base-2`, `base-3`, …: the first that is not `taken`.
pub(super) fn first_free(base: &str, taken: impl Fn(&str) -> bool) -> String {
    std::iter::once(base.to_string())
        .chain((2..).map(|k| format!("{base}-{k}")))
        .find(|p| !taken(p))
        .unwrap_or_else(|| base.to_string())
}

/// Put `text` on the clipboard: `pbcopy` where there is one, else the
/// terminal's OSC 52, when one is `attached`. Says which.
fn copy(text: &str, attached: bool) -> anyhow::Result<&'static str> {
    use std::io::Write;
    if let Ok(mut c) = std::process::Command::new("pbcopy")
        .stdin(std::process::Stdio::piped())
        .spawn()
    {
        if let Some(mut stdin) = c.stdin.take() {
            stdin.write_all(text.as_bytes())?;
        }
        if c.wait()?.success() {
            return Ok("pbcopy");
        }
    }
    anyhow::ensure!(attached, "no terminal attached");
    let mut out = std::io::stdout();
    write!(out, "\x1b]52;c;{}\x07", {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(text)
    })?;
    out.flush()?;
    Ok("terminal clipboard")
}

#[cfg(test)]
#[path = "tests/app.rs"]
mod tests;

/// The label at `i` of a data row's batch file, in the order listed.
fn label_at(p: &Pair, i: usize) -> Option<String> {
    p.label_counts()?.keys().nth(i).cloned()
}
