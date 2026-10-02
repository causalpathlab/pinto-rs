//! Terminal front end of `pinto run`: data files with their coordinates
//! and batch labels, methods, their flags, a confirm popup with the exact
//! commands, then the queue running with its log.

mod batch;
mod data;
mod draw;
mod form;
mod jobs;
mod script;

use crate::tui::browse::{Browser, Outcome};
use crate::tui::shown;
use data::Pair;
use data::{Pick, Side};
use form::{Field, Kind, Method};
use jobs::{Job, Queue};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

/// The methods `pinto run` sets up, in the order listed, each as typed:
/// the short alias of its subcommand.
pub const METHODS: [&str; 3] = ["lc", "cage", "dsvd"];

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

/// A method, whether it is queued, and where it writes.
struct Row {
    form: Method,
    on: bool,
    /// The `--out` prefix: relative to where pinto run started, or
    /// absolute.
    out: String,
    /// Whether `out` was typed by hand, so the output header leaves it.
    typed: bool,
    /// Whether it finished in the last run it was in.
    done: bool,
    /// The `out` of the last run it was in.
    last: Option<String>,
    /// An `out` whose script and batch files a run that did not finish
    /// left: a new run there clears them first.
    leftover: Option<String>,
}

impl Row {
    /// Whether a run at `out` clears what an unfinished one left there.
    fn clears(&self) -> bool {
        self.leftover.as_deref() == Some(self.out.as_str())
    }
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
    text: String,
}

/// One queued fit as the confirm popup shows it.
struct Planned {
    job: Job,
    /// What blocks it from running.
    problem: Option<String>,
    /// The flag clap blamed, to mark its row.
    blamed: Option<String>,
    /// Whether the method already finished earlier in this session.
    again: bool,
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
    /// What every `--out` not typed by hand is named after: see [`under`].
    header: String,
    method_row: usize,
    /// The method whose flags the parameters screen shows.
    param_method: usize,
    field_row: usize,
    advanced: bool,
    filter: String,
    editor: Option<Editor>,
    confirm: Option<Vec<Planned>>,
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
    /// Whether the end of the queue was taken in: see [`App::settle`].
    settled: bool,
    /// Whether the terminal tells shift-enter from enter.
    shift_enter: bool,
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

/// Run the screens; `cli` is pinto's built command, the source of every
/// method's flags and the check of every command line.
pub fn run(cli: clap::Command, start: PathBuf) -> anyhow::Result<()> {
    let mut app = App::new(cli, start)?;
    let ended = crate::tui::with_terminal(|terminal| {
        // Shift-enter told apart from enter, where the terminal can.
        app.shift_enter = enhance_keys(true);
        let result = app.screens_loop(terminal);
        if app.shift_enter {
            enhance_keys(false);
        }
        // Wait for the worker, so no fit it was starting outlives us.
        let ended = app.queue.take().map(Queue::finish);
        result.map(|()| ended)
    })?;
    if let Some((jobs, states)) = ended {
        for line in summary(&jobs, &states) {
            println!("{line}");
        }
    }
    if let Some(manifest) = &app.view {
        let exe = std::env::current_exe()?;
        std::process::Command::new(exe)
            .arg("view")
            .arg(shown(manifest))
            .status()?;
    }
    Ok(())
}

impl App {
    /// Draw and take keys until the user quits.
    fn screens_loop(&mut self, terminal: &mut ratatui::DefaultTerminal) -> anyhow::Result<()> {
        while !self.quit {
            self.poll();
            terminal.draw(|f| self.draw(f))?;
            // Redraw often only while something changes on its own.
            let wait = if self.running() || self.describing > 0 {
                Duration::from_millis(200)
            } else {
                Duration::from_secs(60)
            };
            if event::poll(wait)? {
                if let Event::Key(k) = event::read()? {
                    if k.kind != KeyEventKind::Release {
                        self.key(k);
                    }
                }
            }
        }
        Ok(())
    }

    fn new(cli: clap::Command, start: PathBuf) -> anyhow::Result<Self> {
        let here = std::env::current_dir()?;
        // `..` resolved, symlinks kept: the folders as the user knows them.
        let start = script::lexical(&here.join(start));
        let rows = METHODS
            .iter()
            .map(|m| {
                Ok(Row {
                    form: Method::new(&cli, m)?,
                    on: false,
                    out: free_out(&here, m),
                    typed: false,
                    done: false,
                    last: None,
                    leftover: None,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
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
                text: String::new(),
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
            settled: false,
            shift_enter: false,
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
        match k.code {
            KeyCode::Tab => self.screen = screens[(at + 1) % screens.len()],
            KeyCode::BackTab => self.screen = screens[(at + screens.len() - 1) % screens.len()],
            KeyCode::Enter if k.modifiers.contains(KeyModifiers::SHIFT) => {
                self.screen = screens[(at + 1) % screens.len()];
            }
            KeyCode::Char(c @ '1'..='4') => {
                if let Some(s) = screens.get(c as usize - '1' as usize) {
                    self.screen = *s;
                }
            }
            KeyCode::Char('G') => self.open_confirm(),
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

    fn edit(&mut self, target: Target, text: String) {
        self.editor = Some(Editor { target, text });
    }

    fn editor_key(&mut self, k: KeyEvent) {
        let Some(e) = self.editor.as_mut() else {
            return;
        };
        match k.code {
            KeyCode::Esc => {
                if e.target == Target::Filter {
                    self.filter.clear();
                    self.field_row = 0;
                }
                self.editor = None;
            }
            KeyCode::Enter => {
                let Editor { target, text } = self.editor.take().unwrap();
                match target {
                    Target::Header => {
                        self.header = text.trim().to_string();
                        self.refresh_outs();
                    }
                    Target::Out(i) => {
                        let t = text.trim();
                        let r = &mut self.rows[i];
                        if t.is_empty() {
                            // Cleared: back under the header.
                            r.typed = false;
                            r.out = free_out(&self.here, &under(&self.header, &r.form.name));
                        } else {
                            // A folder: the method's name in it.
                            r.typed = true;
                            r.out = if t.ends_with('/') {
                                under(t, &r.form.name)
                            } else {
                                t.to_string()
                            };
                        }
                    }
                    Target::Field(m, f) => {
                        let field = &mut self.rows[m].form.fields[f];
                        if text.trim().is_empty() && !field.default.is_empty() {
                            // Leaving it out means clap's default, so say that.
                            field.reset();
                            self.message = Some(format!(
                                "--{} cannot be left unset: back to its default {}",
                                field.long, field.default
                            ));
                        } else {
                            field.value = text;
                        }
                    }
                    Target::Filter => {}
                    Target::Name(i) => {
                        let p = &mut self.pairs[i];
                        p.clear_batch();
                        let t = text.trim();
                        if !t.is_empty() {
                            p.name = Some(t.to_string());
                        }
                    }
                    Target::Rename(i, label) => {
                        let p = &mut self.pairs[i];
                        let t = text.trim();
                        if t.is_empty() || t == label {
                            p.renames.remove(&label);
                        } else {
                            p.renames.insert(label, t.to_string());
                        }
                    }
                }
            }
            KeyCode::Backspace | KeyCode::Char(_) => {
                if let KeyCode::Char(c) = k.code {
                    e.text.push(c);
                } else {
                    e.text.pop();
                }
                // The filter narrows the flags as it is typed.
                if e.target == Target::Filter {
                    self.filter.clone_from(&e.text);
                    self.field_row = 0;
                }
            }
            _ => {}
        }
    }

    // ───────────── the file browser ─────────────

    fn browse(&mut self, want: Pick) {
        self.browser = Some(Browser::open(self.browse_dir.clone(), want, None));
    }

    fn browser_key(&mut self, k: KeyEvent) {
        let Some(b) = self.browser.as_mut() else {
            return;
        };
        // Chosen files, or none when cancelled; either way it closes.
        let chosen = match b.key(k) {
            Outcome::Ignored | Outcome::Moved => return,
            Outcome::Cancelled => None,
            Outcome::Chosen(c) => Some(c.files()),
        };
        let Some(b) = self.browser.take() else { return };
        self.browse_dir = b.dir;
        let Some(paths) = chosen else { return };
        match b.want.side() {
            None => self.take_data(paths),
            Some(side) => self.take_sides(&paths, side),
        }
    }

    fn take_data(&mut self, paths: Vec<PathBuf>) {
        let first_new = self.pairs.len();
        for p in paths {
            if self.pairs.iter().any(|q| q.data == p) {
                continue;
            }
            // Opening a large backend takes a while: not here.
            self.read(Task::Data(p.clone()));
            self.pairs.push(Pair::pending(p));
        }
        let added = self.pairs.len() - first_new;
        // Each folder's coordinate and label files go to its data by name,
        // the data taken before counted too, so a file named for one of
        // them goes to no new one.
        let folder = |p: &Pair| p.data.parent().unwrap_or(Path::new(".")).to_path_buf();
        let dirs: std::collections::BTreeSet<PathBuf> =
            self.pairs[first_new..].iter().map(folder).collect();
        for dir in dirs {
            let here: Vec<usize> = (0..self.pairs.len())
                .filter(|&i| folder(&self.pairs[i]) == dir)
                .collect();
            let data: Vec<PathBuf> = here.iter().map(|&i| self.pairs[i].data.clone()).collect();
            let data: Vec<&Path> = data.iter().map(PathBuf::as_path).collect();
            let near = data::near(&dir);
            let alone = near.data <= 1;
            for (side, files) in [(Side::Coord, &near.coords), (Side::Batch, &near.batches)] {
                for (&i, file) in here.iter().zip(data::beside(&data, files, alone)) {
                    if i >= first_new {
                        self.pairs[i].set(side, file);
                    }
                }
            }
        }
        self.pair_row = self.pairs.len().saturating_sub(1);
        let with = |side| self.pairs.iter().filter(|p| p.side(side).is_some()).count();
        self.message = Some(format!(
            "{added} data file{} added; of {}, {} with coordinates, {} with batch labels",
            if added == 1 { "" } else { "s" },
            self.pairs.len(),
            with(Side::Coord),
            with(Side::Batch),
        ));
    }

    /// One coordinate or batch file goes to the data row under the cursor;
    /// several are paired with all rows by name.
    fn take_sides(&mut self, paths: &[PathBuf], side: Side) {
        if let [one] = paths {
            if let Some(p) = self.pairs.get_mut(self.pair_row) {
                p.set(side, Some(one.clone()));
            }
            return;
        }
        for p in &mut self.pairs {
            p.set(side, None);
        }
        let n = self.pairs.len();
        let (what, key) = match side {
            Side::Coord => ("coordinate", 'c'),
            Side::Batch => ("batch", 'b'),
        };
        self.message = Some(match data::assign(&mut self.pairs, paths, side) {
            data::Paired::ByName(_) => format!("{n} {what} files paired by name"),
            data::Paired::InOrder => format!(
                "no names match: {n} {what} files paired in the order listed; check them"
            ),
            data::Paired::Partly(k) => format!(
                "{k} of {n} data files matched a {what} file by name; {key} sets the rest one by one"
            ),
        });
    }

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

    fn methods_key(&mut self, k: KeyEvent) {
        let last = self.rows.len() - 1;
        match k.code {
            KeyCode::Up => self.method_row = self.method_row.saturating_sub(1),
            KeyCode::Down => self.method_row = (self.method_row + 1).min(last),
            KeyCode::Char(' ') => {
                let r = &mut self.rows[self.method_row];
                r.on = !r.on;
                // The parameters screen shows a method its strip lists.
                if !self.param_methods().contains(&self.param_method) {
                    self.param_method = self.param_methods()[0];
                    self.field_row = 0;
                }
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

    /// Methods the parameters screen steps through: the queued ones, or
    /// all when none is.
    fn param_methods(&self) -> Vec<usize> {
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
    fn visible(&self) -> Vec<usize> {
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

    fn params_key(&mut self, k: KeyEvent) {
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
    fn out_of(&self, i: usize) -> (PathBuf, String) {
        let abs = self.here.join(&self.rows[i].out);
        let dir = abs
            .parent()
            .map_or_else(|| self.here.clone(), script::normalize);
        (dir, crate::tui::name(&abs))
    }

    /// The queued fits, each checked.
    fn plan(&self) -> Vec<Planned> {
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
            let argv = r.form.argv(&data, &coords, &batches, &out);
            let job = Job {
                method: r.form.name.clone(),
                dir: dir.clone(),
                out: out.clone(),
                argv,
                made,
                clear: r.clears(),
            };
            // What an unfinished run left is cleared, so only its manifest
            // would be written over.
            let kept = if job.clear { 1 } else { 3 };
            let mut blamed = None;
            let problem = if self.pairs.is_empty() {
                Some("no data files: add some on the Data screen".to_string())
            } else if let Some(p) = &coord_problem {
                Some(p.clone())
            } else if let Err(why) = &batch_args {
                Some(why.clone())
            } else if let Some(there) = jobs::outputs(&job.dir, &job.out)
                .into_iter()
                .take(kept)
                .find(|p| p.exists())
            {
                Some(format!(
                    "{} exists: change --out (o on Methods)",
                    shown(&there)
                ))
            } else if outs.contains(&dir.join(&out)) {
                Some("another queued method writes the same --out".to_string())
            } else if std::fs::metadata(&dir).is_ok_and(|m| !m.is_dir()) {
                // A folder not there yet is made when the fit starts.
                Some(format!("{} is not a folder", shown(&dir)))
            } else {
                form::check(&self.cli, &job.argv).err().inspect(|why| {
                    blamed = form::blamed(why, &r.form.fields).map(str::to_string);
                })
            };
            outs.push(dir.join(&out));
            planned.push(Planned {
                job,
                problem,
                blamed,
                again: r.done,
            });
        }
        planned
    }

    /// Worth saying when no data file has coordinates: the cell graph then
    /// comes from expression.
    fn coord_warning(&self) -> Option<String> {
        (!self.pairs.is_empty() && self.pairs.iter().all(|p| p.coord.is_none())).then(|| {
            "no coordinate files: the cell graph comes from expression (c adds them)".into()
        })
    }

    fn open_confirm(&mut self) {
        if !self.rows.iter().any(|r| r.on) {
            self.message = Some("no method queued: space on the Methods screen picks some".into());
            self.screen = Screen::Methods;
            return;
        }
        if self.running() {
            self.message = Some("fits are still running".into());
            return;
        }
        self.confirm = Some(self.plan());
        self.confirm_scroll = 0;
    }

    fn confirm_key(&mut self, k: KeyEvent) {
        match k.code {
            KeyCode::Esc | KeyCode::Char('q') => self.confirm = None,
            KeyCode::Up => self.confirm_scroll = self.confirm_scroll.saturating_sub(1),
            KeyCode::Down => {
                self.confirm_scroll = (self.confirm_scroll + 1).min(self.confirm_max.get());
            }
            KeyCode::Char('c') => {
                let text = self.copy_text();
                self.message = Some(match copy(&text) {
                    Ok(how) => format!("commands copied ({how})"),
                    Err(e) => format!("cannot copy: {e}"),
                });
            }
            KeyCode::Enter => {
                let Some(planned) = self.confirm.take() else {
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
                        self.settled = false;
                        for r in self.rows.iter_mut().filter(|r| r.on) {
                            r.last = Some(r.out.clone());
                        }
                        self.job_row = 0;
                        self.screen = Screen::Run;
                        self.move_outs_on();
                    }
                    Err(e) => self.message = Some(format!("cannot find pinto: {e}")),
                }
            }
            _ => {}
        }
    }

    /// Once the queue has ended, unqueue the methods it finished, so `G`
    /// again runs only those stopped, failed or not reached, each at the
    /// `--out` it had.
    fn settle(&mut self) {
        let Some(q) = &self.queue else { return };
        if self.settled || !q.finished() {
            return;
        }
        self.settled = true;
        let mut done = Vec::new();
        let mut left = false;
        for (job, state) in q.jobs.iter().zip(q.states()) {
            let Some(r) = self.rows.iter_mut().find(|r| r.form.name == job.method) else {
                continue;
            };
            r.done = state == jobs::State::Done;
            r.leftover = None;
            if r.done {
                r.on = false;
                done.push(job.method.clone());
            } else {
                left = true;
                // Back to the same --out, what the run left cleared then.
                if !job.manifest().exists() {
                    if let Some(last) = r.last.clone() {
                        if !r.typed {
                            r.out.clone_from(&last);
                        }
                        r.leftover = Some(last);
                    }
                }
            }
        }
        if !self.param_methods().contains(&self.param_method) {
            self.param_method = self.param_methods()[0];
            self.field_row = 0;
        }
        if left && !done.is_empty() {
            self.message = Some(format!(
                "{} done and unqueued: G runs the rest (space queues one again)",
                done.join(", ")
            ));
        }
    }

    /// Give each queued method whose `--out` was not typed by hand the
    /// next free name, so `G` again does not aim at the runs just started.
    fn move_outs_on(&mut self) {
        for r in self.rows.iter_mut().filter(|r| r.on && !r.typed) {
            r.out = next_out(&self.here, &under(&self.header, &r.form.name), Some(&r.out));
        }
    }

    /// Name every `--out` not typed by hand after the header, each the
    /// first free one.
    fn refresh_outs(&mut self) {
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

    /// The queued commands as one would type them where pinto run started.
    fn copy_text(&self) -> String {
        self.confirm
            .iter()
            .flatten()
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
            KeyCode::Char('r') => {
                if self.running() {
                    self.message = Some("fits are still running".into());
                } else if self.rows.iter().any(|r| r.on) {
                    self.open_confirm();
                } else {
                    self.message =
                        Some("every method finished: space on Methods queues one again".into());
                }
            }
            KeyCode::Char('v') => {
                let job = &q.jobs[self.job_row.min(last)];
                let state = q.states().get(self.job_row.min(last)).cloned();
                if self.running() {
                    self.message = Some("still running: s stops, or wait for the fits".into());
                } else if state == Some(jobs::State::Done) && job.manifest().exists() {
                    self.view = Some(job.manifest());
                    self.quit = true;
                } else {
                    self.message = Some(format!("{} did not finish", job.method));
                }
            }
            _ => {}
        }
    }
}

/// Ask the terminal to report modifiers on enter (`on`), or stop asking.
/// Returns whether it can.
fn enhance_keys(on: bool) -> bool {
    use ratatui::crossterm::event::{
        KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    };
    use ratatui::crossterm::{execute, terminal::supports_keyboard_enhancement};
    let mut out = std::io::stdout();
    if !on {
        return execute!(out, PopKeyboardEnhancementFlags).is_ok();
    }
    supports_keyboard_enhancement().unwrap_or(false)
        && execute!(
            out,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )
        .is_ok()
}

/// What is printed once the terminal is given back: how each run ended,
/// and its script.
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
                format!("{} {what}; again with: bash {}", j.method, shown(&script))
            } else {
                format!("{} {what}", j.method)
            }
        })
        .collect()
}

/// `stem` named after the output `header`: the stem alone when there is
/// none, `{header}{stem}` when it ends in `/` (a folder), `_`, `-` or
/// `.`, else `{header}_{stem}`.
fn under(header: &str, stem: &str) -> String {
    if header.is_empty() {
        stem.to_string()
    } else if header.ends_with(['/', '_', '-', '.']) {
        format!("{header}{stem}")
    } else {
        format!("{header}_{stem}")
    }
}

/// `{base}`, or `{base}-2`, … : the first prefix in `dir` with no
/// manifest or script yet.
fn free_out(dir: &Path, base: &str) -> String {
    next_out(dir, base, None)
}

/// The first of `{base}`, `{base}-2`, … in `dir` with no manifest or
/// script yet, and not `used` (a run started, its files not written yet).
fn next_out(dir: &Path, base: &str, used: Option<&str>) -> String {
    first_free(base, |p| {
        Some(p) == used || jobs::outputs(dir, p).iter().any(|f| f.exists())
    })
}

/// `base`, else `base-2`, `base-3`, …: the first that is not `taken`.
pub(super) fn first_free(base: &str, taken: impl Fn(&str) -> bool) -> String {
    std::iter::once(base.to_string())
        .chain((2..).map(|k| format!("{base}-{k}")))
        .find(|p| !taken(p))
        .unwrap_or_else(|| base.to_string())
}

/// Put `text` on the clipboard: `pbcopy` where there is one, else the
/// terminal's OSC 52. Says which.
fn copy(text: &str) -> anyhow::Result<&'static str> {
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
