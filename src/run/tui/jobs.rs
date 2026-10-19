//! The queued fits, run one after another on a worker thread: each writes
//! its `{out}.cmd.sh`, then starts pinto in the script's folder with the
//! same command, its log and the frames of its progress bars kept for
//! the screen.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::script;
use legume_numeric::matrix::common_io::{mkdir, remove_file};
use legume_tui::child::{run_one, Failed, Progress, Said, Stopper};

/// Log lines kept for the screen.
const KEEP: usize = 2000;

/// Scores kept in a bar's trend, at most drawn in its sparkline.
pub const TREND: usize = 64;

/// How long a bar goes without a frame before it is taken as cleared:
/// indicatif redraws a live one every 200 ms.
pub const STALE: std::time::Duration = std::time::Duration::from_secs(1);

/// A job's progress bar: where it stands, and the scores it has said.
#[derive(Clone, Debug)]
pub struct Bar {
    pub now: Progress,
    /// When `now` was drawn.
    pub at: std::time::Instant,
    /// One point per score the bar said, the newest last.
    pub trend: Vec<f64>,
    /// The message the newest point came from.
    scored: String,
    /// The bar's name: its message without numbers.
    label: String,
}

impl Bar {
    /// Whether the bar is still drawn, not cleared by its fit.
    pub fn live(&self) -> bool {
        self.at.elapsed() < STALE
    }
}

/// A bar message's name: its words that carry no number, so a sweep's
/// count or score does not make it another bar.
fn label(what: &str) -> String {
    what.split_whitespace()
        .filter(|w| !w.contains('=') && !w.chars().any(|c| c.is_ascii_digit()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// How long a fit interrupted by [`Queue::finish`] has to end before it
/// is killed.
pub(crate) const GRACE: std::time::Duration = std::time::Duration::from_secs(3);

/// One fit to run.
#[derive(Clone, Debug, Default)]
pub struct Job {
    pub method: String,
    /// Its row on the Methods screen.
    pub row: usize,
    /// Where it runs and the script goes.
    pub dir: PathBuf,
    /// The `--out` prefix, a name in `dir`.
    pub out: String,
    /// The command line without the program, paths relative to `dir`.
    pub argv: Vec<String>,
    /// Batch label files written for the run, before its script.
    pub made: Vec<(PathBuf, super::batch::Made)>,
    /// Whether to clear the script and batch files a run at the same
    /// `--out` left unfinished.
    pub clear: bool,
}

/// What a run with `--out` prefix `out` writes in `dir` that a new run
/// must not write over: its manifest, its script, and its batch files.
#[must_use]
pub fn outputs(dir: &Path, out: &str) -> [PathBuf; 3] {
    ["pinto.json", "cmd.sh", "batches"].map(|end| dir.join(format!("{out}.{end}")))
}

impl Job {
    #[must_use]
    pub fn manifest(&self) -> PathBuf {
        outputs(&self.dir, &self.out)[0].clone()
    }

    #[must_use]
    pub fn script(&self) -> PathBuf {
        outputs(&self.dir, &self.out)[1].clone()
    }

    /// What a run that did not finish leaves: its script and batch files.
    #[must_use]
    pub fn leftovers(&self) -> [PathBuf; 2] {
        let [_, script, batches] = outputs(&self.dir, &self.out);
        [script, batches]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum State {
    Waiting,
    Running,
    Done,
    Failed(String),
    Stopped,
}

/// What the screen reads while the queue runs.
#[derive(Default)]
pub struct Shared {
    pub states: Vec<State>,
    pub log: VecDeque<String>,
    /// The last line each job wrote.
    pub last: Vec<String>,
    /// Each job's progress bar, while it runs.
    pub bars: Vec<Option<Bar>>,
    pub finished: bool,
    /// Whether the end was taken in: see [`Queue::take_end`].
    pub seen: bool,
}

/// The queue, running.
pub struct Queue {
    pub jobs: Vec<Job>,
    pub shared: Arc<Mutex<Shared>>,
    stopper: Arc<Stopper>,
    worker: std::thread::JoinHandle<()>,
}

impl Queue {
    /// Start `jobs` in order with `program` (pinto itself).
    pub fn start(jobs: Vec<Job>, program: PathBuf) -> Self {
        let shared = Arc::new(Mutex::new(Shared {
            states: vec![State::Waiting; jobs.len()],
            last: vec![String::new(); jobs.len()],
            bars: vec![None; jobs.len()],
            ..Shared::default()
        }));
        let stopper = Arc::new(Stopper::new());
        let (s, st, js) = (shared.clone(), stopper.clone(), jobs.clone());
        // As the session starts its programs: `--no-detach` passed on.
        let env = crate::tui::child_env();
        let worker = std::thread::spawn(move || {
            for (i, job) in js.iter().enumerate() {
                let state = if st.is_stopped() {
                    State::Stopped
                } else {
                    set(&s, i, State::Running);
                    let mut command = std::process::Command::new(&program);
                    // A group of its own, so a stop reaches what it starts.
                    env.apply(&mut command, legume_tui::Group::Own);
                    run_job(job, command, i, &s, &st)
                };
                set(&s, i, state);
            }
            if let Ok(mut s) = s.lock() {
                s.finished = true;
            }
        });
        Queue {
            jobs,
            shared,
            stopper,
            worker,
        }
    }

    /// Stop the fit running (interrupted, killed when asked again) and
    /// start none after it.
    pub fn stop(&self) {
        self.stopper.stop();
    }

    /// Stop, and wait for the worker until the fit it may have been
    /// starting is stopped too: interrupted, and killed if it has not
    /// ended within [`GRACE`]. The jobs and how each ended.
    pub fn finish(self) -> (Vec<Job>, Vec<State>) {
        self.stop();
        let Queue {
            jobs,
            shared,
            stopper,
            worker,
        } = self;
        let start = std::time::Instant::now();
        while !worker.is_finished() && start.elapsed() < GRACE {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        if !worker.is_finished() {
            stopper.kill();
        }
        let _ = worker.join();
        let states = shared.lock().map(|s| s.states.clone()).unwrap_or_default();
        (jobs, states)
    }

    #[must_use]
    pub fn finished(&self) -> bool {
        self.shared.lock().map_or(true, |s| s.finished)
    }

    /// Whether the queue has ended, true only the first time it is asked
    /// after that.
    pub fn take_end(&self) -> bool {
        self.shared.lock().is_ok_and(|mut s| {
            let new = s.finished && !s.seen;
            s.seen |= s.finished;
            new
        })
    }

    /// Each job's state now.
    #[must_use]
    pub fn states(&self) -> Vec<State> {
        self.shared
            .lock()
            .map(|s| s.states.clone())
            .unwrap_or_default()
    }
}

fn set(s: &Mutex<Shared>, i: usize, state: State) {
    if let Ok(mut s) = s.lock() {
        s.states[i] = state;
    }
}

fn say(s: &Mutex<Shared>, i: usize, line: String) {
    if let Ok(mut s) = s.lock() {
        if s.log.len() >= KEEP {
            s.log.pop_front();
        }
        s.last[i].clone_from(&line);
        s.log.push_back(line);
    }
}

/// A frame `p` of job `i`'s progress bar: where it stands, and a new
/// point of its trend when it says a new score.
fn progress(s: &Mutex<Shared>, i: usize, p: Progress) {
    let label = label(&p.what);
    let Ok(mut s) = s.lock() else { return };
    // A new bar (none before, restarted, another length or name) starts a
    // new trend: its scores are not on the last one's scale.
    let slot = &mut s.bars[i];
    let b = match slot {
        Some(b) if p.pos >= b.now.pos && p.len == b.now.len && label == b.label => b,
        _ => slot.insert(Bar {
            now: Progress::default(),
            at: std::time::Instant::now(),
            trend: Vec::new(),
            scored: String::new(),
            label,
        }),
    };
    // A fit says each score once, and the bar redraws it until the next:
    // a point per message, so a score that holds still counts each time.
    if let Some(v) = p.score().filter(|_| p.what != b.scored) {
        if b.trend.len() >= TREND {
            b.trend.remove(0);
        }
        b.trend.push(v);
        b.scored.clone_from(&p.what);
    }
    b.now = p;
    b.at = std::time::Instant::now();
}

/// Run `job` with `command`, the program set up as the session starts
/// its programs.
fn run_job(
    job: &Job,
    mut command: std::process::Command,
    i: usize,
    s: &Mutex<Shared>,
    stopper: &Stopper,
) -> State {
    if job.manifest().exists() {
        return State::Failed(format!("{} exists", job.manifest().display()));
    }
    if job.clear {
        if let Err(e) = job
            .leftovers()
            .iter()
            .try_for_each(|p| remove_file(&p.to_string_lossy()))
        {
            return State::Failed(format!("cannot clear the unfinished run: {e}"));
        }
    }
    if let Err(e) = mkdir(&job.dir.to_string_lossy()) {
        return State::Failed(format!("cannot make {}: {e}", job.dir.display()));
    }
    if !job.made.is_empty() {
        let written = job
            .made
            .iter()
            .try_for_each(|(path, what)| super::batch::write(path, what));
        if let Err(e) = written {
            return State::Failed(format!("cannot write the batch files: {e}"));
        }
    }
    if let Err(e) = script::write(&job.script(), &job.out, &job.argv) {
        return State::Failed(format!("cannot write the script: {e}"));
    }
    say(
        s,
        i,
        format!("── {} · {}", job.method, crate::tui::shown(&job.script())),
    );
    command.args(&job.argv).current_dir(&job.dir);
    // The log level the script sets, so the run is the one it records.
    if std::env::var_os("RUST_LOG").is_none() {
        command.env("RUST_LOG", script::LOG_LEVEL);
    }
    // Its log, a line at a time, and where its progress bars stand.
    let each = |said| match said {
        Said::Line(line) => say(s, i, line),
        Said::Progress(p) => progress(s, i, p),
        _ => {}
    };
    let ran = run_one(command, stopper, each);
    if let Ok(mut s) = s.lock() {
        s.bars[i] = None;
    }
    match ran {
        Ok(()) => State::Done,
        Err(Failed::Stopped) => State::Stopped,
        Err(Failed::Start(why) | Failed::Exit(why)) => State::Failed(why),
        Err(e) => State::Failed(format!("{e:?}")),
    }
}

#[cfg(test)]
#[path = "tests/jobs.rs"]
mod tests;
