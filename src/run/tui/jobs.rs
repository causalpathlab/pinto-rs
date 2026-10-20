//! The queued fits, run one after another on a worker thread: each writes
//! its `{out}.cmd.sh`, then starts pinto in the script's folder with the
//! same command, its log and the frames of its progress bars kept for
//! the screen.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::script;
use legume_numeric::matrix::common_io::{mkdir, remove_file};
use legume_tui::child::{run_one, Bar, Failed, Said, Stopper};

/// Log lines kept for the screen.
const KEEP: usize = 2000;

/// How long a fit interrupted by [`Queue::finish`] has to end before it
/// is killed.
pub(crate) const STOP_GRACE: std::time::Duration = std::time::Duration::from_secs(3);

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

pub use crate::run::script::outputs;

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

impl Shared {
    /// Line `line` of job `i`'s log, the newest.
    fn push(&mut self, i: usize, line: String) {
        if self.log.len() >= KEEP {
            self.log.pop_front();
        }
        self.last[i].clone_from(&line);
        self.log.push_back(line);
    }
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
    /// ended within [`STOP_GRACE`]. The jobs and how each ended.
    pub fn finish(self) -> (Vec<Job>, Vec<State>) {
        self.stop();
        let Queue {
            jobs,
            shared,
            stopper,
            worker,
        } = self;
        let start = std::time::Instant::now();
        while !worker.is_finished() && start.elapsed() < STOP_GRACE {
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

/// What job `i` said: a line of its log, a frame of its progress bar, or
/// the bar cleared. A line or a clear hides the bar until it is drawn
/// again ([`Bar::track`]); the screen keeps it for legume-tui's grace.
fn heard(s: &Mutex<Shared>, i: usize, said: Said) {
    // One lock: the screen never sees the bar hidden for a line not yet in.
    let Ok(mut s) = s.lock() else { return };
    Bar::track(&mut s.bars[i], &said);
    if let Said::Line(line) = said {
        s.push(i, line);
    }
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
    heard(
        s,
        i,
        Said::Line(format!(
            "── {} · {}",
            job.method,
            crate::tui::shown(&job.script())
        )),
    );
    command.args(&job.argv).current_dir(&job.dir);
    // The log level the script sets, so the run is the one it records.
    if std::env::var_os("RUST_LOG").is_none() {
        command.env("RUST_LOG", script::LOG_LEVEL);
    }
    // Its log, a line at a time, and where its progress bars stand.
    let ran = run_one(command, stopper, |said| heard(s, i, said));
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
