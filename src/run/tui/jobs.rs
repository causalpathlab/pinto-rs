//! The queued fits, run one after another on a worker thread: each writes
//! its `{out}.cmd.sh`, then starts pinto in the script's folder with the
//! same command, its log kept for the screen.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use super::script;
use crate::tui::child::{run_one, Failed, Stopper};
use legume_numeric::matrix::common_io::mkdir;

/// Log lines kept for the screen.
const KEEP: usize = 2000;

/// One fit to run.
#[derive(Clone, Debug)]
pub struct Job {
    pub method: String,
    /// Where it runs and the script goes.
    pub dir: PathBuf,
    /// The `--out` prefix, a name in `dir`.
    pub out: String,
    /// The command line without the program, paths relative to `dir`.
    pub argv: Vec<String>,
    /// Batch label files written for the run, before its script.
    pub made: Vec<(PathBuf, super::batch::Made)>,
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

    /// Where the batch label files written for the run go.
    #[must_use]
    pub fn batches(&self) -> PathBuf {
        outputs(&self.dir, &self.out)[2].clone()
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
    pub finished: bool,
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
            ..Shared::default()
        }));
        let stopper = Arc::new(Stopper::default());
        let (s, st, js) = (shared.clone(), stopper.clone(), jobs.clone());
        let worker = std::thread::spawn(move || {
            for (i, job) in js.iter().enumerate() {
                let state = if st.is_stopped() {
                    State::Stopped
                } else {
                    set(&s, i, State::Running);
                    run_job(job, &program, i, &s, &st)
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

    /// Kill the fit running and start none after it.
    pub fn stop(&self) {
        self.stopper.stop();
    }

    /// Stop, and wait for the worker until the fit it may have been
    /// starting is killed too: the jobs and how each ended.
    pub fn finish(self) -> (Vec<Job>, Vec<State>) {
        self.stop();
        let Queue {
            jobs,
            shared,
            worker,
            ..
        } = self;
        let _ = worker.join();
        let states = shared.lock().map(|s| s.states.clone()).unwrap_or_default();
        (jobs, states)
    }

    #[must_use]
    pub fn finished(&self) -> bool {
        self.shared.lock().map_or(true, |s| s.finished)
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

fn run_job(
    job: &Job,
    program: &std::path::Path,
    i: usize,
    s: &Mutex<Shared>,
    stopper: &Stopper,
) -> State {
    if job.manifest().exists() {
        return State::Failed(format!("{} exists", job.manifest().display()));
    }
    if let Err(e) = mkdir(&job.dir.to_string_lossy()) {
        return State::Failed(format!("cannot make {}: {e}", job.dir.display()));
    }
    if !job.made.is_empty() {
        let written = std::fs::create_dir(job.batches())
            .map_err(anyhow::Error::from)
            .and_then(|()| {
                job.made
                    .iter()
                    .try_for_each(|(path, what)| super::batch::write(path, what))
            });
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
    let mut command = Command::new(program);
    command.args(&job.argv).current_dir(&job.dir);
    // The log level the script sets, so the run is the one it records.
    if std::env::var_os("RUST_LOG").is_none() {
        command.env("RUST_LOG", script::LOG_LEVEL);
    }
    match run_one(command, stopper, |line| say(s, i, line.to_string())) {
        Ok(()) => State::Done,
        Err(Failed::Stopped) => State::Stopped,
        Err(Failed::Error(why)) => State::Failed(why),
    }
}

#[cfg(test)]
#[path = "tests/jobs.rs"]
mod tests;
