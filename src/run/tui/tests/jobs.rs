use super::*;
use std::path::Path;

#[cfg(unix)]
#[test]
fn jobs_run_in_turn_and_leave_their_scripts() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let fake = dir.path().join("fake-pinto");
    std::fs::write(
        &fake,
        "#!/bin/sh\necho \"[t INFO x] working on $1\" >&2\nwhile [ $# -gt 0 ]; do [ \"$1\" = --out ] && touch \"$2.pinto.json\"; shift; done\n",
    )
    .unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let job = |m: &str| {
        Job::fit(
            m,
            0,
            dir.path(),
            m,
            vec![m.into(), "d.zarr".into(), "--out".into(), m.into()],
        )
    };
    let q = Queue::start(vec![job("lc"), job("cage")], fake);
    while !q.finished() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let s = q.shared.lock().unwrap();
    assert_eq!(s.states, [State::Done, State::Done]);
    assert_eq!(s.last, ["working on lc", "working on cage"]);
    drop(s);
    assert!(dir.path().join("lc.cmd.sh").exists());
    assert!(q.jobs.iter().all(|j| j.manifest().exists()));
}

#[cfg(unix)]
#[test]
fn a_job_whose_result_exists_does_not_run() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("r.pinto.json"), "{}").unwrap();
    let job = Job::fit(
        "lc",
        0,
        dir.path(),
        "r",
        vec!["lc".into(), "--out".into(), "r".into()],
    );
    let q = Queue::start(vec![job], PathBuf::from("/bin/false"));
    while !q.finished() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(matches!(
        q.shared.lock().unwrap().states[0],
        State::Failed(_)
    ));
    assert!(!dir.path().join("r.cmd.sh").exists());
}

/// A queue of two jobs, a and b, run by a pinto that is the shell
/// script `body`, its first fit started and past the line `READY` in
/// `body` (a new script can take a while to reach its first line).
#[cfg(unix)]
fn running(dir: &Path, body: &str) -> Queue {
    use std::os::unix::fs::PermissionsExt;
    let ready = dir.join("ready");
    let body = body.replace("READY", &format!(": > '{}'", ready.display()));
    let slow = dir.join("slow-pinto");
    std::fs::write(&slow, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&slow, std::fs::Permissions::from_mode(0o755)).unwrap();
    let job = |m: &str| Job::fit(m, 0, dir, m, vec![m.into(), "--out".into(), m.into()]);
    let q = Queue::start(vec![job("a"), job("b")], slow);
    let t = std::time::Instant::now();
    while !ready.exists() {
        assert!(
            t.elapsed() < std::time::Duration::from_secs(10),
            "never ready"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(q.shared.lock().unwrap().states[0], State::Running);
    q
}

#[cfg(unix)]
#[test]
fn a_stopped_queue_is_waited_for_and_its_fit_interrupted() {
    let dir = tempfile::tempdir().unwrap();
    let q = running(dir.path(), "READY\nexec sleep 30");
    let t = std::time::Instant::now();
    let (_, states) = q.finish();
    // Ended by the interrupt, well before it would be killed.
    assert!(t.elapsed() < STOP_GRACE);
    assert_eq!(states, [State::Stopped, State::Stopped]);
    assert!(!dir.path().join("b.cmd.sh").exists(), "b never started");
}

#[cfg(unix)]
#[test]
fn a_fit_that_ignores_the_interrupt_is_killed_after_a_grace() {
    let dir = tempfile::tempdir().unwrap();
    let q = running(dir.path(), "trap '' INT\nREADY\nsleep 30");
    let t = std::time::Instant::now();
    let (_, states) = q.finish();
    assert!(t.elapsed() >= STOP_GRACE);
    assert!(t.elapsed() < STOP_GRACE + std::time::Duration::from_secs(5));
    assert_eq!(states, [State::Stopped, State::Stopped]);
    assert!(!dir.path().join("b.cmd.sh").exists(), "b never started");
}

#[cfg(unix)]
#[test]
fn batch_files_made_for_a_run_are_written_before_it_starts() {
    let dir = tempfile::tempdir().unwrap();
    let job = Job {
        made: vec![(
            dir.path().join("r.batches/s1.txt"),
            crate::run::tui::batch::Made::Repeat("b1".into(), 2),
        )],
        ..Job::fit(
            "lc",
            0,
            dir.path(),
            "r",
            vec!["lc".into(), "--out".into(), "r".into()],
        )
    };
    let q = Queue::start(vec![job], PathBuf::from("/usr/bin/true"));
    while !q.finished() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(
        std::fs::read_to_string(dir.path().join("r.batches/s1.txt")).unwrap(),
        "b1\nb1\n"
    );
    assert!(dir.path().join("r.cmd.sh").exists());
}

#[cfg(unix)]
#[test]
fn a_new_out_folder_is_made_before_the_script() {
    let dir = tempfile::tempdir().unwrap();
    let at = dir.path().join("res").join("sub");
    let job = Job::fit(
        "lc",
        0,
        &at,
        "r",
        vec!["lc".into(), "--out".into(), "r".into()],
    );
    let q = Queue::start(vec![job], PathBuf::from("true"));
    while !q.finished() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(q.shared.lock().unwrap().states[0], State::Done);
    assert!(at.join("r.cmd.sh").exists());
}

#[cfg(unix)]
#[test]
fn a_run_where_one_did_not_finish_clears_its_script_and_batch_files_first() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("r.cmd.sh"), "old").unwrap();
    std::fs::create_dir(dir.path().join("r.batches")).unwrap();
    std::fs::write(dir.path().join("r.batches/old.txt"), "x\n").unwrap();
    let job = Job {
        clear: true,
        ..Job::fit(
            "lc",
            0,
            dir.path(),
            "r",
            vec!["lc".into(), "--out".into(), "r".into()],
        )
    };
    let q = Queue::start(vec![job], PathBuf::from("/usr/bin/true"));
    while !q.finished() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(q.states(), vec![State::Done]);
    assert_ne!(
        std::fs::read_to_string(dir.path().join("r.cmd.sh")).unwrap(),
        "old"
    );
    assert!(!dir.path().join("r.batches").exists());
}

#[test]
fn a_job_s_bar_and_log_follow_what_it_said() {
    // What the job says reaches its bar and its log; how the bar follows
    // its frames is legume-tui's.
    use legume_tui::child::Progress;
    let s = Mutex::new(Shared {
        bars: vec![None],
        last: vec![String::new()],
        ..Shared::default()
    });
    heard(
        &s,
        0,
        Said::Progress(Progress::new(1, 10, "sweeps 1: moves=9")),
    );
    assert!(s.lock().unwrap().bars[0].as_ref().unwrap().shown);
    heard(&s, 0, Said::Line("level 1 done".into()));
    let st = s.lock().unwrap();
    assert!(!st.bars[0].as_ref().unwrap().shown);
    assert_eq!(st.last[0], "level 1 done");
    assert_eq!(st.log.back().map(String::as_str), Some("level 1 done"));
}

/// A job reading a run (lra from `pinto run` or the viewer) beside it, at
/// the run's own prefix: clearing what an unfinished run of it left takes
/// its own script, never the run's manifest, script or batch files.
#[cfg(unix)]
#[test]
fn a_job_reading_a_run_clears_only_its_own_script() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    for f in ["run.pinto.json", "run.cmd.sh", "run.lra.cmd.sh"] {
        std::fs::write(d.join(f), "kept").unwrap();
    }
    std::fs::create_dir(d.join("run.batches")).unwrap();
    std::fs::write(d.join("run.batches/s1.txt"), "b1\n").unwrap();
    let argv = vec!["lra".into(), "--from".into(), "run.pinto.json".into()];
    let paths = ["run.lra.parquet", "run.lra.cmd.sh", "run.pinto.json"].map(|f| d.join(f));
    let job = Job {
        clear: true,
        ..Job::reading("lra", d, "run", argv, paths)
    };
    let run_own = |p: &PathBuf| {
        p.ends_with("run.batches") || p.ends_with("run.pinto.json") || p.ends_with("run.cmd.sh")
    };
    assert!(!job.leftovers.iter().any(run_own), "{:?}", job.leftovers);
    assert!(!job.in_the_way().iter().any(run_own));
    let q = Queue::start(vec![job], PathBuf::from("/usr/bin/true"));
    while !q.finished() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(q.states(), vec![State::Done]);
    assert!(d.join("run.batches/s1.txt").exists());
    for f in ["run.pinto.json", "run.cmd.sh"] {
        assert_eq!(std::fs::read_to_string(d.join(f)).unwrap(), "kept");
    }
    let script = std::fs::read_to_string(d.join("run.lra.cmd.sh")).unwrap();
    assert!(script.contains("if [ -f run.lra.parquet ]"), "{script}");
}
