use super::*;

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
    let job = |m: &str| Job {
        method: m.into(),
        dir: dir.path().to_path_buf(),
        out: m.into(),
        argv: vec![m.into(), "d.zarr".into(), "--out".into(), m.into()],
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
    let job = Job {
        method: "lc".into(),
        dir: dir.path().to_path_buf(),
        out: "r".into(),
        argv: vec!["lc".into(), "--out".into(), "r".into()],
    };
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

#[cfg(unix)]
#[test]
fn a_stopped_queue_is_waited_for_and_its_fit_killed() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let slow = dir.path().join("slow-pinto");
    std::fs::write(&slow, "#!/bin/sh\nexec sleep 30\n").unwrap();
    std::fs::set_permissions(&slow, std::fs::Permissions::from_mode(0o755)).unwrap();
    let job = |m: &str| Job {
        method: m.into(),
        dir: dir.path().to_path_buf(),
        out: m.into(),
        argv: vec![m.into(), "--out".into(), m.into()],
    };
    let q = Queue::start(vec![job("a"), job("b")], slow);
    while q.shared.lock().unwrap().states[0] != State::Running {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    std::thread::sleep(std::time::Duration::from_millis(50));
    let t = std::time::Instant::now();
    q.stop();
    q.join();
    assert!(t.elapsed() < std::time::Duration::from_secs(5));
    assert!(q.finished());
    assert_eq!(
        q.shared.lock().unwrap().states,
        [State::Stopped, State::Stopped]
    );
    assert!(!dir.path().join("b.cmd.sh").exists(), "b never started");
}
