use super::*;

#[test]
fn log_lines_lose_their_lead_colours_and_redraws() {
    assert_eq!(log_line("[2026-01-01 INFO pinto] fitting"), "fitting");
    assert_eq!(log_line("\u{1b}[32mok\u{1b}[0m  "), "ok");
    assert_eq!(log_line("[not a prefix"), "[not a prefix");
    let mut seen = Vec::new();
    let log = "step 1\rstep 2\n[t INFO x] done\n".as_bytes();
    let last = follow_log(Some(log), |l| seen.push(l.to_string()));
    assert_eq!(seen, ["step 1", "step 2", "done"]);
    assert_eq!(last, "done");
}

#[cfg(unix)]
#[test]
fn a_failed_command_gives_its_last_line_and_a_stopped_one_says_so() {
    let mut c = Command::new("sh");
    c.args(["-c", "echo one >&2; echo 'Error: it broke' >&2; exit 3"]);
    assert_eq!(
        run_one(c, &Stopper::default(), |_| {}),
        Err(Failed::Error("it broke".into()))
    );

    // A child killed while it worked says how it ended, not that its last
    // progress line was the error.
    let mut c = Command::new("sh");
    c.args(["-c", "echo '[t INFO x] writing outputs' >&2; kill -9 $$"]);
    assert_eq!(
        run_one(c, &Stopper::default(), |_| {}),
        Err(Failed::Error(
            "signal: 9 (SIGKILL), after: writing outputs".into()
        ))
    );
    let mut c = Command::new("sh");
    c.args(["-c", "exit 2"]);
    assert_eq!(
        run_one(c, &Stopper::default(), |_| {}),
        Err(Failed::Error("exit status: 2".into()))
    );

    let stopper = Stopper::default();
    stopper.stop();
    let mut c = Command::new("sh");
    c.args(["-c", "sleep 5"]);
    assert_eq!(run_one(c, &stopper, |_| {}), Err(Failed::Stopped));

    let c = Command::new("/no/such/program");
    assert!(matches!(
        run_one(c, &Stopper::default(), |_| {}),
        Err(Failed::Error(why)) if why.starts_with("cannot run program:")
    ));
}
