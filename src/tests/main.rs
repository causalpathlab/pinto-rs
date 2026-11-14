use super::*;

fn out_of(argv: &[&str]) -> String {
    let mut cli = Cli::try_parse_from(argv).unwrap_or_else(|e| panic!("{e}"));
    name_folder_out(&mut cli.commands);
    let (out, _) = cli.commands.out_mut().unwrap();
    out.to_string()
}

#[test]
fn an_out_ending_in_a_slash_is_a_folder_named_for_the_method() {
    assert_eq!(
        out_of(&["pinto", "lc", "d.zarr", "--out", "res/"]),
        "res/lc"
    );
    assert_eq!(
        out_of(&["pinto", "dsvd", "d.zarr", "--out", "a/b/"]),
        "a/b/dsvd"
    );
    assert_eq!(
        out_of(&["pinto", "lc", "d.zarr", "--out", "res/r"]),
        "res/r"
    );
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn lra(extra: &[&str]) -> Result<SrtLrActivityArgs, clap::Error> {
    let mut argv = vec!["pinto", "lra"];
    argv.extend_from_slice(extra);
    match Cli::try_parse_from(argv)?.commands {
        Commands::LrActivity(a) => Ok(*a),
        _ => panic!("not lra"),
    }
}

#[test]
fn lra_methods_parse_and_default_to_contact_effect() {
    use lr_activity::args::LraMethod;
    let a = lra(&["--from", "r.pinto.json"]).unwrap();
    assert_eq!(a.method, LraMethod::ContactEffect);
    assert_eq!(a.input.from.as_deref(), Some("r.pinto.json"));
    assert!(a.out.is_none() && a.input.data_files.is_empty());
    for (name, m) in [
        ("contact-effect", LraMethod::ContactEffect),
        ("coexpression-test", LraMethod::CoexpressionTest),
        ("coexpression-scores", LraMethod::CoexpressionScores),
    ] {
        assert_eq!(
            lra(&["-f", "r.pinto.json", "--method", name])
                .unwrap()
                .method,
            m
        );
    }
    assert!(lra(&["--method", "permutation"]).is_err());
    let a = lra(&["d.zarr", "--coord", "c.csv", "-o", "o"]).unwrap();
    assert_eq!(a.input.data_files, vec!["d.zarr".into()]);
    assert_eq!(a.input.coord_files, vec!["c.csv".into()]);
    // The old names still reach the command.
    assert!(Cli::try_parse_from(["pinto", "lr-activity", "-f", "r.pinto.json"]).is_ok());
}

#[test]
fn removed_lra_flags_are_refused_with_their_replacement() {
    for flag in ["--edge-scores-only", "--lc-prefix"] {
        assert!(lra(&["-f", "r.pinto.json", flag, "1"]).is_err(), "{flag}");
        let msg = check_removed_lra_flags(&strings(&["pinto", "lra", "-f", "r", flag]))
            .unwrap_err()
            .to_string();
        assert!(msg.contains(flag) && msg.contains("removed"), "{msg}");
    }
    let msg = check_removed_lra_flags(&strings(&["pinto", "lr-activity", "--edge-scores-only"]))
        .unwrap_err()
        .to_string();
    assert!(msg.contains("--method coexpression-scores"), "{msg}");
    assert!(check_removed_lra_flags(&strings(&["pinto", "lra", "-f", "r", "--seed", "1"])).is_ok());
    // Other commands are not checked.
    assert!(check_removed_lra_flags(&strings(&["pinto", "lc", "--lc-prefix", "3"])).is_ok());
}

#[test]
fn lra_simulate_parses_its_study_flags() {
    let parse = |extra: &[&str]| {
        let mut argv = vec!["pinto", "lra-simulate"];
        argv.extend_from_slice(extra);
        match Cli::try_parse_from(argv)
            .unwrap_or_else(|e| panic!("{e}"))
            .commands
        {
            Commands::LraSimulate(a) => a,
            _ => panic!("not lra-simulate"),
        }
    };
    let a = parse(&["--from", "r.pinto.json", "-o", "x"]);
    assert_eq!(a.out.as_deref(), Some("x"));
    // The study file and its example are gone.
    for flag in [&["--study", "s.json"][..], &["--example-study"][..]] {
        let argv = [&["pinto", "lra-simulate", "--from", "r"][..], flag].concat();
        assert!(Cli::try_parse_from(argv).is_err(), "{flag:?} still parses");
    }
    let mut cli = Cli::try_parse_from(["pinto", "lra-simulate", "-f", "r", "-o", "res/"]).unwrap();
    name_folder_out(&mut cli.commands);
    assert_eq!(cli.commands.out_mut().unwrap().0.as_ref(), "res/lra");
    // The benchmark flags are gone from lra.
    assert!(lra(&["-f", "r", "--sim-reps", "3"]).is_err());
}

#[test]
fn lra_help_lists_every_verdict() {
    let help = built_cli()
        .find_subcommand_mut("lra")
        .expect("lra")
        .render_long_help()
        .to_string();
    for v in lr_activity::ligand_effect::verdict::Verdict::ALL {
        assert!(
            help.contains(v.code()),
            "{} missing from lra --help",
            v.code()
        );
    }
    for m in ["contact-effect", "coexpression-test", "coexpression-scores"] {
        assert!(help.contains(m), "{m}");
    }
}

/// `pinto run --help` names the methods that run on a fitted run, and the
/// residual's stage-2 fit.
#[cfg(feature = "view")]
#[test]
fn run_help_names_the_methods_on_a_run() {
    let help = built_cli()
        .find_subcommand_mut("run")
        .expect("run")
        .render_long_help()
        .to_string();
    for m in ["lra", "lra-simulate", "residual", "stage-2"] {
        assert!(help.contains(m), "{m} missing from run --help");
    }
}

/// `--knn-spatial 0` is refused as it is parsed, by lra and lra-simulate
/// alike, rather than quietly built as one contact per cell.
#[test]
fn lra_needs_a_contact_per_cell_at_least() {
    for cmd in ["lra", "lra-simulate"] {
        let parse = |k: &str| {
            Cli::try_parse_from([
                "pinto",
                cmd,
                "d.zarr",
                "--coord",
                "c.csv",
                "--knn-spatial",
                k,
            ])
        };
        assert!(parse("0").is_err(), "{cmd}");
        assert!(parse("1").is_ok(), "{cmd}");
    }
}
