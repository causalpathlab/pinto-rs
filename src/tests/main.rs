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
    // as `pinto run` names it
    assert_eq!(
        out_of(&["pinto", "cage-cell", "d.zarr", "--out", "res/"]),
        "res/cage-cell"
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

/// `pinto srt-simulate --help` lists the scenarios by their names, each with
/// what it is, and not by their earlier names, which still parse.
#[test]
fn simulate_help_lists_the_scenarios_by_name() {
    let help = built_cli()
        .find_subcommand_mut("srt-simulate")
        .expect("srt-simulate")
        .render_long_help()
        .to_string();
    let words = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let flat = words(&help);
    // words with their punctuation stripped, a name's hyphens kept
    let bare: Vec<&str> = help
        .split(|c: char| !(c.is_alphanumeric() || c == '-'))
        .collect();
    for sc in crate::simulate::params::SCENARIOS {
        assert!(
            help.lines().any(|l| l.trim() == sc.name),
            "{} not on a line of its own",
            sc.name
        );
        assert!(flat.contains(&words(sc.about)), "{} undescribed", sc.name);
        for old in sc.aliases {
            assert!(!bare.contains(old), "{old} shown");
        }
    }
    assert!(!bare.contains(&"first-donor-unconfounded"), "{help}");
    assert!(flat.contains("--set n_unconfounded_donors=1"), "{help}");
    let p = built_cli()
        .try_get_matches_from(["pinto", "srt-simulate", "--scenario", "L2", "-o", "o"])
        .unwrap();
    let (_, m) = p.subcommand().unwrap();
    assert_eq!(m.get_one::<String>("scenario").unwrap(), "L2");
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

fn lra_simulate(extra: &[&str]) -> LraSimulateArgs {
    let mut argv = vec!["pinto", "lra-simulate"];
    argv.extend_from_slice(extra);
    match Cli::try_parse_from(argv)
        .unwrap_or_else(|e| panic!("{e}"))
        .commands
    {
        Commands::LraSimulate(a) => *a,
        _ => panic!("not lra-simulate"),
    }
}

/// Any of `--one`, `--set`, `--pairs` and `--null-pairs` selects the
/// single-scenario mode, its pair counts ten each unless given; without
/// them the built-in grid runs.
#[test]
fn lra_simulate_selects_one_scenario_by_its_flags() {
    let base = ["--from", "r.pinto.json", "-o", "x"];
    let with = |extra: &[&str]| lra_simulate(&[&base[..], extra].concat());
    assert!(with(&[]).single().unwrap().is_none(), "the grid");
    for extra in [
        &["--one"][..],
        &["--set", "effect_pve=0.05"],
        &["--set=marker=1"],
        &["--pairs", "4"],
        &["--null-pairs=3"],
    ] {
        assert!(with(extra).single().unwrap().is_some(), "{extra:?}");
    }
    let one = with(&["--set", "effect_pve=0.05", "--set", "rep=2", "--pairs=4"])
        .single()
        .unwrap()
        .unwrap();
    assert_eq!((one.planted, one.null), (4, 10));
    assert_eq!((one.scenario.effect_pve, one.scenario.rep), (0.05, 2));
    assert!(Cli::try_parse_from(
        [&["pinto", "lra-simulate"][..], &base, &["--pairs", "0"]].concat()
    )
    .is_err());
}

/// A single-scenario call checks its point and its `--out` before reading
/// any data: the data files here do not exist, and the error is the point's.
#[test]
fn lra_simulate_refuses_a_bad_point_before_reading_data() {
    let data = ["missing.zarr", "--coord", "missing.csv"];
    let run = |extra: &[&str]| {
        lr_activity::fit_lra_simulate(&lra_simulate(&[&data[..], extra].concat()))
            .unwrap_err()
            .to_string()
    };
    let e = run(&[
        "-o",
        "x",
        "--set",
        "effect_pve=0.8",
        "--set",
        "response_field_pve=0.5",
    ]);
    assert!(e.contains("PVE out of range"), "{e}");
    let e = run(&["-o", "x", "--set", "no_such_key=1"]);
    assert!(e.contains("no_such_key"), "{e}");
    let e = run(&["--set", "effect_pve=0.05"]);
    assert!(e.contains("--out"), "{e}");
    let e =
        lr_activity::fit_lra_simulate(&lra_simulate(&["--from", "missing.pinto.json", "--one"]))
            .unwrap_err()
            .to_string();
    assert!(e.contains("--out"), "{e}");
}

#[test]
fn lra_simulate_has_a_version() {
    let e = Cli::try_parse_from(["pinto", "lra-simulate", "--version"]).unwrap_err();
    assert_eq!(e.kind(), clap::error::ErrorKind::DisplayVersion);
}

/// A single scenario never takes the `--from` run's own prefix, whose
/// `{prefix}.lra-simulate.parquet` is the grid's the manifest records.
#[test]
fn lra_simulate_keeps_a_single_scenario_off_the_run_s_table() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("d.zarr"), b"").unwrap();
    let m = dir.path().join("run.pinto.json");
    crate::util::metadata::PintoMetadata {
        command: "lc".into(),
        prefix: "run".into(),
        data_files: Some(vec!["d.zarr".into()]),
        ..Default::default()
    }
    .write(&m)
    .unwrap();
    let from = m.to_string_lossy().into_owned();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let e =
        lr_activity::fit_lra_simulate(&lra_simulate(&["--from", &from, "-o", &prefix, "--one"]))
            .unwrap_err()
            .to_string();
    assert!(e.contains("prefix"), "{e}");
    let other = dir.path().join("other").to_string_lossy().into_owned();
    let e = lr_activity::fit_lra_simulate(&lra_simulate(&["--from", &from, "-o", &other, "--one"]))
        .unwrap_err()
        .to_string();
    assert!(
        !e.contains("prefix"),
        "another prefix gets past the check: {e}"
    );
    // The same folder by a symbolic link is the same prefix, either way round.
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(dir.path(), &link).unwrap();
    let linked = |name: &str| link.join(name).to_string_lossy().into_owned();
    for (from, out) in [
        (linked("run.pinto.json"), prefix.clone()),
        (from, linked("run")),
    ] {
        let e =
            lr_activity::fit_lra_simulate(&lra_simulate(&["--from", &from, "-o", &out, "--one"]))
                .unwrap_err()
                .to_string();
        assert!(e.contains("prefix"), "--from {from} -o {out}: {e}");
    }
    // A folder not made yet and left by `..`, and on a file system that
    // ignores case the name in another case, are the same prefix too.
    let mut same = vec![dir.path().join("new").join("..").join("run")];
    if dir.path().join("RUN.pinto.json").is_file() {
        same.push(dir.path().join("RUN"));
    }
    for out in same {
        let out = out.to_string_lossy().into_owned();
        let e = lr_activity::fit_lra_simulate(&lra_simulate(&[
            "--from",
            &m.to_string_lossy(),
            "-o",
            &out,
            "--one",
        ]))
        .unwrap_err()
        .to_string();
        assert!(e.contains("prefix"), "-o {out}: {e}");
    }
}

/// A single scenario removes an earlier call's table and sidecar under its
/// prefix before it reads any data, so a call that fails there leaves
/// neither to pass for its own.
#[test]
fn lra_simulate_clears_a_single_scenario_s_earlier_outputs_first() {
    let dir = tempfile::tempdir().unwrap();
    // A data file that is there but cannot be read.
    std::fs::write(dir.path().join("d.zarr"), b"").unwrap();
    let m = dir.path().join("run.pinto.json");
    crate::util::metadata::PintoMetadata {
        command: "lc".into(),
        prefix: "run".into(),
        data_files: Some(vec!["d.zarr".into()]),
        ..Default::default()
    }
    .write(&m)
    .unwrap();
    let out = dir.path().join("p").to_string_lossy().into_owned();
    let earlier = [
        format!("{out}.lra-simulate.parquet"),
        format!("{out}.lra-simulate.json"),
    ];
    for f in &earlier {
        std::fs::write(f, "an earlier call's").unwrap();
    }
    let from = m.to_string_lossy().into_owned();
    lr_activity::fit_lra_simulate(&lra_simulate(&["--from", &from, "-o", &out, "--one"]))
        .unwrap_err();
    for f in &earlier {
        assert!(!std::path::Path::new(f).exists(), "{f} is left");
    }
}

/// Draws one command's help, short or long.
type Render = fn(&mut clap::Command) -> clap::builder::StyledStr;

/// Every command's help as `render` draws it, the top level's first, each
/// with the command line that reaches it.
fn every_help(render: Render) -> Vec<(String, String)> {
    fn walk(
        cmd: &mut clap::Command,
        path: String,
        render: Render,
        out: &mut Vec<(String, String)>,
    ) {
        out.push((path.clone(), render(cmd).to_string()));
        for sub in cmd.get_subcommands_mut() {
            let p = format!("{path} {}", sub.get_name());
            walk(sub, p, render, out);
        }
    }
    let mut out = Vec::new();
    walk(&mut built_cli(), "pinto".into(), render, &mut out);
    out
}

/// Every command's help as `render` draws it, then the README.
fn help_and_readme(render: Render) -> Vec<(String, String)> {
    let mut out = every_help(render);
    out.push(("README.md".into(), include_str!("../../README.md").into()));
    out
}

/// `pinto --help` indexes the commands by clap's own list, which names every
/// command, and keeps one quick start; no hand-kept list beside it.
#[test]
fn the_top_level_help_lists_every_command_once() {
    let mut cli = built_cli();
    let help = cli.render_long_help().to_string();
    assert!(!help.contains("SUBCOMMANDS:"), "{help}");
    assert!(!help.contains("All outputs are"), "{help}");
    assert_eq!(help.matches("QUICK START").count(), 1, "{help}");
    for sub in cli.get_subcommands().filter(|s| !s.is_hide_set()) {
        let name = sub.get_name();
        let listed = help
            .lines()
            .filter(|l| l.split_whitespace().next() == Some(name))
            .count();
        assert_eq!(listed, 1, "{name} listed {listed} times:\n{help}");
    }
}

/// Every `pinto <command>` a help text names is a command pinto has, all
/// its features on.
#[cfg(feature = "view")]
#[test]
fn help_names_only_commands_that_exist() {
    let cli = built_cli();
    for (path, help) in every_help(clap::Command::render_long_help) {
        for (at, _) in help.match_indices("`pinto ") {
            let rest = &help[at + "`pinto ".len()..];
            let word: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
                .collect();
            if word.is_empty() {
                continue;
            }
            assert!(
                cli.find_subcommand(&word).is_some(),
                "{path} --help names `pinto {word}`, which is no command"
            );
        }
        assert!(!help.contains("plot::load"), "{path}");
    }
}

/// Help and the README say what pinto does, not how it differs from an
/// earlier release, and number a residual fit's stages one way.
#[test]
fn help_and_readme_carry_no_release_notes() {
    for (path, text) in help_and_readme(clap::Command::render_long_help) {
        for stale in ["no longer", "used to be", "than before", "stage-3"] {
            assert!(!text.contains(stale), "{path} says {stale:?}");
        }
    }
}

/// The annotate stub is gone: its name and alias are unknown commands.
#[test]
fn annotate_is_no_command() {
    for name in ["annotate", "cage-annotate"] {
        let err = built_cli()
            .try_get_matches_from(["pinto", name])
            .expect_err(name);
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::InvalidSubcommand,
            "{name}"
        );
    }
}

#[cfg(feature = "view")]
use crate::session_cli_tests::flat as words;

/// `pinto mcp --help` prints what the server tells an agent, from the one
/// source the server sends.
#[cfg(feature = "view")]
#[test]
fn mcp_help_prints_the_servers_instructions() {
    let help = built_cli()
        .find_subcommand_mut("mcp")
        .expect("mcp")
        .render_long_help()
        .to_string();
    assert!(
        words(&help).contains(&words(&mcp::instructions())),
        "{help}"
    );
    assert!(help.lines().all(|l| l.chars().count() <= 80), "{help}");
}

/// The README keeps no copy of the instructions; it points to the help.
#[cfg(feature = "view")]
#[test]
fn the_readme_points_to_the_instructions_rather_than_copying_them() {
    let readme = include_str!("../../README.md");
    assert!(readme.contains("pinto mcp --help"));
    let readme = words(&readme.replace('>', " "));
    for sentence in mcp::instructions().split(". ").filter(|s| s.len() > 60) {
        assert!(!readme.contains(&words(sentence)), "{sentence}");
    }
}

/// No help, short or long, nor the README or the notices, names a method
/// pinto dropped for its license (the names are spelt in pieces so a search
/// of the tree finds none).
#[test]
fn no_help_names_a_dropped_method() {
    let dropped = [["bank", "sy"].concat(), ["mc", "lust"].concat()];
    let notices = (
        "THIRD_PARTY_NOTICES.md".to_string(),
        include_str!("../../THIRD_PARTY_NOTICES.md").to_string(),
    );
    let texts = [clap::Command::render_help, clap::Command::render_long_help]
        .into_iter()
        .flat_map(help_and_readme)
        .chain([notices]);
    for (path, text) in texts {
        let text = text.to_lowercase();
        for d in &dropped {
            assert!(!text.contains(d.as_str()), "{path} names {d}");
        }
    }
}

/// cage-cell has its own --embedding-l2 default; cage keeps cage's, and
/// either takes the flag when it is given.
#[test]
fn cage_cell_defaults_to_no_embedding_l2_and_cage_keeps_its_own() {
    let l2 = |argv: &[&str]| match Cli::try_parse_from(argv)
        .unwrap_or_else(|e| panic!("{e}"))
        .commands
    {
        Commands::Cage(a) | Commands::CageCell(a) => a.embedding_l2,
        _ => panic!("not cage"),
    };
    assert_eq!(l2(&["pinto", "cage", "d.zarr", "-o", "o"]), 0.0625);
    assert_eq!(l2(&["pinto", "cage-cell", "d.zarr", "-o", "o"]), 0.0);
    for cmd in ["cage", "cage-cell"] {
        assert_eq!(
            l2(&["pinto", cmd, "d.zarr", "-o", "o", "--embedding-l2", "0.25"]),
            0.25
        );
    }
}

/// cage-cell names the flags given on its command line that shape only the
/// pseudobulks it skips, and none it was not given (their defaults are not
/// the user's doing).
#[test]
fn cage_cell_names_only_the_no_op_flags_it_was_given() {
    use cell_activity_graph_embedding::fit::cell_level_no_ops;
    let given = |extra: &[&str]| cage_cell(extra, cell_level_no_ops);
    assert!(given(&[]).is_empty());
    assert_eq!(
        given(&["-d", "3", "--n-hvg", "100"]),
        ["--n-hvg", "--n-pseudobulk"]
    );
    // the batch-effect estimate reads these, so they are not no-ops
    assert_eq!(
        given(&["--num-levels=9", "--must-train-features", "f.txt"]),
        ["--must-train-features"]
    );
    // nor is any of them when modules are seeded from the pseudobulks
    assert!(given(&["-d", "3", "--n-hvg", "100", "--feature-modules", "4"]).is_empty());
}

/// cage-cell takes `--feature-modules` as cage does, before it reads a
/// file.
#[test]
fn cage_cell_takes_feature_modules() {
    use cell_activity_graph_embedding::fit::check_cell_level;
    let modules = ["--feature-modules", "4", "--n-pseudobulk", "3"];
    assert!(cage_cell(&modules, check_cell_level).is_ok());
}

/// cage-cell refuses, before it reads a file, what needs a hierarchy of
/// pseudobulks.
#[test]
fn cage_cell_refuses_hierarchy_flags() {
    use cell_activity_graph_embedding::fit::check_cell_level;
    for extra in [
        &["--chain-levels", "1"][..],
        &["--feature-embedding", "dict.parquet"][..],
    ] {
        let e = cage_cell(extra, |c, m| {
            check_cell_level(c, m).unwrap_err().to_string()
        });
        assert!(
            e.starts_with(&format!("cage-cell: {}", extra[0])),
            "{extra:?}: {e}"
        );
    }
    assert!(cage_cell(&[], check_cell_level).is_ok());
}

/// `f` of `pinto cage-cell` with `extra`, as built, and its matches.
fn cage_cell<T>(extra: &[&str], f: impl Fn(&clap::Command, &clap::ArgMatches) -> T) -> T {
    use clap::CommandFactory;
    let argv = [&["pinto", "cage-cell", "d.zarr", "-o", "o"][..], extra].concat();
    let mut cmd = Cli::command();
    let m = cmd.try_get_matches_from_mut(argv).unwrap();
    f(
        cmd.find_subcommand("cage-cell").unwrap(),
        m.subcommand_matches("cage-cell").unwrap(),
    )
}

/// cage's `--feature-modules` help is pinto's own: cage draws its NCE
/// negatives over pseudobulks, never within a module, and a run takes
/// modules or `--feature-inclusion moran`, not both. `--inclusion-blocks`
/// is a tuning knob, hidden. cage-cell says the same, and how it trains
/// them on cells.
#[test]
fn cage_says_what_its_feature_modules_do_and_hides_the_inclusion_knob() {
    let mut cli = built_cli();
    for cmd in ["cage", "cage-cell"] {
        let sub = cli.find_subcommand_mut(cmd).expect(cmd);
        let arg = |id: &str| {
            sub.get_arguments()
                .find(|a| a.get_id() == id)
                .unwrap_or_else(|| panic!("{cmd}: no {id}"))
                .clone()
        };
        assert!(arg("inclusion_blocks").is_hide_set(), "{cmd}");
        assert!(!arg("feature_inclusion").is_hide_set(), "{cmd}");
        assert!(!arg("feature_modules").is_hide_set(), "{cmd}");
        let help = sub.render_long_help().to_string();
        // the flag's entry: from its line up to the next flag's
        let flag_line = |l: &str| {
            let l = l.trim_start();
            l.starts_with("--") || l.as_bytes().get(2) == Some(&b',')
        };
        let modules: String = help
            .lines()
            .skip_while(|l| !l.trim_start().starts_with("--feature-modules <"))
            .enumerate()
            .take_while(|&(i, l)| i == 0 || !flag_line(l))
            .map(|(_, l)| format!("{l}\n"))
            .collect();
        assert!(!modules.is_empty(), "{cmd}: --feature-modules not listed");
        for wrong in ["own module", "WITHIN a module", "senna", "--multiome"] {
            assert!(!modules.contains(wrong), "{cmd} says {wrong:?}: {modules}");
        }
        for right in [
            "never within a module",
            "--feature-inclusion moran",
            "cage-cell trains on cells",
        ] {
            assert!(modules.contains(right), "{cmd} lacks {right:?}: {modules}");
        }
    }
}

/// Each of pinto's models, and each choice that handles noisy features,
/// is named where a user or an agent picks a method: the README, `pinto
/// run`'s help and the MCP instructions. dsvd is presented beside lc and
/// cage, not after them.
#[cfg(feature = "view")]
#[test]
fn the_models_and_their_feature_choices_are_named_where_methods_are_picked() {
    let mut cli = built_cli();
    let run = cli
        .find_subcommand_mut("run")
        .expect("run")
        .render_long_help()
        .to_string();
    let readme = include_str!("../../README.md");
    let mcp = mcp::instructions();
    for (what, text) in [("README", readme), ("pinto run", &run), ("mcp", &mcp)] {
        for model in ["lc", "dsvd", "cage"] {
            assert!(text.contains(model), "{what} does not name {model}");
        }
    }
    for (what, text) in [("README", readme), ("mcp", &mcp)] {
        for choice in ["feature-inclusion", "feature-modules", "feature-network"] {
            assert!(text.contains(choice), "{what} does not name {choice}");
        }
    }
    assert_eq!(run::MODELS[..2], ["lc", "dsvd"], "dsvd listed beside lc");
    assert!(readme.contains("### `dsvd` (delta-svd), recommended"));
}

/// lc's `--feature-network` help says what the file is and what the
/// modules do, and that pinto brings no network of its own.
#[test]
fn lc_says_what_its_feature_network_is() {
    let mut cli = built_cli();
    let lc = cli.find_subcommand_mut("lc").expect("lc");
    let arg = lc
        .get_arguments()
        .find(|a| a.get_id() == "feature_network")
        .expect("--feature-network")
        .clone();
    let help = arg.get_long_help().expect("long help").to_string();
    assert!(!help.contains("do not come from feature modules"), "{help}");
    for says in ["two-column", "Leiden", "module pairs", "ships no network"] {
        assert!(help.contains(says), "lacks {says:?}: {help}");
    }
    let readme = include_str!("../../README.md");
    assert!(readme.contains("ships no network"));
}

/// The README states `--max-threads`, its default, what it means on a
/// machine with more cores, and lc's Gibbs block layout as the code has it.
#[test]
fn the_readme_states_max_threads_and_lc_blocks() {
    let readme = include_str!("../../README.md");
    let flat = readme.split_whitespace().collect::<Vec<_>>().join(" ");
    let cap = util::common::DEFAULT_MAX_THREADS;
    use crate::link_community::gibbs::GIBBS_MIN_BLOCK_EDGES;
    for said in [
        format!("`--max-threads N` (default {cap})"),
        format!("at most {cap} worker threads"),
        format!("more than {cap} cores"),
        "Raise `--max-threads` to use more cores, including for lc's fine Gibbs".to_string(),
        "the same `--seed` and `--max-threads` give the same lc draws on any machine".to_string(),
        format!("at least {} edges into", 2 * GIBBS_MIN_BLOCK_EDGES),
        format!("blocks of at least {GIBBS_MIN_BLOCK_EDGES} edges"),
        "at most `--max-threads` blocks a component".to_string(),
    ] {
        assert!(flat.contains(&said), "README does not say {said:?}");
    }
}

/// A one-cell manifest at `{dir}/run` naming `command`, the model kind
/// impute dispatches on; no other artifact is written.
fn tiny_model(dir: &std::path::Path, command: &str) -> String {
    use crate::util::metadata::{create_dsvd_metadata, GraphParams, RunInputs};
    let prefix = dir.join("run").to_string_lossy().to_string();
    let data_files: Vec<Box<str>> = vec!["a.zarr".into()];
    let mut meta = create_dsvd_metadata(&RunInputs {
        prefix: &prefix,
        data_files: &data_files,
        coord_file: None,
        coord_columns: &[],
        n_cells: 1,
        n_features: 1,
        n_edges: 0,
        k: 1,
        graph: GraphParams::default(),
        ablation: None,
        delta: None,
    });
    meta.command = command.into();
    meta.write(&dir.join("run.pinto.json")).unwrap();
    prefix
}

/// `pinto impute` of the model at `model` with the extra flags `extra`: its
/// error message.
fn impute_error(model: &str, extra: &[&str]) -> String {
    use clap::CommandFactory;
    let mut argv = vec!["pinto", "impute", "q.zarr", "--model", model, "-o", "o"];
    argv.extend_from_slice(extra);
    let mut cmd = Cli::command();
    let m = cmd.try_get_matches_from_mut(&argv).unwrap();
    let qc_given = util::input::SrtInputArgs::qc_flags_given(
        cmd.find_subcommand("impute").unwrap(),
        m.subcommand_matches("impute").unwrap(),
    );
    let Commands::Impute(a) = Cli::try_parse_from(argv).unwrap().commands else {
        panic!("not impute")
    };
    crate::impute::run_impute(&a, &qc_given)
        .unwrap_err()
        .to_string()
}

/// impute of an lc / dsvd model runs no cell QC, so it refuses every QC
/// flag given on the command line, by name, even at its default value,
/// before it reads a file. `--no-qc` asks for what that path does and is
/// not refused; a cage model's impute runs QC through `pinto predict` and is
/// not refused.
#[test]
fn impute_refuses_qc_flags_on_the_profile_path_only() {
    let dir = tempfile::tempdir().unwrap();
    let lc = tiny_model(dir.path(), "lc");
    let err = impute_error(&lc, &["--qc-mads", "5"]);
    assert!(err.contains("runs no cell QC"), "{err}");
    assert!(err.contains(": --qc-mads would"), "{err}");
    let err = impute_error(&lc, &["--no-qc", "--qc-min-counts", "1", "--qc-histogram"]);
    assert!(
        err.contains(": --qc-min-counts, --qc-histogram would"),
        "{err}"
    );
    // neither --no-qc nor the defaults are reported: the run goes on and
    // fails later
    for extra in [&[][..], &["--no-qc"]] {
        let err = impute_error(&lc, extra);
        assert!(!err.contains("runs no cell QC"), "{err}");
    }

    let cage_dir = tempfile::tempdir().unwrap();
    let cage = tiny_model(cage_dir.path(), "cage");
    let err = impute_error(&cage, &["--qc-mads", "5"]);
    assert!(!err.contains("runs no cell QC"), "{err}");
}

/// impute refuses a feature QC floor as every other command that takes the
/// flag does, as it parses its command line.
#[test]
fn impute_refuses_a_feature_qc_floor() {
    let argv = ["pinto", "impute", "q.zarr", "--model", "m", "-o", "o"];
    let parse =
        |floor: &str| Cli::try_parse_from([&argv[..], &["--qc-feature-min-cells", floor]].concat());
    assert!(parse("0").is_ok());
    let err = parse("5").err().unwrap().to_string();
    assert!(err.contains("--qc-feature-min-cells"), "{err}");
    assert!(err.contains("reads only the cell QC"), "{err}");
}
