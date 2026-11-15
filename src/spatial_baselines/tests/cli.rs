//! The baselines' command lines stay small: a short `--help` each.

/// The baselines whose `-h` is kept short.
pub(crate) const BASELINES: &[&str] = &["stagate", "spatialde2", "spatialde2-pb"];

/// Most lines a baseline's `-h` may take.
const HELP_LINES: usize = 60;

#[test]
fn each_baseline_s_help_is_short() {
    for cmd in BASELINES {
        let help = crate::built_cli()
            .find_subcommand_mut(cmd)
            .unwrap()
            .render_help()
            .to_string();
        let lines = help.lines().count();
        assert!(lines <= HELP_LINES, "{cmd} -h: {lines} lines\n{help}");
        // spatialde2-pb has no --reference and takes -d
        let pb = cmd.ends_with("-pb");
        let flag = |f: &str| help.lines().any(|l| l.trim_start().starts_with(f));
        assert_eq!(flag("--reference"), !pb, "{cmd}");
        assert_eq!(flag("-d, --n-pseudobulk"), pb, "{cmd}");
        assert!(help.contains("--ignore-batch"), "{cmd}");
    }
}

/// `-d` belongs to spatialde2-pb alone, at least 1 as it parses, and
/// spatialde2-pb has no `--reference`, hidden or not (so `pinto run`'s
/// form and the MCP schema do not offer it).
#[test]
fn only_spatialde2_pb_takes_d_and_it_has_no_reference() {
    let parse = |cmd: &str, extra: &[&str]| {
        let mut argv = vec!["pinto", cmd, "d.zarr", "-c", "c.csv", "-o", "x"];
        argv.extend(extra);
        <crate::Cli as clap::Parser>::try_parse_from(argv).map(|c| c.commands)
    };
    for cmd in ["spatialde2", "stagate"] {
        assert!(parse(cmd, &["-d", "5"]).is_err(), "{cmd} -d");
        assert!(parse(cmd, &["--reference"]).is_ok(), "{cmd} --reference");
    }
    let Ok(crate::Commands::Spatialde2Pb(pb)) = parse("spatialde2-pb", &["-d", "5"]) else {
        panic!("spatialde2-pb -d")
    };
    assert_eq!(pb.n_pseudobulk, 5);
    assert!(parse("spatialde2-pb", &["-d", "0"]).is_err(), "-d 0");
    assert!(parse("spatialde2-pb", &["--reference"]).is_err());
    let cli = crate::built_cli();
    let pb = cli.find_subcommand("spatialde2-pb").unwrap();
    assert!(
        pb.get_arguments().all(|a| a.get_id() != "reference"),
        "spatialde2-pb carries a --reference"
    );
}
