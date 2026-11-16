//! The baselines' command lines stay small: a short `--help` each.

/// The baselines whose `-h` is kept short.
pub(crate) const BASELINES: &[&str] = &[
    "stagate",
    "spatialde2",
    "spatialpca",
    "mefisto",
    "spatialde2-pb",
    "stagate-pb",
    "spatialpca-pb",
    "mefisto-pb",
];

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
        // no baseline has --reference; the -pb forms take -d
        let pb = cmd.ends_with("-pb");
        let flag = |f: &str| help.lines().any(|l| l.trim_start().starts_with(f));
        assert!(!flag("--reference"), "{cmd}");
        assert_eq!(flag("-d, --n-pseudobulk"), pb, "{cmd}");
        assert!(help.contains("--ignore-batch"), "{cmd}");
    }
}

/// `-d` belongs to the -pb forms of spatialde2 and stagate, not to them,
/// at least 1 as it parses, and no
/// baseline has a `--reference`, hidden or not (so neither `pinto run`'s
/// form nor the MCP schema offers it): each runs its sparse formulation.
#[test]
fn only_the_pb_forms_take_d_and_no_baseline_has_reference() {
    let parse = |cmd: &str, extra: &[&str]| {
        let mut argv = vec!["pinto", cmd, "d.zarr", "-c", "c.csv", "-o", "x"];
        argv.extend(extra);
        <crate::Cli as clap::Parser>::try_parse_from(argv).map(|c| c.commands)
    };
    for cmd in ["spatialde2", "stagate"] {
        assert!(parse(cmd, &["-d", "5"]).is_err(), "{cmd} -d");
    }
    let Ok(crate::Commands::Spatialde2Pb(pb)) = parse("spatialde2-pb", &["-d", "5"]) else {
        panic!("spatialde2-pb -d")
    };
    assert_eq!(pb.n_pseudobulk, 5);
    assert!(parse("spatialde2-pb", &["-d", "0"]).is_err(), "-d 0");
    let Ok(crate::Commands::StagatePb(pb)) = parse("stagate-pb", &["-d", "5"]) else {
        panic!("stagate-pb -d")
    };
    assert_eq!(pb.n_pseudobulk, 5);
    assert!(parse("stagate-pb", &["-d", "0"]).is_err(), "-d 0");
    let cli = crate::built_cli();
    for cmd in BASELINES {
        assert!(parse(cmd, &["--reference"]).is_err(), "{cmd} --reference");
        let sub = cli.find_subcommand(cmd).unwrap();
        assert!(
            sub.get_arguments().all(|a| a.get_id() != "reference"),
            "{cmd} carries a --reference"
        );
    }
}

/// The factor baselines show two method flags each, at their papers'
/// defaults, `--n-pcs` naming SpatialPCA's; their tuning knobs are hidden
/// (so neither `pinto run`'s form nor the MCP schema offers them), and the
/// `-pb` forms take `-d`.
#[test]
fn factor_baselines_show_few_flags_at_their_papers_defaults() {
    let parse = |argv: &[&str]| {
        let mut full = vec!["pinto"];
        full.extend(argv);
        full.extend(["d.zarr", "-c", "c.csv", "-o", "x"]);
        <crate::Cli as clap::Parser>::try_parse_from(full).map(|c| c.commands)
    };
    let Ok(crate::Commands::Spatialpca(a)) = parse(&["spatialpca"]) else {
        panic!("spatialpca")
    };
    assert_eq!(
        (a.fit.n_factors, a.fit.n_genes, a.fit.refine_k),
        (20, 3000, 6)
    );
    let Ok(crate::Commands::Spatialpca(a)) = parse(&["spatialpca", "--n-pcs", "7"]) else {
        panic!("--n-pcs")
    };
    assert_eq!(a.fit.n_factors, 7);
    // no dense-kernel knobs
    assert!(parse(&["spatialpca", "--bandwidth", "1.5"]).is_err());
    assert!(parse(&["mefisto", "--n-inducing", "100"]).is_err());
    let Ok(crate::Commands::Mefisto(m)) = parse(&["mefisto"]) else {
        panic!("mefisto")
    };
    assert_eq!(
        (m.fit.n_factors, m.fit.n_genes, m.fit.refine_k),
        (4, 2000, 0)
    );
    for cmd in ["spatialpca", "mefisto"] {
        assert!(parse(&[cmd, "-d", "5"]).is_err(), "{cmd} -d");
        let pb = format!("{cmd}-pb");
        assert!(parse(&[&pb, "-d", "5"]).is_ok(), "{pb} -d");
    }
    let cli = crate::built_cli();
    let visible = |cmd: &str| -> Vec<String> {
        cli.find_subcommand(cmd)
            .unwrap()
            .get_arguments()
            .filter(|a| !a.is_hide_set() && !a.is_positional())
            .filter_map(|a| a.get_long().map(str::to_string))
            .collect()
    };
    for cmd in ["spatialpca", "mefisto"] {
        let v = visible(cmd);
        for want in ["n-factors", "n-genes", "n-clusters", "ignore-batch", "seed"] {
            assert!(v.iter().any(|f| f == want), "{cmd}: no --{want}");
        }
        for hidden in ["knn", "cg-tol", "slq-probes", "refine-k", "resolution"] {
            assert!(!v.iter().any(|f| f == hidden), "{cmd}: --{hidden} shown");
        }
    }
    assert!(!visible("mefisto").iter().any(|f| f == "n-inducing"));
}

/// A baseline's `--ignore-batch` help says what it does there: no batch
/// effect is estimated or removed and the counts are used as stored; it
/// does not speak of predict or impute, which cage's does.
#[test]
fn a_baseline_s_ignore_batch_help_is_its_own() {
    let cli = crate::built_cli();
    let help = |cmd: &str| {
        cli.find_subcommand(cmd)
            .unwrap()
            .get_arguments()
            .find(|a| a.get_id() == "ignore_batch")
            .and_then(|a| a.get_long_help())
            .unwrap()
            .to_string()
    };
    for cmd in BASELINES {
        let h = help(cmd);
        assert!(h.contains("used as stored"), "{cmd}: {h}");
        assert!(!h.contains("predict"), "{cmd}: {h}");
    }
    assert!(help("cage").contains("predict"));
}
