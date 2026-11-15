//! The baselines' command lines stay small: a short `--help` each.

/// The baselines whose `-h` is kept short.
pub(crate) const BASELINES: &[&str] = &["stagate", "spatialde2"];

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
        assert!(help.contains("--reference"), "{cmd}");
        assert!(help.contains("--ignore-batch"), "{cmd}");
    }
}
