//! `--feature-modules` on the command line, and the plan a run's flags
//! resolve to before any data is read.

use crate::cell_activity_graph_embedding::args::{CellActivityGraphEmbeddingArgs, ModulePlan};
use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    cage: CellActivityGraphEmbeddingArgs,
}

fn parse(extra: &[&str]) -> Result<CellActivityGraphEmbeddingArgs, clap::Error> {
    Cli::try_parse_from(
        ["x", "d.zarr", "-o", "out"]
            .into_iter()
            .chain(extra.iter().copied()),
    )
    .map(|c| c.cage)
}

fn plan(extra: &[&str]) -> anyhow::Result<ModulePlan> {
    parse(extra).unwrap().module_plan()
}

/// The flag takes `auto`, a count and 0;
/// unset it is not given (the default is decided against the other flags).
/// One module, or a word that is neither, is refused as it is parsed.
#[test]
fn the_flag_parses_auto_a_count_and_zero() {
    let given = |extra: &[&str]| parse(extra).unwrap().feature_modules;
    assert_eq!(given(&[]), None);
    let flag = "--feature-modules";
    assert_eq!(given(&[flag, "auto"]), Some(ModulePlan::Auto));
    assert_eq!(given(&[flag, "AUTO"]), Some(ModulePlan::Auto));
    assert_eq!(given(&[flag, "20"]), Some(ModulePlan::Fixed(20)));
    assert_eq!(given(&[flag, "0"]), Some(ModulePlan::Off));
    for bad in ["1", "many", "-3", "2.5"] {
        assert!(
            parse(&["--feature-modules", bad]).is_err(),
            "{bad} accepted"
        );
    }
}

/// Modules are on by default with their count left to the data; `auto`
/// says the same, a count fixes it, and 0 turns the layer off.
#[test]
fn the_default_plan_is_auto_and_zero_is_off() {
    assert_eq!(plan(&[]).unwrap(), ModulePlan::Auto);
    assert_eq!(
        plan(&["--feature-modules", "auto"]).unwrap(),
        ModulePlan::Auto
    );
    assert_eq!(
        plan(&["--feature-modules", "20"]).unwrap(),
        ModulePlan::Fixed(20)
    );
    assert_eq!(plan(&["--feature-modules", "0"]).unwrap(), ModulePlan::Off);
}

/// The default yields: a run given `--feature-inclusion moran` or a
/// `--feature-embedding` trains without modules, with no error, and so
/// does one that also says `--feature-modules 0`.
#[test]
fn the_default_yields_to_moran_and_to_a_given_dictionary() {
    let moran = ["--feature-inclusion", "moran"];
    let dictionary = ["--feature-embedding", "dict.parquet"];
    for other in [&moran[..], &dictionary[..]] {
        assert_eq!(plan(other).unwrap(), ModulePlan::Off, "{other:?}");
        let off = [other, &["--feature-modules", "0"][..]].concat();
        assert_eq!(plan(&off).unwrap(), ModulePlan::Off, "{off:?}");
    }
    // `all` is the inclusion default, so it turns nothing off
    assert_eq!(
        plan(&["--feature-inclusion", "all"]).unwrap(),
        ModulePlan::Auto
    );
}

/// Modules asked for by the flag (`auto` or a count) beside moran or a
/// dictionary are refused, naming both flags.
#[test]
fn modules_given_beside_moran_or_a_dictionary_are_refused() {
    for (other, value) in [
        ("--feature-inclusion", "moran"),
        ("--feature-embedding", "dict.parquet"),
    ] {
        for modules in ["auto", "20"] {
            let e = plan(&[other, value, "--feature-modules", modules])
                .unwrap_err()
                .to_string();
            assert!(
                e.contains(other) && e.contains("--feature-modules"),
                "{other} with {modules}: {e}"
            );
        }
    }
}
