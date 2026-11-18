//! `--feature-modules` on the fits that read counts as data, and the
//! two-stage path on modules.

use super::*;
use crate::util::metadata::PintoMetadata;
use clap::Parser;
use std::path::Path;

#[test]
fn the_default_yields_to_offset_and_to_a_feature_network() {
    assert_eq!(plan(None, false, false, false).unwrap().0, ModulePlan::Auto);
    assert_eq!(
        plan(None, false, true, false).unwrap(),
        (ModulePlan::Off, "offset")
    );
    assert_eq!(
        plan(None, false, false, true).unwrap(),
        (ModulePlan::Off, "feature-network")
    );
    assert_eq!(
        plan(Some(ModulePlan::Off), false, true, true).unwrap().0,
        ModulePlan::Off
    );
    assert_eq!(
        plan(Some(ModulePlan::Fixed(3)), false, false, false)
            .unwrap()
            .0,
        ModulePlan::Fixed(3)
    );
    for (offset, network) in [(true, false), (false, true)] {
        for asked in [ModulePlan::Auto, ModulePlan::Fixed(3)] {
            assert!(plan(Some(asked), false, offset, network).is_err());
        }
    }
}

/// The fixture's data and coordinates, and a fit's argv on them.
struct Fixture {
    _dir: tempfile::TempDir,
    d: String,
    data: String,
    coords: String,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = crate::test_support::two_sections(dir.path(), 8);
    Fixture {
        d: dir.path().to_str().unwrap().to_string(),
        _dir: dir,
        data,
        coords,
    }
}

impl Fixture {
    fn argv(&self, method: &str, data: &str, out: &str, extra: &[&str]) -> Vec<String> {
        let mut argv: Vec<String> = [
            "pinto",
            "fit",
            method,
            data,
            "-c",
            &self.coords,
            "--coord-column-names",
            "x,y",
            "-o",
            out,
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let own: &[&str] = match method {
            "lc" => &[
                "--n-communities",
                "2",
                "--num-gibbs",
                "2",
                "--num-greedy",
                "1",
                "--num-levels",
                "2",
            ],
            "dsvd" => &["-t", "3", "--num-levels", "2"],
            "spatialpca-pb" => &["-d", "16", "--n-clusters", "2"],
            "spatialpca" => &["--n-clusters", "2"],
            "stagate" | "stagate-pb" => &["--n-epochs", "5", "--n-clusters", "2"],
            "mefisto" | "mefisto-pb" => &["--max-iter", "5", "--n-clusters", "2"],
            _ => &["--n-clusters", "2"],
        };
        argv.extend(own.iter().chain(extra).map(|s| s.to_string()));
        argv
    }
}

/// A fit as `pinto fit` runs it: its modules prepared, the fit, the
/// manifest recorded.
fn fit(argv: &[String]) -> PintoMetadata {
    use crate::{Commands, Fit};
    let mut cli = crate::Cli::try_parse_from(argv).unwrap_or_else(|e| panic!("{e}"));
    let modules = crate::prepare_feature_modules(&mut cli.commands).unwrap();
    let Commands::Fit(f) = &cli.commands else {
        unreachable!()
    };
    let sp = crate::baselines::spatial::spatialpca::run_pb;
    match f {
        Fit::LinkCommunity(a) => crate::link_community::fit::fit_srt_link_community(a),
        Fit::DeltaSvd(a) => crate::svd::fit::fit_srt_delta_svd(a),
        Fit::SpatialpcaPb(a) => sp(a),
        Fit::Spatialpca(a) => crate::baselines::spatial::spatialpca::run(a),
        Fit::Spatialde2(a) => crate::baselines::spatial::spatialde2::run(a),
        Fit::Stagate(a) => crate::baselines::spatial::stagate::run(a),
        Fit::MefistoPb(a) => crate::baselines::spatial::mefisto::run_pb(a),
        _ => unreachable!(),
    }
    .unwrap();
    let (out, info) = modules.expect("a fit with --feature-modules");
    record(&out, &info).unwrap();
    PintoMetadata::read(Path::new(&format!("{out}.pinto.json"))).unwrap()
}

#[test]
fn the_shared_builder_sums_the_counts_within_modules() {
    let f = fixture();
    let out = format!("{}/lc", f.d);
    let meta = fit(&f.argv("lc", &f.data, &out, &["--feature-modules", "2"]));
    let info = meta.feature_modules.expect("recorded");
    assert_eq!(info.setting, "2");
    assert_eq!(info.n_modules, Some(2));
    assert!(info.module_level());
    let files: Vec<Box<str>> = meta
        .data_files
        .unwrap()
        .iter()
        .map(|s| s.as_str().into())
        .collect();
    assert_eq!(files.len(), 2);
    assert!(
        files.iter().all(|p| p.contains(".feature_modules/")),
        "{files:?}"
    );
    let raw: Vec<Box<str>> = f.data.split(',').map(Box::from).collect();
    let kind = data_beans::aux::feature_names::FeatureNameKind::Exact;
    let (genes, modules) = (
        open_fit_data(&raw, &kind).unwrap(),
        open_fit_data(&files, &kind).unwrap(),
    );
    assert_eq!(
        modules.row_names().unwrap(),
        vec!["M0".into(), "M1".into()] as Vec<Box<str>>
    );
    // the cells QC kept, each with its counts in the modules
    let n = modules.num_columns();
    assert!(n > 0 && n <= genes.num_columns());
    assert_eq!(
        info.qc_dropped,
        Some(genes.num_columns() - n),
        "every cell is in the modules or dropped by QC"
    );
    let total = |v: &data_beans::sparse_io_vector::SparseIoVec| -> f32 {
        v.read_columns_csc(0..v.num_columns())
            .unwrap()
            .values()
            .iter()
            .sum()
    };
    if n == genes.num_columns() {
        assert!((total(&genes) - total(&modules)).abs() < 1e-3);
    }
    assert!(Path::new(info.membership.as_deref().unwrap()).exists());
}

#[test]
fn every_method_family_records_its_module_setting() {
    let f = fixture();
    for (method, extra, setting) in [
        ("lc", &[][..], "auto"),
        ("dsvd", &["--feature-modules", "2"][..], "2"),
        ("dsvd", &["--feature-modules", "0"][..], "off"),
        ("stagate", &["--feature-modules", "2"][..], "2"),
        ("spatialpca", &["--feature-modules", "2"][..], "2"),
        ("spatialpca-pb", &[][..], "auto"),
        ("spatialde2", &["--feature-modules", "0"][..], "off"),
    ] {
        let out = format!("{}/{method}-{setting}", f.d);
        let info = fit(&f.argv(method, &f.data, &out, extra))
            .feature_modules
            .unwrap_or_else(|| panic!("{method}: nothing recorded"));
        assert_eq!(info.setting, setting, "{method}");
        if setting == "off" {
            assert!(!info.module_level(), "{method}");
        }
        if setting == "2" {
            assert_eq!(info.n_modules, Some(2), "{method}");
        }
    }
}

#[test]
fn lc_with_a_feature_network_fits_the_features_and_refuses_both() {
    let f = fixture();
    let network = format!("{}/net.tsv", f.d);
    std::fs::write(&network, "GENE0\tGENE1\nGENE1\tGENE2\nGENE0\tGENE2\n").unwrap();
    let out = format!("{}/both", f.d);
    let both = f.argv(
        "lc",
        &f.data,
        &out,
        &["--feature-network", &network, "--feature-modules", "2"],
    );
    let mut cli = crate::Cli::try_parse_from(&both).unwrap();
    assert!(crate::prepare_feature_modules(&mut cli.commands).is_err());
    let alone = f.argv("lc", &f.data, &out, &["--feature-network", &network]);
    let mut cli = crate::Cli::try_parse_from(&alone).unwrap();
    let (_, info) = crate::prepare_feature_modules(&mut cli.commands)
        .unwrap()
        .unwrap();
    assert_eq!(info.setting, "feature-network");
    assert!(!info.module_level());
}

/// Stage 1 on modules, its residual (divide) at module level, and stage 2
/// on the same modules with `--offset`, for `method`.
fn two_stage_on_modules(method: &str) {
    use crate::residual::{run_residual, Operation, ResidualArgs};
    let f = fixture();
    let stage1 = format!("{}/{method}1", f.d);
    let meta = fit(&f.argv(method, &f.data, &stage1, &["--feature-modules", "2"]));
    let files = meta.data_files.clone().unwrap().join(",");
    assert!(meta.feature_modules.unwrap().module_level(), "{method}");

    let resid = format!("{}/{method}-resid", f.d);
    run_residual(&ResidualArgs::try_parse_from(["x", "--from", &stage1, "-o", &resid]).unwrap())
        .unwrap();
    let marker: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{resid}/residual.json")).unwrap())
            .unwrap();
    assert_eq!(
        marker["operation"],
        serde_json::to_value(Operation::Divide).unwrap()
    );
    assert_eq!(marker["feature_level"], "module", "{method}: {marker}");
    assert_eq!(marker["n_modules"], 2, "{method}");

    let stage2 = format!("{}/{method}2", f.d);
    let meta = fit(&f.argv(
        method,
        &files,
        &stage2,
        &["--offset", &resid, "--ignore-batch"],
    ));
    let info = meta.feature_modules.unwrap();
    assert_eq!(info.setting, "offset", "{method}");
    assert!(info.module_level(), "{method}: stage 2 fits the modules");
    assert_eq!(info.n_modules, Some(2), "{method}");
    assert!(meta.offset.is_some(), "{method}");
}

// ResidualArgs parses on its own
#[derive(Parser)]
struct ResidualCli {
    #[command(flatten)]
    args: crate::residual::ResidualArgs,
}

trait ParseResidual: Sized {
    fn try_parse_from<const N: usize>(argv: [&str; N]) -> Result<Self, clap::Error>;
}

impl ParseResidual for crate::residual::ResidualArgs {
    fn try_parse_from<const N: usize>(argv: [&str; N]) -> Result<Self, clap::Error> {
        Ok(ResidualCli::try_parse_from(argv)?.args)
    }
}

#[test]
fn lc_two_stage_on_modules() {
    two_stage_on_modules("lc");
}

#[test]
fn spatialpca_pb_two_stage_on_modules() {
    two_stage_on_modules("spatialpca-pb");
}

#[test]
fn mefisto_pb_two_stage_on_modules() {
    two_stage_on_modules("mefisto-pb");
}

/// spatialpca(-pb) on modules models every module: no SPARK-X screen, and
/// no refusal with fewer than two spatially variable genes.
#[test]
fn spatialpca_on_modules_skips_spark_x() {
    let f = fixture();
    for method in ["spatialpca", "spatialpca-pb"] {
        let out = format!("{}/{method}", f.d);
        let meta = fit(&f.argv(method, &f.data, &out, &["--feature-modules", "2"]));
        assert!(
            meta.feature_modules.as_ref().unwrap().module_level(),
            "{method}"
        );
        let text = std::fs::read_to_string(format!("{out}.baseline.json")).unwrap();
        assert!(
            text.contains("none: every feature module"),
            "{method}: {text}"
        );
        assert!(!text.contains("SPARK-X"), "{method}: {text}");
    }
}

/// `--feature-inclusion moran` alone replaces the default modules on a fit
/// that is not cage; beside an explicit `--feature-modules` it is refused.
#[test]
fn moran_alone_replaces_modules_and_beside_them_is_refused() {
    let f = fixture();
    let out = format!("{}/moran-both", f.d);
    let both = f.argv(
        "dsvd",
        &f.data,
        &out,
        &["--feature-inclusion", "moran", "--feature-modules", "auto"],
    );
    let mut cli = crate::Cli::try_parse_from(&both).unwrap();
    let err = crate::prepare_feature_modules(&mut cli.commands).unwrap_err();
    assert!(err.to_string().contains("alternatives"), "{err}");

    let out = format!("{}/moran", f.d);
    let argv = f.argv("dsvd", &f.data, &out, &["--feature-inclusion", "moran"]);
    let mut cli = crate::Cli::try_parse_from(&argv).unwrap();
    match crate::prepare_feature_modules(&mut cli.commands) {
        Ok(Some((_, info))) => {
            assert_eq!(info.setting, "moran");
            assert!(!info.module_level());
            assert!(Path::new(&format!("{out}.feature_inclusion.parquet")).exists());
        }
        // the fixture may carry too little spatial signal: refused, by name
        Err(e) => {
            eprintln!("moran refused: {e}");
            assert!(e.to_string().contains("posterior probability"), "{e}")
        }
        Ok(None) => panic!("nothing prepared"),
    }
}
