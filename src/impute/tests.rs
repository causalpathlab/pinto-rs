//! What the profile projection must guarantee: a cell whose counts sit on
//! one community's features lands on that community, feature matching goes
//! through the canonicalizer (not raw row names), and cells the model
//! cannot see keep a zero row instead of a made-up assignment.

use super::*;
use crate::test_support::{boxed, numbered, sparse_vec};

/// `{tag}.zarr` in `dir`, its columns named `cell_0`, `cell_1`, ….
fn make_data(
    dir: &tempfile::TempDir,
    tag: &str,
    feature_names: &[&str],
    triplets: &[(u64, u64, f32)],
    n_cells: usize,
) -> SparseIoVec {
    let cells = numbered("cell_", n_cells);
    sparse_vec(dir.path(), tag, &boxed(feature_names), &cells, triplets)
}

/// Two communities with disjoint expression: C0 lives on {a, b}, C1 on
/// {c, d}.
fn disjoint_profiles() -> (Mat, Vec<Box<str>>) {
    let profiles = Mat::from_row_slice(
        4,
        2,
        &[
            5.0, 0.0, // a
            5.0, 0.0, // b
            0.0, 4.0, // c
            0.0, 6.0, // d
        ],
    );
    let features: Vec<Box<str>> = vec!["a".into(), "b".into(), "c".into(), "d".into()];
    (profiles, features)
}

#[test]
fn pure_cells_land_on_their_community() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (profiles, features) = disjoint_profiles();
    // cell_0 expresses only C0 features, cell_1 only C1 features, cell_2 both.
    let data = make_data(
        &dir,
        "q",
        &["a", "b", "c", "d"],
        &[
            (0, 0, 5.0),
            (1, 0, 3.0),
            (2, 1, 4.0),
            (3, 1, 2.0),
            (0, 2, 3.0),
            (2, 2, 3.0),
        ],
        3,
    );
    let prop = project_profile_propensity(
        &data,
        &profiles,
        &features,
        &data_beans::aux::feature_names::FeatureNameKind::Exact,
        100,
        None,
        "test",
    )?;
    assert!(prop[(0, 0)] > 0.99, "pure C0 cell: {}", prop[(0, 0)]);
    assert!(prop[(1, 1)] > 0.99, "pure C1 cell: {}", prop[(1, 1)]);
    assert!(
        prop[(2, 0)] > 0.2 && prop[(2, 1)] > 0.2,
        "mixed cell stays mixed: {:?}",
        (prop[(2, 0)], prop[(2, 1)])
    );
    let row_sum: f32 = prop.row(2).iter().sum();
    assert!((row_sum - 1.0).abs() < 1e-4, "rows are proportions");
    Ok(())
}

#[test]
fn feature_matching_goes_through_the_canonicalizer() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (profiles, features) = disjoint_profiles();
    // Query rows carry `<ensembl>_<symbol>` names; the Feature kind resolves
    // them onto the model's bare symbols.
    let data = make_data(
        &dir,
        "ens",
        &["ENSG01_a", "ENSG02_b", "ENSG03_c", "ENSG04_d"],
        &[(0, 0, 5.0), (1, 0, 3.0)],
        1,
    );
    let prop = project_profile_propensity(
        &data,
        &profiles,
        &features,
        &data_beans::aux::feature_names::FeatureNameKind::Gene { delim: '_' },
        100,
        None,
        "test",
    )?;
    assert!(prop[(0, 0)] > 0.99, "canonicalized match: {}", prop[(0, 0)]);
    Ok(())
}

#[test]
fn locus_rows_on_sex_chromosomes_match_the_model() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (profiles, _) = disjoint_profiles();
    // Model keys as training writes them: `chr` dropped, case kept.
    let features: Vec<Box<str>> = ["X:0-1000", "X:2000-3000", "1:0-1000", "1:2000-3000"]
        .map(Into::into)
        .to_vec();
    // Two files read the way impute reads them, so the loader aligns
    // their rows as loci.
    let peaks = [
        "chrX:0-1000",
        "chrX:2000-3000",
        "chr1:0-1000",
        "chr1:2000-3000",
    ];
    let mut files: Vec<Box<str>> = Vec::new();
    for tag in ["a", "b"] {
        make_data(&dir, tag, &peaks, &[(0, 0, 5.0), (1, 0, 3.0)], 1);
        files.push(
            dir.path()
                .join(format!("{tag}.zarr"))
                .to_string_lossy()
                .into(),
        );
    }
    let data = open_backends("test", &files, false)?;
    let prop = project_profile_propensity(
        &data,
        &profiles,
        &features,
        &data_beans::aux::feature_names::FeatureNameKind::Locus {
            merge_overlapping: false,
        },
        100,
        None,
        "test",
    )?;
    for c in 0..2 {
        assert!(prop[(c, 0)] > 0.99, "chrX peaks matched: {}", prop[(c, 0)]);
    }
    Ok(())
}

#[test]
fn a_cell_off_the_model_axis_keeps_a_zero_row() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (profiles, features) = disjoint_profiles();
    // cell_0 expresses only a feature the model never saw; cell_1 is normal.
    let data = make_data(&dir, "off", &["zzz", "a"], &[(0, 0, 9.0), (1, 1, 4.0)], 2);
    let prop = project_profile_propensity(
        &data,
        &profiles,
        &features,
        &data_beans::aux::feature_names::FeatureNameKind::Exact,
        100,
        None,
        "test",
    )?;
    assert!(prop.row(0).iter().all(|&x| x == 0.0));
    assert!(prop[(1, 0)] > 0.99);
    Ok(())
}

#[test]
fn no_shared_feature_is_an_error_not_a_uniform_guess() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (profiles, features) = disjoint_profiles();
    let data = make_data(&dir, "none", &["x", "y"], &[(0, 0, 1.0)], 1);
    assert!(project_profile_propensity(
        &data,
        &profiles,
        &features,
        &data_beans::aux::feature_names::FeatureNameKind::Exact,
        10,
        None,
        "test",
    )
    .is_err());
    Ok(())
}

fn parse_impute(argv: &[&str]) -> anyhow::Result<ImputeArgs> {
    use clap::Parser;
    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        args: ImputeArgs,
    }
    Ok(Wrap::try_parse_from(argv)?.args)
}

#[test]
fn explicit_reference_data_bypasses_the_manifest() -> anyhow::Result<()> {
    let args = parse_impute(&[
        "impute",
        "q",
        "--model",
        "no/such/model",
        "-o",
        "out",
        "--reference-data",
        "ref.zarr",
    ])?;
    let files = resolve_reference_data(&args, None)?;
    assert_eq!(files, vec![Box::<str>::from("ref.zarr")]);
    Ok(())
}

#[test]
fn a_missing_manifest_names_the_flag_as_the_way_out() -> anyhow::Result<()> {
    let args = parse_impute(&["impute", "q.zarr", "--model", "no/such/model", "-o", "out"])?;
    let err = resolve_reference_data(&args, None).unwrap_err();
    assert!(err.to_string().contains("--reference-data"), "{err}");
    Ok(())
}

////////////////////////////////////////////
// Reference propensity onto the data axis //
////////////////////////////////////////////

#[test]
fn a_reference_cell_the_run_never_placed_gets_a_zero_row() -> anyhow::Result<()> {
    // The run kept b and d (in that order); the files hold a, b, c, d.
    let prop = Mat::from_row_slice(2, 2, &[0.7, 0.3, 0.1, 0.9]);
    let aligned = align_to_columns(
        "p",
        prop,
        &boxed(&["b", "d"]),
        &boxed(&["a", "b", "c", "d"]),
    )?;
    assert_eq!(aligned.nrows(), 4);
    assert_eq!(
        aligned.row(0).iter().copied().collect::<Vec<_>>(),
        vec![0.0, 0.0]
    );
    assert_eq!(
        aligned.row(1).iter().copied().collect::<Vec<_>>(),
        vec![0.7, 0.3]
    );
    assert_eq!(
        aligned.row(2).iter().copied().collect::<Vec<_>>(),
        vec![0.0, 0.0]
    );
    assert_eq!(
        aligned.row(3).iter().copied().collect::<Vec<_>>(),
        vec![0.1, 0.9]
    );
    Ok(())
}

#[test]
fn matching_axes_pass_through_and_foreign_files_are_refused() {
    let prop = Mat::from_row_slice(2, 1, &[1.0, 2.0]);
    let same =
        align_to_columns("p", prop.clone(), &boxed(&["a", "b"]), &boxed(&["a", "b"])).unwrap();
    assert_eq!(same, prop);
    let err =
        align_to_columns("p", prop.clone(), &boxed(&["a", "z"]), &boxed(&["a", "b"])).unwrap_err();
    assert!(err.to_string().contains("1 of its 2 cells"), "{err}");
    let err =
        align_to_columns("p", prop, &boxed(&["a", "b"]), &boxed(&["a", "a", "b"])).unwrap_err();
    assert!(err.to_string().contains("not unique"), "{err}");
}

/// No reference neighbours is refused as an argument, before any data is
/// read.
#[test]
fn zero_reference_neighbours_is_an_argument_error() {
    let e = parse_impute(&[
        "impute",
        "q",
        "--model",
        "m",
        "-o",
        "o",
        "--impute-knn",
        "0",
    ])
    .unwrap_err();
    let kind = e.downcast_ref::<clap::Error>().map(clap::Error::kind);
    assert_eq!(kind, Some(clap::error::ErrorKind::ValueValidation), "{e}");
}
