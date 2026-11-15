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

/// [`project_profile_propensity`] of `data` through `profiles` (rows
/// `features`, matched as `kind`), undivided.
fn project(
    data: &SparseIoVec,
    profiles: &Mat,
    features: &[Box<str>],
    kind: &data_beans::aux::feature_names::FeatureNameKind,
    iters: usize,
) -> anyhow::Result<Mat> {
    let p = ProfileProjection {
        profiles,
        model_features: features,
        feature_kind: kind,
        iters,
        block_size: None,
    };
    project_profile_propensity(data, &p, None, "test")
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
    let prop = project(
        &data,
        &profiles,
        &features,
        &data_beans::aux::feature_names::FeatureNameKind::Exact,
        100,
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
    let prop = project(
        &data,
        &profiles,
        &features,
        &data_beans::aux::feature_names::FeatureNameKind::Gene { delim: '_' },
        100,
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
    let prop = project(
        &data,
        &profiles,
        &features,
        &data_beans::aux::feature_names::FeatureNameKind::Locus {
            merge_overlapping: false,
        },
        100,
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
    let prop = project(
        &data,
        &profiles,
        &features,
        &data_beans::aux::feature_names::FeatureNameKind::Exact,
        100,
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
    assert!(project(
        &data,
        &profiles,
        &features,
        &data_beans::aux::feature_names::FeatureNameKind::Exact,
        10,
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

/// A manifest's recorded reference files resolve beside it, as predict
/// resolves the model's, whatever the working directory; one in neither
/// place is an error naming the flag.
#[test]
fn recorded_reference_files_resolve_beside_the_manifest() -> anyhow::Result<()> {
    use crate::util::metadata::{manifest_path, PintoMetadata, Run};
    let dir = tempfile::tempdir()?;
    let model = dir.path().join("run").to_string_lossy().to_string();
    std::fs::write(dir.path().join("a.zarr"), b"")?;
    let mut meta = PintoMetadata {
        data_files: Some(vec!["a.zarr".into()]),
        ..Default::default()
    };
    meta.write(std::path::Path::new(&manifest_path(&model)))?;
    let args = parse_impute(&["impute", "q.zarr", "--model", &model, "-o", "out"])?;
    let files = resolve_reference_data(&args, Some(&Run::open(&model)?))?;
    assert_eq!(
        std::path::Path::new(files[0].as_ref()),
        dir.path().join("a.zarr")
    );

    meta.data_files = Some(vec!["gone.zarr".into()]);
    meta.write(std::path::Path::new(&manifest_path(&model)))?;
    let err = resolve_reference_data(&args, Some(&Run::open(&model)?)).unwrap_err();
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

mod retrieval {
    use super::super::retrieval::{retrieval_impute, RefDivide};
    use crate::test_support::{boxed, numbered, sparse_vec};
    use crate::util::common::*;
    use data_beans::alg::retrieval_impute::RetrievalImputeConfig;

    /// A reference of four cells over three features, on a propensity that
    /// sets every cell apart, and queries sitting on its cells.
    fn reference(dir: &tempfile::TempDir) -> (SparseIoVec, Mat) {
        let triplets: Vec<(u64, u64, f32)> = (0..4u64)
            .flat_map(|c| (0..3u64).map(move |g| (g, c, (1 + c * 3 + g) as f32)))
            .collect();
        let data = sparse_vec(
            dir.path(),
            "ref",
            &boxed(&["GENE0", "GENE1", "GENE2"]),
            &numbered("cell_", 4),
            &triplets,
        );
        let prop = Mat::from_row_slice(4, 2, &[1.0, 0.0, 0.0, 1.0, 0.7, 0.7, 0.2, 0.9]);
        (data, prop)
    }

    fn cfg(knn: usize) -> RetrievalImputeConfig {
        RetrievalImputeConfig {
            knn,
            temperature: 0.3,
            chunk: 64,
        }
    }

    /// Without a divide it is the upstream retrieval, number for number.
    #[test]
    fn undivided_it_is_the_upstream_retrieval() {
        let dir = tempfile::tempdir().unwrap();
        let (data, prop) = reference(&dir);
        let query = Mat::from_row_slice(3, 2, &[0.9, 0.1, 0.0, 0.0, 0.5, 0.6]);
        let ours = retrieval_impute(&query, &prop, &data, &cfg(3), None).unwrap();
        let upstream =
            data_beans::alg::retrieval_impute::retrieval_impute(&query, &prop, &data, &cfg(3))
                .unwrap();
        assert!((ours - upstream).abs().max() < 1e-6);
    }

    /// Each reference cell's counts are divided by its batch's δ before they
    /// are pooled; a cell with no batch is left as is.
    #[test]
    fn reference_counts_are_divided_by_their_batch_delta() {
        let dir = tempfile::tempdir().unwrap();
        let (data, prop) = reference(&dir);
        let delta = Mat::from_row_slice(3, 2, &[0.5, 2.0, 1.0, 1.0, 4.0, 0.25]);
        let divide = RefDivide {
            delta: delta.clone(),
            batch_of_cell: vec![Some(0), Some(1), Some(1), None],
        };
        // each query is one reference cell, so it retrieves that cell alone
        let imputed = retrieval_impute(&prop, &prop, &data, &cfg(1), Some(&divide)).unwrap();
        for c in 0..4usize {
            for g in 0..3usize {
                let raw = (1 + c * 3 + g) as f32;
                let want = match divide.batch_of_cell[c] {
                    Some(b) => raw / delta[(g, b)],
                    None => raw,
                };
                assert!(
                    (imputed[(c, g)] - want).abs() < 1e-4,
                    "cell {c}, feature {g}"
                );
            }
        }
    }
}

//////////////////////////////////////////
// lc / dsvd: the dictionary's scale    //
//////////////////////////////////////////

/// An lc model of [`crate::test_support::two_sections`] (two batches, the
/// second folded on every third feature) at `dir/run`, with `extra` fit
/// flags: its prefix, data and coordinate files.
fn two_batch_lc(dir: &std::path::Path, extra: &[&str]) -> (String, String, String) {
    use clap::Parser;
    let side = 8;
    let (data, coords) = crate::test_support::two_sections(dir, side);
    let prefix = dir.join("run").to_str().unwrap().to_string();
    let mut argv = vec![
        "lc",
        data.as_str(),
        "-c",
        coords.as_str(),
        "--coord-column-names",
        "x,y",
        "-o",
        prefix.as_str(),
        "--n-communities",
        "4",
        "--num-gibbs",
        "2",
        "--num-greedy",
        "1",
        "--num-levels",
        "2",
    ];
    argv.extend_from_slice(extra);
    crate::link_community::fit::fit_srt_link_community(
        &crate::link_community::args::SrtLinkCommunityArgs::try_parse_from(argv).unwrap(),
    )
    .unwrap();
    (prefix, data, coords)
}

/// The manifest at `prefix`, its batch-adjusted flag set to `adjusted`.
fn set_adjusted(prefix: &str, adjusted: bool) {
    use crate::util::metadata::{manifest_path, PintoMetadata};
    let path = manifest_path(prefix);
    let mut meta = PintoMetadata::read(std::path::Path::new(&path)).unwrap();
    meta.outputs.feature_community_batch_adjusted = adjusted;
    meta.write(std::path::Path::new(&path)).unwrap();
}

/// How far `prop`'s rows (two files of `side²` cells each, a cell's type
/// its half of the grid) split by batch: over the two types, the L1 gap
/// between the batches' mean propensities.
fn batch_split(prop: &Mat, side: usize) -> f32 {
    let n = side * side;
    let mut gap = 0.0;
    for left in [true, false] {
        let mean = |file: usize| {
            let rows: Vec<usize> = (0..n)
                .filter(|c| (c % side < side / 2) == left)
                .map(|c| file * n + c)
                .collect();
            let mut m = vec![0.0f32; prop.ncols()];
            for &r in &rows {
                for (k, x) in m.iter_mut().enumerate() {
                    *x += prop[(r, k)] / rows.len() as f32;
                }
            }
            m
        };
        let (a, b) = (mean(0), mean(1));
        gap += a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>();
    }
    gap
}

/// [`profile_propensities`] of the training data against itself.
fn sides(prefix: &str, data: &str, extra: &[&str]) -> anyhow::Result<ProfileSides> {
    let out = format!("{prefix}.impute");
    let mut argv = vec!["impute", data, "--model", prefix, "-o", out.as_str()];
    argv.extend_from_slice(extra);
    let args = parse_impute(&argv)?;
    let run = crate::util::metadata::Run::open(prefix)?;
    let files = resolve_reference_data(&args, Some(&run))?;
    mkdir_parent(&out)?;
    profile_propensities(&args, Some(&run), &files)
}

/// A multi-batch lc model's profiles are batch-adjusted, so impute divides
/// the reference by the run's δ (and the query, the same cells, by the
/// model's): a type's reference propensity no longer splits by batch, as
/// it does matched raw. A model without the flag (an older fit) is matched
/// raw on both sides, as before.
#[test]
fn an_adjusted_lc_model_puts_both_sides_on_its_profiles_scale() {
    let dir = tempfile::tempdir().unwrap();
    let (prefix, data, _) = two_batch_lc(dir.path(), &[]);
    let (q_adj, _, r_adj, _, divide) = sides(&prefix, &data, &[]).unwrap();
    assert!(divide.is_some(), "the reference is divided");

    set_adjusted(&prefix, false);
    let (q_raw, _, r_raw, ref_data, divide) = sides(&prefix, &data, &[]).unwrap();
    assert!(divide.is_none(), "an older model's reference is not");
    let (adjusted, raw) = (batch_split(&r_adj, 8), batch_split(&r_raw, 8));
    assert!(
        raw > 0.2 && adjusted < 0.5 * raw,
        "reference split by batch: adjusted {adjusted}, raw {raw}"
    );
    assert!(batch_split(&q_adj, 8) < 0.5 * batch_split(&q_raw, 8));

    // flag absent: the undivided projection, through the same profiles
    let (profiles, features) = crate::util::parquet_io::read_feature_community(
        std::path::Path::new(&format!("{prefix}.feature_community.parquet")),
    )
    .unwrap();
    let undivided = project(
        &ref_data,
        &profiles,
        &features,
        &data_beans::aux::feature_names::FeatureNameKind::Exact,
        100,
    )
    .unwrap();
    assert_eq!(r_raw, undivided);
    assert_eq!(q_raw, undivided);
}

/// `--ignore-batch` would match undivided cells to batch-adjusted
/// profiles, so impute refuses it for such a model; an older one takes it.
#[test]
fn ignore_batch_is_refused_for_an_adjusted_model() {
    let dir = tempfile::tempdir().unwrap();
    let (prefix, data, _) = two_batch_lc(dir.path(), &[]);
    let err = sides(&prefix, &data, &["--ignore-batch"])
        .err()
        .expect("refused");
    assert!(err.to_string().contains("--ignore-batch"), "{err}");
    // impute takes no held-out flag, so none is offered
    assert!(!err.to_string().contains("--holdout"), "{err}");
    set_adjusted(&prefix, false);
    assert!(sides(&prefix, &data, &["--ignore-batch"]).is_ok());
}

/// The fit's own arguments run impute, `--auto-batch` among them: its
/// components are never the model's batches, and impute builds no graph
/// to find them, so the query is one batch, new to the model, which
/// nothing estimates: undivided, as `pinto predict` leaves a cage query
/// of one component. The reference is divided by the run's δ.
#[test]
fn impute_takes_the_fits_auto_batch() {
    let dir = tempfile::tempdir().unwrap();
    let auto = ["--auto-batch"];
    let (prefix, data, coords) = two_batch_lc(dir.path(), &auto);
    let path = crate::util::metadata::manifest_path(&prefix);
    let meta = crate::util::metadata::PintoMetadata::read(std::path::Path::new(&path)).unwrap();
    assert!(
        meta.outputs.feature_community_batch_adjusted,
        "two components, a δ"
    );
    let fit_args = [
        "-c",
        coords.as_str(),
        "--coord-column-names",
        "x,y",
        auto[0],
    ];
    let (query, _, _, ref_data, divide) = sides(&prefix, &data, &fit_args).unwrap();
    assert!(divide.is_some(), "the reference is divided");
    let (profiles, features) = crate::util::parquet_io::read_feature_community(
        std::path::Path::new(&format!("{prefix}.feature_community.parquet")),
    )
    .unwrap();
    let undivided = project(
        &ref_data,
        &profiles,
        &features,
        &data_beans::aux::feature_names::FeatureNameKind::Exact,
        100,
    )
    .unwrap();
    assert_eq!(query, undivided);
}
