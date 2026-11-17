//! The model's pair encoder is found from the model prefix alone, so the
//! resolution is checked without a model.

use super::{is_cage_cell, pair_encoder_path, ModelOnlyFlags, PredictArgs};
use clap::{CommandFactory, FromArgMatches, Parser};

#[derive(Parser)]
struct Probe {
    #[command(flatten)]
    args: PredictArgs,
}

/// `pinto predict` of `extra` beside a query and a model: its args and the
/// model-only flags it was given.
fn parse(extra: &[&str]) -> (PredictArgs, ModelOnlyFlags) {
    let argv = [&["predict", "data", "--model", "m", "-o", "o"][..], extra].concat();
    let mut cmd = Probe::command();
    let m = cmd.try_get_matches_from_mut(argv).unwrap();
    let args = Probe::from_arg_matches(&m).unwrap().args;
    (args, ModelOnlyFlags::given(&cmd, &m))
}

#[test]
fn the_pair_encoder_is_required_and_found_beside_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("run").to_string_lossy().to_string();
    let file = format!("{model}.pair_encoder.safetensors");

    let err = pair_encoder_path(&model).unwrap_err();
    assert!(err.to_string().contains(&file), "{err}");

    std::fs::write(&file, b"").unwrap();
    assert_eq!(pair_encoder_path(&model).unwrap(), file);
}

/// The flags that apply to one kind of model are named when given on the
/// command line, whatever their value, and not when left at their defaults.
#[test]
fn model_only_flags_are_named_when_given() {
    let (_, none) = parse(&[]);
    assert!(none.cage.is_empty() && none.holdout.is_empty());
    let (_, given) = parse(&[
        "--no-init-features",
        "--device-no",
        "0",
        "--holdout-unit",
        "cell",
    ]);
    assert_eq!(given.cage, ["--device-no", "--no-init-features"]);
    assert_eq!(given.holdout, ["--holdout-unit"]);
}

/// A model that is not there is refused before any data is read, by its
/// path.
#[test]
fn a_model_that_is_not_there_is_refused_by_its_path() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("nope").to_string_lossy().to_string();
    let (mut args, given) = parse(&[]);
    args.core.model = model.clone().into();
    args.core.common.data_files = vec![dir.path().join("d.zarr").to_string_lossy().into()];
    let e = format!("{:#}", super::predict(&args, &given).unwrap_err());
    assert!(e.contains(&format!("{model}.pinto.json")), "{e}");
    assert!(e.contains("not there"), "{e}");
}

mod batch_effects {
    use super::super::delta::{combine, match_batches, renamed_files, ModelNaming};
    use crate::test_support::boxed;
    use crate::util::common::Mat;
    use crate::util::input::SrtInputArgs;
    use crate::util::metadata::BatchLabels;
    use clap::Parser;

    /// A query's inputs, from its command line `args`.
    fn query(args: &[&str]) -> SrtInputArgs {
        let argv = [&["x"][..], args, &["-o", "o"]].concat();
        SrtInputArgs::try_parse_from(argv).unwrap()
    }

    /// A model's naming: what named its batches, and its data files.
    fn naming(source: Option<BatchLabels>, files: &[String]) -> ModelNaming {
        ModelNaming {
            source,
            files: files.to_vec(),
            names: Vec::new(),
        }
    }

    /// `name`, an empty file under `dir`, as a path.
    fn touch(dir: &std::path::Path, name: &str) -> String {
        let p = dir.join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"").unwrap();
        p.to_str().unwrap().to_string()
    }

    /// `link`, under `dir`, a link to `to`, as a path.
    fn link(dir: &std::path::Path, link: &str, to: &str) -> String {
        let p = dir.join(link);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(to, &p).unwrap();
        p.to_str().unwrap().to_string()
    }

    /// The canonical form of `p`, as a manifest records a data file.
    fn canonical(p: &str) -> String {
        std::fs::canonicalize(p)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string()
    }

    /// Labels match by name, only between labels of one kind: file names
    /// with file names, `--batch-files` labels with those. Components
    /// never match.
    #[test]
    fn query_batches_match_by_name_within_their_kind() {
        use BatchLabels::{AutoBatch, BatchFiles, Files};
        let model = boxed(&["s1", "s2"]);
        let q = boxed(&["s2", "x"]);
        let by_file = query(&["d/s2.zarr", "d/x.zarr"]);
        let by_name = query(&["d/a.zarr", "d/b.zarr", "--batch-files", "a.txt,b.txt"]);
        let auto = query(&["d/s2.zarr", "d/x.zarr", "--auto-batch"]);
        let m = |source| naming(Some(source), &[]);
        assert_eq!(
            match_batches(&q, &by_file, &model, &m(Files)),
            [Some(1), None]
        );
        assert_eq!(
            match_batches(&q, &by_name, &model, &m(BatchFiles)),
            [Some(1), None]
        );
        assert_eq!(
            match_batches(&q, &by_file, &model, &m(BatchFiles)),
            [None, None]
        );
        assert_eq!(match_batches(&q, &by_name, &model, &m(Files)), [None, None]);
        assert_eq!(
            match_batches(&q, &auto, &model, &m(AutoBatch)),
            [None, None]
        );
    }

    /// An older model named its file batches by position (its manifest
    /// says `files`, or records nothing and the labels are exactly its
    /// files' positions): a query file is the batch of the model's data
    /// file it is, through links, whatever either is called. A file that
    /// only shares a name with one is not it.
    #[test]
    fn an_older_models_positions_match_the_query_file_that_is_its_file() {
        use BatchLabels::{FilePositions, Files};
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let (a, b) = (touch(d, "real/a.zarr"), touch(d, "real/b.zarr"));
        let files = vec![canonical(&a), canonical(&b)];
        let model = boxed(&["1", "0"]);
        // s1 is b under another name; the other a is another file
        let s1 = link(d, "links/s1.zarr", &b);
        let other_a = touch(d, "other/a.zarr");
        let q = boxed(&["s1", "a"]);
        for source in [Some(FilePositions), None] {
            let m = naming(source, &files);
            assert_eq!(
                match_batches(&q, &query(&[&s1, &a]), &model, &m),
                [Some(0), Some(1)],
                "{source:?}"
            );
            assert_eq!(
                match_batches(&q, &query(&[&s1, &other_a]), &model, &m),
                [Some(0), None],
                "{source:?}: a file named as the model's a but not it"
            );
        }
        // a model of this pinto whose files are named 0 and 1 is matched
        // by name
        let named = query(&[&touch(d, "q/1.zarr")]);
        assert_eq!(
            match_batches(&boxed(&["1"]), &named, &model, &naming(Some(Files), &files)),
            [Some(0)]
        );
    }

    /// An older model whose manifest records no data files names no file
    /// batch: a query file named like one of its positions is new.
    #[test]
    fn an_older_model_without_its_files_shares_no_file_batch() {
        let model = boxed(&["0", "1"]);
        let q = query(&["d/0.zarr", "d/1.zarr"]);
        for source in [Some(BatchLabels::FilePositions), None] {
            assert_eq!(
                match_batches(&boxed(&["0", "1"]), &q, &model, &naming(source, &[])),
                [None, None],
                "{source:?}"
            );
        }
    }

    /// A model that records nothing of how it labelled its batches matches
    /// `--batch-files` labels by name, even labels that read as its files'
    /// positions; a file's batch only through its files.
    #[test]
    fn unrecorded_labels_are_names_to_batch_files_labels() {
        let files = vec!["d/a.zarr".to_string(), "d/b.zarr".to_string()];
        let model = boxed(&["0", "1"]);
        let q = query(&["d/a.zarr", "d/b.zarr", "--batch-files", "a.txt,b.txt"]);
        assert_eq!(
            match_batches(&boxed(&["1", "0"]), &q, &model, &naming(None, &files)),
            [Some(1), Some(0)]
        );
        let named = boxed(&["s1", "s2"]);
        assert_eq!(
            match_batches(&boxed(&["s2"]), &q, &named, &naming(None, &files)),
            [Some(1)]
        );
    }

    /// A query file of a model file batch's name that is another file is
    /// named, with the model's file (for a warning); one that is the
    /// model's file, through a link, is not.
    #[test]
    fn a_query_file_named_as_a_model_file_but_not_it_is_found() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let a = touch(d, "real/a.zarr");
        let model = ModelNaming {
            source: Some(BatchLabels::Files),
            files: vec![canonical(&a)],
            names: boxed(&["s1"]),
        };
        let same = link(d, "links/s1.zarr", &a);
        let other = touch(d, "other/s1.zarr");
        let files = boxed(&[&same, &other]);
        let names = boxed(&["s1", "s1"]);
        assert_eq!(
            renamed_files(&files, &names, &model),
            [(other.as_str(), model.files[0].as_str())]
        );
    }

    /// A known batch takes the model's column; a new one the query's
    /// estimate times the model-over-estimate ratio of the shared batches,
    /// its spread scaled alike; a row the model lacks keeps the estimate.
    /// Every plane of the fits' table is filled.
    #[test]
    fn a_new_batch_is_carried_into_the_model_frame() {
        use crate::util::batch_effects::RunDelta;
        let planes = |mean: &[f32], sd: f32| RunDelta {
            rows: boxed(&["r0", "r1"]),
            batches: boxed(&["b0", "b1"]),
            mean: Mat::from_row_slice(2, 2, mean),
            sd: Mat::from_element(2, 2, sd),
            log_mean: Mat::from_row_slice(2, 2, mean).map(f32::ln),
            log_sd: Mat::from_element(2, 2, 0.1),
        };
        // model: rows x batches (m0, m1)
        let model = planes(&[0.5, 2.0, 1.0, 1.0], 0.2);
        // query batches (q0 = m1, q1 new); its own estimate pinned over q0, q1
        let estimate = planes(&[0.5, 2.0, 3.0, 1.0 / 3.0], 0.01);
        let matched = [Some(1), None];
        let q = boxed(&["q0", "q1"]);
        let d = combine(&model, &[true, false], Some(&estimate), &matched, &q);
        assert_eq!(d.batches, q);
        assert_eq!(d.mean[(0, 0)], 2.0, "known batch: the model's column");
        assert_eq!(d.sd[(0, 0)], 0.2, "known batch: the model's spread");
        // new: 2.0 (estimate) x 2.0 / 0.5 (model over estimate on q0)
        assert!((d.mean[(0, 1)] - 8.0).abs() < 1e-5, "{}", d.mean[(0, 1)]);
        assert!((d.sd[(0, 1)] - 0.04).abs() < 1e-6, "{}", d.sd[(0, 1)]);
        assert!((d.log_mean[(0, 1)] - 8f32.ln()).abs() < 1e-5);
        assert_eq!(d.log_sd[(0, 1)], 0.1);
        assert_eq!(
            (d.mean[(1, 0)], d.mean[(1, 1)]),
            (3.0, 1.0 / 3.0),
            "no model row"
        );
    }
}

/// The query is divided only by a model's δ: a model without one (one
/// batch, `--ignore-batch`, or an older cage-cell fit) trained its encoder
/// on undivided counts; `--ignore-batch` on predict leaves it unused, and
/// is refused for a model fitted on δ-divided counts.
#[test]
fn the_query_is_divided_only_when_the_model_has_a_delta() {
    use crate::util::batch_effects::RunDelta;
    use crate::util::common::Mat;
    use crate::util::metadata::{manifest_path, PintoMetadata, Run};
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("run").to_string_lossy().to_string();
    assert!(super::model_delta(&model, None, false).unwrap().is_none());

    let file = format!("{model}.delta.parquet");
    RunDelta {
        rows: vec!["GENE1".into()],
        batches: vec!["0".into(), "1".into()],
        mean: Mat::from_row_slice(1, 2, &[0.5, 2.0]),
        sd: Mat::zeros(1, 2),
        log_mean: Mat::zeros(1, 2),
        log_sd: Mat::zeros(1, 2),
    }
    .write(&file)
    .unwrap();
    let mut meta = PintoMetadata::default();
    meta.write(std::path::Path::new(&manifest_path(&model)))
        .unwrap();
    let run = Run::open(&model).unwrap();
    assert!(
        super::model_delta(&model, Some(&run), false)
            .unwrap()
            .is_none(),
        "a delta the manifest does not name"
    );
    meta.outputs.batch_effects = Some(file);
    meta.write(std::path::Path::new(&manifest_path(&model)))
        .unwrap();
    let run = Run::open(&model).unwrap();
    let (d, _) = super::model_delta(&model, Some(&run), false)
        .unwrap()
        .unwrap();
    assert_eq!(d.mean[(0, 1)], 2.0);
    assert!(super::model_delta(&model, Some(&run), true)
        .unwrap()
        .is_none());
    // fitted on δ-divided counts: an undivided query is refused
    meta.outputs.feature_community_batch_adjusted = true;
    meta.write(std::path::Path::new(&manifest_path(&model)))
        .unwrap();
    let run = Run::open(&model).unwrap();
    let err = super::model_delta(&model, Some(&run), true)
        .err()
        .expect("refused");
    assert!(err.to_string().contains("--ignore-batch"), "{err}");
    // a cage query reads no held-out flag, so none is offered
    assert!(!err.to_string().contains("--holdout"), "{err}");
    assert!(super::model_delta(&model, Some(&run), false)
        .unwrap()
        .is_some());
}

/// A manifest's data files: canonical ones as recorded; an older relative
/// entry beside the manifest, whatever the working directory.
#[test]
fn model_data_files_resolve_beside_the_manifest() {
    use crate::util::metadata::{manifest_path, PintoMetadata, Run};
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("run").to_string_lossy().to_string();
    std::fs::write(dir.path().join("a.zarr"), b"").unwrap();
    let meta = PintoMetadata {
        data_files: Some(vec!["a.zarr".into(), "gone.zarr".into()]),
        ..Default::default()
    };
    meta.write(std::path::Path::new(&manifest_path(&model)))
        .unwrap();
    let files = super::model_data_files(Some(&Run::open(&model).unwrap()));
    assert_eq!(
        std::path::Path::new(&files[0]),
        dir.path().join("a.zarr"),
        "a relative entry beside the manifest"
    );
    assert_eq!(files[1], "gone.zarr", "kept as written");
    assert!(super::model_data_files(None).is_empty());
}

/// A model named by its manifest's own path, whatever that file is called,
/// opens as the run its manifest records: the prefix beside the manifest,
/// so its δ (recorded elsewhere) and its files are found. A model named
/// by prefix opens the same run; one without a manifest keeps its prefix.
#[test]
fn a_model_named_by_its_manifest_opens_once_with_its_prefix() {
    use crate::util::metadata::PintoMetadata;
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let delta = dir.path().join("kept").join("run.delta.parquet");
    std::fs::create_dir(dir.path().join("kept")).unwrap();
    std::fs::write(&delta, b"").unwrap();
    std::fs::write(format!("{prefix}.delta.parquet"), b"").unwrap();
    let meta = PintoMetadata {
        command: "cage".into(),
        prefix: "somewhere/else/run".into(),
        outputs: crate::util::metadata::OutputFiles {
            batch_effects: Some("kept/run.delta.parquet".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let manifest = dir.path().join("first.json");
    meta.write(&manifest).unwrap();

    let mut model: Box<str> = manifest.to_string_lossy().into();
    let run = super::open_model(&mut model)
        .unwrap()
        .expect("the manifest opens");
    assert_eq!(model.as_ref(), prefix);
    assert_eq!(run.meta.command, "cage");
    assert_eq!(
        crate::util::batch_effects::run_delta_path(&model, Some(&run))
            .unwrap()
            .map(std::path::PathBuf::from),
        Some(delta),
        "the δ the manifest records, not the one under the prefix"
    );

    let mut bare: Box<str> = format!("{prefix}.pinto.json").into();
    assert!(super::open_model(&mut bare).unwrap().is_none());
    assert_eq!(bare.as_ref(), prefix, "no manifest: the prefix as named");
}

/// Only a manifest that is not there means a model without one: one that
/// is there but cannot be read is an error, not a model found by its files.
#[test]
fn an_unreadable_manifest_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let manifest = crate::util::metadata::manifest_path(&prefix);
    std::fs::write(&manifest, b"{ not json").unwrap();
    let mut model: Box<str> = prefix.as_str().into();
    assert!(
        super::open_model(&mut model).is_err(),
        "a broken manifest was taken as none"
    );
    let mut named: Box<str> = manifest.as_str().into();
    assert!(super::open_model(&mut named).is_err());
}

/// Whether every batch is known is decided from the inputs, before any
/// data is read: per-file labels are known when each file's name is a
/// model batch, `--batch-files` labels when each line names one, and
/// `--auto-batch` components never.
#[test]
fn every_batch_is_known_from_the_inputs() {
    use super::delta::{every_batch_known, ModelNaming};
    use crate::test_support::boxed;
    use crate::util::input::SrtInputArgs;
    use crate::util::metadata::BatchLabels;
    use clap::Parser;
    let dir = tempfile::tempdir().unwrap();
    let file = |n: &str, body: &str| {
        let p = dir.path().join(n);
        std::fs::write(&p, body).unwrap();
        p.to_str().unwrap().to_string()
    };
    let (a, b, new) = (file("a.zarr", ""), file("b.zarr", ""), file("c.zarr", ""));
    let by_file = boxed(&["a", "b"]);
    let known = |args: &[&str], model: &[Box<str>], source| {
        let mut argv = vec!["x"];
        argv.extend_from_slice(args);
        argv.extend_from_slice(&["-o", "o"]);
        let c = SrtInputArgs::try_parse_from(argv).unwrap();
        let naming = ModelNaming {
            source,
            ..Default::default()
        };
        every_batch_known(&c, model, &naming).unwrap()
    };
    let files = Some(BatchLabels::Files);
    assert!(known(&[&b, &a], &by_file, files));
    assert!(known(&[&b], &by_file, files));
    assert!(!known(&[&a, &new], &by_file, files), "a new file");

    let named = boxed(&["s1", "s2"]);
    let batch_files = Some(BatchLabels::BatchFiles);
    let (l1, l2) = (file("a.txt", "s1\ns2\n"), file("b.txt", "s2\ns3\n"));
    assert!(known(&[&a, "--batch-files", &l1], &named, batch_files));
    let both = format!("{l1},{l2}");
    assert!(
        !known(&[&a, &b, "--batch-files", &both], &named, batch_files),
        "s3 is new"
    );
    assert!(!known(&[&a, &b, "--auto-batch"], &by_file, files));
    // against a model labelled by file, a label file is not read at all
    let unread = dir.path().join("not-there.txt");
    assert!(!known(
        &[&a, "--batch-files", unread.to_str().unwrap()],
        &by_file,
        files
    ));
}

/// `--null-from` takes a comma list, as every other file-list flag does,
/// and still takes the files listed after it.
#[test]
fn null_from_takes_a_comma_list() {
    let null_from = |extra: &[&str]| parse(extra).0.scoring.null_from.unwrap();
    let two: Vec<Box<str>> = vec!["a.zarr".into(), "b.zarr".into()];
    assert_eq!(null_from(&["--null-from", "a.zarr,b.zarr"]), two);
    assert_eq!(null_from(&["--null-from", "a.zarr", "b.zarr"]), two);
}

/// Without a manifest, a cage-cell model is known by its files: a pair
/// latent with neither a pair encoder nor a pseudobulk table. A cage
/// model's files are not mistaken for one.
#[test]
fn a_cage_cell_model_is_known_by_its_files_without_a_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("m").to_str().unwrap().to_string();
    let touch = |suffix: &str| std::fs::write(format!("{model}.{suffix}"), b"").unwrap();
    touch("latent.parquet");
    assert!(is_cage_cell(&model, None));
    touch("pb_embedding.parquet");
    assert!(!is_cage_cell(&model, None), "a cage model's files");
    std::fs::remove_file(format!("{model}.pb_embedding.parquet")).unwrap();
    touch("pair_encoder.safetensors");
    assert!(!is_cage_cell(&model, None), "a model with an encoder");
}
