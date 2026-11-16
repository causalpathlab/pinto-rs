//! The rules between `--feature-embedding-mode` and the flags that only one mode
//! reads, checked before any data is opened.

use crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs;
use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    cage: CellActivityGraphEmbeddingArgs,
}

fn parse(argv: &[&str]) -> CellActivityGraphEmbeddingArgs {
    Cli::try_parse_from(
        ["x", "-o", "out", "data.zarr"]
            .into_iter()
            .chain(argv.iter().copied()),
    )
    .unwrap_or_else(|e| panic!("{e}"))
    .cage
}

#[test]
fn the_lora_knobs_need_the_lora_mode() {
    let ok = parse(&[
        "--feature-embedding",
        "d.parquet",
        "--feature-embedding-mode",
        "lora",
        "--lora-rank",
        "4",
    ]);
    assert!(ok.validate_feature_embedding().is_ok());
    let bare = parse(&[
        "--feature-embedding",
        "d.parquet",
        "--feature-embedding-mode",
        "lora",
    ]);
    assert!(bare.validate_feature_embedding().is_ok());
    let wrong = parse(&[
        "--feature-embedding",
        "d.parquet",
        "--feature-embedding-mode",
        "freeze",
        "--lora-rank",
        "4",
    ]);
    let err = wrong.validate_feature_embedding().unwrap_err().to_string();
    assert!(err.contains("--lora-rank"), "{err}");
}

#[test]
fn the_adapter_residual_needs_the_adapt_mode() {
    let ok = parse(&[
        "--feature-embedding",
        "d.parquet",
        "--feature-adapter-residual",
    ]);
    assert!(ok.validate_feature_embedding().is_ok());
    let wrong = parse(&[
        "--feature-embedding",
        "d.parquet",
        "--feature-embedding-mode",
        "lora",
        "--feature-adapter-residual",
    ]);
    assert!(wrong.validate_feature_embedding().is_err());
}

mod embedding_dim {
    use super::parse;
    use crate::cell_activity_graph_embedding::args::DEFAULT_EMBEDDING_DIM;

    fn dim(argv: &[&str], width: Option<usize>) -> anyhow::Result<usize> {
        parse(argv).resolve_embedding_dim(width)
    }

    #[test]
    fn without_a_dictionary_the_flag_or_the_default_decides() {
        assert_eq!(dim(&[], None).unwrap(), DEFAULT_EMBEDDING_DIM);
        // The width cage and cage-cell train at when nothing else decides it.
        assert_eq!(DEFAULT_EMBEDDING_DIM, 64);
        assert_eq!(dim(&["--embedding-dim", "32"], None).unwrap(), 32);
        assert_eq!(
            dim(&["--embedding-dim", "auto"], None).unwrap(),
            DEFAULT_EMBEDDING_DIM
        );
    }

    /// Both help texts of `--embedding-dim` state the width `auto` falls
    /// back to, so they cannot drift from the constant.
    #[test]
    fn the_help_states_the_default_width() {
        let cmd = <super::Cli as clap::CommandFactory>::command();
        let arg = cmd
            .get_arguments()
            .find(|a| a.get_long() == Some("embedding-dim"))
            .unwrap();
        let short = arg.get_help().unwrap().to_string();
        let long = arg.get_long_help().unwrap().to_string();
        assert!(
            short.contains(&format!("else {DEFAULT_EMBEDDING_DIM})")),
            "{short}"
        );
        assert!(
            long.contains(&format!("auto is {DEFAULT_EMBEDDING_DIM};")),
            "{long}"
        );
        // The width doubles as the k-means community count; the help says so.
        assert!(long.contains("--n-communities"), "{long}");
    }

    /// What that help sentence rests on: k-means with no `--n-communities`
    /// carries no count of its own, so the cut reads it off the latent width.
    #[test]
    fn kmeans_without_a_count_leaves_it_to_the_width() {
        use crate::link_community::profiles::EdgeClustering;
        let cut = |argv: &[&str]| super::parse(argv).edge_clustering.resolve(1);
        assert!(matches!(
            cut(&["--edge-cluster-method", "kmeans"]),
            EdgeClustering::Kmeans {
                n_clusters: None,
                ..
            }
        ));
        assert!(matches!(
            cut(&["--edge-cluster-method", "kmeans", "--n-communities", "5"]),
            EdgeClustering::Kmeans {
                n_clusters: Some(5),
                ..
            }
        ));
    }

    #[test]
    fn a_pinned_dictionary_sets_the_width_and_a_conflicting_flag_is_refused() {
        for mode in ["freeze", "free", "lora"] {
            let a = [
                "--feature-embedding",
                "d.parquet",
                "--feature-embedding-mode",
                mode,
            ];
            assert_eq!(dim(&a, Some(128)).unwrap(), 128, "{mode}");
            let b = [&a[..], &["--embedding-dim", "128"]].concat();
            assert_eq!(dim(&b, Some(128)).unwrap(), 128, "{mode}");
            let c = [&a[..], &["--embedding-dim", "16"]].concat();
            let err = dim(&c, Some(128)).unwrap_err().to_string();
            assert!(err.contains("128") && err.contains("16"), "{mode}: {err}");
        }
    }

    #[test]
    fn the_adapter_keeps_its_own_width() {
        let a = ["--feature-embedding", "d.parquet"];
        assert_eq!(dim(&a, Some(128)).unwrap(), DEFAULT_EMBEDDING_DIM);
        assert_eq!(
            dim(&[&a[..], &["--embedding-dim", "32"]].concat(), Some(128)).unwrap(),
            32
        );
    }

    #[test]
    fn the_lora_rank_is_validated_but_not_capped_at_the_width() {
        let a = [
            "--feature-embedding",
            "d.parquet",
            "--feature-embedding-mode",
            "lora",
        ];
        assert!(dim(&a, Some(128)).is_ok());
        // legume-graph-embedding 0.9 does not cap the rank at the width: the
        // default rank on a narrow dictionary is allowed.
        assert!(dim(&a, Some(16)).is_ok());
        assert!(dim(&[&a[..], &["--lora-rank", "4"]].concat(), Some(16)).is_ok());
        assert!(dim(&[&a[..], &["--lora-rank", "0"]].concat(), Some(128)).is_err());
    }
}
