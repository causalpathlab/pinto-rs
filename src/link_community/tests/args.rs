use crate::link_community::args::SrtLinkCommunityArgs;
use clap::Parser;

fn parse(extra: &[&str]) -> Result<SrtLinkCommunityArgs, clap::Error> {
    let base = ["lc", "data.zarr.zip", "-c", "cells.parquet", "-o", "out"];
    SrtLinkCommunityArgs::try_parse_from(base.iter().chain(extra))
}

#[test]
fn alpha_is_one_unless_asked() {
    assert_eq!(parse(&[]).unwrap().alpha, 1.0);
    assert_eq!(parse(&["--alpha", "0"]).unwrap().alpha, 0.0);
    assert_eq!(parse(&["--alpha", "2.5"]).unwrap().alpha, 2.5);
}

/// A prior that is not one would otherwise read as no prior, without a word.
#[test]
fn alpha_must_be_finite_and_not_negative() {
    for bad in ["--alpha=-1", "--alpha=nan", "--alpha=inf"] {
        assert!(parse(&[bad]).is_err(), "{bad}");
    }
}
