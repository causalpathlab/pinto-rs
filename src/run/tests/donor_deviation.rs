use super::*;
use crate::donor_deviation::DonorDeviationArgs;

fn tool_args(v: Value) -> DonorDeviationToolArgs {
    serde_json::from_value(v).unwrap()
}

/// The tool's fields and flags reach the child's command line, which pinto
/// parses into stage 2's arguments.
#[test]
fn the_tool_s_fields_and_flags_reach_the_command_line() {
    let cli = crate::built_cli();
    let a = tool_args(json!({
        "model": "res/s1", "donor_of_batch": "donors.tsv", "out": "res/s2",
        "data": ["a.zarr.zip", "b.zarr.zip"],
        "flags": {"matched-pairs": 20, "pairs": "pb", "device": "cuda"},
        "max_threads": 8,
    }));
    let argv = donor_deviation_argv(&cli, &a, "res/s1", "res/s2").unwrap();
    assert_eq!(argv[..2], ["apply", "donor-deviation"], "{argv:?}");
    assert!(argv.contains(&"--max-threads=8".to_string()), "{argv:?}");
    let parsed: DonorDeviationArgs = super::super::mcp::parsed(&cli, &argv).unwrap();
    assert!(parsed.model.ends_with("res/s1"), "{}", parsed.model);
    assert!(parsed.donor_of_batch.ends_with("donors.tsv"));
    assert!(parsed.out.ends_with("res/s2"));
    assert_eq!(parsed.data.as_ref().map(Vec::len), Some(2));
    assert_eq!(parsed.matched_pairs, 20);
    assert_eq!(parsed.device, crate::util::device::ComputeDevice::Cuda);

    // without data, the run's own files
    let bare = tool_args(json!({"model": "m", "donor_of_batch": "d", "out": "o"}));
    let argv = donor_deviation_argv(&cli, &bare, "m", "o").unwrap();
    let parsed: DonorDeviationArgs = super::super::mcp::parsed(&cli, &argv).unwrap();
    assert!(parsed.data.is_none());
}

/// A field given as a flag, or a flag the command has not, is refused
/// before anything runs.
#[test]
fn a_field_as_a_flag_and_an_unknown_flag_are_refused() {
    let cli = crate::built_cli();
    let as_flag = tool_args(json!({"model": "m", "donor_of_batch": "d", "out": "o",
                                   "flags": {"--model": "other"}}));
    let e = donor_deviation_argv(&cli, &as_flag, "m", "o")
        .unwrap_err()
        .to_string();
    assert!(e.contains("give --model as model"), "{e}");
    let unknown = tool_args(json!({"model": "m", "donor_of_batch": "d", "out": "o",
                                   "flags": {"n-communities": 3}}));
    let e = donor_deviation_argv(&cli, &unknown, "m", "o")
        .unwrap_err()
        .to_string();
    assert!(e.contains("has no flag --n-communities"), "{e}");
}
