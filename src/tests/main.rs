use super::*;

fn out_of(argv: &[&str]) -> String {
    let mut cli = Cli::try_parse_from(argv).unwrap_or_else(|e| panic!("{e}"));
    name_folder_out(&mut cli.commands);
    let (out, _) = cli.commands.out_mut().unwrap();
    out.to_string()
}

#[test]
fn an_out_ending_in_a_slash_is_a_folder_named_for_the_method() {
    assert_eq!(
        out_of(&["pinto", "lc", "d.zarr", "--out", "res/"]),
        "res/lc"
    );
    assert_eq!(
        out_of(&["pinto", "dsvd", "d.zarr", "--out", "a/b/"]),
        "a/b/dsvd"
    );
    assert_eq!(
        out_of(&["pinto", "lc", "d.zarr", "--out", "res/r"]),
        "res/r"
    );
}
