//! Delimited text read and written as Python's `csv` does.

use crate::lr_activity::ligand_effect::reference::table::{format_tsv, parse_delimited, Table};

#[test]
fn quoted_csv_fields() {
    let rows = parse_delimited(
        "\"\",\"a\",b\r\n\"1\",\"x,y\",\"say \"\"hi\"\"\"\n\n2,,z\n",
        ',',
    );
    assert_eq!(
        rows,
        [
            vec!["", "a", "b"],
            vec!["1", "x,y", "say \"hi\""],
            vec!["2", "", "z"],
        ]
    );
}

#[test]
fn header_lookup_and_short_rows() {
    let t = Table::parse("a\tb\tc\n1\t2\n", '\t').unwrap();
    assert_eq!(t.col("c").unwrap(), 2);
    assert!(t.col("d").is_err());
    let r = &t.rows[0];
    assert_eq!(
        crate::lr_activity::ligand_effect::reference::table::field(r, 2),
        ""
    );
}

#[test]
fn tsv_quotes_only_what_needs_it() {
    let rows = vec![["A".to_string(), "x\ty".to_string(), "q\"".to_string()]];
    assert_eq!(
        format_tsv(&["h1", "h2", "h3"], &rows),
        "h1\th2\th3\nA\t\"x\ty\"\t\"q\"\"\"\n"
    );
}
