//! Delimited text as Python's `csv` module reads and writes it: a field
//! that starts with `"` is quoted (`""` inside is one quote), lines end at
//! `\n`, `\r\n` or `\r`, and blank lines are skipped. Writing quotes only
//! the fields that need it, so the tables match the Python builder's.

use anyhow::{Context, Result};

/// A header row and the rows under it.
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Table {
    /// Parse `text` with `delim`; the first non-blank row is the header.
    pub fn parse(text: &str, delim: char) -> Result<Table> {
        let mut rows = parse_delimited(text, delim).into_iter();
        let header = rows.next().context("the table is empty")?;
        Ok(Table {
            header,
            rows: rows.collect(),
        })
    }

    /// Index of column `name` (the last of that name, as a `DictReader`
    /// keeps it).
    pub fn col(&self, name: &str) -> Result<usize> {
        self.header
            .iter()
            .rposition(|h| h == name)
            .with_context(|| format!("no column `{name}` in header {:?}", self.header))
    }
}

/// Field `i` of `row`, empty when the row is short.
pub fn field(row: &[String], i: usize) -> &str {
    row.get(i).map_or("", String::as_str)
}

/// All records of `text`, blank lines skipped.
pub fn parse_delimited(text: &str, delim: char) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut field_start = true;
    let mut in_quotes = false;
    let mut row_has_content = false;
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if field_start => {
                in_quotes = true;
                field_start = false;
                row_has_content = true;
            }
            '\n' | '\r' => {
                if c == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                if row_has_content {
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                }
                field_start = true;
                row_has_content = false;
            }
            c if c == delim => {
                row.push(std::mem::take(&mut field));
                field_start = true;
                row_has_content = true;
            }
            c => {
                field.push(c);
                field_start = false;
                row_has_content = true;
            }
        }
    }
    if row_has_content {
        row.push(field);
        rows.push(row);
    }
    rows
}

/// Tab-separated text with `header`, one line per row, `\n` endings.
pub fn format_tsv<R: AsRef<[String]>>(header: &[&str], rows: &[R]) -> String {
    let mut out = String::new();
    push_row(&mut out, header.iter().copied());
    for row in rows {
        push_row(&mut out, row.as_ref().iter().map(String::as_str));
    }
    out
}

fn push_row<'a>(out: &mut String, fields: impl Iterator<Item = &'a str>) {
    for (i, f) in fields.enumerate() {
        if i > 0 {
            out.push('\t');
        }
        if f.contains(['\t', '"', '\n', '\r']) {
            out.push('"');
            out.push_str(&f.replace('"', "\"\""));
            out.push('"');
        } else {
            out.push_str(f);
        }
    }
    out.push('\n');
}
