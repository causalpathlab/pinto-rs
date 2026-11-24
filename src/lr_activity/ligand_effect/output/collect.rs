//! `pinto lra merge` on effect tables rather than shards: the marginal table
//! and the `--link-community` tables of one run, collected into
//! `{out}.lra.by-link-community.parquet`.
//!
//! Every column of the tables is kept, lined up by name (each part's q-values were
//! corrected within its own table, which its `q_scope` says), and a
//! `link_community` column is added in front: `all` for the marginal table,
//! `K` for `{prefix}.lra.link-community-K.parquet`.

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use std::fs::File;
use std::sync::Arc;

/// The `link_community` value of the marginal table's rows.
pub const MARGINAL_LINK_COMMUNITY: &str = "all";

/// `{out}.lra.by-link-community.parquet`.
pub fn by_link_community_path(out: &str) -> String {
    format!("{out}.lra.by-link-community.parquet")
}

/// Whether `inputs` are effect tables to collect (every one a `.parquet`)
/// rather than shard files to merge.
pub fn are_tables(inputs: &[String]) -> bool {
    !inputs.is_empty() && inputs.iter().all(|path| path.ends_with(".parquet"))
}

/// The link community a table is of, read from its name: `K` for
/// `….lra.link-community-K[.only-TAG].parquet`, `all` for a marginal
/// `….lra[.only-TAG].parquet`; an error for any other table (a curve).
fn link_community_of_table(path: &str) -> anyhow::Result<String> {
    anyhow::ensure!(
        !path.ends_with(".curve.parquet") && !path.ends_with(".celltype.parquet"),
        "{path}: a curve or cell-type table, not an effect table"
    );
    if let Some(k) = super::super::link_community_scope::link_community_in_name(path) {
        return Ok(k.to_string());
    }
    let marginal = path.ends_with(".lra.parquet")
        || path
            .rsplit_once(".lra.only-")
            .is_some_and(|(_, tag)| tag.ends_with(".parquet") && !tag.contains('/'));
    anyhow::ensure!(
        marginal,
        "{path}: not a contact-effect table of the marginal run ({{prefix}}.lra.parquet) \
         or of a link community ({{prefix}}.lra.link-community-K.parquet)"
    );
    Ok(MARGINAL_LINK_COMMUNITY.to_string())
}

/// Collect `tables` into `{out}.lra.by-link-community.parquet`. Their columns
/// are lined up by name: one a table lacks (the marginal table's
/// `--marginal-baselines` columns, which a link community cannot have) is
/// empty on its rows. The path written.
pub fn collect_link_community_tables(tables: &[String], out: &str) -> anyhow::Result<String> {
    // every table's batches with its community, and the columns of them all
    // in order of first appearance after the community's
    let mut fields = vec![Field::new("link_community", DataType::Utf8, false)];
    let mut batches: Vec<(String, RecordBatch)> = Vec::new();
    for table in tables {
        let link_community = link_community_of_table(table)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(table)?)?.build()?;
        for batch in reader {
            let batch = batch?;
            for field in batch.schema().fields() {
                match fields.iter().find(|f| f.name() == field.name()) {
                    Some(known) => anyhow::ensure!(
                        known.data_type() == field.data_type(),
                        "{table}: column {} is {} here, {} in an earlier table",
                        field.name(),
                        field.data_type(),
                        known.data_type()
                    ),
                    None => fields.push(field.as_ref().clone().with_nullable(true)),
                }
            }
            batches.push((link_community.clone(), batch));
        }
    }
    anyhow::ensure!(!batches.is_empty(), "no rows in the tables given");
    let schema = Arc::new(Schema::new(fields));
    let path = by_link_community_path(out);
    legume_numeric::matrix::common_io::mkdir_parent(&path)?;
    let mut writer = ArrowWriter::try_new(File::create(&path)?, schema.clone(), None)?;
    let mut rows = 0;
    for (link_community, batch) in &batches {
        let community_column: ArrayRef = Arc::new(StringArray::from(vec![
            link_community.as_str();
            batch.num_rows()
        ]));
        let columns: Vec<ArrayRef> = std::iter::once(community_column)
            .chain(schema.fields().iter().skip(1).map(|field| {
                match batch.column_by_name(field.name()) {
                    Some(column) => column.clone(),
                    None => arrow_array::new_null_array(field.data_type(), batch.num_rows()),
                }
            }))
            .collect();
        writer.write(&RecordBatch::try_new(schema.clone(), columns)?)?;
        rows += batch.num_rows();
    }
    writer.close()?;
    log::info!(
        "Collected {} tables ({rows} rows) into {path}",
        tables.len()
    );
    Ok(path)
}
