//! Building a parquet table one named column at a time.

use crate::lr_activity::ligand_effect::inference::Estimate;
use legume_numeric::matrix::parquet::{
    parquet_add_bytearray, parquet_add_numeric_column, parquet_add_string_column, ParquetWriter,
};
use parquet::basic::Type as ParquetType;

pub(super) fn f32s<T>(rows: &[T], f: impl Fn(&T) -> f64) -> Vec<f32> {
    rows.iter().map(|r| f(r) as f32).collect()
}

fn i32s<T>(rows: &[T], f: impl Fn(&T) -> usize) -> Vec<i32> {
    rows.iter().map(|r| f(r) as i32).collect()
}

/// A built column's values.
enum Values {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    Str(Vec<Box<str>>),
}

/// Builds a table's columns in order: `est` pushes an estimate's theta
/// and `_se` beside each other, as the curve table lays its points out.
pub(super) struct Columns<'r, T> {
    rows: &'r [T],
    columns: Vec<(String, Values)>,
}

impl<'r, T> Columns<'r, T> {
    pub(super) fn new(rows: &'r [T]) -> Self {
        Self {
            rows,
            columns: Vec::new(),
        }
    }

    fn push(&mut self, name: &str, values: Values) -> &mut Self {
        self.columns.push((name.into(), values));
        self
    }

    pub(super) fn f(&mut self, name: &str, f: impl Fn(&T) -> f64) -> &mut Self {
        let values = Values::F32(f32s(self.rows, f));
        self.push(name, values)
    }

    /// A double column: for a value a float would round away, such as a p
    /// below single precision's smallest, which would become 0.
    pub(super) fn d(&mut self, name: &str, f: impl Fn(&T) -> f64) -> &mut Self {
        let values = Values::F64(self.rows.iter().map(f).collect());
        self.push(name, values)
    }

    pub(super) fn i(&mut self, name: &str, f: impl Fn(&T) -> usize) -> &mut Self {
        let values = Values::I32(i32s(self.rows, f));
        self.push(name, values)
    }

    /// An integer column given whole, a value per row.
    pub(super) fn v(&mut self, name: &str, values: Vec<usize>) -> &mut Self {
        debug_assert_eq!(values.len(), self.rows.len());
        let values = Values::I32(values.into_iter().map(|v| v as i32).collect());
        self.push(name, values)
    }

    pub(super) fn s(&mut self, name: &str, f: impl Fn(&'r T) -> &'r str) -> &mut Self {
        let values = Values::Str(self.rows.iter().map(|r| f(r).into()).collect());
        self.push(name, values)
    }

    /// `name` and `name_se`.
    pub(super) fn est(&mut self, name: &str, f: impl Fn(&T) -> Estimate) -> &mut Self {
        self.f(name, |r| f(r).theta)
            .f(&format!("{name}_se"), |r| f(r).se)
    }

    /// `name`, `name_se`, and the 95% interval as `name_lo` and `name_hi`.
    pub(super) fn est_interval(&mut self, name: &str, f: impl Fn(&T) -> Estimate) -> &mut Self {
        self.est(name, &f)
            .f(&format!("{name}_lo"), |r| f(r).interval().0)
            .f(&format!("{name}_hi"), |r| f(r).interval().1)
    }

    /// The table, keyed by a leading `row` column of row indices, as
    /// [`legume_numeric::matrix::parquet::write_named_table`] lays one out,
    /// with doubles besides.
    pub(super) fn write(&self, path: &str) -> anyhow::Result<()> {
        let names: Vec<Box<str>> = self
            .columns
            .iter()
            .map(|(n, _)| n.as_str().into())
            .collect();
        let types: Vec<ParquetType> = self
            .columns
            .iter()
            .map(|(_, v)| match v {
                Values::F32(_) => ParquetType::FLOAT,
                Values::F64(_) => ParquetType::DOUBLE,
                Values::I32(_) => ParquetType::INT32,
                Values::Str(_) => ParquetType::BYTE_ARRAY,
            })
            .collect();
        let rows: Vec<Box<str>> = (0..self.rows.len()).map(|i| i.to_string().into()).collect();
        let writer = ParquetWriter::new(
            path,
            (rows.len(), names.len()),
            (Some(&rows), Some(&names)),
            Some(&types),
            Some("row"),
        )?;
        let mut file = writer.get_writer()?;
        let mut group = file.next_row_group()?;
        parquet_add_bytearray(&mut group, writer.row_names_vec())?;
        for (_, values) in &self.columns {
            match values {
                Values::F32(d) => parquet_add_numeric_column(&mut group, d)?,
                Values::F64(d) => parquet_add_numeric_column(&mut group, d)?,
                Values::I32(d) => parquet_add_numeric_column(&mut group, d)?,
                Values::Str(d) => parquet_add_string_column(&mut group, d)?,
            }
        }
        group.close()?;
        file.close()?;
        Ok(())
    }
}
