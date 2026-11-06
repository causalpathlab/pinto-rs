//! Building a parquet table one named column at a time.

use crate::lr_activity::ligand_effect::inference::Estimate;
use legume_numeric::matrix::parquet::{write_named_table, Column};

pub(super) fn f32s<T>(rows: &[T], f: impl Fn(&T) -> f64) -> Vec<f32> {
    rows.iter().map(|r| f(r) as f32).collect()
}

fn i32s<T>(rows: &[T], f: impl Fn(&T) -> usize) -> Vec<i32> {
    rows.iter().map(|r| f(r) as i32).collect()
}

/// Builds a table's columns in order: `est` pushes an estimate's theta
/// and `_se` beside each other, as the curve table lays its points out.
pub(super) struct Columns<'r, T> {
    rows: &'r [T],
    f32s: Vec<(String, Vec<f32>)>,
    i32s: Vec<(String, Vec<i32>)>,
    strs: Vec<(String, Vec<Box<str>>)>,
    /// (kind, index into the kind's list) in table order.
    order: Vec<(u8, usize)>,
}

impl<'r, T> Columns<'r, T> {
    pub(super) fn new(rows: &'r [T]) -> Self {
        Self {
            rows,
            f32s: Vec::new(),
            i32s: Vec::new(),
            strs: Vec::new(),
            order: Vec::new(),
        }
    }

    pub(super) fn f(&mut self, name: &str, f: impl Fn(&T) -> f64) -> &mut Self {
        self.order.push((0, self.f32s.len()));
        self.f32s.push((name.into(), f32s(self.rows, f)));
        self
    }

    pub(super) fn i(&mut self, name: &str, f: impl Fn(&T) -> usize) -> &mut Self {
        self.order.push((1, self.i32s.len()));
        self.i32s.push((name.into(), i32s(self.rows, f)));
        self
    }

    pub(super) fn s(&mut self, name: &str, f: impl Fn(&T) -> &str) -> &mut Self {
        self.order.push((2, self.strs.len()));
        self.strs
            .push((name.into(), self.rows.iter().map(|r| f(r).into()).collect()));
        self
    }

    /// `name` and `name_se`.
    pub(super) fn est(&mut self, name: &str, f: impl Fn(&T) -> Estimate) -> &mut Self {
        self.f(name, |r| f(r).theta)
            .f(&format!("{name}_se"), |r| f(r).se)
    }

    pub(super) fn write(&self, path: &str) -> anyhow::Result<()> {
        let table: Vec<(Box<str>, Column<'_>)> = self
            .order
            .iter()
            .map(|&(kind, k)| match kind {
                0 => (self.f32s[k].0.as_str().into(), Column::F32(&self.f32s[k].1)),
                1 => (self.i32s[k].0.as_str().into(), Column::I32(&self.i32s[k].1)),
                _ => (self.strs[k].0.as_str().into(), Column::Str(&self.strs[k].1)),
            })
            .collect();
        let names: Vec<Box<str>> = (0..self.rows.len()).map(|i| i.to_string().into()).collect();
        write_named_table(path, "row", &names, &table)
    }
}
