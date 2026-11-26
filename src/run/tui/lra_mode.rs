//! `pinto lra` with no arguments: pinto run's screens for the analysis of
//! one fitted run. A file browser picks the run (its `.pinto.json`), and
//! `lra` and `lra simulate` read it (`--from`), each writing beside it, on
//! this computer: the forms leave out `--shard`, which splits a run for a
//! cluster. The run's tables open in pinto view's pairs layer.

use super::data::Pick;
use super::{script, App, Screen};
use legume_tui::browse::Browser;
use std::path::{Path, PathBuf};

/// Flags a run on this computer does not take: a shard of a run for a
/// cluster (`pinto lra merge` puts the shards together).
const CLUSTER_ONLY: &[&str] = &["shard"];

impl App {
    /// The screens of `pinto lra`: `lra` queued, the run to choose first.
    pub(super) fn new_lra(cli: clap::Command, start: PathBuf) -> anyhow::Result<Self> {
        let mut app = App::new(cli, start)?;
        app.lra_mode = true;
        app.rows.retain(|r| r.form.from_run);
        anyhow::ensure!(!app.rows.is_empty(), "this pinto has no lra");
        for r in &mut app.rows {
            r.form
                .fields
                .retain(|f| !CLUSTER_ONLY.contains(&f.long.as_str()));
        }
        app.rows[0].on = true;
        app.screen = Screen::Methods;
        // Results go beside the run: no output header to ask for.
        app.editor = None;
        app.browser = Some(Browser::open(app.browse_dir.clone(), Pick::Run, None));
        Ok(app)
    }

    /// The run chosen: every method reads it and writes beside it.
    pub(super) fn take_run(&mut self, manifest: &Path) {
        let shown = script::relative(manifest, &self.here)
            .to_string_lossy()
            .into_owned();
        let prefix = crate::util::metadata::run_prefix(&shown).to_string();
        for r in &mut self.rows {
            if let Some(f) = r.form.field_mut("from") {
                f.value.clone_from(&shown);
            }
            r.out.clone_from(&prefix);
            r.typed = true;
        }
        self.run = Some(manifest.to_path_buf());
        self.message = Some(format!(
            "{} chosen: ctrl+r reviews and runs; f chooses another run",
            crate::tui::name(manifest)
        ));
    }

    /// The tabs' title: the command the screens set up.
    pub(super) fn tab_title(&self) -> &'static str {
        if self.lra_mode {
            " pinto lra   "
        } else {
            " pinto run   "
        }
    }

    /// What the methods read, for the tabs: the run lra analyses, or the
    /// data files the fits take.
    pub(super) fn input_summary(&self) -> String {
        match (&self.run, self.lra_mode) {
            (Some(run), true) => crate::tui::name(run),
            (None, true) => "no run".into(),
            (_, false) => format!("{} data", self.pairs.len()),
        }
    }

    /// `f` on the methods screen: choose another run.
    pub(super) fn choose_run(&mut self) {
        self.browse(Pick::Run);
    }
}

#[cfg(test)]
#[path = "tests/lra_mode.rs"]
mod tests;
