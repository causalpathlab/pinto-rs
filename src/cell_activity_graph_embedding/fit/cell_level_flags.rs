//! The flags of cage's that `cage-cell` refuses, hides or ignores, and the
//! one-level hierarchy it coarsens to.

use super::*;

/// What `cage-cell` does with a flag of cage's that has no meaning on cells.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OnCells {
    /// Off its help: a qualifier of a refused flag, refused with it, or a
    /// setting of cage's pair encoder, which cage-cell does not fit.
    Hide,
    /// Off its help, and warned of when given: it shapes only the HVG
    /// weighting of the projection that builds pseudobulks, or the
    /// pseudobulks themselves, which cage-cell builds only to seed its
    /// feature modules (no warning unless the run trains without them).
    /// (`--num-levels` and the batch
    /// matching shape the batch-effect estimate, which cage-cell makes.)
    Warn,
    /// Off its help, and refused when given: it picks from a hierarchy of
    /// pseudobulks.
    Refuse,
}

/// cage's flags that `cage-cell` hides, warns of or refuses, by argument id.
pub(crate) const CELL_LEVEL_FLAGS: [(&str, OnCells); 9] = [
    ("n_hvg", OnCells::Warn),
    ("feature_list_file", OnCells::Warn),
    ("must_train_features", OnCells::Warn),
    ("n_pseudobulk", OnCells::Warn),
    ("feature_embedding", OnCells::Refuse),
    ("feature_embedding_bias", OnCells::Hide),
    ("feature_embedding_mode", OnCells::Hide),
    ("feature_init_mode", OnCells::Hide),
    ("pair_encoder_steps", OnCells::Hide),
];

/// `arg`, hidden when `cage-cell` has no use for it (every flag of
/// [`CELL_LEVEL_FLAGS`]).
pub(crate) fn hide_on_cells(arg: clap::Arg) -> clap::Arg {
    if CELL_LEVEL_FLAGS.iter().any(|(id, _)| arg.get_id() == *id) {
        arg.hide(true)
    } else {
        arg
    }
}

/// The flags of [`CELL_LEVEL_FLAGS`] marked `what` that `cage-cell` (`cmd`,
/// its matches `m`) was given on its command line; defaults are not the
/// user's doing.
pub(crate) fn cell_level_given(
    cmd: &clap::Command,
    m: &clap::ArgMatches,
    what: OnCells,
) -> Vec<String> {
    let ids = CELL_LEVEL_FLAGS
        .iter()
        .filter(|(_, w)| *w == what)
        .map(|(id, _)| *id);
    crate::util::cli::given_on_cli(cmd, m, ids)
}

/// Refuse the flags `cage-cell` cannot combine with and warn of those that
/// do nothing on cells, from its command line (`cmd`, its matches `m`).
/// Returns the flags given that shape only the pseudobulks seeding
/// `--feature-modules auto`: they will have done nothing if the fit finds
/// no module layer, which only the fit can say.
pub(crate) fn check_cell_level(
    cmd: &clap::Command,
    m: &clap::ArgMatches,
) -> anyhow::Result<Vec<String>> {
    let refused = cell_level_given(cmd, m, OnCells::Refuse);
    anyhow::ensure!(
        refused.is_empty(),
        "cage-cell: {} need a hierarchy of pseudobulks to pick from, which cage-cell \
         does not train on",
        refused.join(", ")
    );
    let no_ops = cell_level_no_ops(cmd, m);
    if !no_ops.is_empty() {
        log::warn!(
            "cage-cell: {} shape only the HVG projection or the pseudobulks, which \
             cage-cell builds only to seed its feature modules, off in this run; they \
             do nothing here",
            no_ops.join(", ")
        );
    }
    Ok(match cell_level_plan(m) {
        Ok(ModulePlan::Auto) => cell_level_given(cmd, m, OnCells::Warn),
        _ => Vec::new(),
    })
}

/// The [`ModulePlan`] of `cage-cell`'s matches `m`.
fn cell_level_plan(m: &clap::ArgMatches) -> anyhow::Result<ModulePlan> {
    use clap::FromArgMatches;
    CellActivityGraphEmbeddingArgs::from_arg_matches(m)?.module_plan()
}

/// The flags given to `cage-cell` (`cmd`, its matches `m`) that do nothing
/// in this run: those marked [`OnCells::Warn`], when the run's
/// [`ModulePlan`] is off. With modules (the default) they shape the
/// pseudobulks that seed them. A plan the run will refuse names none.
pub(crate) fn cell_level_no_ops(cmd: &clap::Command, m: &clap::ArgMatches) -> Vec<String> {
    match cell_level_plan(m) {
        Ok(ModulePlan::Off) => cell_level_given(cmd, m, OnCells::Warn),
        _ => Vec::new(),
    }
}

/// The "hierarchy" of a cell-level fit: every cell its own unit at both
/// levels, so the pair frame's super edges are the deduplicated cell pairs
/// (within batch), the rows cage-cell trains; the one chain level is the
/// cell too, so no pair straddles a group. Nothing is coarsened.
pub(crate) fn cell_level_hierarchy(
    n_cells: usize,
    pairs: &[(usize, usize)],
) -> crate::util::graph_coarsen::MultiLevelCoarsenResult {
    let cells: Vec<usize> = (0..n_cells).collect();
    let (pair_to_sample, n_samples) =
        crate::util::graph_coarsen::cell_labels_to_pair_samples(&cells, pairs);
    crate::util::graph_coarsen::MultiLevelCoarsenResult {
        all_pair_to_sample: vec![pair_to_sample.clone(), pair_to_sample],
        all_num_samples: vec![n_samples; 2],
        all_cell_labels: vec![cells.clone(), cells],
    }
}
