//! `pinto run`: set up fits in the terminal and run them.
//!
//! Pick data files with their coordinates and batch labels, queue one or
//! more methods, change their flags, check the exact commands, and run
//! them in turn. Each command is kept as `{out}.cmd.sh`, a script that
//! runs it again and refuses to run over a result already there.

pub(crate) mod form;
pub(crate) mod lra;
pub(crate) mod mcp;
pub(crate) mod residual;
pub(crate) mod script;
pub(crate) mod sim;
pub(crate) mod tui;

/// The methods `pinto run` sets up, in the order listed, each as typed:
/// the short alias of its subcommand. pinto's own models first, then the
/// spatial baselines (model-level ports of published methods, for
/// comparison), which take the same data, coordinate and batch files and
/// write a run of the same shape.
pub const METHODS: [&str; 6] = ["lc", "cage", "dsvd", "banksy", "spatialde2", "stagate"];

/// The spatial baselines among [`METHODS`].
pub const BASELINES: [&str; 3] = ["banksy", "spatialde2", "stagate"];

/// What `pinto run` also sets up on a fitted run (`--from` its
/// `.pinto.json`), after the fits: ligand-receptor analysis and its power
/// check.
pub const ON_RUNS: [&str; 2] = ["lra", "lra-simulate"];

/// What `pinto run` also sets up on a fitted run, after those: its
/// residual, with a second fit on the residual's files.
pub const RESIDUAL: &str = "residual";

/// The methods the second stage fits the residual with, the first offered
/// first.
pub const STAGE_TWO: [&str; 3] = ["dsvd", "lc", "cage"];

#[derive(clap::Args, Debug)]
pub struct RunArgs {
    #[arg(help = "Folder the file browser starts in [default: .]")]
    pub dir: Option<std::path::PathBuf>,

    #[command(flatten)]
    pub session: legume_tui::cli::SessionArgs,
}

impl RunArgs {
    /// The session to run in. A folder named starts a new one; started
    /// with none, live sessions are offered first.
    pub fn session(&self) -> legume_tui::Options {
        self.session.options(self.dir.is_some())
    }

    /// Where the file browser starts.
    pub fn start(&self) -> std::path::PathBuf {
        self.dir.clone().unwrap_or_else(|| ".".into())
    }
}
