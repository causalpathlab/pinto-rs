//! `pinto run`: set up fits in the terminal and run them.
//!
//! Pick data files with their coordinates and batch labels, queue one or
//! more methods, change their flags, check the exact commands, and run
//! them in turn. Each command is kept as `{out}.cmd.sh`, a script that
//! runs it again and refuses to run over a result already there.

pub(crate) mod form;
pub(crate) mod lra;
pub(crate) mod mcp;
pub(crate) mod script;
pub(crate) mod tui;

/// The methods `pinto run` sets up, in the order listed, each as typed:
/// the short alias of its subcommand.
pub const METHODS: [&str; 3] = ["lc", "cage", "dsvd"];

/// What `pinto run` also sets up on a fitted run (`--from` its
/// `.pinto.json`), after the fits: ligand-receptor analysis and its power
/// check.
pub const ON_RUNS: [&str; 2] = ["lra", "lra-simulate"];

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
