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
pub(crate) mod sim;
pub(crate) mod tui;

/// The methods `pinto run` sets up, in the order listed, each as typed:
/// the short alias of its subcommand. pinto's own models first (lc and
/// dsvd, then cage and its cell-level form), then the spatial baselines
/// (model-level ports of published methods, for comparison, then
/// their `-pb` forms fitted on super-cells), which take
/// the same data, coordinate and batch files and write a run of the same
/// shape.
pub const METHODS: [&str; 12] = [
    "lc",
    "dsvd",
    "cage",
    "cage-cell",
    "spatialde2",
    "stagate",
    "spatialpca",
    "mefisto",
    "spatialde2-pb",
    "stagate-pb",
    "spatialpca-pb",
    "mefisto-pb",
];

/// pinto's own models among [`METHODS`].
pub const MODELS: &[&str] = METHODS.split_at(N_MODELS).0;

/// The spatial baselines among [`METHODS`].
pub const BASELINES: &[&str] = METHODS.split_at(N_MODELS).1;

const N_MODELS: usize = 4;

/// What `pinto run` also sets up on a fitted run (`--from` its
/// `.pinto.json`), after the fits: ligand-receptor analysis and its power
/// check.
pub const ON_RUNS: [&str; 2] = ["lra", "lra simulate"];

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
