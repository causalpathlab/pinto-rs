mod color;
mod data;
mod draft;
mod fixture;
mod follow;
mod gene;
mod grid;
mod heatmap;
mod helpers;
mod index;
pub(super) mod lra;
mod lupin;
mod markers;
mod pdf;
mod render;
mod round;
mod saved;
mod scalebar;
mod structure;

/// `pinto view` with these arguments after the run's prefix.
pub(super) fn view_args(prefix: &str, rest: &[&str]) -> crate::view::ViewArgs {
    use clap::Parser;
    #[derive(Parser)]
    struct Args {
        #[command(flatten)]
        view: crate::view::ViewArgs,
    }
    Args::parse_from(["view", prefix].into_iter().chain(rest.iter().copied())).view
}
