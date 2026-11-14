//! Placeholder: filled by the stagate port.

use super::input::SpatialInputArgs;

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(flatten)]
    pub input: SpatialInputArgs,
}

pub fn run(_args: &Args) -> anyhow::Result<()> {
    anyhow::bail!("stagate: not implemented yet")
}
