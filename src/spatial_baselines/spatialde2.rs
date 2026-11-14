//! Placeholder: filled by the spatialde2 port.

use super::input::SpatialInputArgs;

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(flatten)]
    pub input: SpatialInputArgs,
}

pub fn run(_args: &Args) -> anyhow::Result<()> {
    anyhow::bail!("spatialde2: not implemented yet")
}
