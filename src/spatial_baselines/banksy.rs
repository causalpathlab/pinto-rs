//! Placeholder: filled by the banksy port.

use super::input::SpatialInputArgs;

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(flatten)]
    pub input: SpatialInputArgs,
}

pub fn run(_args: &Args) -> anyhow::Result<()> {
    anyhow::bail!("banksy: not implemented yet")
}
