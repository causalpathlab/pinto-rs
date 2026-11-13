//! `pinto srt-simulate`: resolve the parameters, simulate, write.

use super::panel::{donor_label, grid, read_coords};
use super::params::{SimParams, SCENARIOS};
use super::sim::Sim;
use super::write::{
    write_backend, write_coords, write_domains, write_genes, write_spots, write_states,
};
use log::info;
use serde_json::json;

#[derive(clap::Args, Debug)]
pub struct SimulateArgs {
    /// Named scenario to start from (see the long help for the list)
    #[arg(long, default_value = "L0", long_help = scenario_help())]
    pub scenario: String,

    /// Change any parameter, `key=value`, repeatable (e.g. --set pve_donor=0.3,
    /// --set seed=2, --set nuisance_off=true)
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub set: Vec<String>,

    /// Spot coordinate files, comma-separated, used by the sections in turn;
    /// a square grid when none is given. Only positions are read.
    #[arg(long, value_delimiter = ',')]
    pub coords: Vec<Box<str>>,

    /// Coordinate column names in the coordinate files
    #[arg(long, value_delimiter = ',', default_value = "row_um,col_um")]
    pub coord_column_names: Vec<Box<str>>,

    /// Output directory
    #[arg(short, long)]
    pub out: Box<str>,
}

fn scenario_help() -> String {
    let mut s = String::from("Named scenario to start from:\n");
    for sc in SCENARIOS {
        s.push_str(&format!("  {:<11} {}\n", sc.name, sc.about));
    }
    s.push_str(
        "Any parameter can then be changed with --set key=value; the manifest records them all.\n\
         --set nuisance_off=true with the same seed draws the exact counterfactual.",
    );
    s
}

pub fn resolve_params(args: &SimulateArgs) -> anyhow::Result<SimParams> {
    let p = SimParams::scenario(&args.scenario)?.with_overrides(&args.set)?;
    p.validate()?;
    Ok(p)
}

pub fn run_simulate(args: &SimulateArgs) -> anyhow::Result<()> {
    let p = resolve_params(args)?;
    let mut geometries = args
        .coords
        .iter()
        .map(|f| read_coords(f, &args.coord_column_names))
        .collect::<anyhow::Result<Vec<_>>>()?;
    if geometries.is_empty() {
        geometries.push(grid(p.grid_side, p.grid_spacing_um));
    }
    let sim = Sim::new(p, &geometries);
    let p = &sim.p;
    info!(
        "Simulating {} donors × {} sections, {} spots, {} genes",
        p.n_donors,
        p.sections_per_donor,
        sim.sections.iter().map(|s| s.n()).sum::<usize>(),
        p.n_genes
    );

    let out = args.out.as_ref();
    std::fs::create_dir_all(format!("{out}/sections"))?;
    std::fs::create_dir_all(format!("{out}/truth"))?;
    for (s, sec) in sim.sections.iter().enumerate() {
        let triplets = sim.counts(s);
        let nnz = triplets.len();
        write_backend(
            &format!("{out}/sections/{}.zarr", sec.name),
            &sim.genes.names,
            &sec.barcodes,
            triplets,
        )?;
        write_coords(&format!("{out}/sections/{}.coords.parquet", sec.name), sec)?;
        info!("{}: {} spots, {} nonzero counts", sec.name, sec.n(), nnz);
    }
    write_spots(&format!("{out}/truth/spots.parquet"), &sim)?;
    write_domains(&format!("{out}/truth/domains.parquet"), &sim)?;
    write_states(&format!("{out}/truth/states.parquet"), &sim)?;
    write_genes(&format!("{out}/truth/genes.parquet"), &sim)?;

    let pve = sim.realised_pve();
    let names = ["type", "donor", "inter", "section", "noise"];
    let manifest = json!({
        "command": "srt-simulate",
        "version": env!("CARGO_PKG_VERSION"),
        "scenario": args.scenario,
        "params": p,
        "realised_pve_programme_genes": names.iter().zip(pve)
            .map(|(n, v)| (n.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
        "sections": sim.sections.iter().map(|sec| json!({
            "section": sec.name.as_ref(),
            "donor": donor_label(sec.donor).as_ref(),
            "n_spots": sec.n(),
        })).collect::<Vec<_>>(),
    });
    std::fs::write(
        format!("{out}/manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    info!("Wrote {out}/sections, {out}/truth and {out}/manifest.json");
    Ok(())
}
