//! `pinto srt-simulate`: resolve the parameters, simulate, write.

use super::panel::{donor_label, grid, read_coords};
use super::params::{find_scenario, SimParams, SCENARIOS};
use super::sim::Sim;
use super::write::{
    write_backend, write_coords, write_domains, write_donors, write_genes, write_spots,
    write_states, write_truth_run,
};
use crate::util::common::in_dir;
use log::info;
use serde_json::json;

#[derive(clap::Args, Debug)]
pub struct SimulateArgs {
    /// Named scenario to start from (see the long help for the list)
    #[arg(long, default_value = "unconfounded", long_help = scenario_help())]
    pub scenario: String,

    /// Change any parameter, `key=value`, repeatable (e.g. --set pve_donor=0.3,
    /// --set seed=2, --set nuisance_off=true)
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub set: Vec<String>,

    /// Spot coordinate files, comma-separated, used by the sections in turn;
    /// a square grid when none is given. Only positions are read.
    #[arg(long = "coord", short = 'c', alias = "coords", value_delimiter = ',')]
    pub coords: Vec<Box<str>>,

    /// Coordinate column names in the coordinate files
    #[arg(long, value_delimiter = ',', default_value = "row_um,col_um")]
    pub coord_column_names: Vec<Box<str>>,

    /// Output directory
    #[arg(short, long)]
    pub out: Box<str>,
}

fn scenario_help() -> String {
    let mut s =
        String::from("Named scenario to start from (a cell type is a cell state of the model):\n");
    for sc in SCENARIOS {
        s.push_str(&format!("  {}\n", sc.name));
        for line in wrapped(sc.about, 72) {
            s.push_str(&format!("      {line}\n"));
        }
    }
    s.push_str(
        "Any parameter can then be changed with --set key=value; the manifest records them all.\n\
         --set nuisance_off=true with the same seed draws the exact counterfactual.\n\
         --set n_unconfounded_donors=1 draws one donor from the seed (never the donor-only\n\
         domain's owner) and gives it no donor-wide, donor-by-cell-type or section shift.",
    );
    s
}

/// `text`'s words in lines of at most `width` characters (a longer word
/// alone on its line).
fn wrapped(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        match lines.last_mut() {
            Some(line) if line.chars().count() + 1 + word.chars().count() <= width => {
                line.push(' ');
                line.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    lines
}

/// A section's counts under the simulation folder `out`.
pub fn backend_path(out: &str, section: &str) -> String {
    in_dir(out, &format!("sections/{section}.zarr"))
}

/// A section's spot coordinates under `out`.
pub fn coords_path(out: &str, section: &str) -> String {
    in_dir(out, &format!("sections/{section}.coords.parquet"))
}

/// The manifest under `out`.
pub fn manifest_path(out: &str) -> String {
    in_dir(out, MANIFEST)
}

/// The manifest's file name: written last, it marks the simulation whole.
pub const MANIFEST: &str = "manifest.json";

/// The truth run under `out`, as `pinto view` opens it.
pub fn truth_run_path(out: &str) -> String {
    in_dir(out, "truth/truth.pinto.json")
}

/// The domains table under `out`.
pub fn domains_path(out: &str) -> String {
    in_dir(out, "truth/domains.parquet")
}

/// Where a simulation into `out` is written before it is moved to `out`
/// whole, so a stopped one leaves no `out` behind: `{out}.partial` beside it.
fn partial_path(out: &std::path::Path) -> std::path::PathBuf {
    let mut name = out.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    out.with_file_name(name)
}

/// The scenario's name, an earlier one read as its name, and the
/// parameters it and the `--set`s give.
pub fn resolve_params(args: &SimulateArgs) -> anyhow::Result<(&'static str, SimParams)> {
    let sc = find_scenario(&args.scenario)?;
    let p = (sc.params)().with_overrides(&args.set)?;
    p.validate()?;
    Ok((sc.name, p))
}

pub fn run_simulate(args: &SimulateArgs) -> anyhow::Result<()> {
    let (scenario, p) = resolve_params(args)?;
    let mut geometries = args
        .coords
        .iter()
        .map(|f| read_coords(f, &args.coord_column_names))
        .collect::<anyhow::Result<Vec<_>>>()?;
    if geometries.is_empty() {
        geometries.push(grid(p.grid_side, p.grid_spacing_um));
    }

    // its components, so a trailing slash names the same folder
    let done_path: std::path::PathBuf = std::path::Path::new(args.out.as_ref())
        .components()
        .collect();
    let done = done_path.to_string_lossy().into_owned();
    // a partial folder is only ever an earlier stopped run's
    let staged = partial_path(&done_path).to_string_lossy().into_owned();
    if std::path::Path::new(&staged).exists() {
        std::fs::remove_dir_all(&staged)?;
    }
    // written in place only over a folder that already holds something;
    // missing or empty, the run is staged and moved there whole. A folder
    // `--atomic-outputs` staged is written in place: it is moved at the end.
    let at_final = crate::util::staging::final_str(&done);
    let in_place = at_final != done || done_path.read_dir().is_ok_and(|mut d| d.next().is_some());
    let out = if in_place {
        done.as_str()
    } else {
        staged.as_str()
    };
    // made before simulating, so an out that cannot be made fails at once
    std::fs::create_dir_all(in_dir(out, "sections"))?;
    std::fs::create_dir_all(in_dir(out, "truth"))?;

    info!("{}", p.shares_summary());
    let sim = Sim::new(p, &geometries);
    let p = &sim.p;
    info!(
        "Simulating {} donors × {} sections, {} spots, {} genes",
        p.n_donors,
        p.sections_per_donor,
        sim.sections.iter().map(|s| s.n()).sum::<usize>(),
        p.n_genes
    );

    for (s, sec) in sim.sections.iter().enumerate() {
        let triplets = sim.counts(s);
        let nnz = triplets.len();
        let backend = backend_path(out, &sec.name);
        write_backend(&backend, &sim.genes.names, &sec.barcodes, triplets)?;
        write_coords(&coords_path(out, &sec.name), sec)?;
        info!("{}: {} spots, {} nonzero counts", sec.name, sec.n(), nnz);
    }
    write_spots(&in_dir(out, "truth/spots.parquet"), &sim)?;
    write_donors(&in_dir(out, "truth/donors.parquet"), &sim)?;
    write_domains(&domains_path(out), &sim)?;
    write_states(&in_dir(out, "truth/states.parquet"), &sim)?;
    write_genes(&in_dir(out, "truth/genes.parquet"), &sim)?;
    // relative to the truth run's manifest, so the outputs can be moved whole
    let data_files: Vec<String> = sim
        .sections
        .iter()
        .map(|sec| backend_path("..", &sec.name))
        .collect();
    write_truth_run(
        &in_dir(out, "truth"),
        &in_dir(&done, "truth"),
        &sim,
        &data_files,
    )?;

    let pve = sim.realised_pve();
    let names = super::params::TERMS;
    let manifest = json!({
        "command": "srt-simulate",
        "version": env!("CARGO_PKG_VERSION"),
        "scenario": scenario,
        "params": p,
        "unconfounded_donors": {
            "donors": sim.unconfounded_donors.iter().map(|&d| donor_label(d)).collect::<Vec<_>>(),
            "drawn": p.n_unconfounded_donors > 0,
        },
        "realised_pve_programme_genes": names.iter().zip(pve)
            .map(|(n, v)| (n.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
        "sections": sim.sections.iter().map(|sec| json!({
            "section": sec.name.as_ref(),
            "donor": donor_label(sec.donor).as_ref(),
            "n_spots": sec.n(),
        })).collect::<Vec<_>>(),
    });
    std::fs::write(manifest_path(out), serde_json::to_string_pretty(&manifest)?)?;
    if !in_place {
        if done_path.exists() {
            std::fs::remove_dir(&done_path)?;
        }
        std::fs::rename(out, &done_path)?;
    }
    info!("Wrote {at_final}/sections, {at_final}/truth and {at_final}/manifest.json");
    info!(
        "See the truth with: pinto view {}",
        truth_run_path(&at_final)
    );
    Ok(())
}
