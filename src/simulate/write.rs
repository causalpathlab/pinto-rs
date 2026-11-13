//! Output: one backend and one coordinate file per section, the truth
//! tables, and a manifest.
//!
//! ```text
//! {out}/sections/S1.zarr            counts, genes × spots
//! {out}/sections/S1.coords.parquet  spot, row_um, col_um
//! {out}/truth/spots.parquet         per spot: section, donor, domain and its owner, state, depth, mixture
//! {out}/truth/domains.parquet       owner (empty: shared), composition
//! {out}/truth/states.parquet        owner (empty: shared)
//! {out}/truth/genes.parquet         baseline, dispersion, programmes
//! {out}/manifest.json               every parameter, and the realised PVE
//! ```

use super::panel::{donor_label, Section};
use super::sim::Sim;
use super::truth::Info;
use legume_numeric::matrix::parquet::{write_named_table, Column};

fn f32s(v: impl Iterator<Item = f64>) -> Vec<f32> {
    v.map(|x| x as f32).collect()
}

fn owner(info: &Info) -> Box<str> {
    info.owner.map_or("".into(), donor_label)
}

/// A zarr backend of `genes × barcodes` counts from `(gene, spot, count)`.
pub fn write_backend(
    path: &str,
    genes: &[Box<str>],
    barcodes: &[Box<str>],
    triplets: Vec<(u64, u64, f32)>,
) -> anyhow::Result<()> {
    use data_beans::sparse_io::{create_sparse_from_triplets_owned, SparseIoBackend};
    let nnz = triplets.len();
    let mut backend = create_sparse_from_triplets_owned(
        triplets,
        (genes.len(), barcodes.len(), nnz),
        Some(path),
        Some(&SparseIoBackend::Zarr),
    )?;
    backend.register_row_names_vec(genes);
    backend.register_column_names_vec(barcodes);
    Ok(())
}

pub fn write_coords(path: &str, sec: &Section) -> anyhow::Result<()> {
    let x = f32s(sec.coords.iter().map(|c| c.0));
    let y = f32s(sec.coords.iter().map(|c| c.1));
    write_named_table(
        path,
        "spot",
        &sec.barcodes,
        &[
            ("row_um".into(), Column::F32(&x)),
            ("col_um".into(), Column::F32(&y)),
        ],
    )
}

pub fn write_spots(path: &str, sim: &Sim) -> anyhow::Result<()> {
    let t = &sim.truth;
    let spots: Vec<(usize, usize)> = sim
        .sections
        .iter()
        .enumerate()
        .flat_map(|(s, sec)| (0..sec.n()).map(move |i| (s, i)))
        .collect();
    let strs = |f: &dyn Fn(usize, usize) -> Box<str>| -> Vec<Box<str>> {
        spots.iter().map(|&(s, i)| f(s, i)).collect()
    };
    let nums =
        |f: &dyn Fn(usize, usize) -> f64| -> Vec<f32> { f32s(spots.iter().map(|&(s, i)| f(s, i))) };
    let sec = |s: usize| &sim.sections[s];
    let cell = strs(&|s, i| format!("{}@{}", sec(s).barcodes[i], sec(s).name).into());
    let barcode = strs(&|s, i| sec(s).barcodes[i].clone());
    let section = strs(&|s, _| sec(s).name.clone());
    let donor = strs(&|s, _| donor_label(sec(s).donor));
    let domain = strs(&|s, i| t.domains[t.domain[s][i]].label.clone());
    let domain_owner = strs(&|s, i| owner(&t.domains[t.domain[s][i]]));
    let state = strs(&|s, i| t.states[t.dominant[s][i]].label.clone());
    let depth = nums(&|s, i| sim.depth[s][i]);
    let x = nums(&|s, i| sec(s).coords[i].0);
    let y = nums(&|s, i| sec(s).coords[i].1);
    let mix: Vec<Vec<f32>> = (0..t.n_states())
        .map(|c| nums(&|s, i| t.mixture[s][i][c]))
        .collect();
    let mut cols: Vec<(Box<str>, Column)> = vec![
        ("barcode".into(), Column::Str(&barcode)),
        ("section".into(), Column::Str(&section)),
        ("donor".into(), Column::Str(&donor)),
        ("domain".into(), Column::Str(&domain)),
        ("domain_owner".into(), Column::Str(&domain_owner)),
        ("state".into(), Column::Str(&state)),
        ("depth".into(), Column::F32(&depth)),
        ("row_um".into(), Column::F32(&x)),
        ("col_um".into(), Column::F32(&y)),
    ];
    for (st, m) in t.states.iter().zip(&mix) {
        cols.push((format!("w_{}", st.label).into(), Column::F32(m)));
    }
    write_named_table(path, "cell", &cell, &cols)
}

pub fn write_domains(path: &str, sim: &Sim) -> anyhow::Result<()> {
    let t = &sim.truth;
    let labels: Vec<Box<str>> = t.domains.iter().map(|d| d.label.clone()).collect();
    let owners: Vec<Box<str>> = t.domains.iter().map(owner).collect();
    let comp: Vec<Vec<f32>> = (0..t.n_states())
        .map(|c| f32s(t.composition.iter().map(|k| k[c])))
        .collect();
    let mut cols: Vec<(Box<str>, Column)> = vec![("owner_donor".into(), Column::Str(&owners))];
    for (st, v) in t.states.iter().zip(&comp) {
        cols.push((format!("comp_{}", st.label).into(), Column::F32(v)));
    }
    write_named_table(path, "domain", &labels, &cols)
}

pub fn write_states(path: &str, sim: &Sim) -> anyhow::Result<()> {
    let t = &sim.truth;
    let labels: Vec<Box<str>> = t.states.iter().map(|s| s.label.clone()).collect();
    let owners: Vec<Box<str>> = t.states.iter().map(owner).collect();
    write_named_table(
        path,
        "state",
        &labels,
        &[("owner_donor".into(), Column::Str(&owners))],
    )
}

pub fn write_genes(path: &str, sim: &Sim) -> anyhow::Result<()> {
    let g = &sim.genes;
    let a = f32s(g.a.iter().copied());
    let disp = f32s(g.dispersion.iter().copied());
    let progs: Vec<Vec<f32>> = g
        .programme
        .iter()
        .map(|v| f32s(v.iter().copied()))
        .collect();
    let mut cols: Vec<(Box<str>, Column)> = vec![
        ("log_baseline".into(), Column::F32(&a)),
        ("dispersion".into(), Column::F32(&disp)),
    ];
    for (st, v) in sim.truth.states.iter().zip(&progs) {
        cols.push((format!("programme_{}", st.label).into(), Column::F32(v)));
    }
    write_named_table(path, "gene", &g.names, &cols)
}
