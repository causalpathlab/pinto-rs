//! Output: one backend and one coordinate file per section, the truth
//! tables, and a manifest.
//!
//! ```text
//! {out}/sections/S1.zarr            counts, genes × spots
//! {out}/sections/S1.coords.parquet  spot, row_um, col_um
//! {out}/truth/spots.parquet         per spot: section, donor, domain and its owner, state, depth, mixture;
//!                                   `cell` as pinto names it (`barcode@section`, or the
//!                                   bare barcode with one section), to join onto fits
//! {out}/truth/domains.parquet       owner (empty: shared), composition
//! {out}/truth/states.parquet        owner (empty: shared)
//! {out}/truth/genes.parquet         baseline, dispersion, programmes
//! {out}/truth/truth.pinto.json      the truth as a run `pinto view` opens:
//!                                   truth.cells.parquet and the domain, state
//!                                   and donor levels (truth.*.propensity.parquet)
//! {out}/manifest.json               every parameter, and the realised PVE
//! ```

use super::panel::{donor_label, Section};
use super::sim::Sim;
use super::truth::Info;
use crate::util::common::in_dir;
use data_beans::sparse_io_vector::stacked_column_name;
use legume_numeric::matrix::parquet::{write_named_table, Column};

/// The coordinate columns of every coordinate table written here.
pub const COORD_COLUMNS: [&str; 2] = ["row_um", "col_um"];

fn f32s(v: impl Iterator<Item = f64>) -> Vec<f32> {
    v.map(|x| x as f32).collect()
}

/// Every spot as `(section, spot)`, in section order, with its cell name as
/// pinto names it: `{barcode}@{section}`, or the bare barcode when there is
/// one section (pinto adds the file stem only with several data files).
pub fn spot_index(sim: &Sim) -> (Vec<(usize, usize)>, Vec<Box<str>>) {
    let n = sim.sections.len();
    // as a fit names the cells of the section files `S*.zarr`
    let files: Vec<String> = sim
        .sections
        .iter()
        .map(|s| format!("{}.zarr", s.name))
        .collect();
    let spots: Vec<(usize, usize)> = sim
        .sections
        .iter()
        .enumerate()
        .flat_map(|(s, sec)| (0..sec.n()).map(move |i| (s, i)))
        .collect();
    let cells = spots
        .iter()
        .map(|&(s, i)| {
            stacked_column_name(&sim.sections[s].barcodes[i], &files[s], n)
                .expect("a section file has a name")
        })
        .collect();
    (spots, cells)
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
            (COORD_COLUMNS[0].into(), Column::F32(&x)),
            (COORD_COLUMNS[1].into(), Column::F32(&y)),
        ],
    )
}

pub fn write_spots(path: &str, sim: &Sim) -> anyhow::Result<()> {
    let t = &sim.truth;
    let (spots, cell) = spot_index(sim);
    let strs = |f: &dyn Fn(usize, usize) -> Box<str>| -> Vec<Box<str>> {
        spots.iter().map(|&(s, i)| f(s, i)).collect()
    };
    let nums =
        |f: &dyn Fn(usize, usize) -> f64| -> Vec<f32> { f32s(spots.iter().map(|&(s, i)| f(s, i))) };
    let sec = |s: usize| &sim.sections[s];
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
        (COORD_COLUMNS[0].into(), Column::F32(&x)),
        (COORD_COLUMNS[1].into(), Column::F32(&y)),
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

/// The truth as a run `pinto view` opens like a fit: `{dir}/truth.pinto.json`
/// lists a cells table and three levels, `domain` (one-hot), `state` (each
/// spot's planted mixture) and `donor` (one-hot), with the section backends
/// as its data files so genes can be shown too. Community `C{k}` of a level
/// is named in the manifest by the labels of `domains.parquet`,
/// `states.parquet` or the donors, and the viewer shows those names.
/// `data_files` are relative to `dir`, so the outputs can be moved whole.
/// The files are written in `dir` and the manifest names them in `named`,
/// where `dir` is to be moved once written.
pub fn write_truth_run(
    dir: &str,
    named: &str,
    sim: &Sim,
    data_files: &[String],
) -> anyhow::Result<()> {
    use crate::link_community::outputs::write_propensity_matrix;
    use crate::util::common::Mat;
    use crate::util::metadata::{now_secs, LevelInfo, OutputFiles, PintoMetadata};
    use crate::util::parquet_io::{cells_table_path, write_cells_table};
    let prefix = in_dir(dir, "truth");
    let recorded = in_dir(named, "truth");
    let t = &sim.truth;
    let (spots, cells) = spot_index(sim);
    let mut coords = Mat::zeros(cells.len(), 2);
    for (r, &(s, i)) in spots.iter().enumerate() {
        let (x, y) = sim.sections[s].coords[i];
        coords[(r, 0)] = x as f32;
        coords[(r, 1)] = y as f32;
    }
    let sections: Vec<Box<str>> = spots
        .iter()
        .map(|&(s, _)| sim.sections[s].name.clone())
        .collect();
    let coord_boxed: Vec<Box<str>> = COORD_COLUMNS.iter().map(|&c| c.into()).collect();
    write_cells_table(
        &prefix,
        &cells,
        &coords,
        &coord_boxed,
        &sections,
        &vec![true; cells.len()],
    )?;

    // a level: each spot's weight on each of its communities, in the shared
    // propensity schema, named in the manifest
    let level =
        |j: usize, tag: &str, names: Vec<String>, weight: &dyn Fn(usize, usize, usize) -> f64| {
            let mut m = Mat::zeros(cells.len(), names.len());
            for (r, &(s, i)) in spots.iter().enumerate() {
                for c in 0..names.len() {
                    m[(r, c)] = weight(s, i, c) as f32;
                }
            }
            write_propensity_matrix(&format!("{prefix}.{tag}"), &m, &cells).map(|()| LevelInfo {
                tag: tag.to_string(),
                level_index: j,
                propensity: format!("{recorded}.{tag}.propensity.parquet"),
                link_community: None,
                feature_community: None,
                entropy_present: Some(true),
                community_names: Some(names),
            })
        };
    let one_hot = |a: usize, b: usize| if a == b { 1.0 } else { 0.0 };
    let labels = |infos: &[Info]| infos.iter().map(|x| x.label.to_string()).collect();
    let donors = (0..sim.p.n_donors)
        .map(|d| donor_label(d).to_string())
        .collect();
    let levels = vec![
        level(0, "domain", labels(&t.domains), &|s, i, c| {
            one_hot(t.domain[s][i], c)
        })?,
        level(1, "state", labels(&t.states), &|s, i, c| t.mixture[s][i][c])?,
        level(2, "donor", donors, &|s, _, c| {
            one_hot(sim.sections[s].donor, c)
        })?,
    ];
    let meta = PintoMetadata {
        command: "srt-simulate".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        timestamp: now_secs(),
        prefix: recorded.clone(),
        data_files: Some(data_files.to_vec()),
        n_cells: cells.len(),
        n_features: sim.p.n_genes,
        outputs: OutputFiles {
            cells: Some(cells_table_path(&recorded)),
            coord_columns: Some(COORD_COLUMNS.iter().map(|c| c.to_string()).collect()),
            ..Default::default()
        },
        levels: Some(levels),
        ..Default::default()
    };
    meta.write(std::path::Path::new(&format!("{prefix}.pinto.json")))
}
