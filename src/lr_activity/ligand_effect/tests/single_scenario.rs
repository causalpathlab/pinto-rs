//! `pinto lra-simulate` from data to tables on a small tissue: the built-in
//! grid, pinned, and the single-scenario mode a sweep runs one call per
//! point of.

use super::super::memory::memory_budget;
use super::super::output::{lra_simulate_path, lra_simulate_sidecar_path};
use super::super::pipeline::{alternate_ranks, run_grid, run_simulate_one};
use super::super::reference::{with_test_root, write_empty_reference, Species};
use super::super::simulate::{
    ConfoundShape, ConfounderKind, GridOpts, Scenario, ScenarioKeys, SingleScenario,
};
use super::super::{RunInputs, Settings};
use crate::lr_activity::args::{ReferenceSpecies, RegionJackknife};
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use legume_numeric::matrix::parquet::read_table_columns;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};
use std::path::{Path, PathBuf};

/// Cells per side of each of the two sections.
const SIDE: usize = 20;
const SECTIONS: usize = 2;
/// Distinct ligands, each paired with the one receptor.
const LIGANDS: usize = 6;
const TARGETS: usize = 8;
const IDENTITY_GENES: usize = 30;

fn gene(g: usize) -> Box<str> {
    format!("GENE{}", g + 1).into()
}

/// GENE1 to GENE6 the ligands, GENE7 their receptor, GENE8 to GENE15 the
/// targets, the rest the receivers' identity.
fn n_genes() -> usize {
    LIGANDS + 1 + TARGETS + IDENTITY_GENES
}

fn n_cells() -> usize {
    SECTIONS * SIDE * SIDE
}

/// A cell's row and column within its section.
fn place(c: usize) -> (usize, usize) {
    let i = c % (SIDE * SIDE);
    (i / SIDE, i % SIDE)
}

/// The identity group (by rows), which the identity genes and the receptor
/// follow.
fn group(c: usize) -> usize {
    place(c).0 / 7
}

/// Counts per gene and cell: each ligand more abundant than the one before,
/// so their abundance ranks are their order; the receptor rare in the last
/// group; targets everywhere; identity genes high in their group.
fn triplets() -> Vec<(u64, u64, f32)> {
    let mut rng = SmallRng::seed_from_u64(5);
    let mut out = Vec::new();
    for c in 0..n_cells() {
        for g in 0..n_genes() {
            let rate = match g {
                g if g < LIGANDS => 0.2 + 0.15 * g as f64,
                g if g == LIGANDS && group(c) == 2 => 0.02,
                g if g == LIGANDS => 0.8,
                g if g <= LIGANDS + TARGETS => 1.0,
                g => {
                    let k = g - LIGANDS - 1 - TARGETS;
                    0.3 + 2.0 * f64::from(u8::from(k % 3 == group(c)))
                }
            };
            let x = Poisson::new(rate).unwrap().sample(&mut rng);
            if x > 0.0 {
                out.push((g as u64, c as u64, x as f32));
            }
        }
    }
    out
}

/// What a run reads, written once under a test's folder.
struct Fixture {
    data: SparseIoVec,
    axis: FeatureAxis,
    pairs: Vec<(usize, usize)>,
    coords: Vec<Option<(f32, f32)>>,
    section: Vec<Box<str>>,
    pairs_file: String,
    targets: [Box<str>; 1],
    reference: PathBuf,
}

fn fixture(dir: &Path) -> Fixture {
    let n = n_cells();
    let t = triplets();
    let path = dir.join("d.zarr");
    let mut backend = create_sparse_from_triplets(
        &t,
        (n_genes(), n, t.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )
    .unwrap();
    let names: Vec<Box<str>> = (0..n_genes()).map(gene).collect();
    backend.register_row_names_vec(&names);
    let cells: Vec<Box<str>> = (0..n).map(|c| format!("c{c}").into()).collect();
    backend.register_column_names_vec(&cells);
    let mut data = SparseIoVec::new();
    data.push(Arc::from(backend), None).unwrap();
    let axis = FeatureAxis::resolve_or_identity(&names).unwrap();

    let mut pairs = Vec::new();
    for c in 0..n {
        let (r, col) = place(c);
        if col + 1 < SIDE {
            pairs.push((c, c + 1));
        }
        if r + 1 < SIDE {
            pairs.push((c, c + SIDE));
        }
    }
    let coords = (0..n)
        .map(|c| {
            let (r, col) = place(c);
            let shift = (c / (SIDE * SIDE) * 10 * SIDE) as f32;
            Some((col as f32 * 10.0 + shift, r as f32 * 10.0))
        })
        .collect();
    let section = (0..n)
        .map(|c| format!("s{}", c / (SIDE * SIDE)).into())
        .collect();

    let reference = dir.join("reference");
    write_empty_reference(&reference, Species::Human).unwrap();
    let pairs_path = dir.join("pairs.tsv");
    let mut p = String::from("ligand\treceptor\n");
    for l in 0..LIGANDS {
        p += &format!("{}\t{}\n", gene(l), gene(LIGANDS));
    }
    std::fs::write(&pairs_path, p).unwrap();
    let targets_path = dir.join("targets.tsv");
    let mut t = String::from("receptor\ttarget\tweight\n");
    for g in LIGANDS + 1..=LIGANDS + TARGETS {
        t += &format!("{}\t{}\t1\n", gene(LIGANDS), gene(g));
    }
    std::fs::write(&targets_path, t).unwrap();
    let s = |p: &Path| p.to_string_lossy().into_owned();
    Fixture {
        data,
        axis,
        pairs,
        coords,
        section,
        pairs_file: s(&pairs_path),
        targets: [s(&targets_path).into_boxed_str()],
        reference,
    }
}

impl Fixture {
    /// The run's inputs, writing under `out` and recording in `manifest`.
    fn inputs<'a>(&'a self, out: &'a str, manifest: &'a Path, seed: u64) -> RunInputs<'a> {
        RunInputs {
            settings: Settings {
                out,
                manifest,
                lr_pairs: Some(&self.pairs_file),
                targets: &self.targets,
                species: ReferenceSpecies::Human,
                seed,
                block_size: None,
                only: &[],
                cell_types: None,
                region: None,
                plant_region: None,
                region_jackknife: RegionJackknife::NoRefit,
                marginal: None,
            },
            data: &self.data,
            axis: &self.axis,
            pairs: &self.pairs,
            cell_section: &self.section,
            coords: &self.coords,
            cell_type: None,
            region: None,
        }
    }
}

/// A fresh manifest for a run into `dir`.
fn manifest(dir: &Path) -> PathBuf {
    let path = dir.join("run.pinto.json");
    crate::util::metadata::PintoMetadata {
        command: "lra".into(),
        ..Default::default()
    }
    .write(&path)
    .unwrap();
    path
}

/// A small grid of the built-in kind: no effect and one planted, each
/// unconfounded and under bumps and the real gene (same and opposite sign),
/// on two pairs.
fn small_grid() -> GridOpts {
    GridOpts {
        reps: 1,
        pairs: 2,
        confounders: vec![ConfounderKind::Bumps, ConfounderKind::RealGene],
        scales: vec![32.0],
        shapes: vec![ConfoundShape::Linear],
        marker: false,
        heterogeneous: false,
        ..GridOpts::default()
    }
}

/// The built-in grid's table, pinned: the row count, which pairs ran, and
/// position-weighted sums over the estimate, its SE, its truth and whether
/// it was estimable. A change to the grid, to its
/// seeds or to what one run draws moves them. Values match to a part in
/// 1e5, well above the f32 rounding of the table.
#[test]
fn the_built_in_grid_is_pinned() {
    let dir = tempfile::tempdir().unwrap();
    let f = fixture(dir.path());
    let m = manifest(dir.path());
    let out = dir.path().join("o").to_string_lossy().into_owned();
    with_test_root(&f.reference, || {
        run_grid(&f.inputs(&out, &m, 42), small_grid())
    })
    .unwrap();
    let path = lra_simulate_path(&out);
    let (s, x) = read_table_columns(
        &path,
        &["ligand"],
        &["estimate", "se", "truth", "estimable"],
    )
    .unwrap();
    let mut ligands: Vec<&str> = s[0].iter().map(|l| &**l).collect();
    assert_eq!(ligands.len(), 20);
    ligands.dedup();
    assert_eq!(ligands, ["GENE6", "GENE5"], "the most abundant ligands");
    // Each row weighted by its position, so rows trading values move it.
    let fingerprint = |v: &[f64]| {
        v.iter()
            .enumerate()
            .filter(|(_, x)| x.is_finite())
            .map(|(i, x)| (i + 1) as f64 * x)
            .sum::<f64>()
    };
    let got = x.iter().map(|c| fingerprint(c)).collect::<Vec<_>>();
    let want = [
        6.088_119_713_589_549,
        8.306_056_205_183_268,
        3.749_613_553_285_598_8,
        206.0,
    ];
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-5 * w.abs(), "column {k}: {g} != {w}");
    }
}

// * The single-scenario mode

/// One single-scenario call with `sets` on `planted` and `null` pairs, on
/// a pool of `threads`, into `{dir}/{name}`, handed the manifest `m`; its
/// output prefix, or the call's error.
fn single_call(
    f: &Fixture,
    (dir, m): (&Path, &Path),
    name: &str,
    sets: &[&str],
    pairs: (usize, usize),
    threads: usize,
) -> anyhow::Result<String> {
    let out = dir.join(name).to_string_lossy().into_owned();
    let sets: Vec<String> = sets.iter().map(|s| s.to_string()).collect();
    let one = SingleScenario::new(&sets, pairs.0, pairs.1)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        with_test_root(&f.reference, || {
            run_simulate_one(&f.inputs(&out, m, 42), &one)
        })
    })?;
    Ok(out)
}

/// A confounded point with an effect planted: the keys a sweep varies.
const CONFOUNDED: [&str; 4] = [
    "ligand_field_pve=0.25",
    "response_field_pve=0.25",
    "confounder=bumps",
    "confound_scale=32",
];

fn with_effect(effect: &str) -> Vec<&str> {
    std::iter::once(effect).chain(CONFOUNDED).collect()
}

/// Columns compared between calls: the scores and what they rest on.
const SCORED: [&str; 16] = [
    "effect_pve",
    "truth",
    "truth_exposed",
    "estimable",
    "estimate",
    "se",
    "df",
    "marginal_cor",
    "unadjusted_diff",
    "effect_ipw",
    "celltype_p",
    "moran_p",
    "contact_p",
    "coexpr_p",
    "misclassified",
    "p",
];

/// Even abundance ranks planted, odd ranks null, as many of each as asked
/// and there are.
#[test]
fn labels_alternate_by_abundance_rank() {
    let t = |r: &[(usize, bool)]| r.to_vec();
    assert_eq!(
        alternate_ranks(10, 2, 2),
        t(&[(0, true), (1, false), (2, true), (3, false)])
    );
    assert_eq!(
        alternate_ranks(10, 3, 1),
        t(&[(0, true), (1, false), (2, true), (4, true)])
    );
    assert_eq!(
        alternate_ranks(10, 1, 3),
        t(&[(0, true), (1, false), (3, false), (5, false)])
    );
    assert_eq!(
        alternate_ranks(3, 2, 2),
        t(&[(0, true), (1, false), (2, true)])
    );
}

/// The keys are the table's own column names, a 0/1 column reads as a
/// flag, and what no column names is refused.
#[test]
fn scenario_keys_are_the_table_columns() {
    let keys = |sets: &[&str]| {
        let sets: Vec<String> = sets.iter().map(|s| s.to_string()).collect();
        ScenarioKeys::default().with_overrides(&sets)
    };
    let k = keys(&[
        "effect_pve=0.05",
        "response_field_pve=-0.2",
        "confounder=realgene",
        "confound_shape=quadratic",
        "confound_scale=4",
        "marker=1",
        "heterogeneous=true",
        "dose_order=3",
        "rep=2",
    ])
    .unwrap();
    let s = k.scenario();
    assert_eq!(s.effect_pve, 0.05);
    assert_eq!(s.response_field_pve, -0.2);
    assert_eq!(s.confounder, ConfounderKind::RealGene);
    assert_eq!(s.shape, ConfoundShape::Quadratic);
    assert_eq!(s.scale, 4.0);
    assert!(s.marker && s.heterogeneous && !s.dose_response);
    assert_eq!((s.dose_order, s.rep), (3, 2));
    let d = ScenarioKeys::default().scenario();
    let base = Scenario::default();
    assert_eq!(
        (d.effect_pve, d.scale, d.rep),
        (base.effect_pve, base.scale, base.rep)
    );
    for bad in [
        "scale=4",
        "seed=3",
        "effect_pve",
        "confounder=gaussian",
        "marker=2",
    ] {
        assert!(keys(&[bad]).is_err(), "{bad}");
    }
    // Every key is a column of the table, under the same name.
    let names = serde_json::to_value(ScenarioKeys::default()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.parquet").to_string_lossy().into_owned();
    crate::lr_activity::ligand_effect::output::write_sim(&path, &[], None).unwrap();
    let columns = legume_numeric::matrix::parquet::peek_parquet_field_names(&path).unwrap();
    for key in names.as_object().unwrap().keys() {
        assert!(columns.iter().any(|c| &**c == key), "{key}");
    }
    // And every value is spelt as the table spells it.
    for kind in [
        ConfounderKind::Fourier,
        ConfounderKind::Bumps,
        ConfounderKind::RealGene,
    ] {
        assert_eq!(serde_json::to_value(kind).unwrap(), kind.label());
        let k = keys(&[&format!("confounder={}", kind.label())]).unwrap();
        assert_eq!(k.confounder, kind);
    }
    for shape in [ConfoundShape::Linear, ConfoundShape::Quadratic] {
        assert_eq!(serde_json::to_value(shape).unwrap(), shape.label());
        let k = keys(&[&format!("confound_shape={}", shape.label())]).unwrap();
        assert_eq!(k.confound_shape, shape);
    }
}

/// A point out of range, or a pair count of zero, is refused as it is read.
#[test]
fn an_invalid_point_is_refused() {
    let new = |sets: &[&str], p, n| {
        let sets: Vec<String> = sets.iter().map(|s| s.to_string()).collect();
        SingleScenario::new(&sets, p, n)
    };
    assert!(new(&["effect_pve=0.1"], 10, 10).is_ok());
    for bad in [
        &["effect_pve=0.8", "response_field_pve=0.5"][..],
        &["ligand_field_pve=1"],
        &["effect_pve=-0.1"],
        &["confound_scale=0"],
        &["dose_order=0"],
        &["dose_midpoint=1.5"],
    ] {
        let e = new(bad, 10, 10).unwrap_err().to_string();
        assert!(!e.is_empty(), "{bad:?}");
    }
    assert!(new(&[], 0, 10).is_err() && new(&[], 10, 0).is_err());
}

/// `PINTO_MEMORY_BUDGET` sets the budget; unset, half of what is available.
#[test]
fn the_memory_budget_can_be_set() {
    let gib = 1usize << 30;
    assert_eq!(
        memory_budget(Some("123456789"), Some(16 * gib)).unwrap(),
        123_456_789
    );
    assert_eq!(memory_budget(None, Some(16 * gib)).unwrap(), 8 * gib);
    assert!(memory_budget(Some("lots"), Some(16 * gib)).is_err());
    assert!(memory_budget(Some("0"), Some(16 * gib)).is_err());
}

/// One call holds N1 planted and N0 null rows under the same scenario, on
/// alternate abundance ranks, with the scores a sweep reads, and a sidecar;
/// the same call on one thread or four writes the same bytes, and leaves the
/// run's manifest as it was.
#[test]
fn a_call_holds_both_labels_and_reproduces_byte_for_byte() {
    let dir = tempfile::tempdir().unwrap();
    let f = fixture(dir.path());
    let m = manifest(dir.path());
    let before = std::fs::read(&m).unwrap();
    let sets = with_effect("effect_pve=0.05");
    let four = single_call(&f, (dir.path(), &m), "four", &sets, (2, 2), 4).unwrap();
    assert_eq!(
        std::fs::read(&m).unwrap(),
        before,
        "the manifest is untouched"
    );
    let path = lra_simulate_path(&four);
    let (s, x) = read_table_columns(
        &path,
        &["ligand"],
        &["planted", "effect_pve", "estimate", "se", "t", "p", "df"],
    )
    .unwrap();
    let ligands: Vec<&str> = s[0].iter().map(|l| &**l).collect();
    // GENE6 is the most abundant ligand, GENE3 the fourth.
    assert_eq!(ligands, ["GENE6", "GENE5", "GENE4", "GENE3"]);
    assert_eq!(x[0], [1.0, 0.0, 1.0, 0.0], "planted on even ranks");
    for (i, (&planted, &pve)) in x[0].iter().zip(&x[1]).enumerate() {
        let want = if planted == 1.0 { 0.05 } else { 0.0 };
        assert!((pve - want).abs() < 1e-6, "row {i}: effect_pve {pve}");
        let (est, se, t, p, df) = (x[2][i], x[3][i], x[4][i], x[5][i], x[6][i]);
        if est.is_finite() {
            assert!((t - est / se).abs() <= 1e-4 * t.abs().max(1.0), "row {i}");
            let want = crate::lr_activity::ligand_effect::inference::t_two_sided_p(t, df);
            assert!((p - want).abs() < 1e-5, "row {i}: p {p} vs {want}");
        }
    }
    let (_, ps) = read_table_columns(
        &path,
        &[],
        &["marginal_cor_p", "unadjusted_diff_p", "effect_ipw_p"],
    )
    .unwrap();
    assert!(ps.iter().all(|c| c.len() == 4), "a p column per method");

    let sidecar: serde_json::Value =
        serde_json::from_slice(&std::fs::read(lra_simulate_sidecar_path(&four)).unwrap()).unwrap();
    assert_eq!(sidecar["planted"], 2);
    assert_eq!(sidecar["null"], 2);
    assert_eq!(sidecar["assignment"], "alternate-rank");
    assert_eq!(sidecar["scenario"]["effect_pve"], 0.05);
    assert_eq!(sidecar["scenario"]["confounder"], "bumps");
    assert_eq!(sidecar["seed"], 42);
    assert_eq!(sidecar["version"], env!("CARGO_PKG_VERSION"));
    assert!(sidecar["reference"]["sha256"].is_string(), "{sidecar}");
    let pairs = sidecar["pairs"].as_array().unwrap();
    assert_eq!(pairs.len(), 4);
    for (rank, p) in pairs.iter().enumerate() {
        assert_eq!(p["rank"], rank);
        assert_eq!(p["ligand"], ligands[rank]);
        assert_eq!(p["planted"], rank % 2 == 0);
        assert_eq!(
            p["seed"],
            legume_numeric::matrix::rand_util::mix_seed(42, rank as u64)
        );
    }

    let one = single_call(&f, (dir.path(), &m), "one", &sets, (2, 2), 1).unwrap();
    assert_eq!(
        std::fs::read(lra_simulate_path(&one)).unwrap(),
        std::fs::read(&path).unwrap(),
        "one thread or four, the same bytes"
    );
}

/// A pair's rows depend on its rank, the seed and the scenario alone: a
/// call with another effect size and fewer planted pairs gives the same
/// null rows.
#[test]
fn a_pair_s_rows_do_not_depend_on_the_rest_of_the_call() {
    let dir = tempfile::tempdir().unwrap();
    let f = fixture(dir.path());
    let m = manifest(dir.path());
    let a = single_call(
        &f,
        (dir.path(), &m),
        "a",
        &with_effect("effect_pve=0.05"),
        (2, 2),
        2,
    )
    .unwrap();
    let b = single_call(
        &f,
        (dir.path(), &m),
        "b",
        &with_effect("effect_pve=0.1"),
        (1, 2),
        2,
    )
    .unwrap();
    let read = |out: &str| {
        let (s, x) = read_table_columns(&lra_simulate_path(out), &["ligand"], &SCORED).unwrap();
        let (_, planted) = read_table_columns(&lra_simulate_path(out), &[], &["planted"]).unwrap();
        (
            s.into_iter().next().unwrap(),
            x,
            planted.into_iter().next().unwrap(),
        )
    };
    let ((la, xa, pa), (lb, xb, pb)) = (read(&a), read(&b));
    assert_eq!(pb, [1.0, 0.0, 0.0]);
    let nulls = |l: &[Box<str>], p: &[f64]| -> Vec<(usize, Box<str>)> {
        (0..l.len())
            .filter(|&i| p[i] == 0.0)
            .map(|i| (i, l[i].clone()))
            .collect()
    };
    let (na, nb) = (nulls(&la, &pa), nulls(&lb, &pb));
    assert_eq!(
        na.iter().map(|x| &x.1).collect::<Vec<_>>(),
        nb.iter().map(|x| &x.1).collect::<Vec<_>>()
    );
    for ((ia, l), (ib, _)) in na.iter().zip(&nb) {
        for (k, col) in SCORED.iter().enumerate() {
            let (u, v) = (xa[k][*ia], xb[k][*ib]);
            assert!(u.to_bits() == v.to_bits(), "{l} {col}: {u} vs {v}");
        }
    }
    // The planted pair took the larger effect.
    let truth = SCORED.iter().position(|c| *c == "truth").unwrap();
    assert!(xb[truth][0] != xa[truth][0], "the planted rows differ");
}

/// The real-gene confounder needs a gene outside every pair of the call:
/// with all six ligands benchmarked there is none, and the call fails
/// rather than change what it simulates, an earlier call's outputs under
/// its prefix gone. Without a field to draw, no gene is needed.
#[test]
fn realgene_without_a_free_gene_fails() {
    let dir = tempfile::tempdir().unwrap();
    let f = fixture(dir.path());
    let m = manifest(dir.path());
    let out = dir.path().join("x").to_string_lossy().into_owned();
    let (table, sidecar) = (lra_simulate_path(&out), lra_simulate_sidecar_path(&out));
    for earlier in [&table, &sidecar] {
        std::fs::write(earlier, "an earlier call's").unwrap();
    }
    let sets = [
        "effect_pve=0.05",
        "ligand_field_pve=0.25",
        "confounder=realgene",
    ];
    let e = single_call(&f, (dir.path(), &m), "x", &sets, (3, 3), 2).unwrap_err();
    assert!(format!("{e:#}").contains("realgene"), "{e:#}");
    assert!(!Path::new(&table).exists(), "the earlier table is gone");
    assert!(!Path::new(&sidecar).exists(), "the earlier sidecar is gone");

    let unconfounded = ["effect_pve=0.05", "confounder=realgene"];
    single_call(&f, (dir.path(), &m), "x", &unconfounded, (3, 3), 2).unwrap();
    let made: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
    assert!(made["real_gene_confounder"].is_null(), "{made}");
}
