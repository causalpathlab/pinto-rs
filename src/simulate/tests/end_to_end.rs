use crate::simulate::run::{run_simulate, SimulateArgs};
use data_beans::sparse_io::open_sparse_matrix_by_path;

fn truth_meta(out: &str) -> crate::util::metadata::PintoMetadata {
    let path = std::path::PathBuf::from(format!("{out}/truth/truth.pinto.json"));
    crate::util::metadata::PintoMetadata::read(&path).unwrap()
}

fn args(out: &str, scenario: &str, extra: &[&str]) -> SimulateArgs {
    let mut set: Vec<String> = ["grid_side=12", "n_genes=60", "program_genes=8", "seed=3"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    set.extend(extra.iter().map(|s| s.to_string()));
    SimulateArgs {
        scenario: scenario.into(),
        set,
        coords: Vec::new(),
        coord_column_names: vec!["row_um".into(), "col_um".into()],
        reference_data: None,
        reference_data_labels: None,
        out: out.into(),
    }
}

/// A run writes one readable backend and coordinate file per section, the
/// truth tables, and a manifest holding every parameter.
#[test]
fn a_run_writes_backends_truth_and_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim").to_string_lossy().to_string();
    run_simulate(&args(&out, "donor-confounded-donor-only-cell-type", &[])).unwrap();
    for s in 1..=6 {
        let m = open_sparse_matrix_by_path(&format!("{out}/sections/S{s}.zarr.zip")).unwrap();
        assert_eq!((m.num_rows(), m.num_columns()), (Some(60), Some(144)));
        assert!(std::path::Path::new(&format!("{out}/sections/S{s}.coords.parquet")).exists());
    }
    for t in ["spots", "domains", "states", "genes"] {
        assert!(
            std::path::Path::new(&format!("{out}/truth/{t}.parquet")).exists(),
            "{t}"
        );
    }
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out}/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(m["params"]["n_genes"], 60);
    assert_eq!(m["params"]["patch_kind"], "new-state");
    assert_eq!(m["sections"].as_array().unwrap().len(), 6);
    assert!(m["realized_pve_program_genes"]["type"].as_f64().unwrap() > 0.0);
}

/// The same seed gives the same counts, the scenario named by its earlier
/// name or its own, and the manifest records its own name.
#[test]
fn a_run_is_reproducible() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a").to_string_lossy().to_string();
    let b = dir.path().join("b").to_string_lossy().to_string();
    run_simulate(&args(&a, "donor-confounded", &[])).unwrap();
    run_simulate(&args(&b, "donor-confounded", &[])).unwrap();
    let read = |o: &str| {
        let m = open_sparse_matrix_by_path(&format!("{o}/sections/S3.zarr.zip")).unwrap();
        m.read_columns_csc((0..144).collect::<Vec<usize>>())
            .unwrap()
    };
    assert_eq!(read(&a), read(&b));
    let scenario = |o: &str| {
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{o}/manifest.json")).unwrap())
                .unwrap();
        m["scenario"].clone()
    };
    assert_eq!(scenario(&a), "donor-confounded");
    assert_eq!(scenario(&b), "donor-confounded");
}

#[test]
fn an_impossible_setting_is_refused_before_anything_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("bad").to_string_lossy().to_string();
    assert!(run_simulate(&args(&out, "unconfounded", &["pve_donor=0.9"])).is_err());
    assert!(!std::path::Path::new(&out).exists());
}

/// The truth is written as a run with a cells table and the domain, state
/// and donor levels.
#[test]
fn the_truth_is_written_as_a_run() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim").to_string_lossy().to_string();
    run_simulate(&args(&out, "donor-confounded-donor-only-cell-type", &[])).unwrap();
    let meta = truth_meta(&out);
    let tags: Vec<String> = meta.levels.unwrap().iter().map(|l| l.tag.clone()).collect();
    assert_eq!(tags, ["domain", "state", "donor"]);
    assert_eq!(meta.n_cells, 6 * 144);
    assert_eq!(meta.data_files.unwrap().len(), 6);
}

/// The manifest's data files resolve beside it, and every backend column,
/// read the way pinto reads data files, names a truth cell: with one section
/// (bare barcodes) and with several (`barcode@section`).
#[test]
fn backend_columns_join_onto_the_truth_cells() {
    use crate::util::input::read_expr_data;
    use crate::util::metadata::resolve_beside;
    for (tag, extra) in [
        ("one", vec!["n_donors=1", "sections_per_donor=1"]),
        ("six", vec![]),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join(tag).to_string_lossy().to_string();
        run_simulate(&args(&out, "unconfounded", &extra)).unwrap();
        let manifest = std::path::PathBuf::from(format!("{out}/truth/truth.pinto.json"));
        let files: Vec<Box<str>> = truth_meta(&out)
            .data_files
            .unwrap()
            .iter()
            .map(|f| {
                assert!(std::path::Path::new(f).is_relative(), "{f}");
                let p = resolve_beside(&manifest, f).expect("a data file beside the manifest");
                p.to_string_lossy().into()
            })
            .collect();
        let columns = read_expr_data(&files).unwrap().column_names().unwrap();
        let cells: std::collections::HashSet<Box<str>> = {
            use legume_numeric::matrix::parquet::read_parquet_string_column;
            read_parquet_string_column(&format!("{out}/truth/truth.cells.parquet"), 0)
                .unwrap()
                .into_iter()
                .collect()
        };
        assert_eq!(columns.len(), cells.len(), "{tag}");
        assert!(columns.iter().all(|c| cells.contains(c)), "{tag}");
    }
}

/// `pinto view` opens the truth run, places every spot, and names the
/// communities of each level as the truth tables do.
#[cfg(feature = "view")]
#[test]
fn the_viewer_opens_the_truth() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim").to_string_lossy().to_string();
    run_simulate(&args(&out, "donor-confounded-donor-only-cell-type", &[])).unwrap();
    let run = crate::view::data::Run::open(&format!("{out}/truth/truth")).unwrap();
    let geometry = run.load_geometry().unwrap();
    assert_eq!(geometry.names.len(), 6 * 144);
    assert_eq!(geometry.tiles.len(), 6);
    let levels = run.meta.level_list();
    let domain = run.load_communities(&geometry, &levels[0]).unwrap();
    let names: Vec<&str> = domain
        .names
        .as_ref()
        .unwrap()
        .iter()
        .map(|n| n.as_ref())
        .collect();
    assert_eq!(
        names,
        [
            "shared_domain_1",
            "shared_domain_2",
            "shared_domain_3",
            "shared_domain_4",
            "D3_specific_domain_1"
        ]
    );
}

/// Scoring a simulation's own truth, written as a cell<TAB>label table,
/// gives perfect scores on both tests; labels that follow the donor fail
/// the first.
#[test]
fn scoring_the_truth_and_the_donors() {
    use crate::sim_eval::score::{read_labels, score, TruthSpots};
    use crate::simulate::panel::donor_label;
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim").to_string_lossy().to_string();
    run_simulate(&args(&out, "donor-confounded-donor-only-cell-type", &[])).unwrap();
    let t = TruthSpots::read(&out).unwrap();
    let write = |name: &str, f: &dyn Fn(usize) -> String| -> String {
        let path = dir.path().join(name).to_string_lossy().to_string();
        let text: String = std::iter::once("cell\tlabel\n".to_string())
            .chain((0..t.cells.len()).map(|i| format!("{}\t{}\n", t.cells[i], f(i))))
            .collect();
        std::fs::write(&path, text).unwrap();
        path
    };
    let truth = write("truth.tsv", &|i| t.domain_labels[t.domain[i]].to_string());
    let donors = write("donors.tsv", &|i| donor_label(t.donor[i]).to_string());
    let s = score(&t, &read_labels(&truth, "final", "label").unwrap()).unwrap();
    assert_eq!(
        (s.n_scored, s.donor_conditional_entropy),
        (t.cells.len(), Some(1.0))
    );
    assert!((s.shared_ari - 1.0).abs() < 1e-12);
    assert_eq!(s.patches.len(), 1);
    assert_eq!(s.patches[0].f1, 1.0);
    let s = score(&t, &read_labels(&donors, "final", "label").unwrap()).unwrap();
    assert!(s.donor_conditional_entropy.unwrap().abs() < 1e-9);
    for d in &s.donor_conditional_entropy_by_domain {
        let x = d.donor_conditional_entropy.unwrap();
        assert!(x.abs() < 1e-9, "{}: {x}", d.domain);
    }
}

/// The outputs appear at `out` only once all are written: they are written
/// to `{out}.partial` and moved, and a partial folder an earlier stopped run
/// left is replaced.
#[test]
fn a_simulation_appears_only_when_whole() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim");
    let partial = dir.path().join("sim.partial");
    std::fs::create_dir_all(partial.join("sections")).unwrap();
    std::fs::write(partial.join("sections/stale"), "").unwrap();
    crate::simulate::run::run_simulate(&args(&out.to_string_lossy(), "unconfounded", &[])).unwrap();
    assert!(out.join("manifest.json").is_file());
    assert!(!out.join("sections/stale").exists());
    assert!(!partial.exists());
    let run = std::fs::read_to_string(out.join("truth/truth.pinto.json")).unwrap();
    assert!(
        !run.contains("partial"),
        "the run names where it was moved: {run}"
    );
}

/// An out given with a trailing slash, or already there as an empty folder
/// (as the records' replay script makes it), still gets the whole run, and
/// no partial folder is left beside it.
#[test]
fn a_simulation_into_a_slashed_or_empty_out_is_whole() {
    let dir = tempfile::tempdir().unwrap();
    let slashed = dir.path().join("a");
    crate::simulate::run::run_simulate(&args(
        &format!("{}/", slashed.display()),
        "unconfounded",
        &[],
    ))
    .unwrap();
    assert!(slashed.join("manifest.json").is_file());
    let empty = dir.path().join("b");
    std::fs::create_dir(&empty).unwrap();
    crate::simulate::run::run_simulate(&args(&empty.to_string_lossy(), "unconfounded", &[]))
        .unwrap();
    assert!(empty.join("truth/truth.pinto.json").is_file());
    for left in ["a.partial", "b.partial", "a/.partial"] {
        assert!(!dir.path().join(left).exists(), "{left} left behind");
    }
}

/// Each donor's status is in the truth: a `confounded` column per donor in
/// `donors.parquet` and per spot in `spots.parquet`, and in the manifest the
/// unconfounded donors and whether they were drawn.
#[test]
fn the_truth_records_each_donors_status() {
    use legume_numeric::matrix::parquet::read_parquet_string_columns_by_name;
    let dir = tempfile::tempdir().unwrap();
    for (name, set, drawn) in [
        ("drawn", "n_unconfounded_donors=1", true),
        ("listed", "unconfounded_donors=[1]", false),
    ] {
        let out = dir.path().join(name).to_string_lossy().to_string();
        run_simulate(&args(&out, "donor-confounded", &[set])).unwrap();
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{out}/manifest.json")).unwrap())
                .unwrap();
        assert_eq!(m["unconfounded_donors"]["drawn"], drawn, "{name}");
        let free: Vec<String> = m["unconfounded_donors"]["donors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_str().unwrap().to_string())
            .collect();
        assert_eq!(free.len(), 1, "{name}");
        if !drawn {
            assert_eq!(free, ["D2"]);
        }
        let status = |d: &str| {
            if free.iter().any(|f| f == d) {
                "false"
            } else {
                "true"
            }
        };
        let donors = read_parquet_string_columns_by_name(
            &format!("{out}/truth/donors.parquet"),
            &["donor", "confounded"],
        )
        .unwrap();
        assert_eq!(donors[0].len(), 3, "{name}");
        for (d, c) in donors[0].iter().zip(&donors[1]) {
            assert_eq!(c.as_ref(), status(d), "{name} {d}");
        }
        let spots = read_parquet_string_columns_by_name(
            &format!("{out}/truth/spots.parquet"),
            &["donor", "confounded"],
        )
        .unwrap();
        assert_eq!(spots[0].len(), 6 * 144);
        for (d, c) in spots[0].iter().zip(&spots[1]) {
            assert_eq!(c.as_ref(), status(d), "{name} {d}");
        }
    }
}

/// A donor is confounded only when some shift reaches it: with no nuisance
/// term on (the unconfounded scenario, `nuisance_off`, or every nuisance
/// share at 0) every donor and spot is written `false`.
#[test]
fn a_donor_with_no_shift_is_not_confounded() {
    use legume_numeric::matrix::parquet::read_parquet_string_columns_by_name;
    let dir = tempfile::tempdir().unwrap();
    for (name, scenario, set) in [
        ("default", "unconfounded", &[][..]),
        ("off", "donor-confounded", &["nuisance_off=true"][..]),
        (
            "zero",
            "donor-confounded",
            &["pve_donor=0", "pve_inter=0", "pve_section=0"][..],
        ),
    ] {
        let out = dir.path().join(name).to_string_lossy().to_string();
        run_simulate(&args(&out, scenario, set)).unwrap();
        for table in ["donors", "spots"] {
            let col = read_parquet_string_columns_by_name(
                &format!("{out}/truth/{table}.parquet"),
                &["confounded"],
            )
            .unwrap();
            assert!(!col[0].is_empty(), "{name} {table}");
            assert!(
                col[0].iter().all(|c| c.as_ref() == "false"),
                "{name} {table}"
            );
        }
    }
}

/// FNV-1a over `bytes`, then a separator, so two texts never run together.
fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes.iter().chain(&[0xff]) {
        *h = (*h ^ b as u64).wrapping_mul(0x100000001b3);
    }
}

/// `v` as text a fingerprint can take: the keys in `drop` left out, the out
/// folder `out` read as `OUT`, and every non-integer number at ten digits,
/// so the last bits of a platform's `exp` and `ln` do not move it.
fn stable_json(v: &serde_json::Value, out: &str, drop: &[&str]) -> String {
    use serde_json::Value;
    match v {
        Value::Object(m) => {
            let fields: Vec<String> = m
                .iter()
                .filter(|(k, _)| !drop.contains(&k.as_str()))
                .map(|(k, x)| format!("{k:?}:{}", stable_json(x, out, drop)))
                .collect();
            format!("{{{}}}", fields.join(","))
        }
        Value::Array(a) => {
            let items: Vec<String> = a.iter().map(|x| stable_json(x, out, drop)).collect();
            format!("[{}]", items.join(","))
        }
        Value::Number(n) if n.is_f64() => format!("{:.9e}", n.as_f64().unwrap()),
        Value::String(s) => format!("{:?}", s.replace(out, "OUT")),
        other => other.to_string(),
    }
}

/// A fingerprint of everything a run into `out` wrote that a reader sees:
/// each section's counts with their gene and spot names, every column of
/// every table, and the two manifests but for the version and the time.
/// What several donor-specific domains added is left out (the manifest's
/// `patch_owners`, `patch_kinds` and `patches`, the domains' `kind`), so
/// the fingerprints from before still hold for the rest.
fn run_fingerprint(out: &str) -> u64 {
    use legume_numeric::matrix::parquet::{
        peek_parquet_field_names, read_parquet_string_columns_by_name,
    };
    let mut h: u64 = 0xcbf29ce484222325;
    let json = |path: String| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    };
    let manifest = json(format!("{out}/manifest.json"));
    let added = ["patch_owners", "patch_kinds", "patches"];
    let drop: Vec<&str> = std::iter::once("version").chain(added).collect();
    fnv(&mut h, stable_json(&manifest, out, &drop).as_bytes());
    let run = json(format!("{out}/truth/truth.pinto.json"));
    let run = stable_json(&run, out, &["version", "timestamp"]);
    fnv(&mut h, run.as_bytes());

    let mut tables: Vec<String> = [
        "spots",
        "donors",
        "domains",
        "states",
        "genes",
        "truth.cells",
        "truth.domain.propensity",
        "truth.state.propensity",
        "truth.donor.propensity",
    ]
    .iter()
    .map(|t| format!("{out}/truth/{t}.parquet"))
    .collect();
    for sec in manifest["sections"].as_array().unwrap() {
        let name = sec["section"].as_str().unwrap();
        let m = open_sparse_matrix_by_path(&format!("{out}/sections/{name}.zarr.zip")).unwrap();
        for names in [m.row_names().unwrap(), m.column_names().unwrap()] {
            names.iter().for_each(|n| fnv(&mut h, n.as_bytes()));
        }
        let n = m.num_columns().unwrap();
        let csc = m.read_columns_csc((0..n).collect::<Vec<usize>>()).unwrap();
        for (g, i, y) in csc.triplet_iter() {
            fnv(&mut h, format!("{g} {i} {}", y.to_bits()).as_bytes());
        }
        tables.push(format!("{out}/sections/{name}.coords.parquet"));
    }
    for path in tables {
        let mut fields = peek_parquet_field_names(&path).unwrap();
        if path.ends_with("/domains.parquet") {
            fields.retain(|f| f.as_ref() != "kind");
        }
        let wanted: Vec<&str> = fields.iter().map(|f| f.as_ref()).collect();
        let columns = read_parquet_string_columns_by_name(&path, &wanted).unwrap();
        for (field, column) in fields.iter().zip(&columns) {
            fnv(&mut h, field.as_bytes());
            column.iter().for_each(|x| fnv(&mut h, x.as_bytes()));
        }
    }
    h
}

/// A run without reference data writes what it always did: counts, truth
/// tables and manifests, in every scenario and with a drawn unconfounded
/// donor (the counts checked identical to those from before reference data
/// could be given). Any stream, any order of a sum or any column that moves
/// shows here. The fingerprints are Linux x86-64's: a count is drawn through
/// `exp`, `ln` and the incomplete beta, whose last bits differ between
/// platforms' maths libraries, so a rare count lands one step over
/// elsewhere (macOS ARM); there the test is not run.
#[test]
#[cfg_attr(
    not(all(target_os = "linux", target_arch = "x86_64")),
    ignore = "the fingerprints are Linux x86-64's"
)]
fn a_run_without_reference_data_writes_what_it_always_did() {
    let want: [(&str, &[&str], u64); 6] = [
        ("unconfounded", &[], 0xe1444648c5f12a20),
        ("donor-confounded", &[], 0x37371ab11b1f2d04),
        (
            "donor-confounded-donor-only-cell-type",
            &[],
            0x9c0a1989b67ff53a,
        ),
        ("donor-confounded-donor-only-mix", &[], 0x8238b2b5962d39ab),
        ("composition-shift", &[], 0x8cfc50cfbc357dd6),
        (
            "donor-confounded",
            &["n_unconfounded_donors=1"],
            0x1282d6fa2b88671e,
        ),
    ];
    let dir = tempfile::tempdir().unwrap();
    for (k, (scenario, extra, fingerprint)) in want.into_iter().enumerate() {
        let out = dir.path().join(format!("sim{k}"));
        let out = out.to_string_lossy();
        run_simulate(&args(&out, scenario, extra)).unwrap();
        let h = run_fingerprint(&out);
        assert_eq!(h, fingerprint, "{scenario} {extra:?}: {h:#018x}");
    }
}

/// Three owners named: each its own domain, labelled by its owner, with its
/// kind in the manifest and the domains table; scoring the truth scores
/// each domain within its own owner.
#[test]
fn several_owners_each_have_a_domain_of_their_own() {
    use crate::sim_eval::score::{score, TruthSpots};
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim").to_string_lossy().to_string();
    run_simulate(&args(
        &out,
        "donor-confounded",
        &[
            "patch_owners=0,1,2",
            "patch_kinds=new-state,new-mix,new-state",
        ],
    ))
    .unwrap();
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out}/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(m["params"]["patch_owners"], serde_json::json!([0, 1, 2]));
    assert_eq!(
        m["patches"],
        serde_json::json!([
            {"domain": "D1_specific_domain_1", "owner": "D1", "kind": "new-state"},
            {"domain": "D2_specific_domain_1", "owner": "D2", "kind": "new-mix"},
            {"domain": "D3_specific_domain_1", "owner": "D3", "kind": "new-state"},
        ])
    );
    let t = TruthSpots::read(&out).unwrap();
    let labels: Vec<(Box<str>, i64)> = (0..t.cells.len())
        .map(|i| (t.cells[i].clone(), t.domain[i] as i64))
        .collect();
    let s = score(&t, &labels).unwrap();
    let owners: Vec<&str> = s.patches.iter().map(|p| p.owner.as_str()).collect();
    assert_eq!(owners, ["D1", "D2", "D3"]);
    for p in &s.patches {
        assert_eq!((p.within, p.f1), ("owner_donor", 1.0), "{}", p.domain);
    }
}

/// Owners and kinds drawn from the seed: recorded as the lists drawn, the
/// same for the same seed, and each drawn domain scored within its owner.
#[test]
fn drawn_owners_are_recorded_and_scored_within_each_owner() {
    use crate::sim_eval::pairs::{score_pairs, PairOptions, SpotLayout};
    use crate::sim_eval::score::TruthSpots;
    let dir = tempfile::tempdir().unwrap();
    let mut drawn = Vec::new();
    for (k, seed) in ["seed=3", "seed=3", "seed=11"].into_iter().enumerate() {
        let out = dir
            .path()
            .join(format!("sim{k}"))
            .to_string_lossy()
            .to_string();
        run_simulate(&args(
            &out,
            "donor-confounded",
            &["patch_owners=random", "patch_kinds=random", seed],
        ))
        .unwrap();
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{out}/manifest.json")).unwrap())
                .unwrap();
        let owners = m["params"]["patch_owners"].as_array().unwrap().clone();
        let kinds = m["params"]["patch_kinds"].as_array().unwrap().clone();
        assert!(!owners.is_empty() && owners.len() == kinds.len(), "{m}");
        assert_eq!(m["patches"].as_array().unwrap().len(), owners.len());
        let t = TruthSpots::read(&out).unwrap();
        let layout = SpotLayout::read(&out, &t).unwrap();
        // the truth's own domains as one-hot vectors
        let x = crate::util::common::Mat::from_fn(t.cells.len(), t.domain_labels.len(), |i, j| {
            (t.domain[i] == j) as u8 as f32
        });
        let o = PairOptions::new(2000, 1, 6).unwrap();
        let s = score_pairs(&t, &layout, &t.cells, &x, &o).unwrap();
        assert_eq!(s.patches.len(), owners.len());
        for (p, want) in s.patches.iter().zip(&owners) {
            let owner = crate::simulate::panel::donor_label(want.as_u64().unwrap() as usize);
            assert_eq!(p.owner, owner.as_ref());
            assert_eq!(p.donor_domain_vs_own_donor.auroc, 1.0, "{}", p.domain);
        }
        drawn.push((owners, kinds));
    }
    assert_eq!(drawn[0], drawn[1], "the same seed draws the same");
}
