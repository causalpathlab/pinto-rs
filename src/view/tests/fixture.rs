//! Synthetic runs shared by the view tests.

use crate::util::common::*;
use crate::util::parquet_io::CellTable;
use crate::view::data::{Communities, Geometry};

/// `n` cells on a jittered square lattice, split into `n_batches` batches
/// that all share one coordinate frame (as pinto writes them).
pub(super) fn synth_cells(n: usize, n_batches: usize) -> CellTable {
    let side = (n as f64).sqrt().ceil() as usize;
    let names: Vec<Box<str>> = (0..n).map(|i| format!("c{i}").into_boxed_str()).collect();
    let coords: Vec<(f32, f32)> = (0..n)
        .map(|i| {
            let jitter = ((i * 7919) % 97) as f32 / 97.;
            ((i % side) as f32 + 0.3 * jitter, (i / side) as f32)
        })
        .collect();
    let batches = (n_batches > 1).then(|| {
        (0..n)
            .map(|i| format!("b{}", i % n_batches).into())
            .collect()
    });
    let index = names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();
    CellTable {
        names,
        coords,
        batches,
        index,
        in_graph: None,
        coord_col_names: vec!["x".into(), "y".into()],
    }
}

/// Community = vertical stripe of the lattice; propensity 0.75 on it,
/// the rest spread evenly.
pub(super) fn synth_communities(geom: &Geometry, k: usize) -> Communities {
    let n = geom.n();
    let names = geom.names.clone();
    let mut prop = Mat::from_element(n, k, 0.25 / (k - 1) as f32);
    let mut cluster = Vec::with_capacity(n);
    for i in 0..n {
        let c = (geom.x[i].max(0.) as usize) % k;
        prop[(i, c)] = 0.75;
        cluster.push(c as i64);
    }
    // Half the maximum entropy, ln K, so it quantizes to 128.
    let entropy = Some(vec![0.5 * (k as f32).ln(); n]);
    Communities::join(geom, "final", (prop, cluster, entropy, names))
}
