//! From a baseline's per-cell embedding to spatial domains, the step every
//! baseline shares where its original clusters with a GPL package (mclust,
//! Walktrap, Louvain): pinto's seeded k-means into `--n-clusters` domains,
//! or Leiden without it, and the optional majority refinement over spatial
//! neighbours that SpatialPCA takes from SpaGCN.

use super::graph::Neighbours;
use crate::util::common::Mat;
use legume_numeric::matrix::kmeans::kmeans_centroids_seeded;

/// Neighbours of the kNN graph Leiden runs on, the cell counted
/// (`sc.pp.neighbors`' default).
pub const LEIDEN_NEIGHBOURS: usize = 15;

/// Lloyd rounds of the k-means at most.
pub const KMEANS_MAX_ITER: usize = 300;

/// The domains of the embedding `z` (cells × latent): with `n_clusters`,
/// seeded k-means into that many clusters ([`kmeans_centroids_seeded`]:
/// k-means++ then Lloyd rounds until no cell moves, at most
/// [`KMEANS_MAX_ITER`]); without, Leiden at `resolution` ([`leiden_at`]).
/// The labels and the method's name.
pub fn cluster_embedding(
    z: &Mat,
    n_clusters: Option<usize>,
    resolution: f64,
    seed: u64,
) -> anyhow::Result<(Vec<usize>, &'static str)> {
    Ok(match n_clusters {
        Some(k) => {
            anyhow::ensure!(
                k >= 1 && z.nrows() >= k,
                "k-means into {k} clusters needs at least {k} cells"
            );
            let (_, labels) = kmeans_centroids_seeded(z, k, KMEANS_MAX_ITER, seed);
            (labels, "kmeans")
        }
        None => (leiden_at(z, resolution, seed)?, "leiden"),
    })
}

/// `sc.pp.neighbors` (15 neighbours, the cell counted) then modularity
/// clustering at a fixed `resolution` of the unweighted graph (scanpy's
/// `sc.tl.louvain` default `use_weights=False`), by Leiden (Traag, Waltman
/// and van Eck 2019, Scientific Reports 9:5233,
/// doi:10.1038/s41598-019-41721-w).
pub fn leiden_at(z: &Mat, resolution: f64, seed: u64) -> anyhow::Result<Vec<usize>> {
    use legume_numeric::matrix::knn_graph::{
        compact_labels, modularity_to_cpm_resolution, run_leiden, KnnGraph, KnnGraphArgs,
    };
    let n = z.nrows();
    anyhow::ensure!(n >= 2, "Leiden needs at least 2 cells");
    let args = KnnGraphArgs {
        knn: LEIDEN_NEIGHBOURS - 1,
        block_size: 1000,
        reciprocal: false,
    };
    let (graph, fuzzy) = KnnGraph::from_rows_fuzzy(z, args)?;
    // sc.tl.louvain(use_weights=False): the connectivities' pattern only
    let unit: Vec<f32> = fuzzy
        .iter()
        .map(|&w| f32::from(u8::from(w > 0.0)))
        .collect();
    let (network, total) = graph.to_leiden_network_with(&unit);
    let res = modularity_to_cpm_resolution(resolution, total);
    let mut labels = run_leiden(&network, n, res, Some(seed as usize));
    compact_labels(&mut labels);
    Ok(labels)
}

/// SpaGCN's refinement (Hu et al. 2021, Nature Methods 18:1342,
/// doi:10.1038/s41592-021-01255-8), as SpatialPCA applies it: one
/// synchronous pass in which each cell takes the label held by most of
/// itself and its listed spatial neighbours `nb` (its nearest within its
/// section), keeping its own label on a tie. On a regular lattice with the
/// lattice's own neighbours listed this is the paper's rule.
pub fn refine_by_neighbours(labels: &[usize], nb: &Neighbours) -> Vec<usize> {
    let k = labels.iter().max().map_or(0, |&m| m + 1);
    let mut count = vec![0usize; k];
    labels
        .iter()
        .enumerate()
        .map(|(i, &own)| {
            count.iter_mut().for_each(|c| *c = 0);
            count[own] += 1;
            for &j in &nb.index[i] {
                count[labels[j]] += 1;
            }
            let best = *count.iter().max().unwrap_or(&0);
            if count[own] == best {
                own
            } else {
                count.iter().position(|&c| c == best).unwrap_or(own)
            }
        })
        .collect()
}
