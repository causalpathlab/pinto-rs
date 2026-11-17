use crate::util::common::*;
use crate::util::coverage::{measured_mass, Coverage};
use crate::util::input::read_expr_data;
use clap::Parser;
use dmatrix_gamma::GammaMatrix;
use legume_numeric::matrix::common_io::mkdir_parent;
use legume_numeric::param::dmatrix_gamma;
use legume_numeric::param::io::ParamIo;
use legume_numeric::param::traits::TwoStatParam;

#[derive(Parser, Debug, Clone)]
pub struct SrtPropensityArgs {
    /// How the pair latent becomes link communities. Shared verbatim with
    /// `cage` and `dsvd` — the k-means fallback width is the latent dimension
    /// here.
    #[command(flatten)]
    edge_clustering: crate::util::edge_clustering::EdgeClusterArgs,

    #[arg(
        long,
        default_value_t = crate::util::common::DEFAULT_SEED,
        help = "Random seed for the edge clustering"
    )]
    seed: u64,

    #[arg(
        short = 'z',
        long,
        required = true,
        help = "Latent edge representation file (.latent.parquet)"
    )]
    latent_data_file: Box<str>,

    #[arg(
        short = 'e',
        long,
        required = true,
        help = "Coordinate pair file (.coord_pairs.parquet)",
        long_help = "Coordinate pair file (.coord_pairs.parquet from delta-svd).\n\
                     Must contain left_cell and right_cell columns."
    )]
    coord_pair_file: Box<str>,

    #[arg(
        long = "data-files",
        short_alias = 'd',
        value_delimiter(','),
        help = "Expression data files (.zarr or .h5)",
        long_help = "Expression data files (.zarr or .h5, comma separated). Optional;\n\
                     used for additional per-vertex expression statistics."
    )]
    expr_data_files: Option<Vec<Box<str>>>,

    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        help = "Coordinate column names in coord_pair_file",
        long_help = "Coordinate column names in coord_pair_file (comma separated).\n\
                     Looked up as left_{name} and right_{name}.",
        hide = true
    )]
    coord_column_names: Option<Vec<Box<str>>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Output file prefix. Generates: {out}.propensity.parquet,\n\
                     {out}.link_community.parquet"
    )]
    pub out: Box<str>,
}

pub fn fit_srt_propensity(args: &SrtPropensityArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let MatWithNames {
        rows,
        cols: _,
        mat: proj_mk,
    } = Mat::from_parquet(args.latent_data_file.as_ref())?;

    let pair_names = read_pair_names(&args.coord_pair_file, "left_cell", "right_cell")?;

    // Pair provenance, when the source run was augmented. Positionally
    // parallel to the pair rows, so it can be re-emitted with the re-cut
    // communities; dropping it would make every expression pair read as
    // physically adjacent downstream (lr-activity's spatial filter, the
    // viewer's mesh) through the missing-column fallback.
    let edge_kind: Option<Vec<i32>> =
        if legume_numeric::matrix::parquet::peek_parquet_field_names(&args.coord_pair_file)?
            .iter()
            .any(|c| c.as_ref() == "edge_kind")
        {
            let MatWithNames { mat: kind_mat, .. } = Mat::from_parquet_with_indices_names(
                &args.coord_pair_file,
                Some(0),
                None,
                Some(&["edge_kind".into()]),
            )?;
            Some(kind_mat.iter().map(|&v| v as i32).collect())
        } else {
            None
        };

    if pair_names.len() != rows.len() {
        anyhow::bail!(
            "pair names length {} != latent matrix rows {}",
            pair_names.len(),
            rows.len()
        );
    }

    // Same cut `cage` and `dsvd` make, through the same enum. `proj_mk` is
    // already `[E × K]`, the orientation the routine wants, so the latent is
    // clustered in place off the parquet read.
    let edge_membership = args.edge_clustering.resolve(args.seed).cluster(&proj_mk)?;
    let num_clusters =
        crate::link_community::profiles::realized_communities(&edge_membership, rows.len())?;

    info!("calibrating propensity");

    let mut vertices = pair_names.par_iter().flatten().cloned().collect::<Vec<_>>();
    vertices.par_sort();
    vertices.dedup();

    let vertex_index: HashMap<Box<str>, usize> = vertices
        .iter()
        .enumerate()
        .map(|(i, x)| (x.clone(), i))
        .collect();

    let nvertices = vertices.len();
    info!("{} vertices", nvertices);

    let mut prop_kn = Mat::zeros(num_clusters, nvertices);
    let arc_count_kn = Arc::new(Mutex::new(&mut prop_kn));

    pair_names
        .par_iter()
        .zip(edge_membership.par_iter())
        .for_each(|(vertices, &k)| {
            let indices = vertices
                .iter()
                .filter_map(|x| vertex_index.get(x).copied())
                .collect::<Vec<_>>();

            let mut count_kn = arc_count_kn.lock().expect("lock count kn");

            for v in indices {
                count_kn.column_mut(v)[k] += 1.0;
            }
        });

    prop_kn.sum_to_one_columns_inplace();

    // Propensity output (optionally with coordinates)
    let prop_nk = prop_kn.transpose();

    // with each vertex's coordinates after the shared columns, when given
    let coords = match &args.coord_column_names {
        Some(coord_column_names) => {
            info!("Extracting vertex coordinates from coord_pair_file");
            Some(vertex_coords(
                &args.coord_pair_file,
                coord_column_names,
                &pair_names,
                &vertex_index,
                nvertices,
            )?)
        }
        None => None,
    };
    crate::link_community::outputs::write_propensity_with(
        &args.out,
        &prop_nk,
        &vertices,
        args.coord_column_names.as_deref().zip(coords.as_ref()),
    )?;

    // Per-edge community labels, in the ONE edge-table schema pinto reads
    // (`left_cell` / `right_cell` / `community`).
    {
        info!("Writing link communities");
        // Map name pairs back to vertex indices. Strict rather than filtered:
        // dropping an edge here would silently misalign the edge list from
        // `edge_membership`, which is zipped with it positionally.
        let edges: Vec<(usize, usize)> = pair_names
            .iter()
            .map(|p| -> anyhow::Result<(usize, usize)> {
                let l = *vertex_index
                    .get(&p[0])
                    .ok_or_else(|| anyhow::anyhow!("unknown left cell {}", p[0]))?;
                let r = *vertex_index
                    .get(&p[1])
                    .ok_or_else(|| anyhow::anyhow!("unknown right cell {}", p[1]))?;
                Ok((l, r))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        crate::link_community::outputs::write_link_communities(
            &(args.out.to_string() + ".link_community.parquet"),
            &edges,
            &edge_membership,
            &vertices,
            edge_kind.as_deref(),
        )?;
    }

    if let Some(data_files) = args.expr_data_files.as_ref() {
        info!("Estimate cluster-specific feature expressions");
        let data_vec = read_expr_data(data_files)?;
        let features = data_vec.row_names()?;
        let (mut sum_dk, denom_dk) = feature_stats(&data_vec, &prop_kn, &vertex_index, None)?;

        info!("Applying NB Fisher-info weighting to feature-cluster stats");
        let w = compute_nb_fisher_weights(&data_vec, None)?;
        apply_feature_weights(&mut sum_dk, &w);

        let mut gamma_param = GammaMatrix::new((sum_dk.nrows(), sum_dk.ncols()), 1.0, 1.0);

        gamma_param.update_stat(&sum_dk, &denom_dk);
        gamma_param.calibrate();

        gamma_param.to_melted_parquet(
            &(args.out.to_string() + ".features.parquet"),
            (Some(&features), Some("feature")),
            (None, Some("community")),
        )?;
    }

    {
        use crate::util::metadata::create_prop_metadata;
        let meta = create_prop_metadata(
            args.out.as_ref(),
            args.expr_data_files.as_deref(),
            Some(args.coord_pair_file.as_ref()),
            nvertices,
            num_clusters,
        );
        let meta_path = crate::util::metadata::manifest_path(&args.out);
        meta.write(std::path::Path::new(&meta_path))?;
        info!("Wrote {}", crate::util::staging::final_str(&meta_path));
    }

    info!("Done");
    Ok(())
}

/// Each feature's summed counts per community, and the community mass
/// they are averaged over: `n_features × k` each. A feature is averaged
/// over the cells of the files that measured it.
fn feature_stats(
    data_vec: &SparseIoVec,
    prop_kn: &Mat,
    vertex_index: &HashMap<Box<str>, usize>,
    block_size: Option<usize>,
) -> anyhow::Result<(Mat, Mat)> {
    let data_vertices = data_vec.column_names()?;

    let jobs = legume_numeric::matrix::utils::generate_minibatch_intervals(
        data_vec.num_columns(),
        data_vec.num_rows(),
        block_size,
    );

    let prog_bar = new_progress_bar(jobs.len() as u64).with_message("feature-module blocks");
    // Folded, not collected: `sum_dk` is `[n_features x k]` no matter how few
    // cells a job reads, so collecting one per job makes peak memory scale
    // with the job count. `generate_minibatch_intervals` sizes a block to
    // bound READ work and knows nothing about that, so a wide, sparse input
    // lands on the small-block floor and produces thousands of jobs. Same
    // shape as `fit_feature_community_param`.
    let coverage = Coverage::of(data_vec);
    let n_files = coverage.as_ref().map_or(1, Coverage::n_files);
    let (sum_dk, n_bk) = jobs
        .par_iter()
        .progress_with(prog_bar.clone())
        .try_fold(
            || {
                (
                    Mat::zeros(data_vec.num_rows(), prop_kn.nrows()),
                    Mat::zeros(n_files, prop_kn.nrows()),
                )
            },
            |(mut acc_dk, mut acc_bk), &(lb, ub)| -> anyhow::Result<(Mat, Mat)> {
                let x_dn = data_vec.read_columns_csc(lb..ub)?;
                let mut p_kn = Mat::zeros(prop_kn.nrows(), x_dn.ncols());

                for (i, v) in data_vertices[lb..ub].iter().enumerate() {
                    if let Some(&j) = vertex_index.get(v) {
                        p_kn.column_mut(i).copy_from(&prop_kn.column(j));
                    }
                }

                // Each cell's mass goes to its file: a feature is
                // averaged over the files that measured it.
                for i in 0..p_kn.ncols() {
                    let b = coverage.as_ref().map_or(0, |c| c.file(lb + i));
                    let mut row = acc_bk.row_mut(b);
                    row += p_kn.column(i).transpose();
                }
                acc_dk += x_dn * p_kn.transpose();
                Ok((acc_dk, acc_bk))
            },
        )
        .try_reduce(
            || {
                (
                    Mat::zeros(data_vec.num_rows(), prop_kn.nrows()),
                    Mat::zeros(n_files, prop_kn.nrows()),
                )
            },
            |(mut a_dk, mut a_bk), (b_dk, b_bk)| {
                a_dk += b_dk;
                a_bk += b_bk;
                Ok((a_dk, a_bk))
            },
        )?;
    prog_bar.finish_and_clear();

    let mut denom_dk = Mat::zeros(sum_dk.nrows(), sum_dk.ncols());
    for d in 0..denom_dk.nrows() {
        for (k, m) in measured_mass(&n_bk, coverage.as_ref(), d)
            .into_iter()
            .enumerate()
        {
            denom_dk[(d, k)] = m;
        }
    }
    Ok((sum_dk, denom_dk))
}

/// Each of the `n` vertices' coordinates `names` from the pair table at
/// `path` (`left_{name}` / `right_{name}`, one row per pair of `pairs`),
/// from whichever side of a pair the cell is on.
fn vertex_coords(
    path: &str,
    names: &[Box<str>],
    pairs: &[[Box<str>; 2]],
    vertex_index: &HashMap<Box<str>, usize>,
    n: usize,
) -> anyhow::Result<Mat> {
    let side = |side: &str| -> anyhow::Result<Mat> {
        let cols: Vec<Box<str>> = names
            .iter()
            .map(|name| format!("{side}_{name}").into_boxed_str())
            .collect();
        Ok(Mat::from_parquet_with_indices_names(path, Some(0), Some(&[]), Some(&cols))?.mat)
    };
    let (left, right) = (side("left")?, side("right")?);
    let mut out = Mat::zeros(n, names.len());
    for (k, pair) in pairs.iter().enumerate() {
        for (cell, coords) in [(&pair[0], &left), (&pair[1], &right)] {
            if let Some(&v) = vertex_index.get(cell) {
                out.row_mut(v).copy_from(&coords.row(k));
            }
        }
    }
    Ok(out)
}

/// Each pair's two cell names, `[left, right]`, from the columns named so.
fn read_pair_names(path: &str, left: &str, right: &str) -> anyhow::Result<Vec<[Box<str>; 2]>> {
    let [lefts, rights]: [Vec<Box<str>>; 2] =
        legume_numeric::matrix::parquet::read_parquet_string_columns_by_name(path, &[left, right])?
            .try_into()
            .map_err(|_| anyhow::anyhow!("{path}: two columns asked, not two read"))?;
    // The reader gives an empty cell for a null one; a pair needs both cells.
    anyhow::ensure!(
        !lefts.iter().chain(&rights).any(|c| c.is_empty()),
        "{path}: a pair with no {left} or {right}"
    );
    Ok(lefts.into_iter().zip(rights).map(|(l, r)| [l, r]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_zarr;

    /// Pairs read in the order the flags name the columns, whatever the
    /// file's column order.
    #[test]
    fn pair_names_follow_the_flags_not_the_file_s_column_order() {
        use legume_numeric::matrix::parquet::{write_table, Column};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.parquet").to_string_lossy().into_owned();
        let right: Vec<Box<str>> = vec!["b".into(), "d".into()];
        let left: Vec<Box<str>> = vec!["a".into(), "c".into()];
        write_table(
            &path,
            &[
                ("right_cell".into(), Column::Str(&right)),
                ("left_cell".into(), Column::Str(&left)),
            ],
        )
        .unwrap();
        let pairs = read_pair_names(&path, "left_cell", "right_cell").unwrap();
        assert_eq!(pairs, [["a".into(), "b".into()], ["c".into(), "d".into()]]);
        assert!(read_pair_names(&path, "left_cell", "nowhere").is_err());

        // A pair with an empty (or null) cell is refused, not read as a cell
        // named "".
        let left: Vec<Box<str>> = vec!["a".into(), "".into()];
        write_table(
            &path,
            &[
                ("left_cell".into(), Column::Str(&left)),
                ("right_cell".into(), Column::Str(&right)),
            ],
        )
        .unwrap();
        assert!(read_pair_names(&path, "left_cell", "right_cell").is_err());
    }

    /// A cell is placed from whichever side of a pair it is on: one only
    /// ever on the right is not left at the origin.
    #[test]
    fn a_cell_only_on_the_right_of_its_pairs_is_placed() {
        use legume_numeric::matrix::parquet::{write_table, Column};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.parquet").to_string_lossy().into_owned();
        let s = |v: &[&str]| -> Vec<Box<str>> { v.iter().map(|&x| x.into()).collect() };
        let (left, right) = (s(&["a", "a"]), s(&["b", "c"]));
        write_table(
            &path,
            &[
                ("left_cell".into(), Column::Str(&left)),
                ("right_cell".into(), Column::Str(&right)),
                ("left_x".into(), Column::F32(&[1.0, 1.0])),
                ("left_y".into(), Column::F32(&[2.0, 2.0])),
                ("right_x".into(), Column::F32(&[3.0, 5.0])),
                ("right_y".into(), Column::F32(&[4.0, 6.0])),
            ],
        )
        .unwrap();
        let pairs = read_pair_names(&path, "left_cell", "right_cell").unwrap();
        let index: HashMap<Box<str>, usize> = s(&["a", "b", "c"]).into_iter().zip(0..).collect();
        let xy = vertex_coords(&path, &s(&["x", "y"]), &pairs, &index, 3).unwrap();
        let at = |v: usize| (xy[(v, 0)], xy[(v, 1)]);
        assert_eq!([at(0), at(1), at(2)], [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]);
    }

    #[test]
    fn a_feature_s_sums_divide_by_the_cells_of_files_that_measured_it() {
        let dir = tempfile::tempdir().unwrap();
        // GENE2 is on the first file's panel only: 4 and 0 in its cells.
        let files = vec![
            write_zarr(
                dir.path(),
                "a",
                &["GENE1", "GENE2"],
                2,
                &[(0, 0, 1.), (0, 1, 1.), (1, 0, 4.)],
            ),
            write_zarr(dir.path(), "b", &["GENE1"], 2, &[(0, 0, 1.), (0, 1, 1.)]),
        ];
        let data = read_expr_data(&files).unwrap();
        let names = data.column_names().unwrap();
        let vertex_index: HashMap<Box<str>, usize> = names
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, n)| (n, i))
            .collect();
        // One community holding every cell wholly.
        let prop_kn = Mat::from_element(1, names.len(), 1.);

        let (sum_dk, denom_dk) = feature_stats(&data, &prop_kn, &vertex_index, Some(1)).unwrap();
        let row = |f: &str| {
            data.row_names()
                .unwrap()
                .iter()
                .position(|r| r.as_ref() == f)
                .unwrap()
        };
        let (g1, g2) = (row("GENE1"), row("GENE2"));
        assert_eq!((sum_dk[(g1, 0)], denom_dk[(g1, 0)]), (4., 4.));
        // Over file a's two cells, not all four.
        assert_eq!((sum_dk[(g2, 0)], denom_dk[(g2, 0)]), (4., 2.));
    }
}
