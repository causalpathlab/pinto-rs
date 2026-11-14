//! The inputs every spatial baseline reads, through pinto's data-beans
//! reader ([`crate::util::input::read_data_with_coordinates`]), so cells are
//! named, ordered and split into sections exactly as `lc`, `dsvd` and `cage`
//! name, order and split them.

use crate::util::common::Mat;
use crate::util::input::{read_data_with_coordinates, SRTReadArgs, PRELOAD_HELP};
use data_beans::aux::feature_names::FeatureNameKind;
use nalgebra_sparse::CscMatrix;

/// Data files, their coordinates, the output prefix and the seed.
#[derive(clap::Args, Debug, Clone)]
pub struct SpatialInputArgs {
    /// Expression data files (.zarr or .h5), comma separated; each file is a
    /// section
    #[arg(required = true, value_delimiter(','))]
    pub data_files: Vec<Box<str>>,

    /// Coordinate files, one per data file, comma separated (as `pinto lc -c`)
    #[arg(long = "coord", short = 'c', value_delimiter(','), required = true)]
    pub coord_files: Vec<Box<str>>,

    /// 0-based coordinate column indices (overrides --coord-column-names)
    #[arg(long = "coord-column-indices", value_delimiter(','))]
    pub coord_columns: Option<Vec<usize>>,

    /// Coordinate column names (as `pinto lc`)
    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        default_value = "pxl_row_in_fullres,pxl_col_in_fullres,cell_centroid_x,cell_centroid_y,x_centroid,y_centroid"
    )]
    pub coord_column_names: Vec<Box<str>>,

    /// Header row index in coordinate files (default: detected)
    #[arg(long, hide = true)]
    pub coord_header_row: Option<usize>,

    #[arg(long, default_value_t = false, help = PRELOAD_HELP)]
    pub preload_data: bool,

    /// Output prefix
    #[arg(long, short, required = true)]
    pub out: Box<str>,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

/// Counts and coordinates of every cell, sections kept apart.
pub struct SpatialData {
    /// Raw counts, genes × cells.
    pub counts: CscMatrix<f32>,
    pub genes: Vec<Box<str>>,
    pub cells: Vec<Box<str>>,
    /// Spatial coordinates, cells × dimensions (no batch column).
    pub xy: Mat,
    /// Each cell's section, 0..n_sections, in order of first appearance.
    pub section: Vec<usize>,
    pub n_sections: usize,
}

impl SpatialData {
    /// The cells of each section, in data order.
    pub fn section_cells(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.n_sections];
        for (i, &s) in self.section.iter().enumerate() {
            out[s].push(i);
        }
        out
    }
}

impl SpatialInputArgs {
    pub fn read(&self) -> anyhow::Result<SpatialData> {
        let srt = read_data_with_coordinates(SRTReadArgs {
            data_files: self.data_files.clone(),
            coord_files: self.coord_files.clone(),
            preload_data: self.preload_data,
            coord_columns: self.coord_columns.clone().unwrap_or_default(),
            coord_column_names: self.coord_column_names.clone(),
            batch_files: None,
            header_in_coord: self.coord_header_row,
            feature_kind: FeatureNameKind::Exact,
        })?;
        let cells = srt.data.column_names()?;
        let genes = srt.data.row_names()?;
        let counts = srt.data.read_columns_csc(0..cells.len())?;
        // the reader appends a section index column when there are several
        let has_batch = srt.coordinate_names.last().map(|c| c.as_ref()) == Some("batch");
        let d = srt.coordinates.ncols() - usize::from(has_batch);
        anyhow::ensure!(d >= 1, "no spatial coordinate columns");
        let xy = srt.coordinates.columns(0, d).into_owned();
        let section: Vec<usize> = if has_batch {
            let keys = srt.coordinates.column(d);
            let ids: Vec<u32> = keys.iter().map(|&v| v.to_bits()).collect();
            crate::util::ablation::dense_ids(ids.into_iter())
                .into_iter()
                .map(|s| s as usize)
                .collect()
        } else {
            vec![0; cells.len()]
        };
        let n_sections = section.iter().max().map_or(0, |&m| m + 1);
        log::info!(
            "{} cells, {} genes, {} sections, {d}-D coordinates",
            cells.len(),
            genes.len(),
            n_sections
        );
        Ok(SpatialData {
            counts,
            genes,
            cells,
            xy,
            section,
            n_sections,
        })
    }
}
