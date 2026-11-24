//! What stage 2 reads from a finished stage-1 run, frozen: the cell pairs,
//! each pair's stage-1 embedding and community, each pair's section and
//! donor, and each pair's pooled raw counts.

use crate::util::common::*;
use crate::util::metadata::Run;

/// The stage-1 run's cell pairs, in the order of its `coord_pairs` table
/// (which is also the row order of its `latent` and `link_community`).
pub struct Stage1CellPairs {
    /// Each pair's first cell, as the data's column names name it.
    pub left_cell_of_pair: Vec<Box<str>>,
    /// Each pair's second cell.
    pub right_cell_of_pair: Vec<Box<str>>,
    /// Each pair's section (its first cell's batch label in the run).
    pub section_of_pair: Vec<Box<str>>,
    /// Each pair's stage-1 link community (the run's domain label for it).
    pub community_of_pair: Vec<Box<str>>,
    /// `[pairs × embedding dims]`: each pair's stage-1 embedding, the shared
    /// space the matched other-donor pairs are looked up in.
    pub stage1_pair_embedding: Mat,
}

impl Stage1CellPairs {
    pub fn number_of_pairs(&self) -> usize {
        self.left_cell_of_pair.len()
    }
}

/// The stage-1 run's latent, `{prefix}.latent.parquet` (the manifest records
/// no slot for it).
fn latent_file(run: &Run) -> anyhow::Result<String> {
    let path = format!("{}.latent.parquet", run.prefix);
    anyhow::ensure!(
        std::path::Path::new(&path).is_file(),
        "{path}: the stage-1 run has no pair latent; donor-deviation needs a run that \
         embedded cell pairs (cage or dsvd; lc writes no pair embedding)"
    );
    Ok(path)
}

/// An output the stage-1 run's manifest records, as a path string.
fn recorded_run_file(run: &Run, recorded: Option<&str>, suffix: &str) -> anyhow::Result<String> {
    Ok(run
        .required_recorded_output(recorded, suffix)?
        .to_string_lossy()
        .into_owned())
}

/// Read the stage-1 run's cell pairs, their sections, communities and
/// embedding, and check that the three tables list the same pairs in the
/// same order. Only spatial contacts within one section are kept: a pair
/// joining two sections (whose cells could have two donors), or one the run
/// added for expression similarity, is left out (`lr_activity::io`'s pair
/// reader decides both, as for lra). The embedding is taken at unit length,
/// the scale cage and dsvd write it at.
pub fn read_stage1_cell_pairs(run: &Run) -> anyhow::Result<Stage1CellPairs> {
    use crate::util::parquet_io::{labels, read_link_community_labels, read_named_columns};
    let outputs = &run.meta.outputs;
    let coord_pairs_path =
        recorded_run_file(run, outputs.coord_pairs.as_deref(), "coord_pairs.parquet")?;
    let pair_records = crate::lr_activity::io::read_pair_table(&coord_pairs_path, false)?;
    let cell_pair_ids =
        labels(&read_named_columns(std::path::Path::new(&coord_pairs_path), &["cell_pair"])?[0])?;
    let mut section_of_record = Vec::with_capacity(pair_records.len());
    for record in &pair_records {
        let (left_batch, right_batch) = record.side_batches.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "{coord_pairs_path}: no left_batch/right_batch; donor-deviation needs each \
                 pair's batch, which a run with more than one batch records"
            )
        })?;
        section_of_record
            .push((left_batch == right_batch && record.is_spatial).then_some(left_batch));
    }
    let number_left_out = section_of_record.iter().filter(|s| s.is_none()).count();
    if number_left_out > 0 {
        info!(
            "{number_left_out} of {} cell pairs left out: across sections or expression-similar",
            pair_records.len()
        );
    }

    let community_path = recorded_run_file(
        run,
        outputs.link_community.as_deref(),
        "link_community.parquet",
    )?;
    let community_cells = read_named_columns(
        std::path::Path::new(&community_path),
        &["left_cell", "right_cell"],
    )?;
    let (community_left, community_right) =
        (labels(&community_cells[0])?, labels(&community_cells[1])?);
    anyhow::ensure!(
        community_left.len() == pair_records.len()
            && pair_records.iter().enumerate().all(|(p, record)| {
                community_left[p] == record.left_cell && community_right[p] == record.right_cell
            }),
        "{community_path} and {coord_pairs_path} list different cell pairs"
    );
    let community_label = read_link_community_labels(std::path::Path::new(&community_path))?;

    let latent_path = latent_file(run)?;
    let latent = Mat::from_parquet(&latent_path)?;
    anyhow::ensure!(
        latent.rows == cell_pair_ids,
        "{latent_path} and {coord_pairs_path} list different cell pairs (or in another order)"
    );

    let kept_rows: Vec<usize> = (0..pair_records.len())
        .filter(|&p| section_of_record[p].is_some())
        .collect();
    let mut stage1_pair_embedding = latent.mat.select_rows(kept_rows.iter());
    legume_numeric::matrix::dense_mat_io::l2_normalize_rows_inplace(&mut stage1_pair_embedding);
    Ok(Stage1CellPairs {
        left_cell_of_pair: kept_rows
            .iter()
            .map(|&p| pair_records[p].left_cell.clone())
            .collect(),
        right_cell_of_pair: kept_rows
            .iter()
            .map(|&p| pair_records[p].right_cell.clone())
            .collect(),
        section_of_pair: kept_rows
            .iter()
            .map(|&p| section_of_record[p].expect("kept").clone())
            .collect(),
        community_of_pair: kept_rows
            .iter()
            .map(|&p| community_label[p].to_string().into())
            .collect(),
        stage1_pair_embedding,
    })
}

/// Each section's donor, from a two-column text file (`section donor` per
/// line, separated by a tab, a comma or spaces; a `batch donor` header line
/// is skipped). Every section the run's pairs sit in must be listed.
pub fn read_donor_of_section(path: &str) -> anyhow::Result<HashMap<Box<str>, Box<str>>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("--donor-of-batch {path}: {e}"))?;
    let mut donor_of_section = HashMap::default();
    for (line_number, line) in text.lines().enumerate() {
        let fields: Vec<&str> = line
            .split(|c: char| c == '\t' || c == ',' || c.is_whitespace())
            .filter(|f| !f.is_empty())
            .collect();
        match fields.as_slice() {
            [] => continue,
            ["batch", "donor"] | ["section", "donor"] => continue,
            [section, donor] => {
                if let Some(earlier) =
                    donor_of_section.insert(Box::<str>::from(*section), Box::<str>::from(*donor))
                {
                    anyhow::ensure!(
                        earlier.as_ref() == *donor,
                        "--donor-of-batch {path} line {}: batch {section} is given donor \
                         {donor}, and donor {earlier} on an earlier line",
                        line_number + 1
                    );
                }
            }
            _ => anyhow::bail!(
                "--donor-of-batch {path} line {}: expected two fields (batch, donor), got {}",
                line_number + 1,
                fields.len()
            ),
        }
    }
    Ok(donor_of_section)
}

/// The donors the pairs belong to, sorted, and each pair's donor as an
/// index into them.
pub struct DonorsOfPairs {
    pub donor_names: Vec<Box<str>>,
    pub donor_index_of_pair: Vec<usize>,
}

/// Give every pair its donor through its section. Only donors that hold a
/// pair are listed (a donor in the file but not in the run would otherwise
/// read as one tested and found not to deviate). Refuse fewer than three:
/// with two, a deviation is only a contrast between them, and which of the
/// two deviates is not identified.
pub fn donors_of_pairs(
    section_of_pair: &[Box<str>],
    donor_of_section: &HashMap<Box<str>, Box<str>>,
) -> anyhow::Result<DonorsOfPairs> {
    let donor_of_pair: Vec<&str> = section_of_pair
        .iter()
        .map(|section| {
            donor_of_section
                .get(section)
                .map(|d| d.as_ref())
                .ok_or_else(|| {
                    anyhow::anyhow!("section {section} has no donor in --donor-of-batch")
                })
        })
        .collect::<anyhow::Result<_>>()?;
    let mut donor_names: Vec<Box<str>> = donor_of_pair.iter().map(|&d| d.into()).collect();
    donor_names.sort();
    donor_names.dedup();
    anyhow::ensure!(
        donor_names.len() >= 3,
        "donor-deviation needs at least three donors (found {}): with two, which one \
         deviates is not identified",
        donor_names.len()
    );
    let donor_index: HashMap<&str, usize> = donor_names
        .iter()
        .enumerate()
        .map(|(i, d)| (d.as_ref(), i))
        .collect();
    let donor_index_of_pair = donor_of_pair.iter().map(|d| donor_index[d]).collect();
    Ok(DonorsOfPairs {
        donor_names,
        donor_index_of_pair,
    })
}

/// Each gene's pooled raw counts over the pairs: a pair's count of a gene
/// is the sum of its two cells' counts, the pair activity cage trains on.
pub struct PairCountsByGene {
    pub gene_names: Vec<Box<str>>,
    /// For each gene, the pairs with a nonzero pooled count, as
    /// (pair index, pooled count), sorted by pair index.
    pub nonzero_pairs_of_gene: Vec<Vec<(u32, f32)>>,
}

/// Cells read per block when pooling counts.
const CELLS_PER_READ_BLOCK: usize = 20_000;

/// Pool every pair's two cells' raw counts, gene by gene. `data` holds the
/// cells; a pair whose cell is not in it is an error.
pub fn pool_pair_counts_by_gene(
    data: &SparseIoVec,
    pairs: &Stage1CellPairs,
) -> anyhow::Result<PairCountsByGene> {
    let gene_names = data.row_names()?;
    let cell_names = data.column_names()?;
    let column_of_cell: HashMap<&str, usize> = cell_names
        .iter()
        .enumerate()
        .map(|(j, c)| (c.as_ref(), j))
        .collect();
    let column_of = |cell: &str| -> anyhow::Result<usize> {
        column_of_cell
            .get(cell)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("cell {cell} of the stage-1 pairs is not in the data"))
    };
    // pairs each cell belongs to, so each cell's counts are read once
    let mut pairs_of_cell: Vec<Vec<u32>> = vec![Vec::new(); cell_names.len()];
    for pair in 0..pairs.number_of_pairs() {
        pairs_of_cell[column_of(&pairs.left_cell_of_pair[pair])?].push(pair as u32);
        pairs_of_cell[column_of(&pairs.right_cell_of_pair[pair])?].push(pair as u32);
    }
    // per gene, (pair, a cell's count) for each of the pair's cells with the
    // gene: at most two entries per pair, merged below
    let mut pooled: Vec<Vec<(u32, f32)>> = vec![Vec::new(); gene_names.len()];
    for block_start in (0..cell_names.len()).step_by(CELLS_PER_READ_BLOCK) {
        let block_end = (block_start + CELLS_PER_READ_BLOCK).min(cell_names.len());
        let counts = data.read_columns_csc(block_start..block_end)?;
        for (offset, cell_counts) in counts.col_iter().enumerate() {
            let cell_pairs = &pairs_of_cell[block_start + offset];
            if cell_pairs.is_empty() {
                continue;
            }
            for (&gene, &count) in cell_counts.row_indices().iter().zip(cell_counts.values()) {
                pooled[gene].extend(cell_pairs.iter().map(|&pair| (pair, count)));
            }
        }
    }
    let nonzero_pairs_of_gene = pooled
        .into_par_iter()
        .map(|mut entries| {
            entries.sort_unstable_by_key(|&(pair, _)| pair);
            let mut merged: Vec<(u32, f32)> = Vec::with_capacity(entries.len());
            for (pair, count) in entries {
                match merged.last_mut() {
                    Some(last) if last.0 == pair => last.1 += count,
                    _ => merged.push((pair, count)),
                }
            }
            merged.retain(|&(_, count)| count > 0.0);
            merged
        })
        .collect();
    Ok(PairCountsByGene {
        gene_names,
        nonzero_pairs_of_gene,
    })
}
