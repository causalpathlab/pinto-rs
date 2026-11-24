//! `pinto lra --shard I/N` and `pinto lra merge`: one run's pairs in `N`
//! independent pieces, run as separate jobs (a Slurm array), then put back
//! together.
//!
//! A shard estimates the pairs dealt to it ([`deal`]), weighed by their
//! competent receivers: a ligand's pairs stay together, as they share its
//! base propensities, unless they weigh more than a shard's share. Each
//! pair's estimate depends only on the tissue and its own seed, so a pair
//! comes out of its shard as it would of the whole run. A shard writes
//! no table and records nothing in the manifest: it writes
//! `{out}.lra.shard-I-of-N.cbor` ([`super::link_community_scope::shard_path`]), its pairs' rows before the
//! p-values and co-expression's local fdr.
//!
//! `pinto lra merge` reads every shard of a run, checks they are its `N`
//! shards (one run key, each `I` once, every pair once), and writes the
//! tables a run without `--shard` writes, its p-values and co-expression's
//! local fdr over every pair, recording them in the run's manifest.

use super::baselines::TypeSource;
use super::null_draws::NullRow;
use super::output::CelltypeRow;
use super::reference::hex;
use super::rows::EffectRow;
use crate::util::metadata::LraOptions;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::path::PathBuf;

/// The shard file's layout; a merge refuses any other.
pub const FORMAT: u32 = 2;

/// Shard `index` (0-based) of `count`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shard {
    pub index: usize,
    pub count: usize,
}

impl std::str::FromStr for Shard {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let bad = || format!("{s:?}: give I/N, shard I (0-based) of N, as 0/16");
        let (i, n) = s.split_once('/').ok_or_else(bad)?;
        let index: usize = i.trim().parse().map_err(|_| bad())?;
        let count: usize = n.trim().parse().map_err(|_| bad())?;
        if count == 0 || index >= count {
            return Err(format!("{s:?}: shard I of N needs 0 <= I < N"));
        }
        Ok(Self { index, count })
    }
}

impl std::fmt::Display for Shard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.index, self.count)
    }
}

/// `{out}.lra-simulate.shard-I-of-N.cbor`.
pub fn sim_path(out: &str, shard: Shard) -> String {
    format!(
        "{out}.lra-simulate.shard-{}-of-{}.cbor",
        shard.index, shard.count
    )
}

/// The shard each pair goes to, of `count`, the pairs of `groups` (each
/// ligand's, indices into `weight`) weighing `weight` each: a ligand whose
/// pairs weigh more than a shard's share of the whole is cut into runs of
/// its pairs of at most that share (a pair alone gets what it gets beside
/// its ligand's others, only its base propensities refitted); then the
/// pieces, heaviest first, each to the shard weighing least so far, the
/// lowest on a tie. The same groups, weights and `count` always deal the
/// same way.
pub fn deal(groups: &[Vec<usize>], weight: &[f64], count: usize) -> Vec<usize> {
    let share = weight.iter().sum::<f64>() / count as f64;
    let mut pieces: Vec<(f64, Vec<usize>)> = Vec::new();
    for g in groups {
        let mut piece: (f64, Vec<usize>) = (0.0, Vec::new());
        for &k in g {
            if !piece.1.is_empty() && piece.0 + weight[k] > share {
                pieces.push(std::mem::take(&mut piece));
            }
            piece.0 += weight[k];
            piece.1.push(k);
        }
        if !piece.1.is_empty() {
            pieces.push(piece);
        }
    }
    // heaviest first, then by first pair, so the deal is a function of the
    // inputs alone
    pieces.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1[0].cmp(&b.1[0])));
    let mut load = vec![0f64; count];
    let mut shard_of = vec![usize::MAX; weight.len()];
    for (w, piece) in pieces {
        let s = (0..count)
            .min_by(|&a, &b| load[a].total_cmp(&load[b]).then(a.cmp(&b)))
            .expect("count > 0");
        load[s] += w;
        for k in piece {
            shard_of[k] = s;
        }
    }
    shard_of
}

/// What the tables of a run are recorded with in its manifest, beside the
/// rows: the same for every shard of one run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TableRecord {
    /// The manifest the run records in: the `--from` run's (`from_run`),
    /// shared by every shard, else the shard's own `{out}.pinto.json`, in
    /// whose place a merge writes its own.
    pub manifest: PathBuf,
    pub from_run: bool,
    pub abf_prior_scale: f64,
    /// [`super::Settings::options`] with the identity's left-out genes and
    /// the baselines' type source filled in.
    pub options: LraOptions,
    pub reference_record: Option<PathBuf>,
    /// Pairs the full analysis would estimate.
    pub n_all: usize,
    /// A hash of the tissue the pairs are estimated on ([`tissue_key`]), so
    /// shards of another data set never merge.
    pub tissue: String,
}

/// SHA-256 of the cells' count, the features' names, the spatial contacts,
/// each cell's coordinates and section.
pub fn tissue_key(
    n_cells: usize,
    features: &[Box<str>],
    contacts: &[(usize, usize)],
    coords: &[Option<(f32, f32)>],
    cell_section: &[Box<str>],
) -> String {
    let mut h = sha2::Sha256::new();
    h.update((n_cells as u64).to_le_bytes());
    for f in features {
        h.update(f.as_bytes());
        h.update([0]);
    }
    for &(a, b) in contacts {
        h.update((a as u64).to_le_bytes());
        h.update((b as u64).to_le_bytes());
    }
    for c in coords {
        match c {
            Some((x, y)) => {
                h.update([1]);
                h.update(x.to_le_bytes());
                h.update(y.to_le_bytes());
            }
            None => h.update([0]),
        }
    }
    for s in cell_section {
        h.update(s.as_bytes());
        h.update([0]);
    }
    hex(h.finalize().as_slice())
}

/// What every shard file starts with.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Header {
    pub format: u32,
    pub pinto_version: String,
    pub shard: Shard,
    /// A hash of what decides the run's pieces and how they are finished
    /// ([`run_key`], [`sim_run_key`]): every shard of one run has the same.
    pub run_key: String,
}

impl Header {
    pub fn new(shard: Shard, run_key: String) -> Self {
        Self {
            format: FORMAT,
            pinto_version: env!("CARGO_PKG_VERSION").into(),
            shard,
            run_key,
        }
    }
}

/// A shard file: of `pinto lra` or of `pinto lra simulate`.
#[derive(Serialize, Deserialize)]
pub enum Contents {
    Lra(Box<ShardFile>),
    Simulate(SimShardFile),
}

impl Contents {
    fn header(&self) -> &Header {
        match self {
            Contents::Lra(f) => &f.header,
            Contents::Simulate(f) => &f.header,
        }
    }
}

/// One `pinto lra` shard's file.
#[derive(Serialize, Deserialize)]
pub struct ShardFile {
    pub header: Header,
    pub record: TableRecord,
    /// Each of the shard's pairs: its place in the full list, its rows
    /// before [`super::report::report`], and with `--marginal-baselines`
    /// its cell-type rows.
    pub pairs: Vec<ShardPair>,
    /// With `--marginal-baselines`, where the cell types came from.
    pub celltype_source: Option<TypeSource>,
}

#[derive(Serialize, Deserialize)]
pub struct ShardPair {
    pub index: usize,
    pub rows: Vec<EffectRow>,
    pub celltype: Vec<CelltypeRow>,
    /// With `--null-permutations`, the pair's null rows.
    #[serde(default)]
    pub null: Vec<NullRow>,
}

/// One `pinto lra simulate` shard's file: its runs of the `n_runs` of the
/// whole call, and how the call's tables are finished.
#[derive(Serialize, Deserialize)]
pub struct SimShardFile {
    pub header: Header,
    pub n_runs: usize,
    pub finish: SimFinish,
    pub runs: Vec<SimRun>,
}

/// How a simulate call ends once every run is in.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SimFinish {
    /// The built-in grid: the table, the reference data's provenance (as
    /// JSON) beside it, recorded in the manifest.
    Grid {
        manifest: PathBuf,
        from_run: bool,
        abf_prior_scale: f64,
        plant_region: Option<String>,
        reference_provenance: Option<String>,
    },
    /// One scenario (`--one`): the scored table and the sidecar, `record`
    /// the sidecar's JSON but for its pairs and table.
    One {
        plant_region: Option<String>,
        record: String,
    },
}

/// One run of a simulate call: its place among the call's runs, its row
/// (`None` for a grid scenario that failed), and for `--one` its pair's
/// abundance rank, label and seed.
#[derive(Serialize, Deserialize)]
pub struct SimRun {
    pub index: usize,
    pub row: Option<super::simulate::SimRow>,
    pub rank: usize,
    pub planted: bool,
    pub seed: u64,
}

/// The run key: SHA-256 of pinto's version, the record but for the
/// manifest (each shard of a run on data files has its own) and every
/// pair's name in order.
pub fn run_key<'a>(
    record: &TableRecord,
    pair_names: impl Iterator<Item = (&'a str, &'a str)>,
) -> String {
    let keyed = TableRecord {
        manifest: if record.from_run {
            record.manifest.clone()
        } else {
            PathBuf::new()
        },
        ..record.clone()
    };
    let mut h = sha2::Sha256::new();
    h.update(env!("CARGO_PKG_VERSION"));
    h.update(serde_json::to_vec(&keyed).unwrap_or_default());
    for (l, r) in pair_names {
        h.update(l);
        h.update([0]);
        h.update(r);
        h.update([1]);
    }
    hex(h.finalize().as_slice())
}

/// A simulate call's run key: SHA-256 of pinto's version, the tissue, how
/// the call finishes (its manifest left out when a shard has its own) and
/// `runs`, a description of every run in order.
pub fn sim_run_key(tissue: &str, finish: &SimFinish, runs: &str) -> String {
    let keyed = match finish {
        SimFinish::Grid {
            from_run: false, ..
        } => {
            let mut f = finish.clone();
            if let SimFinish::Grid { manifest, .. } = &mut f {
                *manifest = PathBuf::new();
            }
            f
        }
        _ => finish.clone(),
    };
    let mut h = sha2::Sha256::new();
    h.update(env!("CARGO_PKG_VERSION"));
    h.update(tissue);
    h.update(serde_json::to_vec(&keyed).unwrap_or_default());
    h.update(runs);
    hex(h.finalize().as_slice())
}

/// Write `file` to `path`.
pub fn write(path: &str, file: &Contents) -> anyhow::Result<()> {
    legume_numeric::matrix::common_io::mkdir_parent(path)?;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    ciborium::into_writer(file, &mut w)
        .map_err(|e| anyhow::anyhow!("{path}: could not write the shard: {e}"))?;
    Ok(())
}

/// What a shard file is of, as a merge of it writes.
#[cfg(feature = "view")]
pub enum Kind {
    /// `pinto lra`, with the cell-type rows (`--marginal-baselines`) or not,
    /// of the marginal run or of a `--link-community`.
    Lra {
        celltype: bool,
        link_community: Option<i64>,
    },
    /// `pinto lra simulate`, one scenario (`--one`) or the built-in grid.
    Simulate { one: bool },
}

/// The run key of the shard file at `path`, and what it is of.
#[cfg(feature = "view")]
pub fn peek(path: &str) -> anyhow::Result<(String, Kind)> {
    Ok(match read(path)? {
        Contents::Lra(f) => (
            f.header.run_key,
            Kind::Lra {
                celltype: f.celltype_source.is_some(),
                link_community: f.record.options.link_community,
            },
        ),
        Contents::Simulate(f) => (
            f.header.run_key,
            Kind::Simulate {
                one: matches!(f.finish, SimFinish::One { .. }),
            },
        ),
    })
}

fn read(path: &str) -> anyhow::Result<Contents> {
    let r = std::io::BufReader::new(
        std::fs::File::open(path).map_err(|e| anyhow::anyhow!("{path}: {e}"))?,
    );
    let file: Contents = ciborium::from_reader(r)
        .map_err(|e| anyhow::anyhow!("{path}: not a pinto lra shard file ({e})"))?;
    let format = file.header().format;
    anyhow::ensure!(
        format == FORMAT,
        "{path}: a shard file of layout {format} (this pinto reads {FORMAT})"
    );
    Ok(file)
}

/// The shard files at `paths`, in shard order, checked to be every shard of
/// one run: the same run key and count, each shard once.
pub fn read_set(paths: &[String]) -> anyhow::Result<Vec<Contents>> {
    anyhow::ensure!(!paths.is_empty(), "no shard files given");
    let mut files: Vec<(String, Contents)> = paths
        .iter()
        .map(|p| read(p).map(|f| (p.clone(), f)))
        .collect::<anyhow::Result<_>>()?;
    files.sort_by_key(|(_, f)| f.header().shard.index);
    let (first_path, first) = (&files[0].0, files[0].1.header());
    let count = first.shard.count;
    for (p, f) in &files {
        let h = f.header();
        anyhow::ensure!(
            h.run_key == first.run_key,
            "{p} is a shard of another run than {first_path} (its settings, pairs, tissue or \
             pinto version differ)"
        );
        anyhow::ensure!(
            h.shard.count == count,
            "{p} is shard {} but {first_path} is shard {}",
            h.shard,
            first.shard
        );
    }
    for w in files.windows(2) {
        let (a, b) = (w[0].1.header(), w[1].1.header());
        anyhow::ensure!(
            a.shard.index != b.shard.index,
            "shard {} given twice: {} and {}",
            a.shard,
            w[0].0,
            w[1].0
        );
    }
    let mut have = vec![false; count];
    for (_, f) in &files {
        have[f.header().shard.index] = true;
    }
    let missing: Vec<String> = (0..count)
        .filter(|&i| !have[i])
        .map(|i| format!("{i}/{count}"))
        .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "missing shard(s) {} of {count}",
        missing.join(", ")
    );
    Ok(files.into_iter().map(|(_, f)| f).collect())
}

/// Every shard of a `pinto lra` run, put together: the pair rows and
/// cell-type rows of every pair in pair order.
pub struct Merged {
    pub run_key: String,
    pub record: TableRecord,
    pub count: usize,
    pub rows: Vec<EffectRow>,
    /// With `--marginal-baselines`, every pair's cell-type rows.
    pub celltype: Option<Vec<CelltypeRow>>,
    /// With `--null-permutations`, every pair's null rows.
    pub null: Option<Vec<NullRow>>,
}

/// [`read_set`]'s files as one `pinto lra` run.
pub fn merge_lra(files: Vec<Contents>) -> anyhow::Result<Merged> {
    let count = files.len();
    let files: Vec<ShardFile> = files
        .into_iter()
        .map(|f| match f {
            Contents::Lra(f) => Ok(*f),
            Contents::Simulate(_) => {
                anyhow::bail!("a pinto lra simulate shard among pinto lra shards")
            }
        })
        .collect::<anyhow::Result<_>>()?;
    let run_key = files[0].header.run_key.clone();
    let record = files[0].record.clone();
    let source = files[0].celltype_source;
    let mut pairs: Vec<ShardPair> = files.into_iter().flat_map(|f| f.pairs).collect();
    pairs.sort_by_key(|p| p.index);
    anyhow::ensure!(
        pairs.len() == record.n_all && pairs.iter().enumerate().all(|(k, p)| p.index == k),
        "the shards hold {} of the run's {} pairs, not each once",
        pairs.len(),
        record.n_all
    );
    let rows = pairs.iter_mut().flat_map(|p| p.rows.drain(..)).collect();
    let null = record
        .options
        .null_permutations
        .map(|_| pairs.iter_mut().flat_map(|p| p.null.drain(..)).collect());
    let celltype = source.map(|s| {
        pairs
            .into_iter()
            .flat_map(|p| p.celltype)
            .map(|r| CelltypeRow {
                labels: s.name(),
                ..r
            })
            .collect()
    });
    Ok(Merged {
        run_key,
        record,
        count,
        rows,
        celltype,
        null,
    })
}

/// [`read_set`]'s files as one `pinto lra simulate` call: its run key, how
/// it finishes, and every run in order.
pub fn merge_sim(files: Vec<Contents>) -> anyhow::Result<(String, SimFinish, Vec<SimRun>)> {
    let files: Vec<SimShardFile> = files
        .into_iter()
        .map(|f| match f {
            Contents::Simulate(f) => Ok(f),
            Contents::Lra(_) => anyhow::bail!("a pinto lra shard among pinto lra simulate shards"),
        })
        .collect::<anyhow::Result<_>>()?;
    let run_key = files[0].header.run_key.clone();
    let (n_runs, finish) = (files[0].n_runs, files[0].finish.clone());
    let mut runs: Vec<SimRun> = files.into_iter().flat_map(|f| f.runs).collect();
    runs.sort_by_key(|r| r.index);
    anyhow::ensure!(
        runs.len() == n_runs && runs.iter().enumerate().all(|(k, r)| r.index == k),
        "the shards hold {} of the call's {n_runs} runs, not each once",
        runs.len()
    );
    Ok((run_key, finish, runs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shard_is_i_of_n() {
        let s: Shard = "3/16".parse().unwrap();
        assert_eq!((s.index, s.count), (3, 16));
        assert_eq!(s.to_string(), "3/16");
        for bad in ["16/16", "0/0", "3", "a/4", "-1/4"] {
            assert!(bad.parse::<Shard>().is_err(), "{bad}");
        }
    }

    #[test]
    fn ligands_are_dealt_whole_unless_heavier_than_a_share() {
        // pairs 0..10 in five ligands of sizes 4, 3, 1, 1, 1, each pair of
        // weight one
        let groups = vec![vec![0, 1, 2, 3], vec![4, 5, 6], vec![7], vec![8], vec![9]];
        let ones = vec![1.0; 10];
        let of = deal(&groups, &ones, 3);
        let mut load = [0; 3];
        for &s in &of {
            load[s] += 1;
        }
        // a share is 10/3: the four-pair ligand is cut, the three-pair one is not
        assert!(groups[1].iter().all(|&k| of[k] == of[4]), "{of:?}");
        assert_eq!(load.iter().max(), Some(&4), "{load:?}");
        assert_eq!(of, deal(&groups, &ones, 3), "the same deal every time");
        assert!(deal(&groups, &ones, 1).iter().all(|&s| s == 0));
        // the weights, not the pair counts, are dealt: one heavy pair alone
        let mut heavy = ones.clone();
        heavy[9] = 9.0;
        let of = deal(&groups, &heavy, 2);
        assert!((0..9).all(|k| of[k] != of[9]), "{of:?}");
    }
}
