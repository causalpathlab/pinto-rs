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
//! `{out}.lra.shard-I-of-N.cbor` ([`path`]), its pairs' rows before the
//! q-values and verdicts.
//!
//! `pinto lra merge` reads every shard of a run, checks they are its `N`
//! shards (one run key, each `I` once, every pair once), and writes the
//! tables a run without `--shard` writes, its q-values and verdicts over
//! every pair, recording them in the run's manifest.

use super::baselines::TypeSource;
use super::output::CelltypeRow;
use super::rows::EffectRow;
use crate::util::metadata::LraOptions;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::path::PathBuf;

/// The shard file's layout; a merge refuses any other.
pub const FORMAT: u32 = 1;

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

/// `{out}.lra.shard-I-of-N.cbor`.
pub fn path(out: &str, shard: Shard) -> String {
    format!("{out}.lra.shard-{}-of-{}.cbor", shard.index, shard.count)
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
}

/// One shard's file.
#[derive(Serialize, Deserialize)]
pub struct ShardFile {
    pub format: u32,
    pub pinto_version: String,
    pub shard: Shard,
    /// A hash of the record and the full pair list ([`run_key`]): every
    /// shard of one run has the same.
    pub run_key: String,
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
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Write `file` to `path`.
pub fn write(path: &str, file: &ShardFile) -> anyhow::Result<()> {
    legume_numeric::matrix::common_io::mkdir_parent(path)?;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    ciborium::into_writer(file, &mut w)
        .map_err(|e| anyhow::anyhow!("{path}: could not write the shard: {e}"))?;
    Ok(())
}

fn read(path: &str) -> anyhow::Result<ShardFile> {
    let r = std::io::BufReader::new(
        std::fs::File::open(path).map_err(|e| anyhow::anyhow!("{path}: {e}"))?,
    );
    let file: ShardFile = ciborium::from_reader(r)
        .map_err(|e| anyhow::anyhow!("{path}: not a pinto lra shard file ({e})"))?;
    anyhow::ensure!(
        file.format == FORMAT,
        "{path}: a shard file of layout {} (this pinto reads {FORMAT})",
        file.format
    );
    Ok(file)
}

/// The shards at `paths`, checked to be every shard of one run, in shard
/// order, and the pair rows and cell-type rows of every pair in pair order.
pub struct Merged {
    pub record: TableRecord,
    pub count: usize,
    pub rows: Vec<EffectRow>,
    pub celltype: Option<(TypeSource, Vec<CelltypeRow>)>,
}

pub fn read_all(paths: &[String]) -> anyhow::Result<Merged> {
    anyhow::ensure!(!paths.is_empty(), "no shard files given");
    let mut files: Vec<(String, ShardFile)> = paths
        .iter()
        .map(|p| read(p).map(|f| (p.clone(), f)))
        .collect::<anyhow::Result<_>>()?;
    files.sort_by_key(|(_, f)| f.shard.index);
    let (first_path, first) = &files[0];
    let (count, key) = (first.shard.count, first.run_key.clone());
    for (p, f) in &files {
        anyhow::ensure!(
            f.run_key == key && f.pinto_version == first.pinto_version,
            "{p} is a shard of another run than {first_path} (its settings, pairs or pinto \
             version differ)"
        );
        anyhow::ensure!(
            f.shard.count == count,
            "{p} is shard {} but {first_path} is shard {}",
            f.shard,
            first.shard
        );
    }
    for w in files.windows(2) {
        anyhow::ensure!(
            w[0].1.shard.index != w[1].1.shard.index,
            "shard {} given twice: {} and {}",
            w[0].1.shard,
            w[0].0,
            w[1].0
        );
    }
    let have: Vec<usize> = files.iter().map(|(_, f)| f.shard.index).collect();
    let missing: Vec<String> = (0..count)
        .filter(|i| !have.contains(i))
        .map(|i| format!("{i}/{count}"))
        .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "missing shard(s) {} of {count}",
        missing.join(", ")
    );
    let record = first.record.clone();
    let source = first.celltype_source;
    let mut pairs: Vec<ShardPair> = files.into_iter().flat_map(|(_, f)| f.pairs).collect();
    pairs.sort_by_key(|p| p.index);
    anyhow::ensure!(
        pairs.windows(2).all(|w| w[0].index != w[1].index),
        "a pair is in two shards"
    );
    let rows = pairs.iter_mut().flat_map(|p| p.rows.drain(..)).collect();
    let celltype = source.map(|s| {
        let rows = pairs
            .into_iter()
            .flat_map(|p| p.celltype)
            .map(|r| CelltypeRow {
                labels: s.name(),
                ..r
            })
            .collect();
        (s, rows)
    });
    Ok(Merged {
        record,
        count,
        rows,
        celltype,
    })
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
