//! The lra layer of `pinto view`: a run's contact-effect tables (`pinto
//! lra`'s full table and its `--only` subsets), their pooled pairs as a
//! list to search by gene and sort, and one pair on the
//! map: each receiver's contact exposure to the ligand, exposed or not,
//! receptor-competent or not, and the senders, computed as lra computes
//! them ([`RunFrame`]); with the pair's threshold–response curve
//! ([`super::curve`]).
//!
//! The receivers are read once per run (again for a table made with other
//! options), the first time a pair is drawn; each pair after that costs
//! its ligand's and receptor's rows, and the last few drawn are kept.

use super::color::{Rgb, Theme};
use super::curve::Curve;
use super::data::{Communities, NO_CLUSTER};
use super::gene::{GeneMap, Source};
use super::lupin::same_file;
use super::render::{Layer, Style};
use super::StillPair;
use super::{Base, Level};
use crate::lr_activity::args::LraMethod;
use crate::lr_activity::ligand_effect::frame::{Class, PairMap, RunFrame};
use crate::lr_activity::ligand_effect::only;
use crate::lr_activity::ligand_effect::output::{
    curve_beside, percent_text, read_curves, read_pair_rows, CurveRow, Curves, PairRow, Q_SCOPE_ALL,
};
use crate::util::metadata::{LraOptions, LraSubset};
use std::cell::OnceCell;
use std::path::{Path, PathBuf};

/// Percentile the exposure ramp tops out at.
const EXPOSURE_CLIP: f32 = 99.;

/// Pairs drawn kept, the newest; one drawn again after that is redrawn.
const DRAWN_KEPT: usize = 3;

// * Tables

/// A contact-effect table of the run, and what it was made with.
#[derive(Clone, Debug, PartialEq)]
pub struct TableRef {
    pub path: PathBuf,
    pub curve: Option<PathBuf>,
    /// The `--only` subset it holds; `None` for the full table.
    pub subset: Option<LraSubset>,
    pub options: LraOptions,
}

impl TableRef {
    /// `full table`, or the subset's list.
    pub fn label(&self) -> String {
        match &self.subset {
            None => "full table".into(),
            Some(s) => format!("only {}", s.only.join(",")),
        }
    }

    /// A table named by its path (`--lra-table`): a subset when its name
    /// says so, its curve beside it.
    pub fn at(path: &Path, run: &Run) -> Self {
        let recorded = run.tables().into_iter().find(|t| same_file(&t.path, path));
        if let Some(t) = recorded {
            return t;
        }
        let name = path.to_string_lossy();
        let curve = PathBuf::from(curve_beside(&name));
        // a --link-community table names its community, which its exposure
        // runs through, so it is never read as the marginal run's
        let link_community =
            crate::lr_activity::ligand_effect::link_community_scope::link_community_in_name(&name);
        TableRef {
            path: path.to_path_buf(),
            curve: curve.is_file().then_some(curve),
            subset: (name.contains(".lra.only-") || link_community.is_some()).then(|| LraSubset {
                table: name.to_string(),
                ..Default::default()
            }),
            options: LraOptions {
                link_community,
                ..run.lra_options()
            },
        }
    }
}

use super::data::Run;

impl Run {
    /// The options the run's full table was made with, else lra's defaults.
    pub fn lra_options(&self) -> LraOptions {
        self.meta.outputs.lra.options.clone().unwrap_or_default()
    }

    /// The run's contact-effect tables on disk: the full table first, then
    /// each `--only` subset, as the manifest records them.
    pub fn tables(&self) -> Vec<TableRef> {
        let lra = &self.meta.outputs.lra;
        let file = |p: &str| Some(self.resolve(p)).filter(|p| p.is_file());
        let mut out = Vec::new();
        if let Some(path) = lra.table.as_deref().and_then(file) {
            out.push(TableRef {
                path,
                curve: lra.curve.as_deref().and_then(file),
                subset: None,
                options: self.lra_options(),
            });
        }
        for s in &lra.subsets {
            if let Some(path) = file(&s.table) {
                out.push(TableRef {
                    path,
                    curve: file(&s.curve),
                    options: lra.subset_options(s).unwrap_or_else(|| self.lra_options()),
                    subset: Some(s.clone()),
                });
            }
        }
        out
    }
}

/// A table read: its pooled rows and its curves.
pub struct Table {
    pub r: TableRef,
    pub rows: Vec<PairRow>,
    /// Which pairs its local fdrs were fitted over (its `q_scope`
    /// column).
    pub q_scope: Box<str>,
    curves: Curves,
}

impl Table {
    pub fn load(r: TableRef) -> anyhow::Result<Self> {
        let (rows, q_scope) = read_pair_rows(&r.path.to_string_lossy())
            .map_err(|e| anyhow::anyhow!("{}: {e}", r.path.display()))?;
        let curves = match &r.curve {
            Some(c) => read_curves(&c.to_string_lossy())
                .map_err(|e| anyhow::anyhow!("{}: {e}", c.display()))?,
            None => Curves::default(),
        };
        Ok(Table {
            r,
            rows,
            q_scope,
            curves,
        })
    }

    /// Whether its local fdrs were fitted within a `--only` subset.
    pub fn subset_fitted(&self) -> bool {
        self.r.subset.is_some() || &*self.q_scope != Q_SCOPE_ALL
    }

    /// Row `i`'s threshold–response curve; empty without one.
    pub fn curve(&self, i: usize) -> &[CurveRow] {
        let r = &self.rows[i];
        self.curves
            .get(&(r.ligand.clone(), r.receptor.clone()))
            .map_or(&[], Vec::as_slice)
    }

    /// The row of `name` (`LIGAND:RECEPTOR`, case aside); else the nearest
    /// names, for the error.
    pub fn find(&self, name: &str) -> Result<usize, Vec<String>> {
        let want = name.trim().to_uppercase();
        let (ligand, receptor) = want.split_once(':').unwrap_or((&want, ""));
        self.rows
            .iter()
            .position(|r| &*r.key.0 == ligand && &*r.key.1 == receptor)
            .ok_or_else(|| only::nearest(&want, self.rows.iter().map(PairRow::name)))
    }
}

// * The list

/// How the pair list is ordered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Sort {
    /// Headline p, smallest first.
    #[default]
    P,
    /// Largest change first, either sign.
    Effect,
    /// By ligand, then receptor.
    Name,
}

impl Sort {
    pub fn next(self) -> Sort {
        match self {
            Sort::P => Sort::Effect,
            Sort::Effect => Sort::Name,
            Sort::Name => Sort::P,
        }
    }

    pub fn name(self) -> String {
        super::value_name(&self)
    }
}

/// Which pairs the list shows, and in what order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Filter {
    /// Genes: each word must be in the ligand's or the receptor's name.
    pub search: String,
    pub sort: Sort,
}

impl Filter {
    /// The search's words, upper case.
    fn words(&self) -> Vec<String> {
        self.search
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|w| !w.is_empty())
            .map(str::to_uppercase)
            .collect()
    }
}

/// Whether `row` passes every search word of `words` (upper case).
fn passes(row: &PairRow, words: &[String]) -> bool {
    let (l, r) = &row.key;
    words
        .iter()
        .all(|w| l.contains(w.as_str()) || r.contains(w.as_str()))
}

/// NaN last, whichever way a key sorts.
fn nan_last(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => a.total_cmp(&b),
        (a, b) => a.cmp(&b),
    }
}

/// The rows `f` shows, in its order.
pub fn visible(rows: &[PairRow], f: &Filter) -> Vec<usize> {
    let words = f.words();
    let mut v: Vec<usize> = (0..rows.len())
        .filter(|&i| passes(&rows[i], &words))
        .collect();
    let name = |a: usize, b: usize| rows[a].key.cmp(&rows[b].key);
    match f.sort {
        Sort::P => v.sort_by(|&a, &b| {
            nan_last(rows[a].effect_dr_p, rows[b].effect_dr_p).then_with(|| name(a, b))
        }),
        Sort::Effect => v.sort_by(|&a, &b| {
            nan_last(-rows[a].pct.abs(), -rows[b].pct.abs()).then_with(|| name(a, b))
        }),
        Sort::Name => v.sort_by(|&a, &b| name(a, b)),
    }
    v
}

/// A row's headline as a percent change with its interval; `—` without.
pub fn change(row: &PairRow) -> String {
    if row.pct.is_finite() {
        percent_text(row.pct, row.pct_lo, row.pct_hi)
    } else {
        "not estimable".into()
    }
}

/// A p or a local fdr in a few characters; one too small for the table's
/// f32 reads as below the smallest it holds.
pub fn probability_text(probability: f64) -> String {
    if !probability.is_finite() {
        "—".into()
    } else if probability == 0.0 {
        // Stored as f32, below whose smallest it reads as 0.
        "<1e-45".into()
    } else if probability < 1e-3 {
        format!("{probability:.0e}")
    } else {
        format!("{probability:.3}")
    }
}

// * The pair on the map

/// What a pair's map shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum PairShow {
    /// Senders, and receivers by exposed or not and receptor-competent or not.
    #[default]
    Classes,
    /// Each receiver's contact exposure to the ligand.
    Exposure,
}

impl PairShow {
    pub fn other(self) -> PairShow {
        match self {
            PairShow::Classes => PairShow::Exposure,
            PairShow::Exposure => PairShow::Classes,
        }
    }
}

/// The classes' colours in `theme`, in [`Class::ALL`]'s order; the
/// unexposed receptor-negative receivers are the tissue's dimmed grey.
pub fn class_palette(theme: Theme) -> Vec<Rgb> {
    let own: [Rgb; 4] = match theme {
        Theme::Dark => [
            [255, 196, 46],
            [240, 72, 62],
            [244, 168, 150],
            [86, 146, 238],
        ],
        Theme::Light => [[214, 148, 0], [196, 32, 38], [236, 150, 128], [30, 92, 200]],
    };
    let mut p = own.to_vec();
    p.push(theme.dimmed());
    p
}

/// A run's receivers as lra saw them, matched to the map's cells.
pub struct Receivers {
    pub frame: RunFrame,
    /// Map cell of each receiver row.
    cell_of_row: Vec<Option<u32>>,
    /// The options they were read with.
    options: LraOptions,
}

impl Receivers {
    /// `frame` on `base`'s cells, read with `options`.
    pub fn new(frame: RunFrame, base: &Base, options: &LraOptions) -> Self {
        let geom = &base.geom;
        let cell_of_row = frame
            .cell
            .iter()
            .map(|&c| geom.index.get(&frame.column_names[c]).map(|&i| i as u32))
            .collect();
        Receivers {
            frame,
            cell_of_row,
            options: options.clone(),
        }
    }

    /// Read the receivers of the run `base` shows, made with `options`.
    pub fn open(base: &Base, options: &LraOptions) -> anyhow::Result<Self> {
        Ok(Self::new(
            RunFrame::open(&base.run.manifest, options)?,
            base,
            options,
        ))
    }
}

/// One pair drawn on the map.
pub struct OnMap {
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    pub map: PairMap,
    /// The classes, as a grouping of the map's cells.
    pub classes: Level,
    /// Each map cell's exposure, and whether it is a receiver.
    values: Vec<f32>,
    measured: Vec<bool>,
    /// The exposure as a feature's level, made the first time it is drawn.
    exposure: OnceCell<GeneMap>,
}

impl OnMap {
    /// `row`'s pair on `rx`, drawn on `base`'s cells; `level` is the level
    /// on screen, which the export's command names.
    pub fn new(base: &Base, rx: &Receivers, row: &PairRow, level: usize) -> anyhow::Result<Self> {
        let threshold = row.threshold.is_finite().then_some(row.threshold);
        let map = rx.frame.pair(&row.ligand, &row.receptor, threshold)?;
        let n = base.geom.n();
        let mut group = vec![NO_CLUSTER; n];
        let mut values = vec![0f32; n];
        let mut measured = vec![false; n];
        for (r, cell) in rx.cell_of_row.iter().enumerate() {
            let Some(i) = cell.map(|i| i as usize) else {
                continue;
            };
            group[i] = map.class(r) as u16;
            values[i] = map.exposure[r] as f32;
            measured[i] = true;
        }
        let names: Vec<Box<str>> = Class::ALL.iter().map(|c| c.name().into()).collect();
        let tag = row.name();
        let comm = Communities::from_groups(&tag, group, names);
        let mut classes = base.grouping(level, comm, None);
        classes.palette = class_palette(base.theme());
        Ok(OnMap {
            ligand: row.ligand.clone(),
            receptor: row.receptor.clone(),
            map,
            classes,
            values,
            measured,
            exposure: OnceCell::new(),
        })
    }

    /// The exposure, as a feature's level on `base`'s grid.
    pub fn exposure(&self, base: &Base) -> &GeneMap {
        self.exposure.get_or_init(|| {
            GeneMap::new(
                &self.ligand,
                Source::Observed,
                &self.values,
                Some(&self.measured),
                EXPOSURE_CLIP,
                &base.grid,
            )
            .labelled(&format!("{} contact exposure", self.ligand))
        })
    }

    /// The level, style and feature map that draw it on `base` as `show`
    /// asks.
    pub fn drawn(
        &self,
        base: &Base,
        show: PairShow,
        point: f32,
    ) -> (&Level, Style<'static>, Option<&GeneMap>) {
        let style = Style {
            layer: match show {
                PairShow::Classes => Layer::Argmax,
                PairShow::Exposure => Layer::Gene,
            },
            edges: false,
            focus: None,
            theme: base.theme(),
            point,
        };
        let gene = (show == PairShow::Exposure).then(|| self.exposure(base));
        (&self.classes, style, gene)
    }

    /// Recolour for `theme`.
    pub fn retheme(&mut self, theme: Theme) {
        self.classes.palette = class_palette(theme);
    }

    /// Each class with its colour and receivers, as the legend lists them.
    pub fn legend(&self) -> Vec<(&'static str, Rgb, usize)> {
        let counts = self.map.counts();
        Class::ALL
            .iter()
            .enumerate()
            .map(|(k, c)| (c.name(), self.classes.palette[k], counts[k]))
            .collect()
    }
}

// * The layer

/// A table to show: one of the layer's, or one named by its path (read
/// again when the layer has it, as it may have been written anew).
pub enum Which {
    At(usize),
    Path(Box<TableRef>),
}

/// A table read to be shown, not shown yet ([`LraView::stage`]): a call
/// that checks what it asks of the table first can still change nothing.
pub struct Staged {
    /// Its place among the tables; `None` for one new to them.
    at: Option<usize>,
    r: TableRef,
    /// The table as read now; `None` when the one read before is shown.
    read: Option<Table>,
}

/// A drawn pair's key: its name and threshold, and the receivers' options.
type DrawnKey = (String, u64, LraOptions);

/// The layer's state: which table, which pairs the list shows, the pair
/// drawn and how, the pairs marked for a `--only` run, and what has been
/// read and drawn so far.
pub struct LraView {
    pub tables: Vec<TableRef>,
    /// The table shown, by position in `tables`; it is always read.
    pub at: usize,
    /// Which rows the list shows: see [`LraView::set_filter`].
    filter: Filter,
    /// The rows `filter` shows, in its order.
    shown: Vec<usize>,
    /// The tables read so far, by position in `tables`.
    read: Vec<Option<Table>>,
    /// The row drawn on the map, by position in the table's rows.
    pub selected: Option<usize>,
    pub show: PairShow,
    /// Pairs marked, by name, for `pinto lra --only`.
    pub marked: std::collections::BTreeSet<String>,
    /// The receivers last read, for the options they were read with.
    pub receivers: Option<Receivers>,
    /// The last pairs drawn, the newest last.
    drawn: Vec<(DrawnKey, OnMap)>,
}

impl LraView {
    /// The layer on `tables`, the first one read; `None` without any.
    pub fn new(tables: Vec<TableRef>) -> anyhow::Result<Option<Self>> {
        if tables.is_empty() {
            return Ok(None);
        }
        let mut v = LraView {
            read: Vec::new(),
            tables,
            at: 0,
            filter: Filter::default(),
            shown: Vec::new(),
            selected: None,
            show: PairShow::Classes,
            marked: Default::default(),
            receivers: None,
            drawn: Vec::new(),
        };
        let first = v.stage(Which::At(0))?;
        v.commit(first);
        Ok(Some(v))
    }

    /// Show table `at`, read the first time it is shown; the pair drawn
    /// stays when that table has it. On an error nothing changes.
    pub fn load(&mut self, at: usize) -> anyhow::Result<()> {
        let s = self.stage(Which::At(at))?;
        self.commit(s);
        Ok(())
    }

    /// Show the table at `r`'s path, added to the tables when it is new
    /// and read: one read before is read again, as it may have been
    /// written anew. On an error nothing changes.
    pub fn load_path(&mut self, r: TableRef) -> anyhow::Result<()> {
        let s = self.stage(Which::Path(Box::new(r)))?;
        self.commit(s);
        Ok(())
    }

    /// Read the table `which` names, changing nothing: [`LraView::commit`]
    /// shows it.
    pub fn stage(&self, which: Which) -> anyhow::Result<Staged> {
        let (at, r, fresh) = match which {
            Which::At(at) => {
                let r = self
                    .tables
                    .get(at)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("no table {at}"))?;
                (Some(at), r, false)
            }
            Which::Path(r) => {
                let at = self.tables.iter().position(|t| same_file(&t.path, &r.path));
                (at, *r, true)
            }
        };
        let cached = at
            .and_then(|at| self.read.get(at)?.as_ref())
            .is_some_and(|t| !fresh && t.r == r);
        let read = if cached {
            None
        } else {
            Some(Table::load(r.clone())?)
        };
        Ok(Staged { at, r, read })
    }

    /// List the tables among `tables` the layer has not seen: what a
    /// `pinto lra` that ended since it was opened added to the run.
    pub fn add_tables(&mut self, tables: Vec<TableRef>) {
        for t in tables {
            if !self
                .tables
                .iter()
                .any(|known| same_file(&known.path, &t.path))
            {
                self.tables.push(t);
            }
        }
        self.read.resize_with(self.tables.len(), || None);
    }

    /// The table `s` holds.
    pub fn staged<'a>(&'a self, s: &'a Staged) -> &'a Table {
        s.read
            .as_ref()
            .or_else(|| self.read.get(s.at?)?.as_ref())
            .expect("a staged table is read")
    }

    /// Show the table `s` holds; the pair drawn stays when it has it.
    pub fn commit(&mut self, s: Staged) {
        let name = self.selected_row().map(PairRow::name);
        let at = match s.at {
            Some(at) => {
                self.tables[at] = s.r;
                at
            }
            None => {
                self.tables.push(s.r);
                self.tables.len() - 1
            }
        };
        self.read.resize_with(self.tables.len(), || None);
        if let Some(t) = s.read {
            self.read[at] = Some(t);
        }
        self.at = at;
        self.selected = name.and_then(|n| self.table().find(&n).ok());
        self.refilter();
    }

    /// The table shown.
    pub fn table(&self) -> &Table {
        self.read[self.at]
            .as_ref()
            .expect("the table shown is read")
    }

    pub fn filter(&self) -> &Filter {
        &self.filter
    }

    /// List the rows `f` passes, in its order.
    pub fn set_filter(&mut self, f: Filter) {
        self.filter = f;
        self.refilter();
    }

    fn refilter(&mut self) {
        self.shown = visible(self.rows(), &self.filter);
    }

    pub fn rows(&self) -> &[PairRow] {
        &self.table().rows
    }

    /// The rows the list shows, in order.
    pub fn visible(&self) -> &[usize] {
        &self.shown
    }

    pub fn selected_row(&self) -> Option<&PairRow> {
        self.read.get(self.at)?.as_ref()?.rows.get(self.selected?)
    }

    /// The names of the pairs the list shows, in order.
    pub fn shown_names(&self) -> Vec<String> {
        self.visible()
            .iter()
            .map(|&i| self.rows()[i].name())
            .collect()
    }

    /// The pairs a `--only` run takes: the marked ones, else the one
    /// selected; none when neither.
    pub fn pairs_to_run(&self) -> Vec<String> {
        if self.marked.is_empty() {
            self.selected_row().map(PairRow::name).into_iter().collect()
        } else {
            self.marked.iter().cloned().collect()
        }
    }

    /// The row of `name` in `table`, as [`LraView::select`] finds it:
    /// `None` for `""`, an unknown name an error naming the nearest.
    pub fn resolve(table: &Table, name: &str) -> anyhow::Result<Option<usize>> {
        if name.trim().is_empty() {
            return Ok(None);
        }
        match table.find(name) {
            Ok(i) => Ok(Some(i)),
            Err(near) => anyhow::bail!(
                "no pair {name:?} in {}; nearest: {}",
                table.r.label(),
                near.join(", ")
            ),
        }
    }

    /// Select the pair `name`; an unknown name is an error naming the
    /// nearest. `""` selects none.
    pub fn select(&mut self, name: &str) -> anyhow::Result<()> {
        self.selected = Self::resolve(self.table(), name)?;
        Ok(())
    }

    /// The options of the table shown, which its receivers were made with.
    pub fn options(&self) -> LraOptions {
        self.tables
            .get(self.at)
            .map(|t| t.options.clone())
            .unwrap_or_default()
    }

    fn key(row: &PairRow, options: &LraOptions) -> DrawnKey {
        (row.name(), row.threshold.to_bits(), options.clone())
    }

    /// The row of `table` the pair selected now is, by name.
    pub fn selection_in(&self, table: &Table) -> Option<usize> {
        table.find(&self.selected_row()?.name()).ok()
    }

    /// Whether drawing the selected pair reads the receivers first: none
    /// read yet, or read for a table made with other options.
    pub fn needs_receivers(&self) -> bool {
        self.needs_receivers_for(&self.options())
    }

    /// [`LraView::needs_receivers`] for a table made with `options`.
    pub fn needs_receivers_for(&self, options: &LraOptions) -> bool {
        self.receivers
            .as_ref()
            .is_none_or(|rx| rx.options != *options)
    }

    /// The selected pair on the map, drawn now if it has not been; the
    /// receivers are read first when they have not been for the table's
    /// options (a pass over the run's data). `None` with no pair selected.
    pub fn on_map(&mut self, base: &Base, level: usize) -> anyhow::Result<Option<&OnMap>> {
        let Some(row) = self.selected_row().cloned() else {
            return Ok(None);
        };
        self.draw(base, &self.options(), &row, level)?;
        Ok(self.drawn.last().map(|(_, on)| on))
    }

    /// Draw `row`, of a table made with `options`, when it is not among
    /// the last drawn, the receivers read first when they are not read for
    /// `options`; it is the newest drawn after. Only what is read and drawn
    /// changes: [`LraView::shown`] finds it once its table is shown.
    pub fn draw(
        &mut self,
        base: &Base,
        options: &LraOptions,
        row: &PairRow,
        level: usize,
    ) -> anyhow::Result<()> {
        // the map reads exposure over every contact; a --link-community
        // table's exposure runs through its community's links alone
        if let Some(link_community) = options.link_community {
            anyhow::bail!(
                "a --link-community table's pairs are not drawn yet (its exposure runs \
                 through link community {link_community} alone); read its rows, or draw \
                 the marginal table's"
            );
        }
        let key = Self::key(row, options);
        let on = match self.drawn.iter().position(|(k, _)| *k == key) {
            Some(at) => self.drawn.remove(at).1,
            None => {
                if self.needs_receivers_for(options) {
                    self.receivers = Some(Receivers::open(base, options)?);
                }
                let rx = self.receivers.as_ref().expect("read above");
                OnMap::new(base, rx, row, level)?
            }
        };
        if self.drawn.len() >= DRAWN_KEPT {
            self.drawn.remove(0);
        }
        self.drawn.push((key, on));
        Ok(())
    }

    /// The selected pair as drawn already, without drawing it.
    pub fn shown(&self) -> Option<&OnMap> {
        let key = Self::key(self.selected_row()?, &self.options());
        self.drawn.iter().find(|(k, _)| *k == key).map(|(_, on)| on)
    }

    /// Recolour every pair drawn for `theme`.
    pub fn retheme(&mut self, theme: Theme) {
        for (_, on) in &mut self.drawn {
            on.retheme(theme);
        }
    }

    /// The selected pair as a still's redraw command names it: its table
    /// when that is not the run's first, and with `curve` its curve.
    pub fn still_pair(&self, curve: bool) -> Option<StillPair<'_>> {
        let row = self.selected_row()?;
        let first = self.tables.first().map(|t| t.path.as_path());
        let at = self.tables.get(self.at).map(|t| t.path.as_path());
        Some(StillPair {
            name: row.name(),
            show: self.show,
            table: at.filter(|&t| Some(t) != first),
            curve,
        })
    }

    /// The selected pair's curve, ready to draw.
    pub fn curve(&self) -> Option<Curve<'_>> {
        let table = self.table();
        let i = self.selected?;
        let row = &table.rows[i];
        let mut title = format!("{}:{}  threshold-response", row.ligand, row.receptor);
        if table.subset_fitted() {
            title.push_str("  (subset)");
        }
        Some(Curve {
            title,
            points: table.curve(i),
            top: row.tau_top.is_finite().then_some(row.tau_top),
            // A table from before the average has no tau_top: its effect is
            // a point of the curve, not drawn again.
            average: row.tau_top.is_finite().then_some(row.effect),
        })
    }

    /// Mark or unmark `name` for a `--only` run; whether it is marked now.
    pub fn toggle_mark(&mut self, name: String) -> bool {
        if self.marked.remove(&name) {
            false
        } else {
            self.marked.insert(name);
            true
        }
    }
}

/// `pinto lra --from {manifest}` with `only`, as the viewer starts it: no
/// `-o`, so a subset lands beside the run and the full table is its own.
pub fn lra_command(manifest: &Path, options: &LraOptions, only: &[String]) -> Vec<String> {
    options.argv(
        Some(&manifest.to_string_lossy()),
        None,
        LraMethod::ContactEffect,
        only,
    )
}

/// The table `pinto lra --from {manifest}` writes beside the run at
/// `prefix` with `only` (the full table when empty), and its curve
/// ([`crate::lr_activity::lra_outputs`]).
pub fn lra_table(prefix: &str, only: &[String], options: LraOptions) -> TableRef {
    let list: Vec<Box<str>> = only.iter().map(|s| s.as_str().into()).collect();
    let mut made =
        crate::lr_activity::lra_outputs(LraMethod::ContactEffect, prefix, &list).into_iter();
    let path = made.next().expect("lra writes a table");
    let curve = made.next().expect("and its curve");
    let subset = (!list.is_empty()).then(|| LraSubset {
        only: only::canonical(&list),
        table: path.to_string_lossy().into_owned(),
        curve: curve.to_string_lossy().into_owned(),
        ..Default::default()
    });
    TableRef {
        path,
        curve: Some(curve),
        subset,
        options,
    }
}
