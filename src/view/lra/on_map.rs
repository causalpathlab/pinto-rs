//! One lra pair on the map: the run's receivers matched to the map's cells,
//! the pair's classes as a grouping, and its links: those its exposure is
//! summed over, each carrying its share of a responder's exposure, and,
//! with the run's cell types, those links by sender type and receiver type.

use super::{class_palette, PairShow};
use crate::lr_activity::ligand_effect::frame::{Class, PairMap, RunFrame};
use crate::lr_activity::ligand_effect::link_cell_types::{links_by_cell_type, CellTypeLinks};
use crate::lr_activity::ligand_effect::output::PairRow;
use crate::util::metadata::LraOptions;
use crate::view::args::EdgesShown;
use crate::view::color::{Rgb, Theme};
use crate::view::data::{Communities, NO_CLUSTER};
use crate::view::gene::{GeneMap, Source};
use crate::view::index::{Links, PairLinks};
use crate::view::render::{Layer, Style};
use crate::view::{Base, FigureLook, Level};
use std::cell::OnceCell;

/// Percentile the exposure ramp tops out at.
const EXPOSURE_CLIP: f32 = 99.;

/// A sender cell type → receiver cell type, by their codes in the run's
/// cell types; [`NO_TYPE`] for a type the run lacks, which no cell has.
pub type TypeCodes = (u32, u32);

/// In [`TypeCodes`], a type the receivers' cell types lack.
pub const NO_TYPE: u32 = u32::MAX;

/// The links a pair's map carries: the level's, left to a type pair or not.
pub type LinksOf = (usize, Option<TypeCodes>);

/// A run's receivers as lra saw them, matched to the map's cells.
pub struct Receivers {
    pub frame: RunFrame,
    /// Map cell of each receiver row.
    cell_of_row: Vec<Option<u32>>,
    /// Receiver row of each map cell; [`NO_ROW`] for a cell without.
    row_of_cell: Vec<u32>,
    /// Each row's cell type, a code into the frame's
    /// [`RunFrame::cell_type_names`], when the run has them.
    pub type_of_row: Option<Vec<u32>>,
    /// The options they were read with.
    pub(super) options: LraOptions,
}

/// In [`Receivers::row_of_cell`], a map cell that is no receiver.
const NO_ROW: u32 = u32::MAX;

impl Receivers {
    /// `frame` on `base`'s cells, read with `options`.
    pub fn new(frame: RunFrame, base: &Base, options: &LraOptions) -> Self {
        let geom = &base.geom;
        let cell_of_row: Vec<Option<u32>> = frame
            .cell
            .iter()
            .map(|&c| geom.index.get(&frame.column_names[c]).map(|&i| i as u32))
            .collect();
        let mut row_of_cell = vec![NO_ROW; geom.n()];
        for (r, cell) in cell_of_row.iter().enumerate() {
            if let Some(i) = *cell {
                row_of_cell[i as usize] = r as u32;
            }
        }
        let type_of_row = frame.cell_type_of_rows();
        Receivers {
            frame,
            cell_of_row,
            row_of_cell,
            type_of_row,
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

    /// The receiver row of map cell `cell`, if it is one.
    fn row(&self, cell: u32) -> Option<usize> {
        let r = *self.row_of_cell.get(cell as usize)?;
        (r != NO_ROW).then_some(r as usize)
    }
}

/// One pair drawn on the map.
pub struct OnMap {
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    pub map: PairMap,
    /// The classes, as a grouping of the map's cells, with the links of
    /// `level` carrying the pair.
    pub classes: Level,
    /// Its links by sender type and receiver type, when the run has types.
    pub types: Option<CellTypeLinks>,
    /// Each map cell's exposure, and whether it is a receiver.
    values: Vec<f32>,
    measured: Vec<bool>,
    /// The exposure as a feature's level, made the first time it is drawn.
    exposure: OnceCell<GeneMap>,
    /// The level on screen, whose links `classes` carries, and the type pair
    /// they are left to.
    pub(super) links_of: LinksOf,
}

impl OnMap {
    /// `row`'s pair on `rx`, drawn on `base`'s cells; `links_of` is the
    /// level on screen, which the export's command names, and the type pair
    /// its links are left to.
    pub fn new(
        base: &Base,
        rx: &Receivers,
        row: &PairRow,
        links_of: LinksOf,
    ) -> anyhow::Result<Self> {
        let threshold = row.threshold.is_finite().then_some(row.threshold);
        let map = rx.frame.pair(&row.ligand, &row.receptor, threshold)?;
        let n = base.geom.n();
        let mut values = vec![0f32; n];
        let mut measured = vec![false; n];
        for (r, i) in in_scope_cells(rx, &map) {
            values[i] = map.exposure[r] as f32;
            measured[i] = true;
        }
        let classes = classes_on(base, rx, &map, &row.name(), links_of)?;
        let types = rx.type_of_row.as_ref().zip(rx.frame.cell_type_names());
        let types = types.map(|(codes, names)| {
            links_by_cell_type(rx.frame.exposure_contacts(), &map, codes, names)
        });
        Ok(OnMap {
            ligand: row.ligand.clone(),
            receptor: row.receptor.clone(),
            map,
            classes,
            types,
            values,
            measured,
            exposure: OnceCell::new(),
            links_of,
        })
    }

    /// The same pair with the links of `links_of` (a level, a type pair):
    /// the pair itself is not read again, nor, on the same level, its classes.
    pub(super) fn relink(
        &mut self,
        base: &Base,
        rx: &Receivers,
        links_of: LinksOf,
    ) -> anyhow::Result<()> {
        if links_of.0 == self.links_of.0 {
            if let Some(links) = self.classes.edges.as_mut() {
                links.pair = Some(pair_links(links, rx, &self.map, links_of.1).into());
            }
            self.links_of = links_of;
            return Ok(());
        }
        let mut classes = classes_on(base, rx, &self.map, &self.classes.comm.tag, links_of)?;
        classes.palette = self.classes.palette.clone();
        self.classes = classes;
        self.links_of = links_of;
        Ok(())
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
    /// asks, its links with `edges` (where the level has them).
    pub fn drawn(
        &self,
        base: &Base,
        show: PairShow,
        edges: bool,
        point: f32,
        look: FigureLook,
    ) -> (&Level, Style<'static>, Option<&GeneMap>) {
        let style = Style {
            layer: match show {
                PairShow::Classes => Layer::Argmax,
                PairShow::Exposure => Layer::Gene,
            },
            edges: (edges && self.classes.edges.is_some()).then_some(EdgesShown::All),
            focus: None,
            theme: base.theme(),
            point,
            look,
        };
        let gene = (show == PairShow::Exposure).then(|| self.exposure(base));
        (&self.classes, style, gene)
    }

    /// Recolour for `theme`.
    pub fn retheme(&mut self, theme: Theme) {
        self.classes.palette = class_palette(theme);
    }

    /// The links in scope (contacts the exposure is summed over) and those
    /// carrying the ligand to an exposed receptor-positive receiver; `None`
    /// without the level's links.
    pub fn link_counts(&self) -> Option<(usize, usize)> {
        Some(self.classes.edges.as_ref()?.pair.as_ref()?.counts)
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

/// The pair `map` on `links`, each link's ends matched to `rx`'s rows: in
/// scope when they are contacts its exposure is summed over, carrying the
/// larger share of the two directions ([`PairMap::share`]). With `focus`,
/// only the directions from its sender type to its receiver type count:
/// the other links are left out.
pub(in crate::view) fn pair_links(
    links: &Links,
    rx: &Receivers,
    map: &PairMap,
    focus: Option<TypeCodes>,
) -> PairLinks {
    let type_of_row = rx.type_of_row.as_deref();
    let kept = |from: usize, to: usize| match (focus, type_of_row) {
        (Some((s, r)), Some(t)) => t[from] == s && t[to] == r,
        _ => true,
    };
    let edges = &links.edges;
    let n = edges.a.len();
    let mut out = PairLinks {
        in_scope: vec![false; n],
        share: vec![0.; n],
        a_sends: vec![true; n],
        counts: (0, 0),
    };
    for e in 0..n {
        let (Some(a), Some(b)) = (rx.row(edges.a[e]), rx.row(edges.b[e])) else {
            continue;
        };
        let (keep_ab, keep_ba) = (kept(a, b), kept(b, a));
        if !(keep_ab || keep_ba) || !rx.frame.joined(a, b) {
            continue;
        }
        out.in_scope[e] = true;
        let share = |keep: bool, from, to| if keep { map.share(from, to) } else { 0. };
        let (a_to_b, b_to_a) = (share(keep_ab, a, b), share(keep_ba, b, a));
        out.share[e] = a_to_b.max(b_to_a) as f32;
        // with one direction kept, it runs from the sender type's end
        out.a_sends[e] = if keep_ab != keep_ba {
            keep_ab
        } else {
            a_to_b >= b_to_a
        };
        out.counts.0 += 1;
        out.counts.1 += (out.share[e] > 0.) as usize;
    }
    out
}

/// Each receiver row of `rx` in `map`'s scope, with its map cell.
fn in_scope_cells<'a>(
    rx: &'a Receivers,
    map: &'a PairMap,
) -> impl Iterator<Item = (usize, usize)> + 'a {
    rx.cell_of_row
        .iter()
        .enumerate()
        .filter(|&(r, _)| map.in_scope(r))
        .filter_map(|(r, cell)| cell.map(|i| (r, i as usize)))
}

/// `map`'s classes as a grouping of `base`'s cells named `tag`, made from
/// level `level`, with that level's links carrying the pair; a level without
/// a link table draws its cells alone. A row out of the table's scope is
/// drawn as no receiver.
fn classes_on(
    base: &Base,
    rx: &Receivers,
    map: &PairMap,
    tag: &str,
    (level, focus): LinksOf,
) -> anyhow::Result<Level> {
    let mut group = vec![NO_CLUSTER; base.geom.n()];
    for (r, i) in in_scope_cells(rx, map) {
        group[i] = map.class(r) as u16;
    }
    let names: Vec<Box<str>> = Class::ALL.iter().map(|c| c.name().into()).collect();
    let comm = Communities::from_groups(tag, group, names);
    let mut classes = base.grouping(level, comm, None);
    classes.palette = class_palette(base.theme());
    if base.run.levels[level].link_community.is_some() {
        let links = base.links(level)?;
        let pair = pair_links(&links, rx, map, focus);
        classes.edges = Some(Links {
            pair: Some(pair.into()),
            ..links
        });
    }
    Ok(classes)
}
