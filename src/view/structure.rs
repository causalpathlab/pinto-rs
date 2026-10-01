//! Structure plot: every cell's community mixture as a stacked bar, as
//! admixture plots show ancestry and topic models show topic shares.
//!
//! Following `lupin plot-topic`: communities stack in order of overall
//! prevalence, most prevalent at the bottom; cells are grouped into panels
//! (batches, or a lupin round's cell types or clusters), and within a panel
//! ordered by their dominant community's rank, then by its share,
//! highest first. A pixel column averages the cells that fall on it.

use super::color::{Rgb, Theme};
use super::data::{Communities, NO_CLUSTER};
use super::render::Frame;

/// Pixels between panels.
const GAP: usize = 2;

pub struct Structure {
    /// Panel name and its cells in drawing order.
    pub panels: Vec<(String, Vec<u32>)>,
    /// Communities from the bottom of a bar to the top.
    pub stack: Vec<usize>,
}

/// A panel's place in a rendered frame: first and last pixel column.
pub struct Span {
    pub x0: usize,
    pub x1: usize,
    pub name: String,
}

/// A rendered plot: the picture, where its panels are, and which
/// community each pixel shows ([`NO_CLUSTER`] for none), row-major.
pub struct Drawn {
    pub frame: Frame,
    pub panels: Vec<Span>,
    pub community: Vec<u16>,
}

impl Drawn {
    /// The community at pixel `(x, y)`.
    pub fn at(&self, x: usize, y: usize) -> Option<usize> {
        if x >= self.frame.w || y >= self.frame.h {
            return None;
        }
        let c = self.community[y * self.frame.w + x];
        (c != NO_CLUSTER).then_some(c as usize)
    }
}

impl Structure {
    /// Bars from `comm`'s propensities; panels from `panel_of` (one per
    /// cell, [`NO_CLUSTER`] for none), named `names`, in `order`. Cells
    /// without a propensity row are left out.
    pub fn build(comm: &Communities, panel_of: &[u16], names: &[String], order: &[usize]) -> Self {
        let k = comm.k;
        let n = comm.cluster.len();
        let has = |i: usize| comm.cluster[i] != NO_CLUSTER;

        let mut totals = vec![0u64; k];
        for i in (0..n).filter(|&i| has(i)) {
            for (t, &q) in totals.iter_mut().zip(&comm.prop[i * k..(i + 1) * k]) {
                *t += q as u64;
            }
        }
        let mut stack: Vec<usize> = (0..k).collect();
        stack.sort_by_key(|&c| (std::cmp::Reverse(totals[c]), c));
        let mut rank = vec![0usize; k];
        for (r, &c) in stack.iter().enumerate() {
            rank[c] = r;
        }

        let mut members: Vec<Vec<u32>> = vec![Vec::new(); names.len()];
        for i in (0..n).filter(|&i| has(i)) {
            if let Some(m) = members.get_mut(panel_of[i] as usize) {
                m.push(i as u32);
            }
        }
        let unassigned: Vec<u32> = (0..n)
            .filter(|&i| has(i) && (panel_of[i] as usize) >= names.len())
            .map(|i| i as u32)
            .collect();
        let mut panels: Vec<(String, Vec<u32>)> = order
            .iter()
            .filter(|&&p| p < names.len())
            .filter_map(|&p| {
                let cells = std::mem::take(&mut members[p]);
                if cells.is_empty() {
                    return None;
                }
                Some((names[p].clone(), cells))
            })
            .collect();
        // Cells no group claims go last.
        if !unassigned.is_empty() {
            panels.push(("unassigned".into(), unassigned));
        }
        let panels: Vec<(String, Vec<u32>)> = panels
            .into_iter()
            .map(|(name, mut cells)| {
                sort_cells(&mut cells, comm, &rank);
                (name, cells)
            })
            .collect();
        Structure { panels, stack }
    }

    /// Each panel as a plot of its own, by name, on the same stack.
    pub fn into_panels(self) -> Vec<(String, Structure)> {
        let stack = self.stack;
        self.panels
            .into_iter()
            .map(|(name, cells)| {
                let one = Structure {
                    panels: vec![(name.clone(), cells)],
                    stack: stack.clone(),
                };
                (name, one)
            })
            .collect()
    }

    pub fn n_cells(&self) -> usize {
        self.panels.iter().map(|(_, c)| c.len()).sum()
    }

    /// Panel widths for `w` pixels: in proportion to their cells, at least
    /// one pixel each while there is room, `GAP` between them.
    fn widths(&self, w: usize) -> Vec<usize> {
        let p = self.panels.len();
        let n = self.n_cells().max(1);
        let avail = w.saturating_sub(GAP * p.saturating_sub(1));
        let exact: Vec<f64> = self
            .panels
            .iter()
            .map(|(_, c)| c.len() as f64 * avail as f64 / n as f64)
            .collect();
        let mut out: Vec<usize> = exact.iter().map(|&x| x.floor() as usize).collect();
        // Largest remainders take the pixels left over.
        let mut left = avail.saturating_sub(out.iter().sum());
        let mut by_rest: Vec<usize> = (0..p).collect();
        by_rest.sort_by(|&a, &b| {
            (exact[b] - exact[b].floor()).total_cmp(&(exact[a] - exact[a].floor()))
        });
        for &i in &by_rest {
            if left == 0 {
                break;
            }
            out[i] += 1;
            left -= 1;
        }
        // A panel too small for a pixel borrows one from the widest.
        for i in 0..p {
            if out[i] == 0 {
                let widest = (0..p).max_by_key(|&j| out[j]).unwrap_or(i);
                if out[widest] > 1 {
                    out[widest] -= 1;
                    out[i] = 1;
                }
            }
        }
        out
    }

    /// The plot at `w × h` pixels. With `focus`, communities not in it
    /// are drawn in the dimmed tissue colour.
    pub fn render(
        &self,
        comm: &Communities,
        palette: &[Rgb],
        (w, h): (usize, usize),
        focus: Option<&[bool]>,
        theme: Theme,
    ) -> Drawn {
        let mut frame = Frame::blank(w, h, theme.background());
        let k = comm.k;
        let mut community = vec![NO_CLUSTER; w * h];
        let colour = |c: usize| match focus {
            Some(f) if !f.get(c).copied().unwrap_or(false) => theme.dimmed(),
            _ => palette[c],
        };
        let mut spans = Vec::new();
        let mut x = 0usize;
        let mut mix = vec![0f32; k];
        for ((name, cells), width) in self.panels.iter().zip(self.widths(w)) {
            if width == 0 {
                continue;
            }
            for col in 0..width {
                // The cells under this column; stretched when a panel has
                // fewer cells than pixels.
                let a = col * cells.len() / width;
                let b = ((col + 1) * cells.len() / width)
                    .max(a + 1)
                    .min(cells.len());
                mix.fill(0.);
                for &i in &cells[a..b] {
                    let i = i as usize;
                    for (m, &q) in mix.iter_mut().zip(&comm.prop[i * k..(i + 1) * k]) {
                        *m += q as f32;
                    }
                }
                let total: f32 = mix.iter().sum();
                if total <= 0. || x + col >= w {
                    continue;
                }
                // Bottom up, each community a run of rows.
                let mut from = 0f32;
                for &c in &self.stack {
                    let to = from + mix[c] / total;
                    let y0 = (h as f32 * (1. - to)).round() as usize;
                    let y1 = (h as f32 * (1. - from)).round() as usize;
                    let [r, g, b] = colour(c);
                    for y in y0..y1.min(h) {
                        let o = frame.offset(x + col, y);
                        frame.rgba[o..o + 4].copy_from_slice(&[r, g, b, 255]);
                        community[y * w + x + col] = c as u16;
                    }
                    from = to;
                }
            }
            spans.push(Span {
                x0: x,
                x1: x + width - 1,
                name: name.clone(),
            });
            x += width + GAP;
        }
        Drawn {
            frame,
            panels: spans,
            community,
        }
    }
}

/// Dominant community's rank, then its share, highest first.
fn sort_cells(cells: &mut [u32], comm: &Communities, rank: &[usize]) {
    let k = comm.k;
    cells.sort_by_key(|&i| {
        let i = i as usize;
        let row = &comm.prop[i * k..(i + 1) * k];
        let (c, q) = row
            .iter()
            .enumerate()
            .max_by_key(|&(c, &q)| (q, std::cmp::Reverse(rank[c])))
            .map_or((0, 0), |(c, &q)| (c, q));
        (rank[c], std::cmp::Reverse(q))
    });
}
