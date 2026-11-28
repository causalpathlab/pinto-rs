//! A pair's links reduced by cell type: every contact a receiver's exposure
//! is summed over (a `--link-community` table's community links, else every
//! contact), read in both directions, grouped by the sending end's type and
//! the receiving end's type.
//!
//! For each sender type → receiver type it counts the directed links, the
//! active ones (into an exposed, receptor-competent receiver, from an end
//! that carries the ligand), and sums their shares of the receiver's
//! exposure ([`PairMap::share`]). A responder's shares sum to 1, so a
//! receiver type's shares sum to its responders, and a cell's share is how
//! many responders' exposure that sender type accounts for. Nothing is
//! called: every number is a count or a sum.

use super::contacts::Contacts;
use super::frame::PairMap;

/// One sender type → receiver type of a pair's links.
#[derive(Clone, Debug, PartialEq)]
pub struct TypeLinks {
    pub sender_type: u32,
    pub receiver_type: u32,
    /// Directed links from a sender-type end to a receiver-type end in scope.
    pub links: usize,
    /// Those carrying a share of an exposed, receptor-competent receiver's
    /// exposure.
    pub active: usize,
    /// Their shares, summed: responders' exposure the sender type accounts for.
    pub share: f64,
}

/// A pair's links by cell type.
#[derive(Clone, Debug, PartialEq)]
pub struct CellTypeLinks {
    /// The types, by code.
    pub names: Vec<Box<str>>,
    /// Every sender type → receiver type with a link, the largest share
    /// first (then the most links).
    pub cells: Vec<TypeLinks>,
    /// Exposed, receptor-competent receivers with a share: the shares' total.
    pub responders: usize,
}

impl CellTypeLinks {
    /// `c`'s sender type and receiver type, by name.
    pub fn names_of(&self, c: &TypeLinks) -> (Box<str>, Box<str>) {
        let name = |code: u32| self.names[code as usize].clone();
        (name(c.sender_type), name(c.receiver_type))
    }

    /// `sender → receiver` by name.
    pub fn label(&self, c: &TypeLinks) -> String {
        format!(
            "{} → {}",
            self.names[c.sender_type as usize], self.names[c.receiver_type as usize]
        )
    }
}

/// The sender type → receiver type `text` names, `"SENDER > RECEIVER"` (or
/// `→`), by `names` (case aside); a name may hold `>` itself, as each place
/// it could be split is tried.
pub fn type_pair_named(names: &[Box<str>], text: &str) -> Option<(u32, u32)> {
    let code = |name: &str| {
        let name = name.trim();
        names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(name))
            .map(|p| p as u32)
    };
    text.match_indices(['>', '→']).find_map(|(at, sep)| {
        let (a, b) = (&text[..at], &text[at + sep.len()..]);
        Some((code(a)?, code(b)?))
    })
}

/// `map`'s links over `contacts` (rows), by `type_of_row` (codes into
/// `names`): each row in `map`'s scope as receiver, each of its contacts as
/// sender.
pub fn links_by_cell_type(
    contacts: &Contacts,
    map: &PairMap,
    type_of_row: &[u32],
    names: &[Box<str>],
) -> CellTypeLinks {
    let k = names.len();
    let mut grid: Vec<(usize, usize, f64)> = vec![(0, 0, 0.); k * k];
    let mut responders = 0;
    for to in (0..map.n()).filter(|&r| map.in_scope(r)) {
        let mut any = false;
        for &from in &contacts.adj[to] {
            let from = from as usize;
            let cell = &mut grid[type_of_row[from] as usize * k + type_of_row[to] as usize];
            cell.0 += 1;
            let share = map.share(from, to);
            if share > 0. {
                cell.1 += 1;
                cell.2 += share;
                any = true;
            }
        }
        responders += any as usize;
    }
    let mut cells: Vec<TypeLinks> = grid
        .into_iter()
        .enumerate()
        .filter(|(_, (links, _, _))| *links > 0)
        .map(|(at, (links, active, share))| TypeLinks {
            sender_type: (at / k) as u32,
            receiver_type: (at % k) as u32,
            links,
            active,
            share,
        })
        .collect();
    cells.sort_by(|a, b| b.share.total_cmp(&a.share).then(b.links.cmp(&a.links)));
    CellTypeLinks {
        names: names.to_vec(),
        cells,
        responders,
    }
}
