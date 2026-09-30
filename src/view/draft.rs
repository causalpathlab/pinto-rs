//! Decisions staged on a lupin round before they are sent to
//! `lupin relabel`: a label or keep per cluster, merges, and marker edits.
//!
//! The draft is saved beside its round (`{round}.relabel_draft.json`) on
//! every change, so leaving the viewer loses nothing, and removed once the
//! next round is written. Cluster ids belong to the round they were staged
//! on, so a draft is never applied to another round.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Verdict {
    Label { label: String, rationale: String },
    Keep { label: String, rationale: String },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Merge {
    pub clusters: Vec<i64>,
    pub label: String,
    pub rationale: String,
}

/// Add `feature` to (or drop it from) `label`'s markers.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Mark {
    pub label: String,
    pub feature: String,
    pub add: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Draft {
    /// The round whose cluster ids the decisions name.
    pub round: PathBuf,
    #[serde(default)]
    pub verdicts: BTreeMap<i64, Verdict>,
    #[serde(default)]
    pub merges: Vec<Merge>,
    #[serde(default)]
    pub marks: Vec<Mark>,
}

impl Draft {
    pub fn path_for(round: &Path) -> PathBuf {
        let name = super::lupin::file_name(round);
        let stem = name.trim_end_matches(".lupin.json");
        round.with_file_name(format!("{stem}.relabel_draft.json"))
    }

    /// The draft saved for `round`, or an empty one.
    pub fn load(round: &Path) -> Draft {
        let saved = std::fs::read_to_string(Self::path_for(round))
            .ok()
            .and_then(|t| serde_json::from_str::<Draft>(&t).ok());
        match saved {
            Some(d) if d.round == round => d,
            _ => Draft {
                round: round.to_path_buf(),
                ..Draft::default()
            },
        }
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::path_for(&self.round);
        if self.is_empty() {
            let _ = std::fs::remove_file(path);
            return Ok(());
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn discard(&self) {
        let _ = std::fs::remove_file(Self::path_for(&self.round));
    }

    pub fn is_empty(&self) -> bool {
        self.verdicts.is_empty() && self.merges.is_empty() && self.marks.is_empty()
    }

    pub fn len(&self) -> usize {
        self.verdicts.len() + self.merges.len() + self.marks.len()
    }

    /// The staged merge holding cluster `id`.
    pub fn merge_of(&self, id: i64) -> Option<usize> {
        self.merges.iter().position(|m| m.clusters.contains(&id))
    }

    /// Stage `mark`, replacing an opposite one on the same type and feature;
    /// staging the same mark again takes it back. Whether it is now staged.
    pub fn toggle_mark(&mut self, mark: Mark) -> bool {
        let same = |m: &Mark| {
            super::lupin::label_key(&m.label) == super::lupin::label_key(&mark.label)
                && m.feature.eq_ignore_ascii_case(&mark.feature)
        };
        match self.marks.iter().position(same) {
            Some(i) if self.marks[i].add == mark.add => {
                self.marks.remove(i);
                false
            }
            Some(i) => {
                self.marks[i] = mark;
                true
            }
            None => {
                self.marks.push(mark);
                true
            }
        }
    }

    /// Take back any marker edit staged for `feature`; whether there was one.
    pub fn clear_mark(&mut self, feature: &str) -> bool {
        let before = self.marks.len();
        self.marks
            .retain(|m| !m.feature.eq_ignore_ascii_case(feature));
        self.marks.len() != before
    }

    /// The marker edit staged for `feature`, if any.
    pub fn mark_of(&self, feature: &str) -> Option<&Mark> {
        self.marks
            .iter()
            .find(|m| m.feature.eq_ignore_ascii_case(feature))
    }

    /// Take back everything staged on cluster `id`: its verdict and any
    /// merge it is in.
    pub fn unstage(&mut self, id: i64) -> bool {
        let verdict = self.verdicts.remove(&id).is_some();
        let before = self.merges.len();
        self.merges.retain(|m| !m.clusters.contains(&id));
        verdict || self.merges.len() != before
    }

    /// One line per staged decision, for the confirmation list.
    pub fn summary(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (label, add, genes) in self.marks_by_type() {
            let sign = if add { '+' } else { '-' };
            out.push(format!("{label} markers {sign}{}", genes.join(" ")));
        }
        for m in &self.merges {
            let ids: Vec<String> = m.clusters.iter().map(|c| format!("K{c}")).collect();
            out.push(format!("merge {} → {}", ids.join("+"), m.label));
        }
        for (id, v) in &self.verdicts {
            out.push(match v {
                Verdict::Label { label, .. } => format!("K{id} → {label}"),
                Verdict::Keep { label, .. } => format!("K{id} keep {label}"),
            });
        }
        out
    }

    /// Marker edits grouped per (type, direction), in first-staged order.
    fn marks_by_type(&self) -> Vec<(String, bool, Vec<String>)> {
        let mut out: Vec<(String, bool, Vec<String>)> = Vec::new();
        for m in &self.marks {
            let key = super::lupin::label_key(&m.label);
            match out
                .iter_mut()
                .find(|(l, a, _)| super::lupin::label_key(l) == key && *a == m.add)
            {
                Some((_, _, genes)) => genes.push(m.feature.clone()),
                None => out.push((m.label.clone(), m.add, vec![m.feature.clone()])),
            }
        }
        out
    }

    /// The JSON lines `lupin relabel -d -` reads: marker edits, then merges,
    /// then labels and keeps.
    pub fn decisions(&self) -> String {
        let round = std::fs::canonicalize(&self.round).unwrap_or_else(|_| self.round.clone());
        let round = round.display().to_string();
        let line = |mut v: serde_json::Value| {
            v["decided_by"] = "user".into();
            v["round"] = round.clone().into();
            let mut s = v.to_string();
            s.push('\n');
            s
        };
        let mut out = String::new();
        for (label, add, genes) in self.marks_by_type() {
            let action = if add { "markers_add" } else { "markers_drop" };
            let why = if add { "listed in" } else { "dropped from" };
            out += &line(serde_json::json!({
                "action": action,
                "label": label,
                "features": genes,
                "rationale": format!("{} {why} {label}'s markers in pinto view", genes.join(", ")),
            }));
        }
        for m in &self.merges {
            out += &line(serde_json::json!({
                "action": "merge",
                "clusters": m.clusters,
                "label": m.label,
                "rationale": m.rationale,
            }));
        }
        for (id, v) in &self.verdicts {
            let (action, label, rationale) = match v {
                Verdict::Label { label, rationale } => ("label", label, rationale),
                Verdict::Keep { label, rationale } => ("keep", label, rationale),
            };
            out += &line(serde_json::json!({
                "action": action,
                "cluster": id,
                "label": label,
                "rationale": rationale,
            }));
        }
        out
    }
}
