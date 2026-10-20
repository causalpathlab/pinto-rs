//! `pinto view --follow`: the view an agent has on `pinto mcp`, drawn
//! here as it changes. Watch only: the agent's view is applied as the
//! viewer's own keys would, a move of the person's own pauses following
//! (`F` takes it up again), and nothing goes back to the agent.

use super::App;
use crate::view::{parse_focus, Show};
use legume_mcp::watch::{self, Body, Follow, Live};
use ratatui::DefaultTerminal;
use serde_json::Value;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// The agent's calls the panel lists.
const CALLS: usize = 4;

/// Longest a call's arguments run in the panel.
const ARGS: usize = 40;

/// The running `pinto mcp` server `server` names (its pid, or its folder),
/// or the newest one when it names none.
pub(in crate::view) fn pick(server: &str) -> anyhow::Result<Live> {
    let mut live = watch::live()
        .into_iter()
        .filter(|l| l.meta.server == "pinto" && l.alive);
    let found = if server.is_empty() {
        live.next()
    } else {
        live.find(|l| l.meta.pid.to_string() == server || l.dir == Path::new(server))
    };
    found.ok_or_else(|| match server {
        "" => anyhow::anyhow!("no `pinto mcp` is running: an agent starts one"),
        _ => anyhow::anyhow!("no running `pinto mcp` is {server:?}: give its pid or folder"),
    })
}

/// The run the agent at `server` has open.
pub(in crate::view) fn run_of(server: &Live) -> anyhow::Result<String> {
    watch::state(&server.dir)
        .and_then(|s| s["run"].as_str().map(String::from))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "the agent on `pinto mcp` (pid {}) has opened no run yet",
                server.meta.pid
            )
        })
}

/// The server followed, and what it has done.
pub(super) struct Follower {
    feed: Follow,
    pid: u32,
    /// Whether the agent's view is drawn as it changes.
    pub(super) following: bool,
    /// The agent's view, as last published, and whether it is yet to be
    /// drawn.
    latest: Option<Value>,
    pending: bool,
    /// The agent's last calls, the newest last: the tool, its arguments,
    /// and how it went.
    calls: VecDeque<String>,
    ended: bool,
}

impl Follower {
    /// Follow the server in `dir` from its view now.
    pub(super) fn open(server: &Live) -> anyhow::Result<Self> {
        let feed = watch::follow(&server.dir, watch::From::End)?;
        let latest = watch::state(&server.dir);
        Ok(Follower {
            feed,
            pid: server.meta.pid,
            following: true,
            pending: latest.is_some(),
            latest,
            calls: VecDeque::new(),
            ended: false,
        })
    }

    /// Read what the server did since; true when the panel has news.
    fn read(&mut self) -> bool {
        let mut news = false;
        while let Some(event) = self.feed.try_next() {
            news = true;
            match event.body {
                Body::Call { tool, args, .. } => {
                    if self.calls.len() == CALLS {
                        self.calls.pop_front();
                    }
                    self.calls.push_back(format!("{tool} {}", brief(&args)));
                }
                Body::Result { ok, ms, text, .. } => {
                    if let Some(call) = self.calls.back_mut() {
                        call.push_str(&match ok {
                            true => format!("  ✓ {ms} ms"),
                            false => format!("  ✗ {}", text.lines().next().unwrap_or("")),
                        });
                    }
                }
                Body::State { state, .. } => {
                    self.latest = Some(state);
                    self.pending = true;
                }
                _ => {}
            }
        }
        if !self.ended && !alive(self.pid) {
            self.ended = true;
            news = true;
        }
        news
    }

    /// The panel's lines on the agent: whether it is followed, its calls.
    pub(super) fn lines(&self) -> Vec<String> {
        // The last view stays once the server has ended.
        let how = if self.ended {
            "ended"
        } else if self.following {
            "following"
        } else {
            "paused  F"
        };
        let mut lines = vec![format!("pid {}  {how}", self.pid)];
        lines.extend(self.calls.iter().rev().cloned());
        lines
    }
}

/// `args` in a line: its JSON, cut short.
fn brief(args: &Value) -> String {
    let text = match args.as_object() {
        Some(o) if o.is_empty() => String::new(),
        _ => args.to_string(),
    };
    match text.char_indices().nth(ARGS) {
        Some((at, _)) => format!("{}…", &text[..at]),
        None => text,
    }
}

#[cfg(unix)]
fn alive(pid: u32) -> bool {
    // Signal 0 only asks whether the process is there.
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    let there = unsafe { libc::kill(pid, 0) } == 0;
    there || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn alive(_: u32) -> bool {
    true
}

impl App<'_> {
    /// Read the server followed, and draw the agent's view when it changed
    /// and is followed.
    pub(super) fn follow_tick(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        let Some(f) = self.follower.as_mut() else {
            return Ok(());
        };
        if f.read() {
            self.need_panel = true;
        }
        if f.following && f.pending {
            f.pending = false;
            if let Some(state) = f.latest.clone() {
                if let Err(e) = self.apply_followed(&state, terminal) {
                    self.fail(format!("following the agent: {e}"));
                }
            }
        }
        Ok(())
    }

    /// A move of the person's own: following waits for `F`.
    pub(super) fn pause_following(&mut self) {
        if let Some(f) = self.follower.as_mut().filter(|f| f.following) {
            f.following = false;
            self.status = "paused following the agent  F follows".into();
        }
    }

    /// `F`: the agent's view again, and follow it.
    pub(super) fn resume_following(&mut self) {
        match self.follower.as_mut() {
            Some(f) => {
                f.following = true;
                f.pending = f.latest.is_some();
                self.status = "following the agent".into();
            }
            None => self.status = "F follows an agent: pinto view --follow".into(),
        }
    }

    /// Draw the view `state` describes (`view_map`'s settings), as the
    /// viewer's keys would.
    fn apply_followed(
        &mut self,
        state: &Value,
        terminal: &mut DefaultTerminal,
    ) -> anyhow::Result<()> {
        let text = |key: &str| state[key].as_str();
        let run = text("run").unwrap_or_default();
        if !crate::view::lupin::same_file(Path::new(run), Path::new(&self.base.run.prefix))
            && run != self.base.run.prefix
        {
            self.status = format!("the agent opened {run}: --follow again to see it");
            return Ok(());
        }
        if let Some(tag) = text("level") {
            let i = self.base.run.level_index(Some(tag))?;
            if i != self.cur {
                self.go_level(i, terminal)?;
            }
        }
        let show = match text("show") {
            Some("types") => Show::Types,
            Some("clusters") => Show::Clusters,
            _ => Show::Communities,
        };
        if let (Some(path), true) = (text("round"), show != Show::Communities) {
            let path = PathBuf::from(path);
            let open = self
                .round
                .as_ref()
                .is_some_and(|r| crate::view::lupin::same_file(&r.path, &path));
            if !open {
                self.busy(terminal, "loading the agent's round ...".into())?;
                self.load_round(&path)?;
            }
        }
        if show != self.show {
            self.set_show(show);
        }
        if let Some(layer) = text("layer").and_then(|l| l.parse::<super::Layer>().ok()) {
            if layer != self.layer && layer.check(&self.level().comm).is_ok() {
                self.layer = layer;
                if let super::Layer::Community(c) = layer {
                    self.community = c;
                }
                self.need_map = true;
            }
        }
        let ids: Vec<Box<str>> = state["focus"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(Into::into))
            .collect();
        let focus = parse_focus(&ids, &self.level().comm)?;
        if focus != self.focus {
            self.shown = focus.iter().position(|&f| f);
            self.focus = focus;
            self.list_markers();
            self.need_map = true;
        }
        if let Some(edges) = state["edges"].as_bool() {
            if edges != self.edges {
                self.toggle_edges(terminal)?;
            }
        }
        if let Some(p) = state["point_size"].as_f64() {
            self.point = p as f32;
        }
        self.follow_gene(state);
        if let Some([x0, y0, x1, y1]) = state["bbox"]
            .as_array()
            .and_then(|b| {
                b.iter()
                    .map(|v| v.as_f64().map(|x| x as f32))
                    .collect::<Option<Vec<_>>>()
            })
            .and_then(|b| <[f32; 4]>::try_from(b).ok())
        {
            let (w, h) = self.image_size();
            let r = crate::view::data::Rect::from_corners(x0, y0, x1, y1);
            self.set_cam(super::Viewport::fit(r, w, h).camera());
        }
        self.need_map = true;
        Ok(())
    }

    /// The agent's gene, its source and clip; none draws the groups.
    fn follow_gene(&mut self, state: &Value) {
        if let Some(source) = state["source"].as_str() {
            self.source = match source {
                "expected" => super::Source::Expected,
                _ => super::Source::Observed,
            };
        }
        if let Some(clip) = state["clip"].as_f64() {
            self.clip = clip as f32;
        }
        match state["gene"].as_str().filter(|g| !g.is_empty()) {
            None => self.gene = None,
            Some(feature) => {
                let same = self.gene.as_ref().is_some_and(|g| {
                    *g.feature == *feature
                        && g.clip == self.clip
                        && (g.source == self.source || g.fell_back(self.source))
                });
                if !same {
                    self.show_gene(feature);
                }
            }
        }
    }
}
