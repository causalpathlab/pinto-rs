//! `pinto view --follow`: the view an agent has on `pinto mcp`, drawn
//! here as it changes. Watch only: the agent's view is applied as the
//! viewer's own keys would, a move of the person's own pauses following
//! (`F` takes it up again), and nothing goes back to the agent.

use super::App;
use crate::view::lupin::same_file;
use crate::view::{parse_focus, Followed, Show};
use legume_mcp::watch::{self, Body, Follow, Live};
use ratatui::DefaultTerminal;
use std::collections::VecDeque;
use std::path::Path;

/// The agent's calls the panel lists.
const CALLS: usize = 4;

/// Longest a call's arguments run in the panel.
const ARGS: usize = 40;

/// The running `pinto mcp` server `server` names (its pid, or its folder),
/// or the newest one when it names none, and the run its agent has open.
pub(in crate::view) fn pick_server(server: &str) -> anyhow::Result<(Live, String)> {
    let mut live = watch::live()
        .into_iter()
        .filter(|l| l.meta.server == crate::view::mcp::SERVER && l.alive);
    let live = if server.is_empty() {
        live.next()
    } else {
        live.find(|l| l.meta.pid.to_string() == server || l.dir == Path::new(server))
    }
    .ok_or_else(|| match server {
        "" => anyhow::anyhow!("no `pinto mcp` is running: an agent starts one"),
        _ => anyhow::anyhow!("no running `pinto mcp` is {server:?}: give its pid or folder"),
    })?;
    let run = view_of(&live.dir).map(|v| v.run).ok_or_else(|| {
        anyhow::anyhow!(
            "the agent on `pinto mcp` (pid {}) has opened no run yet",
            live.meta.pid
        )
    })?;
    Ok((live, run))
}

/// The view the server in `dir` published last.
fn view_of(dir: &Path) -> Option<Followed> {
    serde_json::from_value(watch::state(dir)?).ok()
}

/// The server followed, and what it has done.
pub(super) struct Follower {
    feed: Follow,
    pid: u32,
    following: bool,
    /// The agent's view as last published, and as last drawn here.
    latest: Option<Followed>,
    drawn: Option<Followed>,
    /// The agent's last calls, the newest last: the tool, its arguments,
    /// and how it went.
    calls: VecDeque<String>,
    ended: bool,
}

impl Follower {
    /// Follow the server `live` from its view now.
    pub(super) fn open(live: &Live) -> anyhow::Result<Self> {
        Ok(Follower {
            feed: watch::follow(&live.dir, watch::From::End)?,
            pid: live.meta.pid,
            following: true,
            latest: view_of(&live.dir),
            drawn: None,
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
                    if let Ok(view) = serde_json::from_value(state) {
                        self.latest = Some(view);
                    }
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

    /// The agent's view yet to be drawn, while it is followed.
    fn due(&self) -> Option<&Followed> {
        self.latest
            .as_ref()
            .filter(|v| self.following && self.drawn.as_ref() != Some(v))
    }

    /// The panel's line on the agent: its pid, whether it is followed.
    pub(super) fn header(&self) -> String {
        // The last view stays once the server has ended.
        let how = if self.ended {
            "ended"
        } else if self.following {
            "following"
        } else {
            "paused  F"
        };
        format!("pid {}  {how}", self.pid)
    }

    /// The agent's last calls, the newest first.
    pub(super) fn calls(&self) -> impl Iterator<Item = &String> {
        self.calls.iter().rev()
    }
}

/// `args` in a line: its JSON, cut short.
fn brief(args: &serde_json::Value) -> String {
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
        let Some(view) = f.due().cloned() else {
            return Ok(());
        };
        f.drawn = Some(view.clone());
        if let Err(e) = self.apply_followed(&view, terminal) {
            self.fail(format!("following the agent: {e}"));
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
    pub(super) fn resume_following(
        &mut self,
        terminal: &mut DefaultTerminal,
    ) -> anyhow::Result<()> {
        let Some(f) = self.follower.as_mut() else {
            self.status = "F follows an agent: pinto view --follow".into();
            return Ok(());
        };
        f.following = true;
        // What is on screen is the person's now: draw the agent's anew.
        f.drawn = None;
        self.status = "following the agent".into();
        self.follow_tick(terminal)
    }

    /// Draw `view`, as the viewer's keys would.
    fn apply_followed(
        &mut self,
        view: &Followed,
        terminal: &mut DefaultTerminal,
    ) -> anyhow::Result<()> {
        if !same_file(Path::new(&view.run), Path::new(&self.base.run.prefix)) {
            self.status = format!("the agent opened {}: --follow again to see it", view.run);
            return Ok(());
        }
        if view.show == Show::Communities {
            let i = self.base.run.level_index(Some(&view.level))?;
            if i != self.cur {
                self.go_level(i, terminal)?;
            }
        } else if let Some(path) = &view.round {
            if !self
                .round
                .as_ref()
                .is_some_and(|r| same_file(&r.path, Path::new(path)))
            {
                self.busy(terminal, "loading the agent's round ...".into())?;
                self.load_round(Path::new(path))?;
            }
        }
        if view.show != self.show {
            self.set_show(view.show);
        }
        let layer: super::Layer = view.layer.parse()?;
        if layer != self.layer {
            self.set_layer(layer);
            if let super::Layer::Community(c) = layer {
                self.community = c;
            }
        }
        let focus = parse_focus(&view.focus, &self.level().comm)?;
        if focus != self.focus {
            let shown = focus.iter().position(|&f| f);
            self.set_focus(focus, shown);
        }
        if view.edges != self.edges {
            self.toggle_edges(terminal)?;
        }
        self.point = view.point_size;
        self.follow_gene(view);
        let [x0, y0, x1, y1] = view.bbox;
        self.look_at(crate::view::data::Rect::from_corners(x0, y0, x1, y1));
        Ok(())
    }

    /// The agent's gene, its source and clip; none draws the groups.
    fn follow_gene(&mut self, view: &Followed) {
        let tuned = (view.source, view.clip) != (self.source, self.clip);
        (self.source, self.clip) = (view.source, view.clip);
        match &view.gene {
            None => self.gene = None,
            Some(feature) => {
                if tuned || self.gene.as_ref().is_none_or(|g| *g.feature != **feature) {
                    self.show_gene(feature);
                }
            }
        }
    }
}
