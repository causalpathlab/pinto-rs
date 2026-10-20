//! `pinto view --follow`: the view an agent has on `pinto mcp`, drawn
//! here as it changes. Watch only: the agent's view is applied as the
//! viewer's own keys would, a change of the person's own pauses following
//! (`F` takes it up again), and nothing goes back to the agent.
//!
//! What is known of the server (its calls, whether it is followed or has
//! ended) is legume-mcp's [`watch::Agent`]; this draws its view.

use super::App;
use crate::view::lupin::same_file;
use crate::view::{parse_focus, Followed, Show};
use legume_mcp::watch::{self, Agent, Live};
use ratatui::DefaultTerminal;
use std::path::Path;

/// The agent's calls the panel lists.
const CALLS: usize = 4;

/// The running `pinto mcp` server `server` names (its pid, its folder or
/// the folder's name), or the newest one when it names none, and the run
/// its agent has open.
pub(in crate::view) fn pick_server(server: &str) -> anyhow::Result<(Live, String)> {
    // Its pid as `ps` gives it: `01234` names no server.
    let pid = server
        .parse()
        .ok()
        .filter(|p: &u32| p.to_string() == server);
    let named = |l: &Live| {
        server.is_empty()
            || pid.is_some()
            || l.dir.file_name().is_some_and(|n| n == server)
            || same_file(&l.dir, Path::new(server))
    };
    let live = watch::find(crate::view::mcp::SERVER, pid)
        .into_iter()
        .find(named)
        .ok_or_else(|| match server {
            "" => anyhow::anyhow!("no `pinto mcp` is running: an agent starts one"),
            _ => anyhow::anyhow!("no running `pinto mcp` is {server:?}: give its pid or folder"),
        })?;
    let pid = live.meta.pid;
    let view = watch::state(&live.dir).ok_or_else(|| {
        anyhow::anyhow!("the agent on `pinto mcp` (pid {pid}) has opened no run yet")
    })?;
    let run = read(view)
        .map_err(|e| anyhow::anyhow!("pid {pid}: {e}"))?
        .run;
    Ok((live, run))
}

/// A view the server published.
fn read(view: serde_json::Value) -> anyhow::Result<Followed> {
    serde_json::from_value(view).map_err(|e| {
        anyhow::anyhow!("cannot read the agent's view ({e}): is its `pinto mcp` this version?")
    })
}

/// The server followed, and its view as drawn here.
pub(super) struct Follower {
    agent: Agent,
    /// The agent's view as last published, and whether it is yet to be
    /// drawn here.
    latest: Option<Followed>,
    due: bool,
    /// Whether the last view published could not be read, and whether the
    /// person was told so since one could.
    unreadable: bool,
    told: bool,
}

impl Follower {
    /// Follow the server `live` from its view now.
    pub(super) fn open(live: &Live) -> anyhow::Result<Self> {
        Ok(Follower {
            agent: Agent::new(&live.dir, false)?,
            latest: None,
            due: false,
            unreadable: false,
            told: false,
        })
    }

    /// Read what the server did since: whether the panel has news, and why
    /// a view it published cannot be read, the first time since one could.
    fn read(&mut self) -> (bool, Option<String>) {
        let news = self.agent.poll();
        let Some(view) = self.agent.take_new_state() else {
            return (news, None);
        };
        match read(view) {
            Ok(view) => {
                (self.latest, self.due) = (Some(view), true);
                (self.unreadable, self.told) = (false, false);
                (news, None)
            }
            Err(e) => {
                self.unreadable = true;
                let first = !std::mem::replace(&mut self.told, true);
                (news, first.then(|| e.to_string()))
            }
        }
    }

    /// The panel's lines on the agent, at most `width` wide: the server,
    /// its pid and whether it is followed; what to do, when there is
    /// something; then its last calls. The last view stays once the server
    /// has ended.
    pub(super) fn lines(&self, width: usize) -> Vec<String> {
        let mut lines = self.agent.lines(1, width);
        let mut say = Vec::new();
        if !self.agent.following() {
            say.push("F follows");
        }
        if self.unreadable {
            say.push("cannot read its view");
        }
        if !say.is_empty() {
            lines.push(crate::tui::style::short(&say.join(" · "), width));
        }
        lines.extend(self.agent.call_lines(CALLS, width));
        lines
    }
}

impl App<'_> {
    /// Read the server followed, and draw the agent's view when it changed
    /// and is followed. Relabelling or a dialog keeps the view as it is
    /// until it is done.
    pub(super) fn follow_tick(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        let busy = self.relabel.is_some() || self.modal.is_some();
        let Some(f) = self.follower.as_mut() else {
            return Ok(());
        };
        let (news, unreadable) = f.read();
        let view = (f.agent.following() && !busy && std::mem::take(&mut f.due))
            .then(|| f.latest.clone())
            .flatten();
        self.need_panel |= news;
        if let Some(why) = unreadable {
            self.fail(why);
        }
        if let Some(view) = view {
            if let Err(e) = self.apply_followed(&view, terminal) {
                self.fail(format!("following the agent: {e}"));
            }
        }
        Ok(())
    }

    /// A change of the person's own: following waits for `F`.
    pub(super) fn pause_following(&mut self) {
        if let Some(f) = self.follower.as_mut().filter(|f| f.agent.following()) {
            f.agent.pause();
            self.status = "paused following the agent  F follows".into();
        }
    }

    /// `F`: the agent's view again, and follow it.
    pub(super) fn resume_following(
        &mut self,
        terminal: &mut DefaultTerminal,
    ) -> anyhow::Result<()> {
        if self.relabel.is_some() {
            self.status = "leave relabel mode first".into();
            return Ok(());
        }
        let Some(f) = self.follower.as_mut() else {
            self.status = "F follows an agent: pinto view --follow".into();
            return Ok(());
        };
        // What is on screen is the person's now: legume-mcp offers the
        // agent's view again, and the tick below draws it.
        f.agent.resume();
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
                // Another round is another grouping, as `,` and `.` show it.
                self.set_show(view.show);
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
        // A round's groupings have no edges of their own.
        if view.show == Show::Communities && view.edges != self.edges {
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
