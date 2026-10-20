# PINTO

**P**roximity-based **I**nteraction **N**etwork analysis to dissect **T**issue
**O**rganizations.

PINTO segments spatial transcriptomics tissue into coherent regions by
clustering cell-cell edges of a spatial KNN graph. Each edge carries an
expression profile, and a collapsed Gibbs sampler assigns edges to communities;
per-cell soft membership falls out of the edge labels.

This is the standalone crate extracted from
[`legume-rs`](https://github.com/causalpathlab/legume-rs). The crates.io
package is `pinto-rs`; the installed binary is `pinto`. Its terminal
screens use [`legume-tui`](https://crates.io/crates/legume-tui), which the
other legume viewers share.

## Installation

```sh
cargo install pinto-rs
# optional: --features hdf5 / cuda / metal
```

## Subcommands

### `lc` (link-community), recommended

```sh
pinto lc data.h5 -c tissue_positions.csv -o out
pinto lc data.h5 -c coords.csv -o out --n-communities 25
pinto lc data.h5 -o out   # expression-only
```

### `dsvd` (delta-svd)

```sh
pinto dsvd data.h5 -c coords.csv -o out
```

### `prop` (propensity)

```sh
pinto prop -z out.latent.parquet -e out.coord_pairs.parquet -o prop
```

### `cage` / `predict` / `impute` / `lr-activity`

See `pinto --help` and per-subcommand `--help`.

### `run`

Set up fits in the terminal and run them. Pick data files (coordinate and
batch files beside them are paired by sample name), queue `lc`, `cage` and
`dsvd`, and change any flag; each method's form is read from its own
`--help`. ctrl+r shows the exact commands, checked as pinto would parse them,
and runs them in turn with their log on screen; each fit shows a progress
bar, plus a sparkline of its score (loss, NLL or moves) once it has two
values and the row has room. Tab or shift-enter moves to the next screen
(shift-enter only where the terminal can tell it apart from enter).
Once a run ends, the methods it finished are unqueued (marked ✓), so
ctrl+r again, or `r` on the Run screen, runs only the ones stopped, failed
or not reached; space queues one again. Each reruns into its old `--out`,
after first clearing the script and batch files its unfinished run left
there.

```sh
pinto run data/        # the file browser starts in data/
bash lc.cmd.sh         # each run is saved as {out}.cmd.sh to run again
```

It first asks for an output header, what every result of the run is named
after: `exp1` gives `exp1_lc`, `exp1_cage`, …, and `results/` gives
`results/lc`, … (the folder is made when the fit starts). Esc leaves each
`--out` as the method's name. `O` on the Methods screen changes the header
later, and `o` names one `--out` by hand. Anywhere in pinto, an `--out`
ending in `/` is a folder: `pinto lc … --out res/` writes `res/lc.*`, and the
folder is made as needed.

Each data file is its own batch unless you say otherwise: `n` puts all its
cells in one named batch (two files given one name are one batch), `b` takes
a label file, and `e` renames its labels. The Data screen lists the batches
this makes. Label files the run needs beyond the ones given are written to
`{out}.batches/`.

A saved script refuses to run over an existing `{out}.pinto.json`, and
`pinto run` never writes over a script. `v` opens a finished run in
`pinto view`.

### `view`

Zoomable map of a run's communities, in the terminal. Pan and zoom from the
whole section down to single cells and their edges; switch between argmax,
soft mixture, entropy and single-community layers, and between levels.

```sh
pinto view out                 # interactive; ? lists the keys
pinto view out --summary       # what the run holds
pinto view out --png map.png --layer soft --width 3000
pinto view out --png zoom.png --bbox 8200,6800,8800,7250 --edges
pinto view out --png big.png --point-size 2     # larger cell discs
pinto view out --pdf fig.pdf --focus C3,C17     # figure page: map, scale bar,
                                                # legend, focused markers
pinto view out --png gene.png --gene SYMBOL     # one gene: observed ln(1+count)
pinto view out --png gene.png --gene SYMBOL --expected --clip 95
```

In the viewer, arrows or a drag pan, the wheel or `z`/`Z` zoom, `<`/`>`
shrink or grow the cell discs (`--point-size` for images), `l`/`L` step
through levels, and `Esc` steps back (chart, gene, focus); `q` quits. Keys
follow one rule: lowercase looks, uppercase decides or writes (and enters a
mode), Shift reverses, and a mode refuses keys it doesn't use. Click a cell
or legend entry (or step with `]`/`[`) to show one group with its marker
features, then click a marker to map that gene (`g`/`G` step through them,
`o` switches observed/model-expected, `p` the ramp top p99/p95); `s` exports
the current view as PNG, PDF, and a `.txt` with
the command that redraws it. The scale bar's units are guessed from the
coordinate columns (`--units um|px|none` to override). Map colours follow the
terminal's background (`--theme light|dark` to choose; images default to dark);
`t` switches between dark and a light background for paper, and what `s`
saves is drawn in it.
Drag the side panel's left edge to widen or narrow it (the annotation panel
keeps a width of its own). Errors, such as a lupin run that fails, open in a
popup over the map until a key or click.

Without a run, `pinto view` opens a browser to pick a `*.pinto.json` (↑↓,
Enter, ← up, type to narrow, `~` home).

#### Annotation with lupin

Cell types come from [lupin](https://crates.io/crates/lupin-rs) (0.6.1 or
later, on the `PATH` or at `$PINTO_LUPIN`); the viewer runs it and never
writes labels itself. lupin re-reads the run's data files, so for `.h5` data
install it with HDF5 as pinto was: `cargo install lupin-rs --features hdf5`.

- `A` picks a marker panel (`gene<TAB>type` lines) and annotates the level
  on screen with `lupin annotate --level … --method enrichment`, writing a
  round `{prefix}.{level}.a{k}.lupin.json`.
- `c`/`C` step the map through communities → the round's cell types → its
  clusters; `.`/`,` step through the rounds made from the run.
- `R` relabels the round's clusters, one at a time (`→`/`←`, or click):
  `↑`/`↓` choose a gene, `y` makes it a marker of the working type (`Tab`
  picks the type), `n` drops it from its type, space clears the mark;
  `L` labels, `K` keeps the label, `M` merges (`↑`/`↓` and space, or clicks,
  choose the clusters; Enter names them), `u` takes a decision back. Each
  needs a rationale. Decisions stay in a draft beside the round until `S`
  sends them to `lupin relabel --next`, after a confirmation; `P` previews
  what they would change.

```sh
pinto view out --round out.final.a1.r1.lupin.json --show types --png types.png
```

#### Batches side by side

A run of several batches lays them out in one tiled map; `b` jumps from one
to the next. `w` shows every batch in a grid instead, each fitted to its own
tile and numbered, so a small section is drawn as large as a big one. Point
at a tile (or choose with the arrows) and click it or press Enter to open
that batch; `w` or `Esc` goes back to the map. Drag a tile onto another
place (or Shift and an arrow) to rearrange the batches. Once a batch is
open, the grid keeps the map's zoom and pan: every tile shows its own batch
the way the map shows that one. In the grid, the wheel (around the pointer),
`z`/`Z` and `<`/`>` change only the batch pointed at, which then keeps its
own zoom and point size (the side panel shows them); hold Alt to change
every batch together. `0` fits every batch again and drops their own
settings, and opening a batch opens it at its tile's zoom. With the structure plot on
(`H`), each tile has its own batch's bars under it, on one stack order across
batches; click a community in them to see where it lies. `s` saves the grid
as a PNG with a `.txt` listing the batches by tile number.

#### Structure plot and heatmap

`H` (or shift-enter, where the terminal tells it from enter) steps through
the charts: the map with a structure plot under it, then a
gene heatmap in place of the map, then the map again (`Esc` goes straight
back). Both follow the grouping on screen (communities, or with `c` the
round's cell types or clusters):

- **structure plot**: each cell's community propensities (the share of its
  edges in each link community) as a stacked bar, communities in order of
  overall prevalence, cells in panels by group (or batch) and within a panel
  by dominant community, then its share. Click a community in it (or in the
  side panel's list) to see where it lies: on a map of communities it is
  focused as a legend click would; on a map of a round's groups the map
  shows its propensity. Click again or `Esc` to let go;
- **heatmap**: each group's top genes by their margin over the next highest
  group, so each row peaks in its own column: the model proposes candidates,
  and the groups' mean observed ln(1 + count) chooses among them (the model's
  rates alone when there are no data files). Values are z-scored per gene and
  clipped to ±2.5. `+`/`-` change the genes per group; clicking a gene maps
  it. The heatmap takes only its own keys, `c`, `s`, `f`, `?` and `q`.

`s` saves the view: the map as PNG, PDF and `.txt`, the structure plot as
a PNG and a PDF figure, the heatmap as a table and a PDF figure. Saved
files are listed with thumbnails on the left of the map, kept in
`.pinto-view/` in the working directory (`f` hides them).

By default (`--graphics auto`) the viewer asks the terminal what it
supports. It uses kitty graphics on kitty and Ghostty, iTerm2 images on
iTerm2, WezTerm and VS Code, and sixel on foot and xterm. Everywhere else,
and inside tmux, it draws coloured quadrant blocks (2×2 pixels per
character). `--graphics` picks one instead: `sixel` and `iterm2` for those
protocols, `quadrants` for the blocks above, `symbols` for block
boundaries at eighths of a character, `blocks` for half-blocks, `kitty`
for kitty graphics through a temporary file (inline over ssh), and
`kitty-inline` for kitty graphics always inline. Inside tmux,
`kitty-inline` goes through passthrough (best effort; needs
`allow-passthrough` on).

### Sessions

`pinto run` and `pinto view` keep going when the terminal goes: a dropped
connection, or ctrl+alt+d, detaches them (the first time, it asks for a
name), and the fits or lupin run on. Started again with nothing named, each
offers the live sessions to return to; ctrl+alt+s switches to another one.
On return the screen is drawn again for the terminal it is on now, its
image protocol asked anew. What a session ends with while nobody is
attached (the fits' summary, an error) is said by the next `pinto sessions`,
`attach`, `kill` or start. `--no-detach` (or `LEGUME_TUI_NO_DETACH=1`) runs
in this terminal only.

```sh
pinto sessions         # the live ones, with what each is doing
pinto attach fits      # back to one, by its name or id
pinto rename run-1 fits
pinto kill fits        # stops it as ctrl+c does: the fits running are stopped
```

### `mcp`: agents (MCP)

`pinto mcp` serves `pinto run` and the viewer over the
[Model Context Protocol](https://modelcontextprotocol.io) on stdin/stdout,
so any MCP client can fit a run and look at it as `pinto view` shows it, with
any model behind it (Claude, GPT, Gemini, or a local one). The agent sets up
fits as `pinto run` does, gets the map as an image with its legend, and asks
about groups and cells.

#### Connect a client

The server is one command, `pinto mcp`. A client needs that, and where the
project is: the server keeps its records (below) in its working directory, or
in `LEGUME_MCP_PROJECT` when set. Set it whenever the client may start servers
somewhere else (desktop apps usually do). If the client cannot find `pinto`,
give its full path (`which pinto`).

Most clients (Claude Desktop, Cursor's `.cursor/mcp.json`, Gemini CLI's
`settings.json`, and others) take this shape:

```json
{
  "mcpServers": {
    "pinto": {
      "command": "pinto",
      "args": ["mcp"],
      "env": { "LEGUME_MCP_PROJECT": "/path/to/project" }
    }
  }
}
```

Claude Code:

```sh
claude mcp add pinto -e LEGUME_MCP_PROJECT="$PWD" -- pinto mcp
```

VS Code (`.vscode/mcp.json`):

```json
{ "servers": { "pinto": { "type": "stdio", "command": "pinto", "args": ["mcp"],
                          "env": { "LEGUME_MCP_PROJECT": "${workspaceFolder}" } } } }
```

Codex CLI (`~/.codex/config.toml`):

```toml
[mcp_servers.pinto]
command = "pinto"
args = ["mcp"]
env = { LEGUME_MCP_PROJECT = "/path/to/project" }
```

Key names differ between clients; the server needs only the command, its
argument and, optionally, the environment. Nothing is sent anywhere by pinto:
the model sees what the tools return (maps as PNG images among it), and where
that goes is up to the client and model you choose (a local model keeps it on
the machine). An image reaches the model only if the client passes images on.

#### What the agent does

The server tells the agent how to work when it connects (the MCP
`instructions`). Some clients do not pass those to the model; with one of them,
put this in the agent's own instructions (an `AGENTS.md`, a system prompt):

> Use pinto's MCP tools to fit spatial link-community models and to look at
> them on the tissue. A fit: `run_methods` lists the methods (lc is the
> recommended one) and, with a method, its flags; `run_fit` runs one on data
> files with their coordinates, in the background for anything but a small
> data set (follow its job with `job_status` / `job_wait`), and never writes
> over a run. Looking: start with `view_open` on a run prefix (the `out` of a
> fit, or its `.pinto.json`): it lists the run's levels, batches, groups and
> any lupin rounds. `view_map` draws the map as a PNG with its legend; its
> arguments stay set until changed, so pass only what changes. `view_markers`
> lists a group's top features, `view_cell` names the cell under a pixel of
> the last map (or at a world point, or by name), and `view_export` writes the
> current view as PNG/PDF files. World coordinates are those `view_open`
> reports, with y running down the map. lupin rounds are read here as they are
> on disk; annotating and relabelling them is lupin's own server.

| Tool | Does |
|---|---|
| `run_methods` | pinto's fitting methods (what `pinto run` sets up); with a method, its flags: name, help, default, choices |
| `run_fit` | fits one method on data files with their coordinates (and batches), as `pinto run` would: the command checked as pinto parses it, run as a child `pinto`, in the background with `background` (a job), never over an existing run |
| `view_open` | opens a run; its cells, bounds, batches, levels, groups with their map colours, data files and lupin rounds |
| `view_map` | draws the map as a PNG with its state and legend; level, layer, focus, gene (source, clip), round and `show`, theme, edges, point size, box / fit / batch / center / zoom, width; settings stay until changed, and a call that fails changes nothing |
| `view_markers` | a group's top features, by fold over the other groups |
| `view_cell` | the cell under a pixel of the last map, at a world point, or by name: batch, group, strongest propensities, entropy |
| `view_export` | writes the view as a PNG and/or a PDF figure (only `.png`/`.pdf`, replacing one only with `overwrite`), by running the `pinto view` command it returns |
| `notebook` | (from legume-mcp) an entry in the day's lab notebook |
| `job_status`, `job_wait`, `job_cancel` | (from legume-mcp) follow a `run_fit` in the background, or stop it |

Focus takes a level's communities (`C3` or `3`), and when a round is shown,
lupin's cluster ids (`K3` or `3`) or cell types.

#### Records

The server keeps the project's records, laid out after Noble's guide to
organizing computational biology projects [1] (see
[legume-mcp](https://crates.io/crates/legume-mcp)):

- `results/<YYYY-MM-DD>/runall.sh`: every step that wrote something, in order,
  with paths relative to the project. Each `run_fit` is its `pinto lc` (or
  `cage`, `dsvd`) command and each `view_export` its `pinto view --png/--pdf`
  command, so `bash results/<day>/runall.sh` makes the fits and figures again
  without the agent.
- `results/<YYYY-MM-DD>/notebook.md`: the agent's plan, observations and
  conclusions.
- `results/INDEX.tsv`: one row per output with a one-line summary.

`LEGUME_MCP_RECORDS=off` keeps none.

#### Watching

While an agent works, the server writes a live feed to
`$XDG_RUNTIME_DIR/legume-mcp/pinto-<pid>/` (`LEGUME_MCP_WATCH=off` writes
none), with its view after each change. Follow it in a viewer of your own:

```sh
pinto view --follow          # the newest `pinto mcp`; --follow PID for another
```

The map follows the agent's view as it changes, and the panel lists its last
calls. A change of your own (a key, a click, a zoom) pauses following, and `F`
follows again. Nothing goes back to the agent. The viewer is a session like any
other, so you can detach it and attach again later.

[1] W. S. Noble, "A quick guide to organizing computational biology projects",
PLoS Comput Biol 5(7): e1000424 (2009). doi:10.1371/journal.pcbi.1000424

## Input data

- **Expression:** `.h5` or `.zarr` via `data-beans`
  (`data-beans from-mtx … --backend hdf5 -o data.h5`).
- **Coordinates:** CSV/TSV/parquet (Visium / Xenium column names recognized).
- **Batch (optional):** `-b labels.txt`.

## License

MIT
