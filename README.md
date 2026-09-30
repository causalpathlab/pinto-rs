# PINTO

**P**roximity-based **I**nteraction **N**etwork analysis to dissect **T**issue
**O**rganizations.

PINTO segments spatial transcriptomics tissue into coherent regions by
clustering cell-cell edges of a spatial KNN graph. Each edge carries an
expression profile, and a collapsed Gibbs sampler assigns edges to communities;
per-cell soft membership falls out of the edge labels.

This is the standalone crate extracted from
[`legume-rs`](https://github.com/causalpathlab/legume-rs). The crates.io
package is `pinto-rs`; the installed binary is `pinto`.

## Installation

```sh
cargo install pinto-rs
# optional: --features hdf5 / cuda / metal
```

## Subcommands

### `lc` (link-community) — recommended

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

### `view`

Zoomable map of a run's communities, in the terminal. Pan and zoom from the
whole section down to single cells and their edges; switch between argmax,
soft mixture, entropy and single-community layers, and between levels.

```sh
pinto view out                 # interactive; ? lists the keys
pinto view out --summary       # what the run holds
pinto view out --png map.png --layer soft --width 3000
pinto view out --png zoom.png --bbox 8200,6800,8800,7250 --edges
pinto view out --pdf fig.pdf --focus C3,C17     # figure page: map, scale bar,
                                                # legend, focused markers
pinto view out --png cd3e.png --gene CD3E       # one gene: observed ln(1+count)
pinto view out --png cd3e.png --gene CD3E --expected --clip 95
```

In the viewer, arrows or a drag pan, the wheel or `z`/`Z` zoom, and `Esc`
steps back (plot, gene, selection); `q` quits. Click a cell or legend entry to show one community (with its
marker features), then click a marker to map that gene (`g`/`G` step through
them, `o` switches observed/model-expected, `p` the ramp top p99/p95); `s` exports the current view as PNG, PDF, and a `.txt` with
the command that redraws it. The scale bar's units are guessed from the
coordinate columns (`--units um|px|none` to override). Map colours follow the
terminal's background (`--theme light|dark` to choose; images default to dark).

Without a run, `pinto view` opens a browser to pick a `*.pinto.json` (↑↓,
Enter, ← up, type to narrow, `~` home).

#### Annotation with lupin

Cell types come from [lupin](https://crates.io/crates/lupin-rs) (0.2.1 or
later, on the `PATH` or at `$PINTO_LUPIN`); the viewer runs it and never
writes labels itself.

- `A` picks a marker panel (`gene<TAB>type` lines) and annotates the level
  on screen with `lupin annotate --level … --method enrichment`, writing a
  round `{prefix}.{level}.a{k}.lupin.json`.
- `a` steps the map through communities → the round's cell types → its
  clusters; `n`/`N` step through the rounds made from the run.
- `R` relabels the round's clusters, one at a time (`[ ]`, or click):
  `L` labels, `K` keeps the label, `M` merges (click or `[ ]`/space to add,
  Enter to label), and `+`/`-` add or drop the mapped gene from the target
  type's markers (Tab picks the type). Each needs a rationale. Decisions stay
  in a draft beside the round until `S` sends them to `lupin relabel --next`,
  after a confirmation; `P` previews what they would change.

```sh
pinto view out --round out.final.a1.r1.lupin.json --show types --png types.png
```

#### Structure plot and heatmap

`t` adds a structure plot under the map and `h` shows a gene heatmap in place
of it (press again, or `Esc`, for the map); both follow the grouping on
screen (communities, or with `a` the round's cell types or clusters):

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
  clipped to ±2.5. `+`/`-` change the genes per
  group; clicking a gene maps it.

`s` saves the view: the map as PNG, PDF and `.txt`, the structure plot as a
PNG, the heatmap as a table. Saved files are listed with thumbnails on the
left of the map, kept in `.pinto-view/` in the working directory (`f` hides
them).

Full-resolution images need a terminal with kitty graphics (kitty, Ghostty,
WezTerm) or sixel; elsewhere the map is drawn with coloured quadrant blocks,
2×2 pixels per character (`--graphics symbols` for boundaries at eighths of a
character, `blocks` for half-blocks).

## Input data

- **Expression:** `.h5` or `.zarr` via `data-beans`
  (`data-beans from-mtx … --backend hdf5 -o data.h5`).
- **Coordinates:** CSV/TSV/parquet (Visium / Xenium column names recognized).
- **Batch (optional):** `-b labels.txt`.

## License

MIT
