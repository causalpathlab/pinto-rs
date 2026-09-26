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
```

In the viewer, click a cell or legend entry to show one community (with its
marker features); `s` exports the current view as PNG, PDF, and a `.txt` with
the command that redraws it. The scale bar's units are guessed from the
coordinate columns (`--units um|px|none` to override).

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
