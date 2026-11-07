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

### `cage` / `predict` / `impute`

See `pinto --help` and per-subcommand `--help`.

### `lra`: ligand-receptor analysis

`pinto lra` asks, per ligand-receptor pair, whether cells touching
ligand-carrying neighbours change their response programme compared with
similar cells that do not. It reads any prior run's manifest (lc, cage,
dsvd, ...) or data files with coordinates:

```sh
pinto lra --from out/run1.pinto.json          # writes out/run1.lra.parquet
pinto lra data.zarr --coord cells.csv -o out/lra
pinto lra --from out/run1.pinto.json --lr-pairs my_pairs.tsv
pinto lra --from out/run1.pinto.json --only L1,L2:R1_R2
```

Pairs and response programmes come from a reference (LIANA+ consensus
pairs; receptor -> signalling -> TF -> target programmes from OmniPath and
CollecTRI), downloaded once into a cache. `--lr-pairs` and `--targets` add
to it. A complex is named by its subunits joined with `_` (`R1_R2`);
every subunit must be in the data, and the complex counts as its scarcest
subunit, for ligands and receptors alike. In a `--targets` GMT file a set
name with `_` keys a ligand or receptor the pair list uses whole
(`R1_R2`), otherwise a `ligand_receptor` pair, split at the `_`
that names a listed pair (else the first). `PINTO_REFERENCE_DIR=<dir>`
moves the cache, `PINTO_OFFLINE=1` never downloads,
`PINTO_REFERENCE_REFRESH=1` rebuilds it. The analysis's fixed settings are
recorded in the manifest under `outputs.lra.settings`.
With `--from`, cell types of the run's newest lupin round, when it has
one, adjust the estimate as confounders and strata and add one row per
receiver cell type (`type:<name>`); without a round there are none. The
labels come from expression-derived clusters: the receiver identity
projection leaves the ligand, receptor and target genes out, but the
clustering behind the labels may have used them.

`--only` estimates a few pairs instead of all of them: comma-separated
`LIGAND:RECEPTOR` entries (a complex joins its subunits with `_`,
`L2:R1_R2`) or bare ligands for all of their pairs, matched on
genes (case and subunit order aside) among the reference pairs and
`--lr-pairs`; an entry no pair answers is an error that names the nearest
pairs. The receivers and each pair's estimate are those of the full run
(the receiver identity still leaves out every pair's genes), so a subset
takes seconds. A subset never writes over the full table: it goes to
`{out}.lra.only-<tag>.parquet` (and `.curve.parquet`), `<tag>` the list
itself when short (`l1+l2`) or a short hash of it, with or without
`-o`, and the manifest records it apart, under `outputs.lra.subsets`, the
full table staying `outputs.lra.table`. Its q-values are corrected within
the subset only: the `q_scope` column (`all pairs` in a full table) and each
row's `warnings` say so.

Two other methods test co-expression within link communities (they need an
lc run and `--lr-pairs`): `--method coexpression-test` (permutation test,
`{out}.lr_activity.parquet`) and `--method coexpression-scores`
(descriptive log odds per batch, `{out}.lr_scores.parquet`).

Reading `{out}.lra.parquet`: one row per pair and section (`all` is the
pooled row). `effect_competent` is the headline, a log-rate change of the
response programme (`exp(x) - 1` is the percent change); `verdict` sums up
the evidence and `verdict_note` says why in words. For comparison, and in no
verdict: `marginal_cor` (the correlation of response and contact ligand),
`crude` (exposed minus unexposed, no adjustment), each with `_se`; `effect_ipw`
(overlap weights alone) with `_se`, `_lo` and `_hi`, an interval no control
guards, so not a test of the pair; and `effect_outcome` (the outcome regression
alone), a point estimate with no SE or interval, because it can target a
different contrast when exposure is misclassified. `crude`,
`effect_ipw` and `effect_outcome` are averaged over exposure thresholds as the
headline is, each by its own evidence. `pinto view` shows the pairs and draws
each on the tissue (see `view`, `I`):

| verdict | meaning |
|---|---|
| `supported` | receivers next to ligand-carrying cells change their response programme, ligand and receptor are co-expressed across contacts, and the effect exceeds every negative control |
| `effect_without_coexpression` | receivers respond to neighbouring ligand, but ligand and receptor are not co-expressed across contacts beyond chance |
| `colocation_only` | ligand and receptor sit together across contacts, but receivers show no change in their response programme |
| `no_evidence` | neither a response nor co-expression at this false discovery rate |
| `control_failed` | an effect appeared, but it does not exceed a negative control (receivers without the receptor, or a matched background programme), or one of these could not be checked: likely confounding |
| `celltype_inseparable` | being exposed to the ligand is nearly the same as having a neighbour of one cell type, so the ligand cannot be told apart from that cell type |
| `insufficient_data` | too few exposed or unexposed receptor-carrying receivers, or too little overlap between them, to estimate an effect |

A verdict is deliberately conservative. A pair has an estimate only if its
receptor-competent receivers fill both arms at the middle competence threshold.
The threshold on neighbouring ligand is not chosen: each pair is estimated at
any detected ligand and at the quartiles of its positive doses, and those
estimates are averaged by their approximate Bayes factors, so doses that fit the
response better count for more and disagreement between them widens the
interval. Each control is averaged by its own evidence. `effect_log10_bf` is the
evidence for an effect at any threshold, and `tau_top` and `tau_top_weight` say
which threshold the average leans on most.

A significant effect then has to exceed every control, the same way, each
comparison corrected among the pairs with a significant effect (so their q is
empty for the rest): the effect in receptor-negative receivers
(`effect_contrast_q`) and the effect on matched background programmes (`effect_over_background_q`): many programmes of random
genes with similar expression, where the pair has to stand out from their spread. A hidden field can move
the controls too, so a control that is itself significant does not rule out a
larger effect; it is named in `warnings`. A control that could not run does not
pass, except a background programme where none exists: no match in the pool, or a section's
row, which has none.

### `lra-simulate`: power and calibration check

Can this tissue detect ligand effects of a given size, and does the
analysis stay honest under confounding? `pinto lra-simulate` keeps the real
tissue, plants effects of known size with and without hidden spatial
confounders, and reports power, type I error, bias and coverage per
scenario. It writes `{out}.lra-simulate.parquet`, and `pinto lra` on the
same run then points to it.

By default the hidden field drives the receptor too (`confound_receptor` in
the study): the receptor is simulated per identity bin from the real one,
scaled by `e^{βZ}` at the response's confounding strength β, and competence
is read from it as `pinto lra` reads it. Every run also scores the
co-expression statistics that do not adjust for confounders, each tested at
5%: cell-type label permutation (CellPhoneDB v2, Efremova 2020; CellChat's
product score, Jin 2021; identity bins coarsened to ≤ 8 types, Bonferroni
over type pairs), bivariate Moran's I over contacts (SpatialDM, Li 2023), a
contact co-expression score `Σ L_i R_j` (NICHES, Raredon 2022; COMMOT, Cang
2023), and lra's own co-expression log odds. The summary ends with a table
of each method's type I error and power per scenario, their means over the
confounded nulls and the planted effects, and the baselines' run time.

```sh
pinto lra-simulate --from out/run1.pinto.json
pinto lra-simulate --example-study > study.json   # edit, then:
pinto lra-simulate --from out/run1.pinto.json --study study.json
```

### Ablations, `compare` and held-out scoring

An ablation is a fit with one thing taken away or scrambled. Set it against
its parent with `pinto compare`. lc, cage and dsvd all take the flags; only
one ablation is allowed per run.

```sh
pinto lc data*.zarr.zip -c coords... --seed 42 -o base
pinto lc data*.zarr.zip -c coords... --seed 42 -o perm \
    --permute-coords within-batch --ablation-parent base.pinto.json
pinto compare base.pinto.json perm.pinto.json -o base_vs_perm
```

- `--permute-coords within-batch` shuffles cells among their own section's
  positions. `block` moves whole square tiles (`--permute-block` per side).
  Only the graph sees the scrambled positions; the written coordinates stay
  the cells' own.
- `--drop-features FILE|REGEX` drops features before QC.
- `--mask-cells FILE` masks the listed cells after QC. By default
  (`--mask-mode cut-edges`) the graph is built with them and their edges are
  then cut, so cells on either side of a masked region are not joined across
  it.
- The manifest records `ablation: {parent, kind, spec, seed}`.

`pinto compare A B` joins the two runs' labels on cell name. It reports ARI,
NMI, a Hungarian matching on Jaccard, and A's communities that vanish, split
or merge in B. A community's partners are the communities holding at least
`--tau` of its cells. Writes `{out}.compare.json` and
`{out}.contingency.parquet`.

Held-out scoring: fit lc without a donor, then
`pinto predict donor.zarr.zip -c donor.cells.parquet --model fit -o held_out`.
Each cell pair is split in two: its counts are thinned, or with
`--ablate-features` the listed features are hidden from the fit, as
`senna predict` does it. The pair's community mix is fitted on one part and
the other part is scored. The donor's batch shift δ comes from the fitting
parts only. `{out}.predictive.parquet` uses the column names cage and senna
use, with the training-composition null (`--null-from`), plus a pooled K=1
profile. The manifest's `holdout` block has the totals, the ceiling and a
per-section breakdown.

`pinto compare --paired A B` sets two models' held-out scores against each
other. A and B are `pinto predict` runs on the same held-out cells, each side a
comma list for leave-one-donor-out folds. Units are joined on their cell pair.
The donor is the replicate: `--donor-map` groups sections into donors. Writes
`{out}.paired.json` with each donor's difference in nats per count, exact sign
and Wilcoxon tests over donors, and a donor bootstrap interval.

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

After the fits, the Methods screen lists `lra` and `lra-simulate`, which run
on a fitted run instead of data files: set `--from` to the run's
`.pinto.json` (or its prefix) on the Parameters screen, with `--lr-pairs`,
`--targets` or `--only` as `pinto lra` takes them (files typed as
seen from where `pinto run` started). They are run and saved as
`{out}.cmd.sh` like the fits, write `{out}.lra.parquet` (a subset's
`{out}.lra.only-<tag>.parquet`) or `{out}.lra-simulate.parquet`, and
record them in the `--from` run's manifest; `v` opens that run.

A saved script refuses to run over what its run writes (a fit's
`{out}.pinto.json`, lra's table), and `pinto run` never writes over a script.
`v` opens a finished run in `pinto view`.

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
pinto view out --png pair.png --pair L1:R3  # an lra pair: senders, exposed
                                                # and receptor+ receivers
pinto view out --png pair.png --pair L1:R3 --pair-show exposure
pinto view out --png curve.png --pdf curve.pdf --pair L1:R3 --curve
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

#### Ligand-receptor pairs (`pinto lra`)

`I` opens the run's lra table (the full table, or with `T` each `--only`
subset in turn) as a list in the side panel: one line per pair with its
change, q and verdict, sorted by q (`o`: by effect, by name), one verdict at
a time with `v`/`V` (supported, effect_without_coexpression,
colocation_only, no_evidence, control_failed, celltype_inseparable,
insufficient_data, or all), and `/` keeps the pairs naming a gene. `↑`/`↓` (or a click) pick a
pair: the panel gives its effect as a percent change (`exp(x) - 1`) with its
95% CI, q, coexpression q, threshold, counts and verdict note (a subset's q
marked subset-corrected), each negative control's percent change ± SE with the
q of the effect exceeding it (empty unless the effect is significant), and the
map draws it, computed as lra computes it:
senders (cells carrying the ligand), and receivers exposed or not (their
contact exposure to the ligand above the table's threshold) and
receptor-competent or not (their identity bin detects the receptor). `e`
draws the exposure itself instead; `c` puts the pair's threshold–response
curve (the effect at each exposure threshold, pointwise 95% whiskers,
simultaneous 95% band, the table's threshold ringed) under the map. The
first pair drawn reads the run's receivers once (a pass over its data); each
pair after that reads two genes.

Space or `m` marks pairs, and `M` marks all the filter shows (a verdict, or
one gene's pairs after `/`); `L` runs `pinto lra --only` on the marked ones,
or on the pair under the cursor when none is marked, in the background beside
the run (`{prefix}.lra.only-<tag>.parquet`, its script
`{prefix}.lra.only-<tag>.cmd.sh`). The panel shows its progress bar, `X` stops
it, and its table opens in the list when it is done. On a run without an lra table, `I` then `L` runs the full `pinto lra`.
`s` saves the pair's map, and its curve when shown, as PNG and PDF with the
`pinto view --pair` command that redraws them.

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
> over a run. Ablations are fits with run_fit's ablation flags
> (permute-coords, drop-features, mask-cells with ablation-parent);
> `run_region` turns a polygon into the cell list mask-cells takes;
> `run_compare` sets two runs' communities against each other (ARI, NMI,
> matched communities, and which vanish, split or merge); `run_predict` scores
> held-out cells on a fitted lc model. Ligand-receptor analysis: `run_lra` on
> a run asks, per ligand-receptor pair, whether receivers touching
> ligand-carrying neighbours change their response programme compared with
> similar cells that do not (adjusted for region, section, cell identity,
> neighbour make-up, depth and segmentation bleed, with negative controls);
> `run_lra_simulate` plants effects of known size on the same tissue and
> reports, per scenario, power and type I error beside the co-expression
> baselines (`example_study: true` returns the study it takes). Run
> lra-simulate to check power and calibration before trusting lra on a new
> tissue. `run_lra` with `only` estimates a few pairs into a subset table of
> its own (its q corrected within the subset). `view_lra` lists a run's pairs
> with their verdicts and, given a pair, makes `view_map` draw it: senders,
> and receivers exposed or not and receptor-competent or not; `view_export`
> with `curve: true` writes its threshold-response curve. Each pair's verdict
> is one of: supported: receivers next to
> ligand-carrying cells change their response programme, ligand and receptor
> are co-expressed across contacts, and the effect exceeds every negative control;
> effect_without_coexpression: receivers respond to neighbouring ligand, but
> ligand and receptor are not co-expressed across contacts beyond chance;
> colocation_only: ligand and receptor sit together across contacts, but
> receivers show no change in their response programme; no_evidence: neither a
> response nor co-expression at this false discovery rate; control_failed: an
> effect appeared, but it does not exceed a negative control (receivers without the
> receptor, or a matched background programme), or one of these could not be checked: likely confounding; celltype_inseparable: being exposed to the ligand is
> nearly the same as having a neighbour of one cell type, so the ligand cannot
> be told apart from that cell type; insufficient_data: too few exposed or
> unexposed receptor-carrying receivers, or too little overlap between them,
> to estimate an effect. Looking: start with `view_open` on a run prefix (the
> `out` of a fit, or its `.pinto.json`): it lists the run's levels, batches,
> groups and any lupin rounds. `view_map` draws the map as a PNG with its
> legend; its arguments stay set until changed, so pass only what changes.
> `view_markers` lists a group's top features, `view_cell` names the cell
> under a pixel of the last map (or at a world point, or by name), and
> `view_export` writes the current view as PNG/PDF files. World coordinates
> are those `view_open` reports, with y running down the map. lupin rounds are
> read here as they are on disk; annotating and relabelling them is lupin's
> own server.

| Tool | Does |
|---|---|
| `run_methods` | pinto's fitting methods (what `pinto run` sets up); with a method, its flags: name, help, default, choices |
| `run_fit` | fits one method on data files with their coordinates (and batches), with the command `pinto run` would build, checked as pinto parses it and run as a child `pinto`: a background job by default (`background: false` waits), never over an existing run |
| `run_predict` | `pinto predict` as a recorded job: on an lc model, held-out pairs scored by count thinning or `ablate-features`, beside the training null, a pooled K=1 profile and the ceiling; returns the manifest's `holdout` block when run to its end |
| `run_region` | the cells of a run inside a polygon (world coordinates as `view_map` shows them, or a GeoJSON file in section coordinates) written as `{out}.cells.txt` for `run_fit`'s `mask-cells` |
| `run_compare` | `pinto compare` on two runs (prefix or `.pinto.json`): ARI, NMI, a Hungarian community matching with each pair's Jaccard, and A's communities that vanish, split or merge in B (partners hold ≥ `tau` of a community's cells); writes `{out}.compare.json` and `{out}.contingency.parquet`. With `paired: true`, two models' `run_predict` scores on the same held-out cells, tested over donors (`{out}.paired.json`) |
| `run_lra` | `pinto lra --from <run>` as a recorded job, or `pinto lra` on `data_files` with `coord_files` (`batch_files`, `knn_spatial`): `method` (contact-effect, coexpression-test, coexpression-scores), `out`, `lr_pairs`, `targets`, `species`, `seed`, `only`, and for the co-expression methods `n_permutations`, `json_fwer_threshold`, `min_gene_count`; writes `{out}.lra.parquet` and `{out}.lra.curve.parquet` (slots `lra.table` and `lra.curve` in the run's manifest; with `only`, `{out}.lra.only-<tag>.parquet` under `lra.subsets`, its q corrected within the subset) and replies with the verdict counts, the pairs whose effect survived the controls (percent change with its 95% CI, q) and, when the tissue has no power check yet, `run_lra_simulate` as the next call |
| `run_lra_simulate` | `pinto lra-simulate --from <run>` as a recorded job, or on `data_files` with `coord_files` (`batch_files`, `knn_spatial`): `study` (a JSON object, written as `{out}.study.json` and passed as `--study`), `out`, `species`, `seed`; writes `{out}.lra-simulate.parquet` (slot `lra.simulate`) and replies with each method's type I error and power per scenario, the estimable shares and the means over the confounded nulls and planted effects. `example_study: true` returns the built-in study and runs nothing |
| `view_open` | opens a run; its cells, bounds, batches, levels, groups with their map colours, data files and lupin rounds |
| `view_map` | draws the map as a PNG with its state and legend; level, layer, focus, gene (source, clip), round and `show`, theme, edges, point size, box / fit / batch / center / zoom, width; settings stay until changed, and a call that fails changes nothing |
| `view_markers` | a group's top features, by fold over the other groups |
| `view_cell` | the cell under a pixel of the last map, at a world point, or by name: batch, group, strongest propensities, entropy |
| `view_lra` | the run's lra tables (the full one and `--only` subsets, `table`) and their pairs: verdict counts and a list filtered by `verdict` and `search` (genes), `sort`ed by q, effect or name, each with its percent change and q; `pair` returns that pair's row and makes `view_map` draw it (senders, receivers exposed or not and receptor-competent or not; `show: exposure` the exposure itself) until a group layer, gene or `pair: ""` is asked for; settings stay until changed |
| `view_export` | writes the view as a PNG and/or a PDF figure (only `.png`/`.pdf`, replacing one only with `overwrite`), by running the `pinto view` command it returns; with `curve: true`, the selected lra pair's threshold-response curve |
| `notebook` | (from legume-mcp) an entry in the day's lab notebook |
| `job_status`, `job_wait`, `job_cancel` | (from legume-mcp) follow a `run_fit` (or `run_predict`, `run_lra`, `run_lra_simulate`) in the background, or stop it |

Focus takes a level's communities (`C3` or `3`), and when a round is shown,
lupin's cluster ids (`K3` or `3`) or cell types.

#### Records

The server keeps the project's records, laid out after Noble's guide to
organizing computational biology projects [1] (see
[legume-mcp](https://crates.io/crates/legume-mcp)):

- `results/<YYYY-MM-DD>/runall.sh`: every step that wrote something, in order,
  with paths relative to the project. Each `run_fit` is its `pinto lc` (or
  `cage`, `dsvd`) command, each `run_lra` its `pinto lra` command (the study
  file `run_lra_simulate` writes kept as a heredoc), and each `view_export`
  its `pinto view --png/--pdf` command, so `bash results/<day>/runall.sh`
  makes the fits and figures again without the agent.
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
