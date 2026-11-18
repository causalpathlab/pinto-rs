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

The commands sit under a few verbs:

| command | what it does |
|---|---|
| `pinto fit <method>` | a fit: `lc`, `dsvd`, `cage`, `cage-cell`, and the spatial baselines `spatialde2`, `stagate`, `spatialpca`, `mefisto` (and each as `-pb`) |
| `pinto apply <op>` | on a fitted run: `residual`, `predict`, `impute`, `propensity` |
| `pinto lra`, `pinto lra simulate` | ligand-receptor analysis, and its power and calibration check |
| `pinto sim <op>` | simulation with a planted truth: `new`, `score`, `pairs`, `baseline` |
| `pinto compare`, `pinto view`, `pinto run`, `pinto mcp` | as before |
| `pinto session <op>` | the detachable sessions: `list`, `attach`, `kill`, `rename` |

The sections below are headed by the method's own name.

Every run writes `{out}.pinto.json`, the manifest the other commands find
its files by. Commands that record into a run's manifest (lra and its power
check, several at once on one fit among them) take turns on it through a
hidden lock file beside it (`.run1.pinto.json.lock` beside
`run1.pinto.json`), which is kept. It is not an output: a run copied or
packaged without it loses nothing.

### `lc` (link-community), recommended

```sh
pinto fit lc data.h5 -c tissue_positions.csv -o out
pinto fit lc data.h5 -c coords.csv -o out --n-communities 25
pinto fit lc data.h5 -o out   # expression-only
```

Without `--batch-files` or `--auto-batch`, each data file is its own batch,
named by the file less its data extension (`sampleA.zarr` and
`sampleA.zarr.zip` are batch `sampleA`), so the names do not depend on the
order the files are given in. Two data files of the same name in one run
are refused. `pinto apply predict` and `pinto apply impute` give a query file the
model's batch of the same name, warning when it is not the model's own
file.

With several batches (data files, or `--batch-files` labels), lc estimates
batch effects δ, as dsvd and cage do, and divides every count of its edge
profiles by its batch's δ; `{out}.feature_community.parquet` stays on raw
counts. The manifest names `{out}.delta.parquet`, and
`{out}.delta_cells.parquet` with each cell's batch. `--ignore-batch` fits
without δ.

lc's fine Gibbs and greedy steps group the edges by connected component and
split a component of at least 512 edges into near-equal blocks of at least
256 edges, at most `--max-threads` blocks a component; the coarse cascade
levels before them sweep their edges in one sequence. Each block is swept in
order with its own random stream; the blocks, split or whole components
alike, run in parallel against one snapshot of the others and see their moves
only at the end of a sweep. The block layout and the Gibbs draws depend on
`--seed`, the data and `--max-threads`, which the manifest records
(`max_threads`), not on the machine: the same `--seed` and `--max-threads`
give the same lc draws on any machine, whatever threads run them.

`--max-threads N` (default 16), on every command that runs a fit, a
simulation or a comparison, is the most worker threads a run uses: rayon's own count (`RAYON_NUM_THREADS` when set, else the core
count), at most N. By default pinto takes at most 16 worker threads, so on a
machine with more than 16 cores a default run leaves the other cores free.
Raise `--max-threads` to use more cores, including for lc's fine Gibbs, which
then splits a large component into up to that many blocks.
`RAYON_NUM_THREADS` can lower the threads below `--max-threads` without
changing lc's draws; it cannot raise them above it. The commands that
`pinto run`, `pinto view` and `pinto mcp` start take their own
`--max-threads` unless the job gives one.

`--atomic-outputs` (off by default) makes a run's outputs complete or absent.
The run writes them in a hidden folder beside their final place,
`.{name}.partial-{pid}-{random}`, and renames them into place only when it
succeeds. The files that mark a run complete come last: the manifest
`{out}.pinto.json`, srt-simulate's `manifest.json`, residual's `residual.json`
and a single lra-simulate scenario's `{out}.lra-simulate.json`. An earlier
run's are taken away before the first output moves and the new ones put there
after the last, so a manifest on disk means the run completed.

Every command takes the flag. These stage their outputs: the fits (lc,
delta-svd, propensity, cage, cage-cell, predict, impute and the spatial
baselines, spatialde2, stagate, spatialpca, mefisto and their `-pb` forms),
lra, lra-simulate, residual, srt-simulate, srt-sim-baseline, compare, and
srt-sim-score and srt-sim-pairs when given an `--out`. The rest have nothing
to stage: `pinto view` writes the stills it draws (`--png`, `--pdf`) in place,
as without the flag, and `pinto run`, `pinto mcp`, attach, sessions, kill and
rename write no outputs of their own.

A run that fails or panics removes the hidden folder, with any folder it had
to make above it, and leaves nothing under the final names; one that is killed
leaves the hidden folder behind. Should the final move itself fail part way,
the folder is kept with the outputs not yet moved, the markers and stores the
move had taken aside are put back (a file already renamed over is not), and
the error names the folder. The paths a run records (the
manifest's outputs among them) and the paths its messages name are the final
ones, and what it prints on stdout at its end comes once the outputs are in
place. An earlier run's file that this run does not write is left as it was.

Some places take no rename, and a run that would write there is refused. An
earlier manifest that could only be written in place (a link, read-only,
another user's or hard-linked) cannot be taken away and put back whole: a fit
is refused at its start, before it does any work, and any run before its
first output moves; move the manifest away, or run without the flag. So is an
output whose final name is taken by a link, by a folder where a file would
land or by a file where a folder would. Two runs with the flag must not write
the same `--out` at the same time: each one's outputs are whole, but nothing
keeps one run's manifest from landing beside the other's files.

lra changes the manifest it records into (its `--from` run's, or the one at
its `--out`) only once its own outputs are in place, on the manifest as it is
then, so what other runs recorded meanwhile is kept; as without the flag, a
manifest already there that cannot be changed costs a warning, and one that
has to be made and cannot be is an error. srt-simulate and residual stage
their folder, moved there whole when it is missing or empty, else file by file
into it. A link to a folder, and an `--out` of `.` or `..`, are staged beside
the folder they lead to (the folder a run stands in is filled, never
replaced); the root, and a folder on another file system than the one it is
in, are refused before the run starts. A job of a `pinto run` session takes
the flag from the session, which has no row for it in a method's form. The
`pinto mcp` tools that start a fit, a prediction, a simulation, a baseline, a
residual, an lra or a comparison take `atomic_outputs`, true or false; left
out, it is as the server was started.

Feature modules: `--feature-network pairs.tsv` takes a feature-feature network
you supply, a two-column file of feature pairs named as the data's features
(tab-separated, or comma-separated for `.csv`; lines naming features the data
lacks are skipped). pinto ships no network of its own. lc clusters the network
into feature modules (Leiden; Traag, Waltman and van Eck 2019) and profiles
each cell pair by how far its two cells co-express pairs of modules, in place
of the random projection of all features. It writes
`{out}.feature_graph.parquet`, the pairs it kept.

```sh
pinto fit lc data.h5 -c coords.csv -o out --feature-network pairs.tsv
```

### `dsvd` (delta-svd), recommended

```sh
pinto fit dsvd data.h5 -c coords.csv -o out
```

dsvd scales each cell's counts to a total of 1e4 and takes their log
(after the batch divide, when there are batches), so cells are compared at
one sequencing depth. For each spatial cell pair and feature it then takes a shared
channel (the two cells' values summed) and a difference channel (their
absolute difference), pools them over graph-coarsened samples of pairs, and factors the
pooled Poisson-Gamma means by a randomized SVD; each pair's latent code is cut
into link communities. It stands beside lc as a first fit: it writes the same
propensity, link-community and feature-community tables, so `pinto view`,
`lra`, `residual` and `prop` read a dsvd run as they read an lc run.

### `prop` (propensity)

```sh
pinto apply propensity -z out.latent.parquet -e out.coord_pairs.parquet -o prop
```

### `cage`, `cage-cell` / `predict` / `impute`

```sh
pinto fit cage data.h5 -c coords.csv -o out
pinto fit cage data.h5 -c coords.csv -o out --feature-inclusion moran
pinto fit cage data.h5 -c coords.csv -o out --feature-modules 20
pinto fit cage data.h5 -c coords.csv -o out --feature-modules 0
pinto fit cage-cell data.h5 -c coords.csv -o out
pinto fit cage-cell data.h5 -c coords.csv -o out --feature-modules 20
```

cage learns an embedding of super-cells (pseudobulks of neighbouring cells) on
the coarsened spatial graph, one feature at a time, and places every cell and
cell pair on it with a small encoder. The encoder trains on the super-cells'
pairs, each thinned at random to a cell pair's depth, and then only encodes
the cell pairs. Where most features are noise, a run takes one of two
remedies, never both. By default the features are embedded through M learned
modules (sparsemax membership, Martins and Astudillo 2016), writing the module
tables. `--feature-modules auto`, the default, takes M from the data: features
with no grouping evidence share one background module, and the rest are the
communities Leiden (Traag, Waltman and van Eck 2019) finds in the
nearest-neighbour graph of their profiles over the finest pseudobulks; the
resolved M is logged, and a run that finds fewer than two modules trains
without them. `--feature-modules 20` fixes M, and `--feature-modules 0` turns
the modules off. `--feature-inclusion moran` instead trains each feature with
its probability of spatial signal (a jackknifed Moran's I, Moran 1950, through
the two-group local fdr of Efron et al. 2001), writing
`{out}.feature_inclusion.parquet`; given alone it turns the default modules
off, and given beside an explicit `--feature-modules` it is refused.
`cage-cell` is cage trained on cell pairs, for comparison with cell-level
methods: every within-batch cell pair has an embedding row of its own (the
latent), scored against other pairs of its batch, and a cell's embedding is
the mean of its pairs' rows; no encoder is fitted. It trains until each pair's
row has been updated `--pair-updates` times on average (default 200; the step
count is that times the pair count over the pairs a step touches, logged), with
`--epochs` (default 1000 here) a cap, a learning rate of 0.02 (cage's 0.005),
and an epoch drawing about four positives per pair. It estimates and removes
batch effects as cage does (`--ignore-batch` runs it without), and takes either
remedy as cage does: its modules start from pseudobulks built for that alone,
and training pools the sparse counts of the cells of the pairs each step draws.
`pinto apply predict` applies a cage run to new data (a cage-cell model is refused
for now); see each command's `--help`.
A query batch the model trained on (the same `--batch-files` label, or a
data file of the same name as one of the model's) is divided by the model's
own batch effects; only a new batch is estimated on the query. `pinto apply impute` divides the
reference cells of a cage run by the run's batch effects before pooling
them; for lc and dsvd runs it stays on raw counts, since their
feature-community profiles are fitted on raw counts.

### `lra`: ligand-receptor analysis

`pinto lra` asks, per ligand-receptor pair, whether cells touching
ligand-carrying neighbours change their response program compared with
similar cells that do not. It reads any prior run's manifest (lc, cage,
dsvd, ...) or data files with coordinates:

```sh
pinto lra --from out/run1.pinto.json          # writes out/run1.lra.parquet
pinto lra data.zarr --coord cells.csv -o out/lra
pinto lra --from out/run1.pinto.json --lr-pairs my_pairs.tsv
pinto lra --from out/run1.pinto.json --only L1,L2:R1_R2
```

Pairs and response programs come from a reference (LIANA+ consensus
pairs; receptor -> signalling -> TF -> target programs from OmniPath and
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
one, adjust the estimate as confounders (the receiver's own type and the
share of each type among its neighbours) and strata and add one row per
receiver cell type (`type:<name>`); without a round there are none. The
labels come from expression-derived clusters: the receiver identity
projection leaves the ligand, receptor and target genes out, but the
clustering behind the labels may have used them. `--cell-types` (a `cell`,
`cell_type` table) stands in for the round.

`--region` (a `cell`, `region` table, such as a residual fit's communities)
adds an indicator per region within each section to the adjustment, which the
smooth fields cannot give at a sharp edge, and a row per region
(`region:<name>`, its q corrected within the region). Every section and region
row also carries `effect_vs_rest` (with `_se`, `_df`, `_p`): its effect minus
that on every other receiver, from the same fit, the test that an effect is
the region's or the donor's own. A region row with p < 0.2 is checked by a
delete-one-tile jackknife over its sections, which widens its SEs and never
narrows them (`jackknife_ratio`); `--region-jackknife no-refit` (the default)
keeps the nuisance fits, `refit` refits them per tile and is conservative.
Neither removes confounding finer than the smooth fields follow: a hidden
spatial field at that scale that drives both a ligand and its response
inside a region biases the region's row, as it biases the pooled row, so a
region row carries the same limit as every other. A
`--from` run fitted on a residual's subtracted (excess) counts is read on
its raw counts, and its cell types come from the stage-1 run; a stage-2 fit
under `--offset` (`--residual divide`) is read on its own data, the raw
counts already, its cell types from the stage-1 run.

`--only` estimates a few pairs instead of all of them: comma-separated
`LIGAND:RECEPTOR` entries (a complex joins its subunits with `_`,
`L2:R1_R2`) or bare ligands for all of their pairs, matched on
genes (case and subunit order aside) among the reference pairs and
`--lr-pairs`; an entry no pair answers is an error that names the nearest
pairs. The receivers and each pair's estimate are those of the full run
(the receiver identity still leaves out every pair's genes), so a subset
takes seconds. A subset never writes over the full table: it goes to
`{out}.lra.only-<tag>.parquet` (and `.curve.parquet`, and `.celltype.parquet`
with `--marginal-baselines`), `<tag>` the list
itself when short (`l1+l2`) or a short hash of it, with or without
`-o`, and the manifest records it apart, under `outputs.lra.subsets`, the
full table staying `outputs.lra.table`. Its q-values are corrected within
the subset only: the `q_scope` column (`all pairs` in a full table) and each
row's `warnings` say so.

`--marginal-baselines` also scores each pair on the co-expression statistics
`lra-simulate` compares the estimate with, over the whole tissue: cell-type
label permutation (CellPhoneDB v2, Efremova 2020) with CellChat's
law-of-mass-action score, mean ligand in the sender type times mean receptor
in the receiver type (Jin 2021), a type pair tested when at least 10% of each
side detects its gene, the pair's p the smallest type-pair p,
Bonferroni-corrected over the tested type pairs; bivariate Moran's I of ligand
and receptor over the contacts (SpatialDM, Li 2023); and the contact
co-expression score `Σ L_i R_j` (NICHES, Raredon 2022; COMMOT, Cang 2023).
Each is one-sided, its null the cell-type labels (or, for the contact
statistics, the ligand within each section) permuted up to
`--marginal-permutations` times (default 2000; Moran's I and the contact
score stop early once their verdict at 5% is settled, while the cell-type
test runs every permutation, since its per-type-pair p and q need them). The cell types are the run's own (`--cell-types` or the
lupin round), `unassigned` permuted with the rest but never a tested type;
a run with none falls back to the identity bins coarsened to at most 8 types,
as `lra-simulate` does, and the manifest's lra options (`marginal_baselines`:
`cell-types` or `identity-bins`) and the log say which. Many types need more
permutations: with `m` tested type pairs the cell-type test can reject only
when `m / (1 + N) < 0.05`. The pooled rows gain `celltype_score`,
`celltype_p`, `celltype_tested`, `moran_i`, `moran_p`, `contact_score` and
`contact_p` (NaN, and `celltype_tested` 0, on the other rows), and
`{out}.lra.celltype.parquet` (beside a `--only` subset's table for a subset),
recorded in the manifest as `outputs.lra.celltype` (a subset's beside it
under `outputs.lra.subsets`), holds one row per pair and tested sender type
x receiver type: `ligand`,
`receptor`, `sender_type`, `receiver_type`, `n_sender`, `n_receiver`,
`mean_ligand_sender`, `mean_receptor_receiver` (`log1p` of the
depth-normalized counts), `ligand_detected_sender` and
`receptor_detected_receiver` (the detection shares that let it be tested),
`score`, `p`, `q` (BH within the pair's tested type pairs), `permutations`
and `labels`. None of these adjusts for confounding, unlike the estimate: a
ligand made by one cell type and a receptor carried by another score as an
interaction whatever the receivers do, which is what `lra-simulate` shows
them doing. They are for comparison and enter no verdict. The flag is off by
default, as the permutations cost more than the estimate on a full reference.

Two other methods test co-expression within link communities (they need an
lc run and `--lr-pairs`): `--method coexpression-test` (permutation test,
`{out}.lr_activity.parquet`) and `--method coexpression-scores`
(descriptive log odds per batch, `{out}.lr_scores.parquet`).

Reading `{out}.lra.parquet`: one row per pair and section (`all` is the
pooled row). `effect_competent` is the headline, a log-rate change of the
response program (`exp(x) - 1` is the percent change); `verdict` sums up
the evidence and `verdict_note` says why in words. For comparison, and in no
verdict: `marginal_cor` (the correlation of response and contact ligand),
`unadjusted_diff` (the mean response of exposed competent receivers minus that
of unexposed ones, every receiver weighted 1 and nothing adjusted for), each with `_se`; `effect_ipw`
(overlap weights alone) with `_se`, `_lo` and `_hi`, an interval no control
guards, so not a test of the pair; and `effect_outcome` (the outcome regression
alone), a point estimate with no SE or interval, because it can target a
different contrast when exposure is misclassified. `unadjusted_diff`,
`effect_ipw` and `effect_outcome` are averaged over exposure thresholds as the
headline is, each by its own evidence. `pinto view` shows the pairs and draws
each on the tissue (see `view`, `I`):

| verdict | meaning |
|---|---|
| `supported` | receivers next to ligand-carrying cells change their response program, ligand and receptor are co-expressed across contacts, and the effect exceeds every negative control |
| `effect_without_coexpression` | receivers respond to neighbouring ligand, but ligand and receptor are not co-expressed across contacts beyond chance |
| `colocation_only` | ligand and receptor sit together across contacts, but receivers show no change in their response program |
| `no_evidence` | neither a response nor co-expression at this false discovery rate |
| `control_failed` | an effect appeared, but it does not exceed the negative control (receivers without the receptor), or that could not be checked: likely confounding |
| `celltype_inseparable` | being exposed to the ligand is nearly the same as having a neighbour of one cell type, so the ligand cannot be told apart from that cell type |
| `insufficient_data` | too few exposed or unexposed receptor-carrying receivers, or too little overlap between them, to estimate an effect |

A verdict is deliberately conservative. A pair has an estimate only if its
receptor-competent receivers fill both arms at the middle competence threshold.
The threshold on neighbouring ligand is not chosen: each pair is estimated at
any detected ligand and at the quartiles of its positive doses, and those
estimates are averaged by their approximate Bayes factors, so doses that fit the
response better count for more and disagreement between them widens the
interval. The control is averaged by its own evidence. The prior on an effect
is set from the pair's own standard errors (ten times their median over the
thresholds), so the weights do not depend on the response's units.
`effect_log10_bf` is the evidence for an effect at any threshold against that
prior, and `tau_top` and `tau_top_weight` say which threshold the average leans
on most.

Every test reads its t reference on the effective number of variance tiles,
`3 (Σ s²)² / Σ s⁴` over the tiles' sums, between 2 and the tiles that carry the
row (`effect_competent_df` is that count less one). A row whose signal sits in
a few of many tiles, as a region of one donor does, is tested on those few.

A significant effect then has to exceed the negative control, the same way:
the effect in receptor-negative receivers (`effect_contrast_q`), the
comparison corrected among the pairs with a significant effect (so its q is
empty for the rest). A hidden field can move the control too, so a control
that is itself significant does not rule out a larger effect; it is named in
`warnings`. A control that could not run does not pass.

### `lra-simulate`: power and calibration check

Can this tissue detect ligand effects of a given size, and does the
analysis stay honest under confounding? `pinto lra simulate` keeps the real
tissue, plants effects of known size with and without hidden spatial
confounders, and reports power, type I error, bias and coverage per
scenario. It writes `{out}.lra-simulate.parquet`, and `pinto lra` on the
same run then points to it.

The receptor stays the real one: the hidden field never touches it, so the
receivers analyzed are a fixed set of cell types. Every run also scores the
co-expression statistics that do not adjust for confounders, each tested at
5%: cell-type label permutation (CellPhoneDB v2, Efremova 2020; CellChat's
product score, Jin 2021; identity bins coarsened to ≤ 8 types, Bonferroni
over type pairs), bivariate Moran's I over contacts (SpatialDM, Li 2023), a
contact co-expression score `Σ L_i R_j` (NICHES, Raredon 2022; COMMOT, Cang
2023), and lra's own co-expression log odds. The summary ends with a table
of each method's type I error and power per scenario, their means over the
confounded nulls and the planted effects, and the baselines' run time.

```sh
pinto lra simulate --from out/run1.pinto.json
pinto lra simulate --from out/run1.pinto.json --region region.tsv --plant-region D3_specific_domain_1
```

`--lr-pairs` and `--targets` add pairs and programs as for `pinto lra` (for
gene names no reference knows, such as a simulation's). `--plant-region`
plants every effect in one `--region` label only, and reads the estimate, its
truth and its verdict on that region's row beside the pooled one (`region_*`
columns); the summary and the table (`planted_region`) name the region.

**One scenario per call.** A sweep over scenarios runs one call per point
and seed, many at once, and scores each call by AUROC and AUPRC. `--set
KEY=VALUE` (repeatable), `--pairs N`, `--null-pairs N` or `--one` selects
this mode in place of the built-in grid:

```sh
pinto lra simulate --from out/run1.pinto.json -o sweep/p07-s3 --seed 3 \
  --set effect_pve=0.02 --set ligand_field_pve=0.25 --set response_field_pve=-0.25 \
  --set confounder=bumps --set confound_scale=32 --pairs 10 --null-pairs 10
```

The keys are the table's own column names, so a sweep axis is a column:
`effect_pve` (the planted effect's share of the program genes' log-rate
variance), `ligand_field_pve` and `response_field_pve` (the hidden field's
share of the producer log odds and, signed, of the response),
`confound_scale` (the field's length scale, in contact lengths),
`confounder` (`fourier`, `bumps` or `realgene`), `confound_shape`
(`linear` or `quadratic`), `heterogeneous`, `marker` and `dose_response`
(`true`/`false` or `1`/`0`), `dose_midpoint` and `dose_order` (the dose
response's half-way quantile and Hill order) and `rep`. Unset keys are no
effect and no confounding; a point out of range is refused before any data
is read, and the mode needs `-o`.

Every call holds both labels: the pairs fit to benchmark are ranked by
ligand abundance and the top `--pairs` plus `--null-pairs` (default 10
each) taken, some carrying the effect and the rest none, all under the
same hidden field. The fewer label is spread evenly among the more, the
`k`-th of `few` at rank `floor((k + 1/2) m / few)` of the `m` taken, as
Bresenham's line algorithm spreads its steps (Bresenham 1965, IBM Systems
Journal 4(1):25-30): both labels then have the same mean abundance rank to
within one, whatever the counts, so abundance does not sort them beyond
that. Equal counts alternate, even ranks planted, each planted pair one
rank above a null one. With fewer pairs to benchmark the call takes what
there is, each label keeping its asked share, and warns, as it does below
20 rows. `realgene` draws its field from the most abundant gene outside every
pair of the call, and fails when there is none rather than change the
point. `--plant-region` applies as above; the scores below stay the pooled
row's, the region's being `region_estimate` and `region_se`. `-o` must not
be the `--from` run's prefix, whose table is the grid's.

The call writes `{out}.lra-simulate.parquet`, the table above plus
`planted` (0/1), `t` (estimate over SE, a one-sided score as the planted
effect is positive), `p` (its two-sided p on the t reference) and
`marginal_cor_p`, `unadjusted_diff_p` and `effect_ipw_p`. These five are
doubles computed at full precision (the grid's columns stay floats), so a
p far below a float's smallest value is kept rather than written as 0. A row not estimable
stays, with NaN scores. Then, last and atomically, it writes
`{out}.lra-simulate.json`, whose presence marks the call complete: the
resolved scenario, the counts planted and null and the rule that labelled
them, each pair's ligand, receptor, rank, label, seed and run time, the
real-gene confounder, `--seed`, `rep`, the pinto version, the reference
record (path and sha256) and the table's sha256. `field_sha256` names the
hidden field the call drew, and each pair's the field it ran under, the
same for every pair. It records nothing in the run's manifest.

Each pair's draws start from `mix_seed(--seed, rank)`, with `rep` mixed in,
and the hidden field from `--seed` and `rep` alone, so a pair's rows depend on
its rank and label alone, not on the other pairs of the call (save under
`realgene`, whose gene can change with `--pairs` and `--null-pairs`; the
counts also decide which ranks take which label), and the same flags give the
same table, byte for byte, on any number of threads. These streams are not the
built-in grid's: a point of the grid and the same point run alone give
different draws. Pairs run at once as memory allows within pinto's worker
threads (at most `--max-threads`, 16 by default); a runner starting
several calls side by side sets `PINTO_MEMORY_BUDGET` (bytes) per call, since
each process otherwise takes half the free memory for itself.

### `srt-simulate`, `srt-sim-score`, `srt-sim-pairs`, `srt-sim-baseline`: multi-donor data with a planted truth

Do methods find the same domains in every donor, and keep a domain that
only one donor has, when donor effects are as large as cell-type effects?
`pinto sim new` makes several donors' sections with a known answer,
`pinto sim score-by-set` scores any labelling against it,
`pinto sim score-by-pair` scores a latent mapping on pairs of spots, and
`pinto sim baseline` runs the two usual workflows (pool every donor, or
analyze each donor alone) for comparison.

**What is simulated.** Location sets a spot's domain (smooth random fields
on real or grid coordinates), the domain sets its mix of cell states, and
the states set expression. One donor can own an extra domain, a new cell
state or a new mix of shared ones, which is biology to keep. Counts are
negative binomial through a low-rank Gaussian copula, or, with
`--reference-data`, drawn from a real data set (see below). With one donor and no
nuisance this reduces to SRTsim's reference-free domains (Zhu, Shang and
Zhou 2023); the copula follows scDesign3 (Song et al. 2023) in form.

**Reference data (`--reference-data`).** `pinto sim new --reference-data
ref.zarr.zip` takes the sections' expression from a real data set (genes by
cells; `.zarr.zip`, `.zarr` or `.h5`; convert an AnnData file with
`data-beans from-h5ad ref.h5ad -o ref`). data-beans fits each gene's count
distribution in each of the data set's cell groups (a Poisson mixture), its
coupling to depth, and a Gaussian copula of the highly variable genes; the
planted cell states take the cell groups largest first
(`--reference-data-labels` names them, else k-means), so a spot of one state
draws its group's counts, its depth and noise the data set's own. The
donor, donor-by-cell-type and section shifts are shares of the data set's
variance between states, and `realized_pve_expressed_genes` in the manifest
says how close they came. The manifest records the data set's size and a
fingerprint of its content, never its path; `--set` refuses what the
reference data replaces (`n_genes`, `log_mean`, `dispersion`,
`copula_rank`, ...).

```sh
pinto sim new --scenario donor-confounded --reference-data fan2019.zarr.zip -o sim
```

`pinto lra simulate --reference-data ref.zarr.zip` draws the program genes
of its planted effects the same way: each identity bin takes the cell group
whose mean counts over those genes correlate best with the bin's, and each
program gene the data set has is drawn from that group's count distribution
around the tissue's own mean, correlated with the other program genes as in
the data set (the Poisson log-normal draw otherwise). Its record names the
data set by size and fingerprint (`{out}.lra-simulate.reference-data.json`
beside the grid's table, `reference_data` in a single scenario's JSON).

**Ligand-receptor pairs.** The planted genes (GENE1, GENE2, ...) are named
by no ligand-receptor reference (with `--reference-data` the genes are real, and
lra's own reference applies too), so `sim new` also writes `truth/lr_pairs.tsv` and
`truth/targets.tsv`: per state, a ligand of its program alone and a
receptor of the next state's alone, with the rest of the receiving state's
program as the receptor's response program (no ligand or receptor is
anyone's target). No ligand-receptor effect is planted, so on the
simulation they are nulls; `lra simulate` plants its own:

```sh
pinto lra simulate --from res/lc.pinto.json \
  --lr-pairs sim/truth/lr_pairs.tsv --targets sim/truth/targets.tsv
```
`--set nuisance_off=true` with the same seed draws the exact counterfactual
without donor or section effects.

**The variance shares.** Each spot's log rate is a sum of terms, each
holding a set share of the log rate's variance:

| dial | share of |
|---|---|
| `pve_type` | the cell types (each a cell state of the model), the signal that defines domains |
| `pve_donor` | donor-wide effects, the same in every cell type |
| `pve_inter` | donor-by-cell-type effects |
| `pve_section` | per-section effects |
| (none) | spot noise: what the four shares leave |

Noise is the remainder, so raising one share with the others fixed shrinks
the noise, not the other shares.

**The size of the donor effect** is its share of the log rate's variance,
`pve_donor + pve_inter`: donor-wide plus donor-by-cell-type, the two terms
that differ between donors. `pve_section` is the per-section share on top,
so the whole nuisance is `N = pve_donor + pve_inter + pve_section`. Wherever
a result says "the donor effect" or labels a sweep by it, it means
`pve_donor + pve_inter` at a stated `pve_type`, and how the two donor terms
split it should be stated too, since donor-by-cell-type effects are the harder
ones. A sweep over the nuisance shares has to choose what gives way, and
the choice changes what a trend along N means:

| held fixed | set | as N grows |
|---|---|---|
| `pve_type` | `pve_type = t` | the noise shrinks; the cell-type signal grows against it |
| the type-to-noise ratio | `pve_type = t × (1 − N)` | type and noise shrink together |
| the noise share | `pve_type = 1 − noise − N` | the cell-type signal shrinks |

Every run logs the shares it has, noise spelt out as the remainder. A
nuisance share that is set but off is named with the reason: with
`nuisance_off`, or when it has nothing to vary across (one donor or
section, or every one unconfounded), the term is dropped and its share does not
pass to the noise. With nuisance on, the log gives the type-to-noise ratio
here and with the nuisance shares at 0. The `nuisance_off` counterfactual
keeps the noise share, so its ratio is the run's own. The manifest records
the shares realized on the program genes. Sections can be freed of the
section shift (`unconfounded_sections`), and donors of every shift: an
unconfounded donor has no donor-wide, donor-by-cell-type or section shift on
any of its sections. `--set n_unconfounded_donors=1` gives any scenario one
such donor, drawn from the seed on a stream of its own (so the rest of the
data is drawn as without it), never the owner of a domain only one donor
has; more than the eligible donors is refused. `unconfounded_donors=[0]`
names them instead (one way or the other, not both). `truth/donors.parquet`
and each spot in `truth/spots.parquet` say whether the donor is
`confounded` (some donor-wide, donor-by-cell-type or section shift reaches it;
`false` throughout when no nuisance term is on), and the manifest lists the unconfounded donors and whether
they were drawn.

**Named scenarios** set the rest; `--set key=value` changes any parameter,
and `pinto sim new --help` lists them. Each has 3 donors with 2
sections each, and 4 spatial domains shared by every donor, each 95% one
cell type, with the cell type 30% of the log rate's variance:

| scenario | donor and section shifts | a domain only one donor has |
|---|---|---|
| `unconfounded` (the default) | none | none |
| `donor-confounded` | donor-wide 15%, donor-by-cell-type 15%, section 2% | none |
| `donor-confounded-donor-only-cell-type` | donor-wide 15%, donor-by-cell-type 15%, section 2% | the third donor's, about 15% of its spots, a cell type no other donor has (its program at cosine 0.3 to the first shared cell type's) |
| `donor-confounded-donor-only-mix` | donor-wide 15%, donor-by-cell-type 15%, section 2% | the third donor's, about 15% of its spots, the first two shared cell types in equal parts, a mix no shared domain uses |
| `composition-shift` | none, but donors differ in domain proportions (log-scale spread 0.5) | the third donor's, about 15% of its spots, a cell type no other donor has |

```sh
pinto sim new --scenario donor-confounded-donor-only-cell-type \
  --set pve_donor=0.2 --set seed=2 -o sim
pinto view sim/truth/truth.pinto.json          # the planted domains, states and donors
# one shift-free donor, drawn from the seed
pinto sim new --scenario donor-confounded-donor-only-cell-type \
  --set n_unconfounded_donors=1 -o sim-one-free
```

**Outputs.** `sim/sections/S*.zarr.zip` (counts) and `S*.coords.parquet`
(`row_um`, `col_um`) per section; `sim/truth/` with each spot's section,
donor and whether it is confounded, domain and its owner, state and mixture,
each donor's status (`donors.parquet`), and the truth as a run that
`pinto view` opens; `sim/manifest.json` with every parameter and the
unconfounded donors. A missing or
empty `sim` is written as `sim.partial` and moved into place once whole, so
a stopped run leaves no `sim`; a `sim` that already holds files is written
in place.

**Scoring.** `pinto sim score-by-set --truth sim --labels <runs or tables>`
runs two tests:

1. *Donor-invariant domains*, on spots of shared domains, labels against
   domains: `shared_ari` (adjusted Rand index); `shared_ami` (adjusted
   mutual information, Vinh, Epps and Bailey 2010: the expected MI exact
   under the hypergeometric model, arithmetic-mean normalization; 0 chance,
   1 the same partition); `shared_nmi` (normalized mutual information,
   arithmetic mean; not corrected for chance, so many small labels raise
   it); and `donor_conditional_entropy`, `H(donor | label, domain) /
   H(donor | domain)`, 0 to 1 (1: labels mix the donors fully; 0: the label
   sets the donor), pooled over the shared domains by their spots and for
   each (`donor_conditional_entropy_by_domain`; a domain in one donor is
   skipped with a note, and the pooled value is null when none is left).
2. *Donor-specific domains*, each within its owning donor's spots only
   (`within: "owner_donor"`), so a label is not marked down for spots it
   holds in other donors: its best-matching label there, and that label's
   `f1`, `precision` (`prec` in the table; low: the owner's shared spots
   absorbed) and `recall` (in the `-o` JSON) over the owner's spots; and
   `owner_share` (`owner`), over every spot, the share of the label's spots
   in the owner (1: the label is confined to the owning donor).

**Scoring on pairs.** A clustering has its own preferences (how many
groups, how big), so `pinto sim score-by-pair --truth sim --vectors <inputs>`
scores the per-cell vectors themselves: a run's propensity at `--level`, a
cage run's `{prefix}.cell_embedding.parquet`, or any table with a `cell`
column and numeric columns. Two spots' similarity is the cosine of their
vectors, scored by AUROC (Mann-Whitney, ties counting a half; Hanley and
McNeil 1982) and AUPRC (average precision) on four sets of pairs, each
reported with its prevalence (the AUPRC of chance):
`shared_domain_across_donors`, shared-domain spots of two donors, positive
when they share a domain; `shared_domain_within_donor`, the same within one
donor, the reference, with `donor_gap` (within less across) what the donor
costs the mapping; `patches`, for each donor-specific domain,
`donor_domain_vs_own_donor`, pairs of its spots against one of its spots
with a spot of the same donor outside it, half and half (`negatives:
"same_donor_outside_domain"`, so telling donors apart scores 0.5); and
`neighbour_same_domain`, each spot's `--knn` (6) nearest neighbours in its
section, positive when they share a domain. The sampled sets take `--pairs`
(20000) pairs each from `--seed`, so a score is reproducible; the table
shows the AUROCs and `-o` writes every number as JSON.

**Scoring another method.** Any tool can be scored:

1. Export the sections, e.g. `data-beans to-h5ad` or `data-beans to-mtx` on
   each `sim/sections/S*.zarr.zip`, and fit them with that tool.
2. Write a table with a `cell` column and a label column (`--label-column`,
   default `label`). With several sections a cell is named
   `barcode@section` (e.g. `S1_00001@S1`), as pinto names them; with one,
   the bare barcode. `sim/truth/spots.parquet` lists every name.
3. `pinto sim score-by-set --truth sim --labels that.tsv`. Unlabelled cells
   (empty, `NA`, `-1`, `unassigned`) are left out and counted.

**Baselines.** `pinto sim baseline --truth sim --method <m> -o m.tsv`
writes such a table:

- `pooled`: the scanpy recipe (Wolf, Angerer and Theis 2018) on all
  sections together, no batch handling: normalization, seurat-flavour HVGs
  (Satija et al. 2015), scaling, PCA, a kNN graph and Leiden (Traag,
  Waltman and van Eck 2019) toward the true number of domains;
- `harmony`: the same with the PCs corrected for the donor by Harmony
  (Korsunsky et al. 2019);
- `per-donor`: the recipe on each donor alone, toward the domains that
  donor holds, then each cluster labelled by marker correlation with the
  largest donor's clusters, as scmap-cluster projects onto reference
  centroids (Kiselev, Yiu and Hemberg 2018).

Agents get the same loop as MCP tools: `run_simulate`, `run_sim_score_by_set`,
`run_sim_score_by_pair` and `run_sim_baseline` (see `mcp` below).

### Spatial baselines: `spatialde2`, `stagate`, `spatialpca`, `mefisto` and their `-pb` forms

Published spatial-domain methods, ported to run on pinto's inputs so they
can be scored beside `lc`, `cage` and `dsvd`:

- `spatialde2`: SpatialDE2's tissue segmentation, a Poisson hidden Markov
  random field with a Dirichlet-process prior over regions (Kats,
  Vento-Tormo and Stegle 2021, bioRxiv 10.1101/2021.10.27.466045);
- `stagate`: a graph-attention autoencoder over the spatial graph (STAGATE;
  Dong and Zhang 2022, Nature Communications 13:1739), its embedding
  clustered by pinto's own methods: seeded k-means into exactly
  `--n-clusters` domains, or Leiden when `--n-clusters` is not given. With
  `--n-clusters` this differs from the STAGATE tutorial, which fits a
  Gaussian mixture in R (a GPL package, not ported) at that step;
- `spatialpca`: SpatialPCA, probabilistic PCA whose spatial PCs carry a
  spatial kernel prior (Shang and Zhou 2022, Nature Communications
  13:7203, doi:10.1038/s41467-022-34879-1), its PCs clustered by pinto's
  k-means or Leiden (in place of the paper's Walktrap and Louvain) and
  refined by the paper's majority vote over each cell's 6 nearest;
- `mefisto`: MEFISTO, multi-group Bayesian factor analysis with spatially
  smooth factors (Velten et al. 2022, Nature Methods 19:179,
  doi:10.1038/s41592-021-01343-9), the sections as its groups, its factors
  (each scaled by its weights' norm) clustered by pinto's k-means or
  Leiden (the paper does not cluster).

`spatialde2` and `stagate` are model-level ports of its reference code: the same preprocessing,
model, loss or updates, defaults, stopping rule and clustering step (but
for stagate's fixed-K clustering, above), not bit-for-bit output; each
module's documentation names the repository and commit read, the function
behind each step, and every deliberate deviation. The license texts of
SpatialDE, STAGATE_pyG and STAGATE (MIT) and of scanpy (BSD 3-Clause), with
their copyright lines, are in `THIRD_PARTY_NOTICES.md`. They take the flags `lc`
does for its inputs: the data files (one section each), `-c` coordinate
files, `--coord-column-names`, `-b` batch files, `-o` the output prefix and
`--seed`; spatial graphs are built within a section only. `--n-clusters`
sets the number of domains as each original takes it (see each `--help` for
its default and what it does when not given).

Every baseline runs pinto's sparse formulation of its method only, and
none has a `--reference` mode: the original dense formulations cost too
much time and memory at real cell counts. For `spatialde2` the graph is
the reference's own `neighbors=k` option (`--neighbors`, 100 by default),
in place of its default graph, every pair of cells in a section
connected. For `stagate` the sparse form is the same model and gives the
same numbers as the original's dense input up to float rounding; it keeps
the input sparse on the CPU, while on a GPU `stagate` holds the selected
genes dense, and the manifest and `baseline.json` record the layout held
as `input_layout`. Segmentation runs on every gene by default, as
SpatialDE2's `tissue_segmentation` does; `--n-top-genes` passes a list of
the most expressed.

`spatialpca` and `mefisto` are written from their papers and supplements
alone (their reference code is GPL or otherwise copyleft, and none of it is
read or translated), over one shared sparse factor core: the latent factor
model `Y = W Z + E` with a spatial prior on each factor. The papers'
dense-kernel priors (SpatialPCA's Gaussian
kernel with its low-rank approximation, MEFISTO's Gaussian processes on
inducing cells) cost too much time and memory at real cell counts, so
pinto implements the papers' models with that prior replaced by a sparse
precision on the per-section kNN graph, `(I + rho L) cbar(rho)`, and
offers the `-pb` forms for scale. Its strength is fitted
with the rest (one `rho` for SpatialPCA, one per factor for MEFISTO), so
no `N × N` matrix and no `O(N^3)` step is met: determinants and traces come
from the graph's spectrum (exact on sections of at most 2000 cells,
stochastic Lanczos quadrature above), solves from conjugate gradients.
SpatialPCA's preprocessing is the paper's, each step written from its
own paper: the spatially variable genes by SPARK-X (Zhu, Sun and Zhou
2021, Genome Biology 22:184, doi:10.1186/s13059-021-02404-0) at a
Benjamini-Yekutieli FDR of 0.05, at most `--n-genes` (3000), then
SCTransform's Pearson residuals (Hafemeister and Satija 2019, Genome
Biology 20:296, doi:10.1186/s13059-019-1874-1), standardized per gene
(the paper takes SPARK, a GPL package, on small data sets; SPARK-X serves
all here). MEFISTO models the most variable genes (`--n-genes`), as its
paper does. Each module's documentation lists every
deviation. The visible flags are the number of factors (`--n-factors`,
`--n-pcs` for `spatialpca`; 20 and 4 by default) and of genes (3000 and
2000); the papers' tuning knobs are hidden.

`spatialde2-pb` fits `spatialde2` on super-cells and unravels the fit
back to cells. The cells are coarsened as `cage` coarsens them, to about
`-d` (`--n-pseudobulk`) super-cells within each section, each with its
cells' counts summed and their centroid as coordinates, on the cell graph
collapsed onto them (a super-edge's weight sums the cell-cell edges it
collapses, so the super-cells' model is the cells' model with each
super-cell's cells sharing a region). Each cell's region posterior is then
re-inferred on its own counts and the cell graph with the fit held fixed.
`{out}.cell_pb.parquet` records each cell's super-cell and that
super-cell's label (broadcasting it is a diagnostic, not the output), and
the manifest's `baseline` block records the input. The bar it was held
to: on a simulated section at one super-cell per eight cells, within 0.1
ARI of the cell-level fit and better than the broadcast labels at domain
boundaries.

`spatialpca-pb` and `mefisto-pb` fit the sparse model on the same
super-cells and collapsed graph, then unravel to cells by conditional
re-inference with the fit's loadings held fixed: SpatialPCA's `W` (its
noise, `tau` and `rho` refitted on the cells), MEFISTO's weights, ARD and
sparsity (its noise, factors and each factor's `rho` refitted). Both were
held to the same bar and met it.

`stagate-pb` trains `stagate` (sparse) on the same super-cells, on the
cell graph folded onto them, unweighted (STAGATE's attention takes no edge
weights) and symmetric. The trained encoder then embeds every cell from
its own counts on the cell graph, made symmetric the same way; the domains
are clustered on the super-cells' embedding (k-means or Leiden, as
`stagate`), each cell takes the nearest domain centroid, and the domains
no cell takes are dropped, so `n_domains` counts those present. The
centroids are taken on the cells' embedding, grouped by their
super-cells' domains, since single cells' sparser counts sit on another
scale than the summed counts the encoder was trained on. It was held to
the same bar and met it.

One deviation is shared: with two or more batches (data files, `-b`
labels, or `--auto-batch` components) the counts are divided by the
input's own batch effects (δ, estimated as `cage` estimates them and
written to `{out}.delta.parquet`) before the method's own preprocessing.
The baselines take cage's batch flags under the same names and defaults
(`--batch-files`, `--auto-batch`, `--ignore-batch`, `--cnv-clones`). A
baseline reads its input through the same preprocessing as lc, dsvd and
cage: the same cell QC (on by default; `--no-qc` keeps every cell), the
same `--auto-batch` components on pinto's spatial graph, the same δ, the
same `--offset` (a stage-2 fit, on the cells QC kept), the same ablations
(`--permute-coords`, `--drop-features`, `--mask-cells`) and the same cells
table; the `-pb` forms coarsen along pinto's cell-pair graph.
`--ignore-batch` runs the original method on the counts as stored.

All fit the whole tissue in memory, so each estimates its peak before its
large allocations, logs it, and stops with an error when the estimate
exceeds the memory budget: `PINTO_MEMORY_BUDGET` (bytes) when set, else
half the memory available to the process at the start (cgroup limits
included). The first check comes from the files' stored shape (cells,
genes and nonzeros), with nothing preloaded: the counts are read after
it, preloaded as the other commands read them, and a file that does not
store its nonzero count leaves that read unchecked. Under `--auto-batch`
the batches are known only once the read has found the components; the
check charges each section as one batch.
`stagate` grows with the cells, the edges and the genes, and far faster
with a `--rad-cutoff` that takes many neighbours. Lower `--n-top-genes`,
`--k-cutoff` or `--rad-cutoff` (`--neighbors` or `--n-clusters` for
`spatialde2`), train `stagate` on a GPU (`--device cuda`, in a build with
the `cuda` feature), lower `--n-genes` for `spatialpca` and `mefisto`
(which grow linearly with the cells; their `-pb` forms fit on
super-cells), or raise
`PINTO_MEMORY_BUDGET` knowingly; a runner starting several at once gives
each its share.

```sh
pinto fit stagate s1.zarr,s2.zarr -c s1.coords.parquet,s2.coords.parquet -o res/stagate --n-clusters 4
pinto sim score-by-set --truth sim --labels res/stagate
pinto sim score-by-pair --truth sim --vectors res/stagate.cell_embedding.parquet
```

Each writes a pinto run, so `pinto view`, `compare`, `srt-sim-score` and
`srt-sim-pairs` read it as they read a fit: `{out}.pinto.json` (the
manifest), `{out}.propensity.parquet` (cell × domain membership: the
method's posterior where it has one, else one-hot), `{out}.cells.parquet`,
`{out}.cell_embedding.parquet` (the method's per-cell vectors),
`{out}.labels.tsv` (`cell<TAB>label`) and `{out}.baseline.json` (the
settings, as run). `pinto run` lists them after the fits, and over MCP
`run_methods` and `run_fit` take them as they take `lc`. `pinto apply residual`
takes one as its stage-1 run, as it takes a fit.

### `residual`: what a fitted run leaves unexplained

`pinto apply residual --from FIT -o OUT` (or `--run FIT`) takes a fitted run and
what its communities do not explain into a second stage of a two-stage
fit, meant to pick up structure the first stage absorbed, such as a domain
only one donor has.

From the run's per-cell propensities it refits raw-count community profiles
`p_k` (lc's feature-community statistic with unit weights) and each cell's
proportions `θ_i` (the per-cell EM of `pinto apply impute`), so a cell's
expectation is `μ_gi = n_i Σ_k θ_ik p_{k,b}(g)`. What is left is taken one
of two ways, `--residual`:

- **`divide`** (alias `multiplicative`, the default): stage 2 fits the raw
  counts `y` with `μ` as a multiplicative offset, i.e. models `y / μ`.
  Nothing per entry is written, only the low-rank pieces `μ` is rebuilt
  from: `{OUT}/stage1.cells.parquet` (`n_i`, batch and `θ_i` per cell),
  `{OUT}/stage1.profiles.parquet` (`p_{k,b}(g)` per row) and
  `{OUT}/stage1.features.parquet` (per row the pooled share
  `p̄_g = Σ_i μ_gi / Σ_i n_i`, the negative-binomial dispersion `φ_g` by the
  method of moments, `E[(y − μ)²] = μ + φ μ²`, and the Pearson scale
  `σ²_gb`, the mean of `μ + φ_g μ²` over batch `b`'s cells). Fit the run's
  own data files with `--offset OUT` (lc, dsvd, cage, cage-cell): dsvd and
  cage divide each count by `d_gi = μ_gi / (n_i p̄_g)` where they would
  divide by a batch effect (their projections too: pinto divides, then
  projects); lc scores its edge profiles against the stage-1 expectation
  projected the same way, rates `D_kg / M_kg` with `M_kg` the community's
  summed offset in place of its volume (`--feature-network` is not taken
  with it); every such fit's written `feature_community` holds the same
  rates relative to the stage-1 expectation,
  `λ̂_gk = Σ_i π_ik y_gi / Σ_i π_ik d_gi` (the raw counts over their expected
  mass, in closed form through `θ`); and
  `--feature-inclusion moran` reads Pearson residuals
  `(y − μ) / σ_gb`, the zeros' share in closed form. Such a fit takes no
  batch effect: `--batch-files`, `--auto-batch` and `--cnv-clones` are
  refused, and its manifest records the folder as `offset`. `pinto lra` on it
  reads its own (raw) data and the stage-1 run's cell types.
- **`subtract`** (alias `additive`): per data file the excess
  `round(max(y − μ, 0))` (or the floor, `--excess floor`) on the rows every
  file shares as `{OUT}/{stem}.zarr.zip`, which a fit reads as it reads data;
  `pinto lra` given a fit on them reads the raw counts, not the excess.

Either way the run's coordinates of each file's cells go to
`{OUT}/{stem}.coords.parquet`, and `{OUT}/residual.json` records the
`operation`, the counts in (and out, under subtract), the stage-1 run and
the raw data files. Data is read as the run read it: exactly by name, or
with the `--feature-name-mode` the run was fitted with. Every file's name
must be distinct, and nothing at `OUT` is written over.

The run is taken by what its manifest records, not by the command that
wrote it, so a spatial baseline's run is taken as a fit's is. It must
record its data files (`data_files`), a propensity table
(`outputs.propensity`: cell × community, a soft posterior or one-hot
labels), a cells table (`outputs.cells`: each cell's coordinates and, with
more than one batch, its `batch`) and its coordinate columns
(`outputs.coord_columns`). A run that lacks one is refused, the missing
record named; nothing else of the run is read.

By default (`--adapt batch`) each community's profile is re-estimated per
batch as the run knew it (its cells table's `batch` column, written when the
run had more than one: its `--batch-files`, else its data files), shrunk
toward the pooled profile by a Dirichlet prior worth `--adapt-cells` cells
of the community (default 50; MacKay and Peto 1995). A shift that moves a whole community within a batch,
as donor, donor-by-type and section nuisance do, is absorbed into `μ`; a
local patch, a minority of its community's weight in that batch, stays in
the residual. No batch fold from the stage-1 run enters `μ`.
`--adapt none` keeps the pooled profiles. `residual.json` records `adapt`,
`adapt_cells`, `excess` (under subtract) and the formula.

**What it removes, and what it does not.** `residual` does not estimate a
donor effect or subtract one. With `--adapt batch` it absorbs only what
moves a whole stage-1 community within one batch; with `--adapt none` it
absorbs nothing beyond the pooled profiles, and donor nuisance stays in
the residual. Everything rests on one identifying assumption: donor-specific
biology is local. A whole community shifted in one donor is read as
nuisance, so a donor-specific difference that covers a whole community is
absorbed and never found.

**What it cannot do:**

- **Correct a stage-1 run that is split by donor.** If stage 1 already gave
  each donor its own communities, those communities carry the donor effect,
  and nothing downstream separates nuisance from biology. This happens once
  the donor effect is about as large as the cell-type effect: in
  `srt-simulate` terms, once `pve_donor + pve_inter` approaches `pve_type`.
  The fix is a donor-invariant stage 1, not a better residual.
- **Find a patch stage 1 already named.** A patch that stage 1 gave its own
  community is explained by that community's profile and leaves no excess;
  look for it among the stage-1 communities.
- **Find a donor-specific mix of shared cell states.** The shared profiles
  explain such a mix exactly, so it never appears in a count residual; it
  too can only be found among the stage-1 communities.

**Check before trusting a residual community confined to one donor:** look
at the stage-1 communities' donor make-up first. If they already follow the
donor, the residual's one-donor communities are nuisance. On a simulation,
`srt-sim-pairs` shows it directly: a stage-1 representation at chance on
`shared_domain_across_donors` pairs but not on `shared_domain_within_donor`
ones (a large `donor_gap`) is
split by donor.

```sh
pinto apply residual --from out/lc1 -o out/resid          # --residual divide
pinto fit lc S1.zarr,S2.zarr \
  -c out/resid/S1.coords.parquet,out/resid/S2.coords.parquet \
  --coord-column-names x,y --offset out/resid -o out/lc2

pinto apply residual --from out/lc1 --residual subtract -o out/sub
pinto fit lc out/sub/S1.zarr,out/sub/S2.zarr \
  -c out/sub/S1.coords.parquet,out/sub/S2.coords.parquet \
  --coord-column-names x,y -o out/lc2s   # the first run's coordinate names
```

Each coordinate file holds the run's first two coordinates under the run's
column names (with a `spot` column first), so the second fit reads them by
those names: `--coord-column-names` with the run's names, whatever columns
the first fit picked them from.

Agents run it as the MCP tool `run_residual`, whose reply lists the files
and flags for `run_fit` (see `mcp` below).

### Ablations, `compare` and held-out scoring

An ablation is a fit with one thing taken away or scrambled. Set it against
its parent with `pinto compare`. lc, cage and dsvd all take the flags; only
one ablation is allowed per run.

```sh
pinto fit lc data*.zarr.zip -c coords... --seed 42 -o base
pinto fit lc data*.zarr.zip -c coords... --seed 42 -o perm \
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
`pinto apply predict donor.zarr.zip -c donor.cells.parquet --model fit -o held_out`.
Each cell pair is split in two: its counts are thinned, or with
`--ablate-features` the listed features are hidden from the fit, as
`senna predict` does it. The pair's community mix is fitted on one part and
the other part is scored. The donor's batch shift δ comes from the fitting
parts only. `{out}.predictive.parquet` uses the column names cage and senna
use, with the training-composition null (`--null-from`), plus a pooled K=1
profile. The manifest's `holdout` block has the totals, the ceiling and a
per-section breakdown.

`pinto compare --paired A B` sets two models' held-out scores against each
other. A and B are `pinto apply predict` runs on the same held-out cells, each side a
comma list for leave-one-donor-out folds. Units are joined on their cell pair.
The donor is the replicate: `--donor-map` groups sections into donors. Writes
`{out}.paired.json` with each donor's difference in nats per count, exact sign
and Wilcoxon tests over donors, and a donor bootstrap interval.

### `run`

Set up fits in the terminal and run them. Pick data files (coordinate and
batch files beside them are paired by sample name), queue `lc`, `dsvd`, `cage`
and `cage-cell` (or the spatial baselines `spatialde2`, `stagate`,
`spatialpca` and `mefisto` and their `-pb` forms, listed after them), and change any flag; each method's form is read from its own
`--help`. ctrl+r shows the exact commands, checked as pinto would parse them,
and runs them in turn with their log on screen; each fit shows a progress bar,
plus a sparkline of its score (loss, NLL or moves) once it has two values and
the row has room. Tab or shift-enter moves to the next screen (shift-enter
only where the terminal can tell it apart from enter). Once a run ends, the
methods it finished are unqueued (marked ✓), so ctrl+r again, or `r` on the
Run screen, runs only the ones stopped, failed or not reached; space queues
one again. Each reruns into its old `--out`, after first clearing the script
and batch files its unfinished run left there (a residual's, the files it had
written in `{out}/`).

```sh
pinto run data/        # the file browser starts in data/
bash lc.cmd.sh         # each run is saved as {out}.cmd.sh to run again
```

It first asks for an output header, what every result of the run is named
after: `exp1` gives `exp1_lc`, `exp1_cage`, …, and `results/` gives
`results/lc`, … (the folder is made when the fit starts). Esc leaves each
`--out` as the method's name. `O` on the Methods screen changes the header
later, and `o` names one `--out` by hand. Anywhere in pinto, an `--out`
ending in `/` is a folder: `pinto fit lc … --out res/` writes `res/lc.*`, and the
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

Below them, `residual` sets up a two-stage fit on a fitted run (any that
recorded its propensity; see `residual` above). Its `--from` (`--run` also
works) names the run, and it writes the counts the run leaves unexplained
to `{out}/`; it is done once `{out}/residual.json` is there, and nothing it would write may be
there already. Queuing it queues the `stage-2` row too: under
`--residual divide` a fit on the run's own data files with `--offset` the
residual's folder, under subtract on each data file's `{stem}.zarr.zip`, in the
run's order with its `{stem}.coords.parquet`, and `--coord-column-names`
set to the run's coordinate columns; the data picked on the Data screen are not passed to
it. The stage-2 fit is dsvd unless `m` on its row switches it to lc or cage,
each keeping its own flags on the Parameters screen; space takes it off.
With lra queued on the same run, space on `stage-2 lra` queues lra on the
stage-2 fit as well, starting from that lra's flags: it reads the raw counts
of the run's data files through `residual.json`, so the two tables compare.
Each stage-2 job starts only once the one it reads is done; when that one
fails or stops, it is not started. A residual (or stage-2 fit) that
finished is read where it wrote, as it ran, so ctrl+r again runs only the
rest, on what it wrote; its row's `--out` moves on like a finished fit's.
A residual stopped or failed part-way reruns at its `--out`, its files
there cleared first. All of them are saved as
`{out}.cmd.sh` like the fits, and `c` in the review copies the commands in
the order they run.

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
pinto view out --png gene.png --gene SYMBOL     # one gene: observed log1p CP10k
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
coordinate columns; Space Ranger's image pixels read in µm when its
`scalefactors_json.json` sits beside the coordinates or in `spatial/`
(`--units um|px|none` to override). Map colours follow the
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
95% CI, q, coexpression q, the threshold its average weighs most, counts,
the effect minus the control with its two-sided q, and its warnings (else the verdict's note; a subset's q marked
subset-corrected), and the map draws it, computed as lra computes it:
senders (cells carrying the ligand), and receivers exposed or not (their
contact exposure to the ligand above the table's threshold) and
receptor-competent or not (their identity bin detects the receptor). `e`
draws the exposure itself instead; `c` puts the pair's threshold–response
curve (the effect at each exposure threshold, pointwise 95% whiskers,
simultaneous 95% band; the table's effect, averaged over thresholds, dashed
across it, and the threshold that average weighs most ringed) under the map. The
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

#### Layouts of the cells

`u` draws the cells laid out by a table of theirs in place of the tissue:
UMAP, then tUMAP, t-SNE, PHATE, the diffusion map, ForceAtlas2, LLE and the
stress layout, and after the last the tissue again. `U` steps through the
tables: each level's propensity, then the cell embedding when the run has
one; any `N × K` table of the cells will do. Every layout is legume-numeric's
coarsened layout: the cells' kNN graph is coarsened into at most 1,000
pseudobulks, the layout is run on the pseudobulk graph, and each cell is
placed from the pseudobulks near it in that graph, with `senna layout`'s
settings: a propensity is laid out by its square roots (Hellinger), an
embedding by each cell's direction (cosine). Colours, genes, edges and
the other layers draw on a layout as they draw on the tissue; cells the
table lacks are left out, and a layout has no scale bar. A layout is made on
a thread of its own, once per table and layout, and kept until the view
closes.

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
  and the groups' mean observed log1p CP10k, ln(1 + 1e4 · count / the cell's
  total count), chooses among them (the model's rates alone when there are
  no data files). Values are z-scored per gene and
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
attached (the fits' summary, an error) is said by the next `pinto session list`,
`attach`, `kill` or start. `--no-detach` (or `LEGUME_TUI_NO_DETACH=1`) runs
in this terminal only.

```sh
pinto session list             # the live ones, with what each is doing
pinto session attach fits      # back to one, by its name or id
pinto session rename run-1 fits
pinto session kill fits        # stops it as ctrl+c does: the fits running are stopped
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
`instructions`); `pinto mcp --help` prints them. Some clients do not pass those
to the model; with one of them, put that text in the agent's own instructions
(an `AGENTS.md`, a system prompt).

The tools mirror the command line: a field passing a flag is described by the
flag's short help, and a flag hidden from `--help` is in no tool's schema nor in
`run_methods` (`run_fit` still takes it by name in `flags`).

| Tool | Does |
|---|---|
| `run_methods` | pinto's fitting methods (what `pinto run` sets up: `lc`, `dsvd`, `cage`, `cage-cell`, then the spatial baselines `spatialde2`, `stagate`, `spatialpca`, `mefisto`, `spatialde2-pb`, `stagate-pb`, `spatialpca-pb`, `mefisto-pb`); with a method, the flags its `--help` shows: name, short help, default, choices (hidden flags are in no tool schema) |
| `run_fit` | fits one method on data files with their coordinates (and batches), with the command `pinto run` would build, checked as pinto parses it and run as a child `pinto`: a background job by default (`background: false` waits), never over an existing run; `flags` takes the flags `run_methods` lists by name |
| `run_predict` | `pinto apply predict` as a recorded job: on an lc model, held-out pairs scored by count thinning or `ablate-features`, beside the training null, a pooled K=1 profile and the ceiling; returns the manifest's `holdout` block when run to its end |
| `run_region` | the cells of a run inside a polygon (world coordinates as `view_map` shows them, or a GeoJSON file in section coordinates) written as `{out}.cells.txt` for `run_fit`'s `mask-cells` |
| `run_compare` | `pinto compare` on two runs (prefix or `.pinto.json`): ARI, NMI, a Hungarian community matching with each pair's Jaccard, and A's communities that vanish, split or merge in B (partners hold ≥ `tau` of a community's cells); writes `{out}.compare.json` and `{out}.contingency.parquet`. With `paired: true`, two models' `run_predict` scores on the same held-out cells, tested over donors (`{out}.paired.json`) |
| `run_lra` | `pinto lra --from <run>` as a recorded job, or `pinto lra` on `data_files` with `coord_files` (`batch_files`, `knn_spatial`): `method` (contact-effect, coexpression-test, coexpression-scores), `out`, `lr_pairs`, `targets`, `species`, `seed`, `only`, and for the co-expression methods `n_permutations` and `min_gene_count`, and for the contact effect `marginal_baselines` (`--marginal-baselines`, also writing `{out}.lra.celltype.parquet`, slot `lra.celltype`) and `marginal_permutations` (only with `marginal_baselines`); writes `{out}.lra.parquet` and `{out}.lra.curve.parquet` (slots `lra.table` and `lra.curve` in the run's manifest; with `only`, `{out}.lra.only-<tag>.parquet` under `lra.subsets`, its q corrected within the subset) and replies with the verdict counts, the pairs whose effect survived the controls (percent change with its 95% CI, q) and, when the tissue has no power check yet, `run_lra_simulate` as the next call (with the same `lr_pairs`, `targets` and `species`); called again on a finished `out`, it reads the tables back, and refuses a call whose options differ from those the manifest records for them |
| `run_lra_simulate` | `pinto lra simulate --from <run>` as a recorded job, or on `data_files` with `coord_files` (`batch_files`, `knn_spatial`): `out`, `species`, `seed`; writes `{out}.lra-simulate.parquet` (slot `lra.simulate`) and replies with each method's type I error and power per scenario, the estimable shares and the means over the confounded nulls and planted effects. `one`, `set`, `pairs` and `null_pairs` run one scenario instead (`--one`, `--set`, `--pairs`, `--null-pairs`): the table with its `planted` label and p columns, then `{out}.lra-simulate.json`, nothing recorded in the manifest |
| `run_residual` | `pinto apply residual` as a recorded job: `run` (a fit's or a spatial baseline's: any run that recorded its data files, propensity, cells table and coordinate columns), `out`, `data_files`, `residual` (`divide`, the default, or `subtract`), `excess`, `adapt`, `adapt_cells`, `feature_name_mode`; the run is checked before it starts (a run that lacks one of those records is refused) and nothing is written over; writes `{out}/{stem}.coords.parquet` per data file and `{out}/residual.json`, under divide `{out}/stage1.{cells,profiles,features}.parquet`, under subtract `{out}/{stem}.zarr.zip` per data file; the reply lists stage 2 for `run_fit` as `fit.data` (the run's own files under divide, the excess files under subtract) with `fit.coords`, and as `fit.flags` the coordinate columns and, under divide, `offset`; `run_lra` on that second fit reads the run's raw counts |
| `run_simulate` | `pinto sim new` as a recorded job: `scenario`, `set` (`key=value`, any parameter), `coords` with `coord_column_names`, `out`; the settings and coordinate files are checked before it starts, an existing `out` is refused, and the reply names the truth run `<out>/truth/truth.pinto.json` for `view_open` and `run_sim_score_by_set`, and lists the sections to fit as `fit.data` with `fit.coords` and the flags to read them as `fit.flags` ; it also gives the run's variance `shares` and a `shares_note` that spells them out (noise as the remainder, any share set but off) for the agent to relay |
| `run_sim_score_by_set` | `pinto sim score-by-set` in the call: `truth` (the simulation's truth run or folder), `labels` (runs or label tables), `level`, `label_column`; replies with the two tests as a table, one row per labelling and donor-specific domain, and as JSON; a labelling that cannot be scored is listed under `failed` and the rest are scored |
| `run_sim_score_by_pair` | `pinto sim score-by-pair` in the call: `truth`, `vectors` (runs at `level`, or vector tables such as a cage run's `cell_embedding`), `pairs`, `seed`, `knn`; replies with each set's AUROC as a table, one row per input and donor-specific domain, and every AUROC, AUPRC and prevalence as JSON; an input that cannot be scored is listed under `failed` and the rest are scored |
| `run_sim_baseline` | `pinto sim baseline` as a recorded job: `truth`, `method` (`pooled`, `harmony` or `per-donor`), `n_clusters`, `seed`, `out`; writes `<out>.tsv`, a cell/label table the reply names as `labels` for `run_sim_score_by_set` |
| `view_open` | opens a run; its cells, bounds, batches, levels, groups with their map colours, data files and lupin rounds |
| `view_map` | draws the map as a PNG with its state and legend; level, layer, focus, gene (source, clip), round and `show`, theme, edges, point size, box / fit / batch / center / zoom, width; settings stay until changed, and a call that fails changes nothing |
| `view_markers` | a group's top features, by fold over the other groups |
| `view_cell` | the cell under a pixel of the last map, at a world point, or by name: batch, group, strongest propensities, entropy |
| `view_lra` | the run's lra tables (the full one and `--only` subsets, `table`) and their pairs: verdict counts and a list filtered by `verdict` and `search` (genes), `sort`ed by q, effect or name, each with its percent change and q; `pair` returns that pair's row and makes `view_map` draw it (senders, receivers exposed or not and receptor-competent or not; `show: exposure` the exposure itself) until a group layer, gene or `pair: ""` is asked for; settings stay until changed |
| `view_export` | writes the view as a PNG and/or a PDF figure (only `.png`/`.pdf`, replacing one only with `overwrite`), by running the `pinto view` command it returns; with `curve: true`, the selected lra pair's threshold-response curve |
| `notebook` | (from legume-mcp) an entry in the day's lab notebook |
| `job_status`, `job_wait`, `job_cancel` | (from legume-mcp) follow a `run_fit` (or `run_predict`, `run_lra`, `run_lra_simulate`, `run_residual`, `run_simulate`, `run_sim_baseline`) in the background, or stop it |

Focus takes a level's communities (`C3` or `3`), and when a round is shown,
lupin's cluster ids (`K3` or `3`) or cell types.

#### Records

The server keeps the project's records, laid out after Noble's guide to
organizing computational biology projects [1] (see
[legume-mcp](https://crates.io/crates/legume-mcp)):

- `results/<YYYY-MM-DD>/runall.sh`: every step that wrote something, in order,
  with paths relative to the project. Each `run_fit` is its `pinto fit lc` (or
  `cage`, `dsvd`, a baseline's) command, each `run_lra` its `pinto lra` command, and each `view_export`
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
  (`data-beans from-mtx … --backend hdf5 -o data.h5`). Convert an AnnData
  `.h5ad` first with `data-beans from-h5ad` and pass the result: pinto no
  longer converts one on the fly, though a `{base}.db.zarr` an earlier
  version cached beside it is still used, so delete a stale cache.
- **Coordinates:** CSV/TSV/parquet (Visium / Xenium column names recognized).
- **Batch (optional):** `-b labels.txt`.

## License

MIT
