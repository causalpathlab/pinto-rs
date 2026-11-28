//! The tools `pinto mcp` lists, with their JSON schemas.

use super::MAX_SIDE;
use legume_mcp::ToolSpec;
use serde_json::json;

pub(super) fn specs() -> Vec<ToolSpec> {
    let spec = ToolSpec::new;
    let group = json!({
        "type": ["string", "integer"],
        "description": "C3 or 3 for a community, or its name when the level names them; a lupin cluster id (K3 or 3) or a cell type when a round is shown",
    });
    vec![
        spec(
            "view_open",
            "Open a pinto run and summarize it: cells, coordinates, bounds, batches, \
             levels, the level's groups with their map colours, data files and lupin rounds. \
             Opening another run closes the one open.",
            json!({
                "type": "object",
                "properties": {
                    "prefix": {"type": "string", "description": "Output prefix of the run (the -o it was fit with) or its .pinto.json"},
                    "level": {"type": "string", "description": "Level tag to show (L1, L2, …, final) [default: the last]"},
                },
                "required": ["prefix"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_map",
            "Draw the map as a PNG, with the state and legend as JSON. Every argument is \
             optional and stays set until changed; with none, the same view is drawn again.",
            json!({
                "type": "object",
                "properties": {
                    "level": {"type": "string", "description": "Level tag; shows its communities"},
                    "round": {"type": "string", "description": "A lupin round ({out}.lupin.json); shows its clusters unless show says otherwise"},
                    "show": {"enum": ["communities", "types", "clusters"], "description": "What groups the cells; types and clusters are the round's (the newest round when none is given)"},
                    "layer": {
                        "anyOf": [
                            {"enum": ["argmax", "soft", "entropy"]},
                            {"type": "string", "pattern": "^[Cc]?[0-9]+$"},
                        ],
                        "description": "argmax (most likely group), soft (colours mixed by propensity), entropy (propensity entropy / ln K), or C<k> (one community's propensity, e.g. C3)",
                    },
                    "focus": {"type": "array", "items": group, "description": "Groups to show; the rest are dimmed. [] shows every group"},
                    "gene": {"type": "string", "description": "Colour cells by one feature, by symbol or full name; \"\" shows the groups again"},
                    "source": {"enum": ["observed", "expected"], "description": "The gene's observed log1p CP10k (ln(1 + 1e4 * count / the cell's total count)), or the model's expected level"},
                    "clip": {"type": "number", "minimum": 50, "maximum": 100, "description": "Top of the gene's colour ramp, a percentile of its positive values [default: 99]"},
                    "edges": {"type": "boolean", "description": "Draw adjacent cell pairs, where zoomed in far enough: by link community, or, on a round's types or clusters, a link within one group (isotypic) in its colour and one between two (heterotypic) in the page's text colour, on top"},
                    "edges_shown": {"enum": ["all", "heterotypic", "isotypic"], "description": "On a round's types or clusters, which links edges draws [default: all]; the reply's links counts both kinds on the tissue and in the window"},
                    "heterotypic_colour": {"type": "string", "description": "Heterotypic links' colour as #rrggbb (e.g. #d7191c); \"\" for the page's text colour"},
                    "edge_width": {"type": "number", "minimum": 0.25, "maximum": 8, "description": "Edge line width, × the default [default: 1]"},
                    "point_size": {"type": "number", "minimum": 0.25, "maximum": 8, "description": "Cell disc size, × the default"},
                    "point_alpha": {"type": "number", "minimum": 0, "maximum": 1, "description": "Cell opacity, 0 to 1: below 1 cells fade towards the page and edges under them show [default: 1]"},
                    "theme": {"enum": ["dark", "light"], "description": "Map colours for a dark or light background [default: dark]"},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "World window [x0, y0, x1, y1]"},
                    "fit": {"type": "boolean", "description": "true: the whole tissue"},
                    "batch": {"type": "string", "description": "Fit one batch, by the name view_open lists"},
                    "center": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "Move the window's centre to this world point"},
                    "zoom": {"type": "number", "exclusiveMinimum": 0, "description": "Magnify the window about its centre: 2 zooms in, 0.5 out"},
                    "width": {"type": "integer", "minimum": 1, "maximum": MAX_SIDE, "description": "Image width in pixels, this call only [default: 1024]"},
                    "curve": {"type": "boolean", "description": "Also return the selected lra pair's threshold-response curve (view_lra pair) as a second image, as the terminal draws it under the map; this call only"},
                },
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_markers",
            "A group's top marker features, by fold over the other groups.",
            json!({
                "type": "object",
                "properties": {
                    "group": group,
                    "n": {"type": "integer", "minimum": 1, "description": "How many [default: 12]"},
                },
                "required": ["group"],
                "additionalProperties": false,
            }),
        )
        .read_only(),
        spec(
            "view_cell",
            "The cell nearest a pixel of the last map, a world point, or by name: \
             its batch, position, group, strongest propensities and entropy.",
            json!({
                "type": "object",
                "properties": {
                    "px": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[column, row] pixel of the last view_map image"},
                    "at": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[x, y] world point"},
                    "name": {"type": "string", "description": "Cell name"},
                },
                "additionalProperties": false,
            }),
        )
        .read_only(),
        spec(
            "view_lra",
            "The run's ligand-receptor tables (pinto lra: the full table and any --only \
             subsets) and their pooled pairs: how many have an estimate, and a list searched by \
             gene, sorted by p, effect or name, each with its change (percent, exp(effect) - 1, \
             with 95% CI) and p. With pair, returns that pair's row (effect, SE, p, the effect \
             minus the receptor-negative control with its p, co-expression's local fdr, \
             warnings, threshold, receivers) and view_map draws it from then on: senders (ligand+), \
             receivers exposed or not (contact exposure above the table's threshold) and \
             receptor-competent or not, as lra computed them (show: exposure draws the \
             exposure itself). Arguments stay set until changed. lra judges nothing. A subset's \
             co-expression lfdr is fitted within the subset only (q_scope). view_export curve: true writes the pair's \
             threshold-response curve. mark, unmark, mark_shown and clear_marks keep a set of \
             marked pairs (listed as marked); start runs pinto lra --only on them (or on the \
             pair selected when none is marked) with the shown table's options, as a background \
             job (job_status follows it; view_lra table then opens the subset), as space, M \
             and ctrl+r do in the terminal.",
            json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "full, a subset by its --only list (L1,L2), or a table's path [default: the full table]"},
                    "search": {"type": "string", "description": "Genes: each word must be in the ligand's or receptor's name; \"\" for all"},
                    "sort": {"enum": ["p", "effect", "name"], "description": "Order of the list [default: p]"},
                    "limit": {"type": "integer", "minimum": 0, "description": "Pairs listed, this call only [default: 20]"},
                    "pair": {"type": "string", "description": "LIGAND:RECEPTOR to select and draw on the map; \"\" draws groups again"},
                    "show": {"enum": ["classes", "exposure"], "description": "What the pair's map shows [default: classes]"},
                    "mark": {"type": "array", "items": {"type": "string"}, "description": "LIGAND:RECEPTOR pairs to mark, in the table shown"},
                    "unmark": {"type": "array", "items": {"type": "string"}, "description": "LIGAND:RECEPTOR pairs to unmark"},
                    "mark_shown": {"type": "boolean", "description": "Mark every pair the list shows (after search)"},
                    "clear_marks": {"type": "boolean", "description": "Unmark every pair first"},
                    "start": {"type": "boolean", "description": "Start pinto lra --only on the marked pairs (else the selected one) as a background job"},
                },
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_export",
            "Write the current view as a PNG and/or a one-page PDF figure (title, scale \
             bar, legend, focused groups' markers); with curve, the selected lra pair's \
             threshold-response curve instead. Refuses to replace a file unless overwrite.",
            json!({
                "type": "object",
                "properties": {
                    "png": {"type": "string", "description": "PNG path, ending .png"},
                    "pdf": {"type": "string", "description": "PDF path, ending .pdf"},
                    "width": {"type": "integer", "minimum": 1, "maximum": MAX_SIDE, "description": "Image width in pixels [default: 2400]"},
                    "overwrite": {"type": "boolean", "description": "Replace files that exist"},
                    "curve": {"type": "boolean", "description": "Write the selected lra pair's threshold-response curve (view_lra pair) instead of the map"},
                },
                "additionalProperties": false,
            }),
        ),
    ]
}
