use super::draw::error_lines;
use super::*;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::text::Line;

#[test]
fn an_error_popup_wraps_its_message_to_fit() {
    let msg = format!(
        "lupin annotate: {} is unreadable\nsecond line",
        "x/".repeat(60)
    );
    let lines: Vec<String> = error_lines(&msg, 40)
        .iter()
        .map(ToString::to_string)
        .collect();
    assert!(lines.iter().all(|l| l.chars().count() <= 38), "{lines:?}");
    let text: String = lines.iter().map(|l| l.trim_start()).collect();
    assert!(text.contains(&"x/".repeat(60)), "{lines:?}");
    assert!(lines.iter().any(|l| l.contains("second line")));
    assert!(lines.last().unwrap().contains("closes"));
}

/// The lra layer on screen: `I` opens it, the pair list fills the panel,
/// a pair picked is drawn in its classes and its curve goes under the
/// map, and the list filters by verdict and genes.
#[test]
fn the_lra_layer_lists_pairs_and_draws_the_one_picked() {
    use super::super::tests::lra::{lra_run, receivers, WARNING};
    use clap::Parser;
    #[derive(Parser)]
    struct Args {
        #[command(flatten)]
        view: ViewArgs,
    }
    let dir = tempfile::tempdir().unwrap();
    let (prefix, contacts) = lra_run(dir.path());
    let args = Args::parse_from(["view", prefix.as_str()]).view;
    let base = Base::load(&prefix, "auto", Theme::Dark).unwrap();
    let level = base.level(0, false).unwrap();
    let session = legume_tui::Session::plain(&crate::tui::TOOL);
    let gfx = (Gfx::Cells(Glyphs::HalfBlocks, None), (8., 16.));
    let mut app = App::new(&base, level, &args, gfx, &session);
    app.map = Rect::new(0, 0, 80, 48);
    app.canvas = app.map;
    app.side = Rect::new(80, 0, WIDE_PANEL, 48);
    let key = |c: KeyCode| KeyEvent::new(c, KeyModifiers::NONE);
    let text =
        |app: &App| -> Vec<String> { app.panel(48).0.iter().map(ToString::to_string).collect() };

    app.toggle_lra(None).unwrap();
    assert!(app.lra_on && app.wide());
    let lines = text(&app);
    let all = lines.join("\n");
    assert!(all.contains("pairs full table · 3"), "{all}");
    assert!(all.contains("T 1/2"), "{all}");
    assert!(all.contains("L1:R1") && all.contains("supported"), "{all}");
    assert!(all.contains("I or Esc leave"), "{all}");
    // The list is clickable, a row per pair.
    let (_, picks) = app.panel(48);
    let pairs = picks
        .iter()
        .filter(|(_, p)| matches!(p, Pick::Pair(_)))
        .count();
    assert_eq!(pairs, 3);
    // Its columns line up whatever the verdict: each verdict starts its
    // column's width from the row's end, after a space.
    use super::lra::VERDICT_W;
    let (lines, _) = app.panel(48);
    for (at, _) in picks.iter().filter(|(_, p)| matches!(p, Pick::Pair(_))) {
        let row: Vec<char> = lines[*at].to_string().chars().collect();
        let start = row.len() - VERDICT_W;
        assert!(
            row[start - 1] == ' ' && row[start].is_alphabetic(),
            "{row:?}"
        );
    }
    assert!(app.lra_drawn().is_none(), "nothing picked yet");

    // ↓ picks the first pair, by q; it is drawn in its classes.
    app.lra.as_mut().unwrap().receivers = Some(receivers(&base, contacts));
    assert!(app.lra_key(key(KeyCode::Down), None).unwrap());
    // Drawn with the map, not by the key.
    assert!(app.lra_drawn().is_none());
    app.draw_pending_pair(None);
    assert!(app.error.is_none(), "{:?}", app.error);
    let all = text(&app).join("\n");
    assert!(
        all.contains("▸ L1:R1") || all.contains("▸L1:R1") || all.contains("▸ L1"),
        "{all}"
    );
    assert!(all.contains("L1 → R1  supported"), "{all}");
    assert!(all.contains("+25% (95% CI +20% to +30%)"), "{all}");
    assert!(all.contains("sender (ligand+)"), "{all}");
    // Every line fits the panel, its words whole: the pair's reasons read
    // in full, and which threshold the average leans on.
    let room = usize::from(app.side.width) - 1;
    for l in text(&app) {
        assert!(l.chars().count() <= room, "{l:?} is wider than {room}");
    }
    let flat = all.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(flat.contains(WARNING), "{all}");
    assert!(flat.contains("weighs τ 2 most (80%)"), "{all}");
    // The effect against the control, signed: a small q can mean it
    // falls short of it, as here of the receptor-negative receivers.
    assert!(
        flat.contains("effect − control: receptor− -4.9% q 0.010") && !flat.contains("background"),
        "{all}"
    );
    let (level, style) = app.drawn();
    assert_eq!(level.comm.tag, "L1:R1");
    let vp = app.viewport();
    let frame = base.render(level, &style, app.drawn_gene(), &vp);
    let sender = super::super::lra::class_palette(Theme::Dark)[0];
    assert!(
        frame.rgba.chunks(4).any(|p| p[..3] == sender),
        "senders are drawn in their colour"
    );

    // e: the exposure; c: the curve under the map, taller frame.
    app.lra_key(key(KeyCode::Char('e')), None).unwrap();
    assert!(app.drawn_gene().is_some());
    assert!(text(&app).join("\n").contains("L1 contact exposure"));
    app.lra_key(key(KeyCode::Char('c')), None).unwrap();
    assert!(app.curve_shows());
    let frame = base.render(level_of(&app), &app.drawn().1, app.drawn_gene(), &vp);
    let h = frame.h;
    app.canvas = Rect::new(0, 0, 80, 64);
    let tall = app.with_curve(frame);
    assert!(tall.h > h && tall.rgba.len() == tall.w * tall.h * 4);

    // v: one verdict at a time; / genes, typed and kept.
    app.lra_key(key(KeyCode::Char('v')), None).unwrap();
    let all = text(&app).join("\n");
    assert!(all.contains("verdict supported (1)"), "{all}");
    app.lra_key(key(KeyCode::Char('v')), None).unwrap();
    app.lra_key(key(KeyCode::Char('V')), None).unwrap();
    app.lra_key(key(KeyCode::Char('V')), None).unwrap();
    assert_eq!(app.lra.as_ref().unwrap().filter().verdict, None);
    app.lra_key(key(KeyCode::Char('/')), None).unwrap();
    for c in "g0".chars() {
        app.lra_key(key(KeyCode::Char(c)), None).unwrap();
    }
    app.lra_key(key(KeyCode::Enter), None).unwrap();
    assert_eq!(app.lra.as_ref().unwrap().visible().len(), 1);
    // space marks the pair under the cursor for `pinto lra --only`.
    app.lra_key(key(KeyCode::Down), None).unwrap();
    app.lra_key(key(KeyCode::Char(' ')), None).unwrap();
    assert!(app.lra.as_ref().unwrap().marked.contains("L1:G0"));
    assert!(text(&app).join("\n").contains("*L1:G0"));

    // T: the subset table, subset-corrected.
    app.lra_key(key(KeyCode::Char('/')), None).unwrap();
    app.lra_key(key(KeyCode::Esc), None).unwrap();
    app.lra_key(key(KeyCode::Char('T')), None).unwrap();
    let all = text(&app).join("\n");
    assert!(all.contains("only L1:R1"), "{all}");
    assert!(all.contains("subset-corrected"), "{all}");
    // The export's redraw command names the pair and its table.
    app.lra.as_mut().unwrap().select("L1:R1").unwrap();
    app.draw_pair(None);
    let still = app.still_pair(false).expect("a pair is drawn");
    assert_eq!(still.name, "L1:R1");
    assert!(still.table.is_some());

    // I leaves the layer: the groups are drawn again.
    app.lra_key(key(KeyCode::Char('I')), None).unwrap();
    assert!(!app.lra_on && app.lra_drawn().is_none() && !app.curve_shows());
}

/// The layer's keys at the edges: Ctrl and Alt chords are the viewer's
/// (Ctrl+C quits), space marks successive pairs from none selected, `T`
/// draws the pair it keeps, and the curve is saved at a still's size
/// whatever room the screen gives it.
#[test]
fn the_lra_layers_keys_mark_in_turn_and_leave_chords_alone() {
    use super::super::tests::lra::{lra_run, receivers};
    use clap::Parser;
    #[derive(Parser)]
    struct Args {
        #[command(flatten)]
        view: ViewArgs,
    }
    let dir = tempfile::tempdir().unwrap();
    let (prefix, contacts) = lra_run(dir.path());
    let args = Args::parse_from(["view", prefix.as_str()]).view;
    let base = Base::load(&prefix, "auto", Theme::Dark).unwrap();
    let level = base.level(0, false).unwrap();
    let session = legume_tui::Session::plain(&crate::tui::TOOL);
    let gfx = (Gfx::Cells(Glyphs::HalfBlocks, None), (8., 16.));
    let mut app = App::new(&base, level, &args, gfx, &session);
    app.map = Rect::new(0, 0, 80, 12);
    app.canvas = app.map;
    app.side = Rect::new(80, 0, WIDE_PANEL, 48);
    let key = |c: KeyCode| KeyEvent::new(c, KeyModifiers::NONE);
    app.toggle_lra(None).unwrap();
    app.lra.as_mut().unwrap().receivers = Some(receivers(&base, contacts));

    // Ctrl+C is not `c`, Ctrl+L not `L`: the viewer's to handle.
    for c in ['c', 'l', 'L'] {
        let chord = KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL);
        assert!(!app.lra_key(chord, None).unwrap(), "ctrl+{c}");
    }
    let alt = KeyEvent::new(KeyCode::Char('e'), KeyModifiers::ALT);
    assert!(!app.lra_key(alt, None).unwrap());
    assert!(!app.lra_curve && app.lra_job.is_none());

    // Space from no pair selected marks the first, then the second.
    let v = app.lra.as_ref().unwrap();
    assert!(v.selected.is_none());
    let order: Vec<String> = v.visible().iter().map(|&i| v.rows()[i].name()).collect();
    app.lra_key(key(KeyCode::Char(' ')), None).unwrap();
    app.lra_key(key(KeyCode::Char(' ')), None).unwrap();
    let marked: Vec<&String> = app.lra.as_ref().unwrap().marked.iter().collect();
    let mut want = vec![&order[0], &order[1]];
    want.sort();
    assert_eq!(marked, want);

    // M marks all the filter shows; again, it unmarks them.
    app.lra_key(key(KeyCode::Char('M')), None).unwrap();
    let v = app.lra.as_ref().unwrap();
    assert_eq!(v.marked.len(), v.visible().len());
    let lines = app.lra_lines(40);
    let said = |l: &Line| {
        l.spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<String>()
    };
    assert!(lines
        .iter()
        .any(|(l, _)| said(l).contains("marked  L runs them")));
    app.lra_key(key(KeyCode::Char('M')), None).unwrap();
    assert!(app.lra.as_ref().unwrap().marked.is_empty());
    // With none marked, L runs the pair under the cursor.
    app.lra.as_mut().unwrap().select("L1:R1").unwrap();
    assert_eq!(app.pairs_to_run(), vec!["L1:R1".to_string()]);
    app.lra.as_mut().unwrap().marked.insert("L1:G0".into());
    assert_eq!(app.pairs_to_run(), vec!["L1:G0".to_string()]);
    app.lra.as_mut().unwrap().marked.clear();

    // T keeps the pair selected and queues it to be drawn.
    app.lra.as_mut().unwrap().select("L1:R1").unwrap();
    app.draw_pair(None);
    app.pair_pending = false;
    app.lra_key(key(KeyCode::Char('T')), None).unwrap();
    assert_eq!(app.lra.as_ref().unwrap().at, 1);
    assert!(app.pair_pending, "the pair is drawn in the new table");
    app.draw_pending_pair(None);
    assert!(app.lra_drawn().is_some());

    // The curve's export, with no room under a 12-row map.
    app.lra_key(key(KeyCode::Char('c')), None).unwrap();
    assert!(app.curve_shows() && app.bars.height == 0);
    app.lra.as_mut().unwrap().load(0).unwrap();
    let stem = dir.path().join("fig-curve");
    app.export_curve(&stem.to_string_lossy()).unwrap();
    // The IHDR's width and height, big-endian after the signature.
    let png = std::fs::read(format!("{}.png", stem.display())).unwrap();
    let be = |at: usize| u32::from_be_bytes(png[at..at + 4].try_into().unwrap());
    let w = (super::super::curve::STILL_WIDTH * args.export_scale.max(1)) as u32;
    assert_eq!((be(16), be(20)), (w, w * 3 / 5));
}

fn level_of<'a>(app: &'a App) -> &'a Level {
    app.drawn().0
}
