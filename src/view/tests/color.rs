use crate::view::color::*;

#[test]
fn palette_colours_are_distinct() {
    let p = Theme::Dark.palette(60);
    for i in 0..p.len() {
        for j in i + 1..p.len() {
            assert_ne!(p[i], p[j], "{i} and {j}");
        }
    }
}

#[test]
fn srgb_round_trips() {
    for v in 0..=255u8 {
        assert_eq!(encode(decode(v)), v);
    }
}

#[test]
fn table_encoder_matches_the_formula_within_one_step() {
    for i in 0..=1000 {
        let c = i as f32 / 1000.;
        assert!((encode_fast(c) as i32 - encode(c) as i32).abs() <= 1, "{c}");
    }
}

#[test]
fn ramp_ends_are_the_first_and_last_stops() {
    let v = viridis();
    assert_eq!(v.at(0.), [0x44, 0x01, 0x54]);
    assert_eq!(v.at(1.), [0xfd, 0xe7, 0x25]);
}

#[test]
fn both_themes_give_distinct_palettes_and_read_their_background() {
    for theme in [Theme::Dark, Theme::Light] {
        let p = theme.palette(40);
        for i in 0..p.len() {
            for j in i + 1..p.len() {
                assert_ne!(p[i], p[j], "{theme:?}: {i} and {j}");
            }
        }
        assert_eq!(Theme::for_background(theme.background()), theme);
    }
    assert_eq!(Theme::for_background([255, 255, 255]), Theme::Light);
    assert_eq!(Theme::for_background([30, 30, 30]), Theme::Dark);
}

#[test]
fn ramps_start_on_the_background_side() {
    // Low values fade toward the background: dark ramps start dark on a
    // dark theme and light on a light one.
    let lum = |c: [u8; 3]| c.iter().map(|&v| v as u32).sum::<u32>();
    for (theme, starts_light) in [(Theme::Dark, false), (Theme::Light, true)] {
        for ramp in [theme.magma(), theme.expression()] {
            let (low, high) = (lum(ramp.at(0.)), lum(ramp.at(1.)));
            assert_eq!(low > high, starts_light, "{theme:?}");
        }
    }
}
