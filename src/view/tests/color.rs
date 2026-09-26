use crate::view::color::*;

#[test]
fn palette_colours_are_distinct() {
    let p = palette(60);
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
