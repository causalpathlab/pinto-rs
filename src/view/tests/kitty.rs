use crate::view::kitty::*;
use crate::view::render::Frame;
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;

fn frame(w: usize, h: usize) -> Frame {
    Frame {
        w,
        h,
        rgba: vec![7; w * h * 4],
        background: crate::view::color::Theme::Dark.background(),
    }
}

fn decode_inline(bytes: &[u8]) -> (String, Vec<u8>) {
    // First escape's keys, and every chunk's payload concatenated.
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let mut keys = String::new();
    let mut payload = String::new();
    for part in text.split("\x1b_G").skip(1) {
        let body = part.split("\x1b\\").next().unwrap();
        let (k, p) = body.split_once(';').unwrap_or((body, ""));
        if keys.is_empty() {
            keys = k.to_string();
        }
        if !k.starts_with("a=d") {
            payload.push_str(p);
        }
    }
    (keys, B64.decode(payload).unwrap())
}

#[test]
fn direct_frames_round_trip_through_zlib_chunks() {
    let f = frame(300, 200);
    let mut out = Vec::new();
    let mut k = Kitty::new(Transport::Direct);
    k.show(&mut out, &f, (2, 3), (40, 20)).unwrap();

    let (keys, z) = decode_inline(&out);
    assert!(keys.contains("s=300,v=200"));
    assert!(keys.contains("c=40,r=20"));
    assert!(keys.contains("o=z") && keys.contains("q=2"));
    let mut raw = Vec::new();
    std::io::Read::read_to_end(&mut flate2::read::ZlibDecoder::new(&z[..]), &mut raw).unwrap();
    assert_eq!(raw, f.rgba);
}

#[test]
fn frames_alternate_ids_and_delete_the_previous_one() {
    let f = frame(4, 4);
    let mut k = Kitty::new(Transport::Direct);
    let mut first = Vec::new();
    k.show(&mut first, &f, (0, 0), (1, 1)).unwrap();
    let mut second = Vec::new();
    k.show(&mut second, &f, (0, 0), (1, 1)).unwrap();

    let first = String::from_utf8(first).unwrap();
    let second = String::from_utf8(second).unwrap();
    assert!(first.contains(&format!("i={}", IDS[0])) && !first.contains("a=d"));
    assert!(second.contains(&format!("i={}", IDS[1])));
    assert!(second.contains(&format!("a=d,d=I,i={}", IDS[0])));
}

#[test]
fn inside_tmux_commands_are_wrapped_with_escapes_doubled() {
    let f = frame(2, 2);
    let mut out = Vec::new();
    let mut k = Kitty::new(Transport::Direct).through_tmux(true);
    k.show(&mut out, &f, (0, 0), (1, 1)).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("\x1bPtmux;\x1b\x1b_G"));
    assert!(text.contains("\x1b\x1b\\\x1b\\"));
    // No graphics command escapes tmux unwrapped.
    assert!(!text.replace("\x1b\x1b_G", "").contains("\x1b_G"));
}
