use crate::view::color::*;

#[test]
fn a_theme_reads_its_background_and_the_key_flips_it() {
    for theme in [Theme::Dark, Theme::Light] {
        assert_eq!(Theme::for_background(theme.background()), theme);
    }
    assert_eq!(Theme::for_background([255, 255, 255]), Theme::Light);
    assert_eq!(Theme::for_background([30, 30, 30]), Theme::Dark);
    assert_eq!(Theme::Dark.other(), Theme::Light);
    assert_eq!(Theme::Light.other(), Theme::Dark);
    // A light map, for paper, is drawn on a light background.
    assert_eq!(
        Theme::for_background(Theme::Dark.other().background()),
        Theme::Light
    );
}

/// pinto's maps keep the colours they had before they were drawn with
/// legume-tui's: a few of each theme's, and of the ramps and the scale.
/// legume-tui pins these too; this is pinto's guard that a release of it
/// leaves pinto's maps as they were.
#[test]
fn the_colours_are_pinto_s_own() {
    type Pinned = ([u8; 3], [u8; 3], [u8; 3], [[u8; 3]; 3], [u8; 3], [u8; 3]);
    let pinned: [(Theme, Pinned); 2] = [
        (
            Theme::Dark,
            (
                [17, 17, 20],
                [40, 40, 46],
                [70, 70, 76],
                [[235, 129, 130], [35, 151, 97], [200, 183, 255]],
                [182, 54, 122],
                [182, 55, 122],
            ),
        ),
        (
            Theme::Light,
            (
                [238, 238, 236],
                [212, 212, 216],
                [175, 175, 181],
                [[195, 78, 84], [0, 113, 57], [159, 133, 229]],
                [204, 67, 111],
                [228, 80, 100],
            ),
        ),
    ];
    for (t, (bg, dimmed, none, palette, magma, levels)) in pinned {
        assert_eq!(t.background(), bg, "{t:?}");
        assert_eq!(t.dimmed(), dimmed, "{t:?}");
        assert_eq!(t.no_community(), none, "{t:?}");
        assert_eq!(t.palette(3), palette, "{t:?}");
        assert_eq!(t.magma().at(0.5), magma, "{t:?}");
        assert_eq!(t.expression().at(0.5), levels, "{t:?}");
    }
    assert_eq!(Theme::Dark.bar(), ([255; 3], [0; 3]));
    assert_eq!(Theme::Light.bar(), ([0; 3], [255; 3]));
    assert_eq!(diverging(-1.), [20, 103, 194]);
    assert_eq!(diverging(0.5), [219, 153, 145]);
    assert_eq!(viridis().at(0.5), [33, 145, 140]);
}
