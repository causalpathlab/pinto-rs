//! What `pinto mcp` publishes for `pinto view --follow`.

use crate::view::mcp::MapArgs;
use crate::view::{EdgesShown, Followed, Show};

/// The view published names its settings as `view_map` does, with values
/// it takes (and the run), so the two cannot drift apart.
#[test]
fn the_view_followed_is_view_maps_settings() {
    let view = Followed {
        run: "/runs/x".into(),
        level: "L2".into(),
        round: Some("/runs/x.final.a1.lupin.json".into()),
        show: Show::Clusters,
        layer: "C3".into(),
        focus: vec!["K3".into()],
        gene: Some("ENSG01_CD74".into()),
        source: crate::view::gene::Source::Expected,
        clip: 95.,
        edges: true,
        edges_shown: EdgesShown::Heterotypic,
        heterotypic_colour: Some("#d7191c".into()),
        edge_width: 1.5,
        point_size: 2.,
        point_alpha: 0.5,
        bbox: [0., 1., 10., 11.],
    };
    let mut json = serde_json::to_value(&view).unwrap();
    assert_eq!(json["show"], "clusters");
    assert_eq!(json["source"], "expected");
    assert_eq!(json["edges_shown"], "heterotypic");
    json.as_object_mut().unwrap().remove("run");
    serde_json::from_value::<MapArgs>(json.clone()).unwrap_or_else(|e| panic!("{e}: {json}"));
}
