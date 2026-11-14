//! Each ported baseline carries its source's MIT notice: the crate-root
//! notices file names the project, its URL and the commit the port follows,
//! with the license text and its copyright lines; the module's own header
//! names the source and points to that file.

/// The notices file at the crate root, as `cargo package` ships it.
const NOTICES_FILE: &str = "THIRD_PARTY_NOTICES.md";
const NOTICES: &str = include_str!("../../../THIRD_PARTY_NOTICES.md");

/// The MIT terms below the copyright lines, as each source's LICENSE words
/// them.
const MIT_TERMS: &str =
    "Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.";

struct Port {
    module: &'static str,
    source: &'static str,
    url: &'static str,
    commit: &'static str,
    copyright: &'static [&'static str],
}

const PORTS: [Port; 2] = [
    Port {
        module: include_str!("../spatialde2.rs"),
        source: "SpatialDE",
        url: "https://github.com/PMBio/SpatialDE",
        commit: "6dbdb559a566d639da0f6fd3acc9e17104c0674f",
        copyright: &[
            "Copyright (c) 2020 Ilia Kats <i.kats@dkfz-heidelberg.de>, Stegle Group",
            "Copyright (c) 2018 Teichmann Group",
        ],
    },
    Port {
        module: include_str!("../stagate.rs"),
        source: "STAGATE_pyG",
        url: "https://github.com/QIFEIDKN/STAGATE_pyG",
        commit: "ae1158ca8cf1eb6bb8ee198298552d44c9ac21db",
        copyright: &["Copyright (c) 2021 Kangning Dong"],
    },
];

/// The section of the notices file for one source: from its heading to the
/// next one.
fn section<'a>(notices: &'a str, source: &str) -> &'a str {
    let heading = format!("## {source}\n");
    let start = notices
        .find(&heading)
        .unwrap_or_else(|| panic!("no `{heading}` in {NOTICES_FILE}"));
    let rest = &notices[start + heading.len()..];
    &rest[..rest.find("\n## ").unwrap_or(rest.len())]
}

#[test]
fn each_ported_baseline_has_its_mit_notice_at_the_crate_root() {
    for p in &PORTS {
        let s = section(NOTICES, p.source);
        let fixed = [p.url, p.commit, "MIT License", MIT_TERMS];
        for want in fixed.into_iter().chain(p.copyright.iter().copied()) {
            assert!(s.contains(want), "{}: no {want:?}", p.source);
        }
    }
}

#[test]
fn each_ported_module_header_names_its_source_and_the_notices() {
    for p in &PORTS {
        // The plain comment above the module's `//!` documentation.
        let header = p.module.split("//!").next().unwrap_or_default();
        for want in [p.source, p.url, "MIT", NOTICES_FILE] {
            assert!(header.contains(want), "{}: header lacks {want:?}", p.source);
        }
    }
}
