//! Each ported source carries its notice: the crate-root notices file names
//! the project, its URL and the commit the port follows, with the license
//! text and its copyright lines; each module that follows it names the
//! source in its own header and points to that file.

/// The notices file at the crate root, as `cargo package` ships it.
const NOTICES_FILE: &str = "THIRD_PARTY_NOTICES.md";
const NOTICES: &str = include_str!("../../THIRD_PARTY_NOTICES.md");

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

/// The BSD 3-Clause terms below the copyright lines, as scanpy's LICENSE
/// words them.
const BSD3_TERMS: &str = "Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.";

const MIT: (&str, &str) = ("MIT License", MIT_TERMS);
const BSD3: (&str, &str) = ("BSD 3-Clause License", BSD3_TERMS);

struct Port {
    /// The modules that follow the source.
    modules: &'static [&'static str],
    source: &'static str,
    url: &'static str,
    commit: &'static str,
    /// The license's title and the terms below its copyright lines.
    license: (&'static str, &'static str),
    copyright: &'static [&'static str],
}

const STAGATE_RS: &str = include_str!("spatial/stagate.rs");

const PORTS: [Port; 4] = [
    Port {
        modules: &[include_str!("spatial/spatialde2.rs")],
        source: "SpatialDE",
        url: "https://github.com/PMBio/SpatialDE",
        commit: "6dbdb559a566d639da0f6fd3acc9e17104c0674f",
        license: MIT,
        copyright: &[
            "Copyright (c) 2020 Ilia Kats <i.kats@dkfz-heidelberg.de>, Stegle Group",
            "Copyright (c) 2018 Teichmann Group",
        ],
    },
    Port {
        modules: &[STAGATE_RS],
        source: "STAGATE_pyG",
        url: "https://github.com/QIFEIDKN/STAGATE_pyG",
        commit: "ae1158ca8cf1eb6bb8ee198298552d44c9ac21db",
        license: MIT,
        copyright: &["Copyright (c) 2021 Kangning Dong"],
    },
    Port {
        modules: &[STAGATE_RS],
        source: "STAGATE",
        url: "https://github.com/zhanglabtools/STAGATE",
        commit: "d6dd9f4747ff4f0356347ee9f662d9404bd68143",
        license: MIT,
        copyright: &["Copyright (c) 2021 Kangning Dong"],
    },
    Port {
        modules: &[STAGATE_RS, include_str!("single_cell.rs")],
        source: "scanpy",
        url: "https://github.com/scverse/scanpy",
        commit: "3962176e93ba148bc0f61443c517877d736f2b3b",
        license: BSD3,
        copyright: &[
            "Copyright (c) 2025 scverse®",
            "Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab",
        ],
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
fn each_ported_source_has_its_notice_at_the_crate_root() {
    for p in &PORTS {
        let s = section(NOTICES, p.source);
        let fixed = [p.url, p.commit, p.license.0, p.license.1];
        for want in fixed.into_iter().chain(p.copyright.iter().copied()) {
            assert!(s.contains(want), "{}: no {want:?}", p.source);
        }
    }
}

#[test]
fn each_ported_module_header_names_its_source_and_the_notices() {
    for p in &PORTS {
        let license = p.license.0.trim_end_matches(" License");
        for module in p.modules {
            // The plain comment above the module's `//!` documentation.
            let header = module.split("//!").next().unwrap_or_default();
            for want in [p.source, p.url, license, NOTICES_FILE] {
                assert!(header.contains(want), "{}: header lacks {want:?}", p.source);
            }
        }
    }
}
