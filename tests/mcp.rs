//! `pinto mcp` driven as an agent's client drives it: JSON-RPC lines on
//! its stdin, answers on its stdout, over a run written to disk.

#![cfg(feature = "view")]

mod common;
mod reference;

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    id: u64,
}

impl Client {
    fn start(dir: &std::path::Path) -> Self {
        Self::start_with(dir, &[])
    }

    /// The server with `env` set, for it and the children it runs.
    fn start_with(dir: &std::path::Path, env: &[(&str, &std::ffi::OsStr)]) -> Self {
        Self::start_as(dir, &[], env)
    }

    /// The server started as `pinto {flags} mcp`, with `env` set.
    fn start_as(dir: &std::path::Path, flags: &[&str], env: &[(&str, &std::ffi::OsStr)]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_pinto"))
            .args(flags)
            .arg("mcp")
            .envs(env.iter().copied())
            .current_dir(dir)
            // The feed `pinto view --follow` reads, kept with the test.
            .env("LEGUME_MCP_WATCH", dir.join("watch"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Client {
            child,
            stdin,
            stdout,
            id: 0,
        }
    }

    fn send(&mut self, msg: &Value) {
        self.send_raw(msg.to_string().as_bytes());
    }

    fn send_raw(&mut self, line: &[u8]) {
        self.stdin.write_all(line).unwrap();
        self.stdin.write_all(b"\n").unwrap();
        self.stdin.flush().unwrap();
    }

    fn read(&mut self) -> Value {
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line:?}"))
    }

    /// The reply to `method`, after any notifications, which are returned
    /// with it.
    fn request(&mut self, method: &str, params: Value) -> (Value, Vec<Value>) {
        self.id += 1;
        let id = self.id;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let mut notes = Vec::new();
        loop {
            let msg = self.read();
            if msg.get("id").is_some() {
                assert_eq!(msg["id"], id);
                return (msg, notes);
            }
            notes.push(msg);
        }
    }

    fn call(&mut self, tool: &str, args: Value) -> Value {
        let (reply, _) = self.request("tools/call", json!({"name": tool, "arguments": args}));
        reply["result"].clone()
    }

    /// A call that must succeed: its structured content, and its image if
    /// it had one.
    fn ok(&mut self, tool: &str, args: Value) -> (Value, Option<image::RgbaImage>) {
        let result = self.call(tool, args.clone());
        assert_eq!(result["isError"], false, "{tool} {args}: {result}");
        let image = result["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["type"] == "image")
            .map(|c| {
                use base64::Engine;
                assert_eq!(c["mimeType"], "image/png");
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(c["data"].as_str().unwrap())
                    .unwrap();
                image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
                    .unwrap()
                    .to_rgba8()
            });
        // The state of a map comes as text beside its image.
        let state = match &result["structuredContent"] {
            Value::Null => result["content"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["type"] == "text")
                .map(|c| serde_json::from_str(c["text"].as_str().unwrap()).unwrap())
                .unwrap_or(Value::Null),
            v => v.clone(),
        };
        (state, image)
    }

    /// A call that must fail as a tool does: its message.
    fn fails(&mut self, tool: &str, args: Value) -> String {
        let result = self.call(tool, args.clone());
        assert_eq!(result["isError"], true, "{tool} {args}: {result}");
        result["content"][0]["text"].as_str().unwrap().to_string()
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn an_agent_opens_a_run_looks_at_it_and_asks_about_it() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = common::write_run(dir.path(), 400);
    let mut c = Client::start(dir.path());

    let (init, _) = c.request(
        "initialize",
        json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}),
    );
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(init["result"]["serverInfo"]["name"], "pinto");
    assert!(init["result"]["instructions"]
        .as_str()
        .unwrap()
        .contains("view_open"));
    // A notification is not answered: the next line is the ping's.
    c.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
    let (pong, _) = c.request("ping", json!({}));
    assert_eq!(pong["result"], json!({}));

    let (list, _) = c.request("tools/list", json!({}));
    let names: Vec<&str> = list["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    // pinto run's, the view's own, then what the protocol adds (the
    // records' notebook, the jobs).
    assert_eq!(
        names[..18],
        [
            "run_methods",
            "run_fit",
            "run_predict",
            "run_region",
            "run_compare",
            "run_lra",
            "run_lra_simulate",
            "run_residual",
            "run_simulate",
            "run_sim_score",
            "run_sim_pairs",
            "run_sim_baseline",
            "view_open",
            "view_map",
            "view_markers",
            "view_cell",
            "view_lra",
            "view_export"
        ]
    );

    // A batch, and bytes that are not UTF-8, are answered, and the server
    // carries on.
    c.send(&json!([{"jsonrpc": "2.0", "id": 99, "method": "ping"}]));
    assert_eq!(c.read()["error"]["code"], -32600);
    c.send_raw(b"{\"id\": 98, \xff}");
    assert_eq!(c.read()["error"]["code"], -32700);

    // Before a run is open, a tool fails as a tool, not as the protocol.
    assert!(c.fails("view_map", json!({})).contains("view_open"));
    let (unknown, _) = c.request("tools/call", json!({"name": "nope", "arguments": {}}));
    assert_eq!(unknown["error"]["code"], -32602);

    // Opening reports progress when asked for it.
    let (opened, notes) = c.request(
        "tools/call",
        json!({"name": "view_open", "arguments": {"prefix": prefix}, "_meta": {"progressToken": 7}}),
    );
    assert!(notes
        .iter()
        .any(|n| n["method"] == "notifications/progress" && n["params"]["progressToken"] == 7));
    let summary = &opened["result"]["structuredContent"];
    assert_eq!(summary["cells"], 400);
    assert_eq!(summary["k"], 2);
    assert_eq!(summary["groups"].as_array().unwrap().len(), 2);

    // The whole tissue, 1024 wide, square as the grid is.
    let (state, img) = c.ok("view_map", json!({}));
    let img = img.unwrap();
    assert_eq!((img.width(), img.height()), (1024, 1024));
    assert_eq!(state["image"], json!([1024, 1024]));
    // The scale as the panel says it, with its unit.
    assert!(
        state["scale"].as_str().unwrap().ends_with("/px"),
        "{}",
        state["scale"]
    );
    let groups = state["legend"]["groups"].as_array().unwrap();
    assert!(groups
        .iter()
        .all(|g| g["colour"].as_str().unwrap().starts_with('#')));

    // Settings stick: a focus, then a zoom that keeps it.
    let (state, _) = c.ok("view_map", json!({"focus": ["C1"], "width": 200}));
    assert_eq!(state["focus"], json!(["C1"]));
    let before = state["window"].clone();
    let (state, img) = c.ok("view_map", json!({"zoom": 2}));
    assert_eq!(state["focus"], json!(["C1"]));
    assert_eq!(img.unwrap().width(), 1024, "width is for one call only");
    let span = |w: &Value| w[2].as_f64().unwrap() - w[0].as_f64().unwrap();
    assert!((span(&before) / span(&state["window"]) - 2.).abs() < 1e-3);

    // The cell under the image's centre, and the same one by name.
    let (cell, _) = c.ok("view_cell", json!({"px": [512, 512]}));
    let name = cell["cell"].as_str().unwrap().to_string();
    let (by_name, _) = c.ok("view_cell", json!({"name": name}));
    assert_eq!(by_name["at"], cell["at"]);
    assert!(by_name["group"].as_str().unwrap().starts_with('C'));
    assert!(c
        .fails("view_cell", json!({"name": "nobody"}))
        .contains("nobody"));

    // A call that fails changes nothing: not the edges it could not load
    // (the run has no link communities), nor the focus or window asked
    // for with them.
    let (before, _) = c.ok("view_map", json!({}));
    c.fails("view_map", json!({"edges": true, "focus": ["C0"]}));
    c.fails("view_map", json!({"focus": ["C0"], "fit": true, "zoom": 0}));
    let (after, _) = c.ok("view_map", json!({}));
    assert_eq!(after, before);
    assert_eq!(after["edges"], false);

    // Bad arguments fail with what is wrong.
    assert!(c.fails("view_map", json!({"layer": "C9"})).contains("K=2"));
    assert!(c.fails("view_map", json!({"focus": ["C5"]})).contains("C5"));
    assert!(c
        .fails("view_map", json!({"colour": "red"}))
        .contains("colour"));
    assert!(c
        .fails("view_map", json!({"show": "types"}))
        .contains("round"));
    for window in [
        json!({"bbox": [1, 1, 1, 5]}),
        json!({"bbox": [-3e38, 0, 3e38, 1]}),
        json!({"zoom": 1e-38}),
    ] {
        let msg = c.fails("view_map", window.clone());
        assert!(msg.contains("no area"), "{window}: {msg}");
    }

    // Export writes the view, then refuses to write over it.
    let (wrote, _) = c.ok("view_export", json!({"png": "map.png", "width": 300}));
    let path = wrote["wrote"][0].as_str().unwrap();
    assert!(std::path::Path::new(path).is_absolute());
    let exported = image::open(path).unwrap().to_rgba8();
    assert_eq!(exported.width(), 300);
    // Its command draws the same image again, without the agent.
    let command = wrote["command"].as_str().unwrap();
    let again = dir.path().join("again.png");
    let command = command
        .replacen("pinto ", &format!("{} ", env!("CARGO_BIN_EXE_pinto")), 1)
        .replace(path, again.to_str().unwrap());
    let status = Command::new("sh").arg("-c").arg(&command).status().unwrap();
    assert!(status.success(), "{command}");
    assert_eq!(
        image::open(&again).unwrap().to_rgba8(),
        exported,
        "{command}"
    );
    assert!(c
        .fails("view_export", json!({"png": "map.png"}))
        .contains("exists"));
    c.ok("view_export", json!({"png": "map.png", "overwrite": true}));
    // Nothing but a figure is written, not even with overwrite.
    let manifest = format!("{prefix}.pinto.json");
    let msg = c.fails("view_export", json!({"png": manifest, "overwrite": true}));
    assert!(msg.contains(".png file"), "{msg}");
    assert!(std::fs::read_to_string(&manifest).unwrap().starts_with('{'));
}

/// A person follows the agent from `pinto view --follow`: the agent's
/// view as it changes, paused by a move of their own, taken up again by
/// `F`, and its last view kept once the server is gone.
#[cfg(unix)]
#[test]
fn a_person_follows_the_agent_in_the_viewer() {
    use legume_tui::testing::Private;
    use std::time::Duration;
    let dir = tempfile::tempdir().unwrap();
    let prefix = common::write_run(dir.path(), 400);
    let watch = dir.path().join("watch");
    let watch = watch.to_str().unwrap();
    let mut c = Client::start(dir.path());
    c.request("initialize", json!({"protocolVersion": "2025-06-18"}));

    // Before the agent opens a run there is nothing to follow.
    let sessions = Private::new(env!("CARGO_BIN_EXE_pinto")).patience(Duration::from_secs(30));
    let env = [("LEGUME_MCP_WATCH", watch)];
    let (ok, said) = {
        let mut cmd = sessions.command();
        cmd.args(["view", "--follow", "--no-detach"])
            .env("LEGUME_MCP_WATCH", watch);
        let out = cmd.output().unwrap();
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    assert!(!ok && said.contains("opened no run"), "{said}");

    c.ok("view_open", json!({"prefix": prefix}));
    c.ok("view_map", json!({"layer": "soft"}));
    let mut t = sessions.term(&["view", "--follow", "--no-detach"], &env);
    t.wait_screen("following");
    t.wait_screen("layer  soft");

    // The agent's next move shows, and its call joins the panel's.
    let calls = |t: &legume_tui::testing::Term| t.screen().matches("view_map").count();
    let before = calls(&t);
    c.ok("view_map", json!({"layer": "C1"}));
    t.wait_screen("layer  C1");
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while calls(&t) <= before && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(calls(&t) > before, "{}", t.screen());

    // A zoom of the person's own pauses following, but the agent's calls
    // still join the panel; F takes following up again.
    t.type_(b"z");
    t.wait_screen("paused");
    let before = calls(&t);
    c.ok("view_map", json!({"layer": "argmax"}));
    std::thread::sleep(Duration::from_millis(800));
    assert!(t.screen().contains("layer  C1"), "{}", t.screen());
    assert!(
        calls(&t) > before,
        "the calls stay listed while paused: {}",
        t.screen()
    );
    t.type_(b"F");
    t.wait_screen("layer  argmax");

    // The server gone, its last view stays.
    drop(c);
    t.wait_screen("ended");
    assert!(t.screen().contains("layer  argmax"));
    t.type_(b"q");
    assert_eq!(t.exit_code(), 0);
}

/// An agent sets up a fit as `pinto run` does: the methods and their
/// flags, a command pinto would parse, never over a run, and a fit that
/// fails says why, in the foreground and as a background job.
#[test]
fn an_agent_sets_up_fits_as_pinto_run_does() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = common::write_run(dir.path(), 100);
    let mut c = Client::start(dir.path());
    c.request("initialize", json!({"protocolVersion": "2025-06-18"}));

    let (methods, _) = c.ok("run_methods", json!({}));
    let names: Vec<&str> = methods["methods"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["method"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "lc",
            "dsvd",
            "cage",
            "cage-cell",
            "spatialde2",
            "stagate",
            "spatialpca",
            "mefisto",
            "spatialde2-pb",
            "stagate-pb",
            "spatialpca-pb",
            "mefisto-pb",
        ]
    );
    // No baseline offers a --reference: each runs its sparse formulation.
    for m in &names[4..] {
        let (b, _) = c.ok("run_methods", json!({"method": m}));
        let flags = b["flags"].as_array().unwrap();
        assert!(!flags.is_empty(), "{m}: {b}");
        assert!(!flags.iter().any(|f| f["flag"] == "--reference"), "{m}");
    }
    let (lc, _) = c.ok("run_methods", json!({"method": "lc"}));
    assert_eq!(lc["coords"], true);
    let flags = lc["flags"].as_array().unwrap();
    assert!(flags.iter().any(|f| f["flag"] == "--n-communities"), "{lc}");
    // Flags pinto run fills from its own fields are not listed as flags.
    assert!(!flags
        .iter()
        .any(|f| f["flag"] == "--out" || f["flag"] == "--coord"));

    let fit = |extra: Value| {
        let mut args = json!({"method": "lc", "data": ["nowhere.zarr.zip"], "out": "fits/a"});
        args.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        args
    };
    assert!(c
        .fails(
            "run_fit",
            json!({"method": "svd", "data": ["d"], "out": "x"})
        )
        .contains("one of lc"));
    // Two ablations in one fit are refused when the call is made, by clap,
    // before any job starts.
    let msg = c.fails(
        "run_fit",
        fit(json!({"flags": {"drop-features": "^MT-", "mask-cells": "c.txt"}})),
    );
    assert!(msg.contains("cannot be used with"), "{msg}");
    let msg = c.fails("run_fit", fit(json!({"flags": {"no-such-flag": 1}})));
    assert!(
        msg.contains("--no-such-flag") && msg.contains("run_methods"),
        "{msg}"
    );
    let msg = c.fails("run_fit", fit(json!({"flags": {"n-communities": "many"}})));
    assert!(
        msg.contains("pinto fit lc --n-communities"),
        "clap's complaint, its flag named: {msg}"
    );
    // `pinto run`'s other outputs count as a run there too, not only its
    // manifest.
    std::fs::create_dir_all(dir.path().join("fits")).unwrap();
    std::fs::write(dir.path().join("fits/c.cmd.sh"), "").unwrap();
    let msg = c.fails("run_fit", fit(json!({"out": "fits/c"})));
    assert!(msg.contains("c.cmd.sh exists"), "{msg}");
    // A switch is true or false; coordinates come one per data file.
    let msg = c.fails("run_fit", fit(json!({"flags": {"no-qc": 1}})));
    assert!(msg.contains("--no-qc is a switch"), "{msg}");
    let msg = c.fails("run_fit", fit(json!({"coords": ["a.csv", "b.csv"]})));
    assert!(msg.contains("one per data file"), "{msg}");
    // The run already there is not written over.
    let msg = c.fails("run_fit", fit(json!({"out": prefix})));
    assert!(msg.contains("exists"), "{msg}");

    // A fit that fails says so: at once, or through its job.
    let msg = c.fails("run_fit", fit(json!({"background": false})));
    assert!(msg.contains("nowhere.zarr.zip"), "{msg}");
    // --atomic-outputs is the tool's own field, as --max-threads is.
    let msg = c.fails("run_fit", fit(json!({"flags": {"atomic-outputs": true}})));
    assert!(msg.contains("as atomic_outputs, not as a flag"), "{msg}");
    // In the background unless asked otherwise.
    let (job, _) = c.ok(
        "run_fit",
        fit(json!({"out": "fits/b", "atomic_outputs": true})),
    );
    let id = job["job"].as_str().unwrap().to_string();
    // Paths go from the root, so the records replay it from the project.
    let command = job["command"].as_str().unwrap();
    assert!(
        command.starts_with("pinto fit lc /") && command.contains("/nowhere.zarr.zip"),
        "{command}"
    );
    assert!(command.ends_with(" --atomic-outputs"), "{command}");
    assert!(job["writes"][0]
        .as_str()
        .unwrap()
        .ends_with("fits/b.pinto.json"));
    assert!(job["note"].as_str().unwrap().contains(&id));
    let (waited, _) = c.ok("job_wait", json!({"id": id, "seconds": 60}));
    assert_eq!(waited["state"], "failed", "{waited}");
}

#[test]
fn an_agent_compares_two_runs() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = common::write_run(dir.path(), 100);
    let mut c = Client::start(dir.path());
    c.request("initialize", json!({"protocolVersion": "2025-06-18"}));

    // The ablation flags are the method's own, listed as any other.
    let (lc, _) = c.ok("run_methods", json!({"method": "lc"}));
    let flags = lc["flags"].as_array().unwrap();
    for flag in [
        "--permute-coords",
        "--drop-features",
        "--mask-cells",
        "--ablation-parent",
    ] {
        assert!(flags.iter().any(|f| f["flag"] == flag), "{flag}");
    }

    let out = dir.path().join("cmp").to_string_lossy().into_owned();
    let (same, _) = c.ok(
        "run_compare",
        json!({"a": prefix, "b": format!("{prefix}.propensity.parquet"), "out": out,
               "atomic_outputs": true}),
    );
    assert!(
        same["command"]
            .as_str()
            .is_some_and(|c| c.contains("--atomic-outputs")),
        "{same}"
    );
    assert_eq!(same["ari"], 1.0);
    assert_eq!(same["n_shared"], 100);
    assert!(same["vanished"].as_array().unwrap().is_empty());
    assert!(std::path::Path::new(&format!("{out}.compare.json")).exists());
    assert!(std::path::Path::new(&format!("{out}.contingency.parquet")).exists());
    assert!(c
        .fails("run_compare", json!({"a": prefix, "b": prefix, "out": out}))
        .contains("written over"));
    assert!(c
        .fails(
            "run_compare",
            json!({"a": prefix, "b": prefix, "level_b": "L9", "out": "x"})
        )
        .contains("L9"));
}

/// A server started with `--atomic-outputs` gives the flag to the runs its
/// tools start, once; a tool's own `atomic_outputs` says otherwise, on or
/// off.
#[test]
fn a_server_started_with_atomic_outputs_gives_it_to_its_tools_runs() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = common::write_run(dir.path(), 100);
    let mut c = Client::start_as(dir.path(), &["--atomic-outputs"], &[]);
    c.request("initialize", json!({"protocolVersion": "2025-06-18"}));
    for (name, own, flagged) in [
        ("unsaid", None, 1),
        ("on", Some(true), 1),
        ("off", Some(false), 0),
    ] {
        let out = dir.path().join(name).to_string_lossy().into_owned();
        let mut args = json!({"a": prefix, "b": prefix, "out": out});
        if let Some(own) = own {
            args["atomic_outputs"] = own.into();
        }
        let (reply, _) = c.ok("run_compare", args);
        let command = reply["command"].as_str().unwrap();
        assert_eq!(
            command.matches("--atomic-outputs").count(),
            flagged,
            "{name}: {command}"
        );
        assert!(Path::new(&format!("{out}.compare.json")).is_file());
        assert!(Path::new(&format!("{out}.contingency.parquet")).is_file());
    }
    // nothing is left staged beside the comparisons
    let left: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".partial-"))
        .collect();
    assert!(left.is_empty(), "{left:?}");
}

#[test]
fn an_agent_marks_a_region_to_mask() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = common::write_run(dir.path(), 100);
    let mut c = Client::start(dir.path());
    c.request("initialize", json!({"protocolVersion": "2025-06-18"}));

    // The 10 × 10 grid's left two columns (x 0 and 1), in world coordinates.
    let out = dir.path().join("left").to_string_lossy().into_owned();
    let (picked, _) = c.ok(
        "run_region",
        json!({"run": prefix, "polygon": [[-0.5, -0.5], [1.5, -0.5], [1.5, 9.5], [-0.5, 9.5]], "out": out}),
    );
    assert_eq!(picked["cells"], 20);
    assert_eq!(picked["of"], 100);
    let list = std::fs::read_to_string(format!("{out}.cells.txt")).unwrap();
    assert_eq!(list.lines().count(), 21, "a header and 20 cells");
    assert!(picked["note"].as_str().unwrap().contains("mask-cells"));

    // The same region as GeoJSON in section coordinates (one section here).
    let gj = dir.path().join("left.geojson");
    std::fs::write(
        &gj,
        r#"{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[-0.5,-0.5],[1.5,-0.5],[1.5,9.5],[-0.5,9.5]]]}}"#,
    )
    .unwrap();
    let out2 = dir.path().join("left2").to_string_lossy().into_owned();
    let (again, _) = c.ok(
        "run_region",
        json!({"run": prefix, "geojson": gj, "out": out2}),
    );
    assert_eq!(again["cells"], 20);
    assert!(c
        .fails("run_region", json!({"run": prefix, "out": "x"}))
        .contains("polygon"));

    // run_predict checks its command as pinto parses it before running.
    let msg = c.fails(
        "run_predict",
        json!({"model": prefix, "data": ["d.zarr"], "out": "p", "flags": {"no-such": 1}}),
    );
    assert!(msg.contains("--no-such"), "{msg}");
}

/// lra over the server: calls refused as pinto would refuse them, before
/// anything runs.
#[test]
fn an_agent_sets_up_a_ligand_receptor_analysis() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = common::write_run(dir.path(), 100);
    let mut c = Client::start(dir.path());
    c.request("initialize", json!({"protocolVersion": "2025-06-18"}));

    let run = format!("{prefix}.pinto.json");
    let msg = c.fails("run_lra", json!({"run": run, "fdr_level": 0.1}));
    assert!(msg.contains("unknown field `fdr_level`"), "{msg}");
    let msg = c.fails("run_lra", json!({"run": run, "method": "both"}));
    assert!(msg.contains("contact-effect"), "{msg}");
    let msg = c.fails(
        "run_lra",
        json!({"run": run, "species": "fish", "out": "x"}),
    );
    assert!(msg.contains("pinto lra") && msg.contains("fish"), "{msg}");
    let msg = c.fails("run_lra", json!({"run": "nowhere.pinto.json"}));
    assert!(msg.contains("not found"), "{msg}");
    let msg = c.fails("run_lra", json!({"run": run, "only": "SPP1", "out": "x"}));
    assert!(msg.contains("sequence"), "{msg}");
    // A run with no lra table has none to list.
    c.ok("view_open", json!({"prefix": prefix}));
    let msg = c.fails("view_lra", json!({}));
    assert!(
        msg.contains("no lra table") && msg.contains("run_lra"),
        "{msg}"
    );
    let msg = c.fails("view_export", json!({"png": "c.png", "curve": true}));
    assert!(msg.contains("no lra pair"), "{msg}");
    // The study and its example were removed.
    for gone in ["study", "example_study"] {
        let msg = c.fails("run_lra_simulate", json!({"run": run, gone: true}));
        assert!(msg.contains(&format!("unknown field `{gone}`")), "{msg}");
    }
    let msg = c.fails("run_lra_simulate", json!({}));
    assert!(
        msg.contains("give run, a run's .pinto.json, or data_files with coord_files"),
        "{msg}"
    );
    // A table already at out that cannot be read is not written over: the
    // reply says how to run anew.
    std::fs::write(dir.path().join("old.lra-simulate.parquet"), "").unwrap();
    let msg = c.fails("run_lra_simulate", json!({"run": run, "out": "old"}));
    assert!(msg.contains("give another out"), "{msg}");
    assert!(!dir.path().join("x.lra.parquet").exists());
    // A call is checked as pinto parses it before any finished table at its
    // out is read: a value pinto does not take is refused as such.
    for f in ["old.lra.parquet", "old.lra.curve.parquet"] {
        std::fs::write(dir.path().join(f), "").unwrap();
    }
    for tool in ["run_lra", "run_lra_simulate"] {
        let msg = c.fails(
            tool,
            json!({"run": run, "out": "old", "species": "martian"}),
        );
        assert!(
            msg.contains("martian") && msg.contains("pinto lra"),
            "{tool}: {msg}"
        );
    }
    // Permutations of baselines not asked for, or with a co-expression
    // method, are refused before anything runs.
    let msg = c.fails(
        "run_lra",
        json!({"run": run, "out": "y", "marginal_permutations": 500}),
    );
    assert!(msg.contains("marginal_baselines"), "{msg}");
    let msg = c.fails(
        "run_lra",
        json!({"run": run, "out": "y", "method": "coexpression-test", "marginal_baselines": true}),
    );
    assert!(msg.contains("--marginal-baselines applies to"), "{msg}");
    assert!(!dir.path().join("y.lr_activity.parquet").exists());
}

/// A single lra-simulate scenario with no out takes a prefix of its own,
/// even for a run in today's results folder, whose own prefix holds the
/// built-in grid's table; a grid call keeps that slot.
#[test]
fn a_single_scenario_takes_a_prefix_of_its_own() {
    let dir = tempfile::tempdir().unwrap();
    let reference = reference::empty_reference(dir.path());
    let mut c = Client::start_with(
        dir.path(),
        &[
            ("PINTO_REFERENCE_DIR", reference.as_os_str()),
            ("PINTO_OFFLINE", std::ffi::OsStr::new("1")),
        ],
    );
    c.request("initialize", json!({"protocolVersion": "2025-06-18"}));
    // Today's folder, from a call on data of another name.
    let (job, _) = c.ok(
        "run_lra_simulate",
        json!({"data_files": ["d.zarr"], "coord_files": ["d.csv"], "one": true}),
    );
    let first = job["writes"][0].as_str().unwrap_or_else(|| panic!("{job}"));
    let day = dir.path().join(Path::new(first).parent().unwrap());
    let prefix = common::write_run(&day, 100);
    let run = format!("{prefix}.pinto.json");
    // Another run's prefix is no free name either.
    std::fs::write(day.join("run-scenario.pinto.json"), "{}").unwrap();
    // Each file written, by its name in today's folder.
    let mut writes = |args: Value| -> Vec<String> {
        let (job, _) = c.ok("run_lra_simulate", args);
        let writes = job["writes"].as_array().unwrap_or_else(|| panic!("{job}"));
        writes
            .iter()
            .map(|w| {
                let w = Path::new(w.as_str().unwrap());
                assert_eq!(w.parent().unwrap().file_name(), day.file_name(), "{job}");
                w.file_name().unwrap().to_string_lossy().into_owned()
            })
            .collect()
    };
    for single in [
        json!({"run": run, "one": true}),
        json!({"run": run, "pairs": 4}),
        json!({"run": run, "null_pairs": 4}),
    ] {
        let made = writes(single.clone());
        assert_eq!(made.len(), 2, "the table and its sidecar: {made:?}");
        for w in &made {
            for taken in ["run.", "run-scenario."] {
                assert!(!w.starts_with(taken), "{single}: {w} is a run's slot");
            }
        }
    }
    assert_eq!(writes(json!({"run": run})), ["run.lra-simulate.parquet"]);
}

/// A simulation over the server: settings checked before anything runs, the
/// simulation a background job that never writes over one, its truth run
/// opened in the viewer, and the truth scored against itself.
#[test]
fn an_agent_simulates_looks_and_scores() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = Client::start(dir.path());
    c.request("initialize", json!({"protocolVersion": "2025-06-18"}));
    let tiny = json!(["grid_side=8", "n_genes=40", "programme_genes=5"]);

    let msg = c.fails("run_simulate", json!({"set": ["pve_type=0"], "out": "bad"}));
    assert!(msg.contains("pve_type must be positive"), "{msg}");
    assert!(!dir.path().join("bad").exists(), "refused before it ran");

    let (job, _) = c.ok(
        "run_simulate",
        json!({"scenario": "donor-confounded-donor-only-cell-type", "set": tiny, "out": "sim"}),
    );
    let id = job["job"].as_str().unwrap().to_string();
    let truth = job["truth"].as_str().unwrap().to_string();
    // The reply spells out the variance shares, noise as the remainder.
    assert!(job["shares"]["noise"].as_f64().unwrap() > 0.0, "{job}");
    assert!(
        job["shares_note"]
            .as_str()
            .unwrap()
            .contains("(the remainder)"),
        "{job}"
    );
    assert!(truth.ends_with("sim/truth/truth.pinto.json"), "{job}");
    assert!(job["note"].as_str().unwrap().contains("view_open"), "{job}");
    let (waited, _) = c.ok("job_wait", json!({"id": id, "seconds": 120}));
    assert_eq!(waited["state"], "done", "{waited}");
    assert!(
        waited["summary"].as_str().unwrap().contains(&truth),
        "the finished job names the truth run: {waited}"
    );
    assert!(dir.path().join("sim/manifest.json").is_file());
    // The reply lists the sections to fit, with their coordinates in the
    // same order, all written; the finished job names them too.
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("sim/manifest.json")).unwrap(),
    )
    .unwrap();
    let sections = manifest["sections"].as_array().unwrap();
    let (data, coords) = (
        job["fit"]["data"].as_array().unwrap(),
        job["fit"]["coords"].as_array().unwrap(),
    );
    assert_eq!(
        (data.len(), coords.len()),
        (sections.len(), sections.len()),
        "{job}"
    );
    for ((d, xy), sec) in data.iter().zip(coords).zip(sections) {
        let name = sec["section"].as_str().unwrap();
        let (d, xy) = (d.as_str().unwrap(), xy.as_str().unwrap());
        assert!(d.ends_with(&format!("sim/sections/{name}.zarr")), "{d}");
        assert!(
            xy.ends_with(&format!("sim/sections/{name}.coords.parquet")),
            "{xy}"
        );
        assert!(Path::new(d).exists() && Path::new(xy).is_file(), "{d} {xy}");
    }
    assert!(!dir.path().join("sim.partial").exists(), "{job}");

    let msg = c.fails("run_simulate", json!({"set": tiny, "out": "sim"}));
    assert!(
        msg.contains("exists") && msg.contains("give another out"),
        "{msg}"
    );

    let (opened, _) = c.ok("view_open", json!({"prefix": truth}));
    assert!(
        opened["levels"]
            .as_array()
            .unwrap()
            .contains(&json!("domain")),
        "{opened}"
    );
    let batches: Vec<&Value> = opened["batches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| &b["batch"])
        .collect();
    let names: Vec<&Value> = sections.iter().map(|s| &s["section"]).collect();
    assert_eq!(batches, names, "one batch per section: {opened}");

    // The reply's truth is what run_sim_score takes, as is the out folder.
    for given in [truth.as_str(), "sim"] {
        let (scored, _) = c.ok(
            "run_sim_score",
            json!({"truth": given, "labels": [truth], "level": "domain"}),
        );
        let s = &scored["scores"][0]["score"];
        assert_eq!(s["shared_ari"], 1.0, "{scored}");
        assert_eq!(s["donor_dependence"], 0.0, "{scored}");
        assert_eq!(s["patches"][0]["f1"], 1.0, "{scored}");
    }
    // So it is for run_sim_pairs, the truth run's domains as the vectors.
    let (paired, _) = c.ok(
        "run_sim_pairs",
        json!({"truth": truth, "vectors": [truth], "level": "domain", "pairs": 1000}),
    );
    let s = &paired["scores"][0]["score"];
    assert_eq!(s["cross_donor"]["auroc"], 1.0, "{paired}");
    assert_eq!(s["spatial"]["auroc"], 1.0, "{paired}");
    // A baseline's bad setting, or a simulation it cannot read, is refused
    // before any job.
    let msg = c.fails("run_sim_baseline", json!({"truth": truth, "k": 0}));
    assert!(msg.contains("k must be at least 1"), "{msg}");
    std::fs::create_dir_all(dir.path().join("half/truth")).unwrap();
    let msg = c.fails("run_sim_baseline", json!({"truth": "half"}));
    assert!(msg.contains("give run_simulate's truth"), "{msg}");
    // A table scores like a run: the truth's own spots by their domain.
    let spots = Path::new(&truth).with_file_name("spots.parquet");
    let (scored, _) = c.ok(
        "run_sim_score",
        json!({"truth": truth, "labels": [spots], "label_column": "domain"}),
    );
    assert_eq!(scored["scores"][0]["score"]["shared_ari"], 1.0, "{scored}");

    // A labelling that cannot be scored is named; the others are scored.
    let (scored, _) = c.ok(
        "run_sim_score",
        json!({"truth": truth, "labels": [truth, "nowhere.tsv"], "level": "domain"}),
    );
    assert_eq!(scored["scores"].as_array().unwrap().len(), 1, "{scored}");
    assert_eq!(scored["failed"][0]["labels"], "nowhere.tsv", "{scored}");
    let msg = c.fails(
        "run_sim_score",
        json!({"truth": truth, "labels": ["nowhere.tsv"]}),
    );
    assert!(msg.contains("nowhere.tsv"), "{msg}");
    let msg = c.fails("run_sim_score", json!({"truth": " ", "labels": [truth]}));
    assert!(msg.contains("give run_simulate's truth"), "{msg}");

    // A simulation that fails once launched says so through its job, or at
    // once when waited on, and leaves its out free for a retry.
    std::fs::write(dir.path().join("blocker"), "").unwrap();
    let (job, _) = c.ok("run_simulate", json!({"set": tiny, "out": "blocker/sim"}));
    let id = job["job"].as_str().unwrap().to_string();
    let (waited, _) = c.ok("job_wait", json!({"id": id, "seconds": 120}));
    assert_eq!(waited["state"], "failed", "{waited}");
    let msg = c.fails(
        "run_simulate",
        json!({"set": tiny, "out": "blocker/sim", "background": false}),
    );
    assert!(!msg.contains("give another out"), "{msg}");
}

/// run_residual over the server, on a run written to disk: a run that
/// records no propensity output is refused before any job; with no out
/// the residual takes today's free name, passing over a file in the way,
/// and an -o that is a file is refused. (What the job writes, and its
/// command line, are `pinto residual`'s and `run residual`'s own tests.)
#[test]
fn an_agent_takes_a_residual_under_a_free_name() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = Client::start(dir.path());
    c.request("initialize", json!({"protocolVersion": "2025-06-18"}));
    let prefix = common::write_run(dir.path(), 16);
    let run = format!("{prefix}.pinto.json");
    let mut meta: Value = serde_json::from_str(&std::fs::read_to_string(&run).unwrap()).unwrap();
    // The run's data files and coordinate columns, as a fit records them.
    let data = dir.path().join("raw/S1.zarr");
    std::fs::create_dir_all(&data).unwrap();
    meta["data_files"] = json!([data.to_string_lossy()]);
    meta["outputs"]["coord_columns"] = json!(["x", "y"]);
    std::fs::write(&run, meta.to_string()).unwrap();

    // Without a propensity output: refused, and no job started.
    let bare = dir.path().join("bare.pinto.json");
    let mut no_propensity = meta;
    no_propensity["prefix"] = json!(dir.path().join("bare").to_string_lossy());
    no_propensity["outputs"]
        .as_object_mut()
        .unwrap()
        .remove("propensity");
    std::fs::write(&bare, no_propensity.to_string()).unwrap();
    let msg = c.fails("run_residual", json!({"run": bare, "out": "bad"}));
    assert!(msg.contains("records no propensity output"), "{msg}");
    assert!(!dir.path().join("bad").exists(), "refused before it ran");
    let (jobs, _) = c.ok("job_status", json!({}));
    assert!(!jobs.to_string().contains("residual of"), "{jobs}");

    // Today's free name passes over a file in the way; an -o that is a
    // file is refused before any job.
    c.fails("run_residual", json!({"run": bare}));
    let day = std::fs::read_dir(dir.path().join("results"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.is_dir())
        .unwrap();
    let in_the_way = day.join("residual");
    std::fs::write(&in_the_way, "").unwrap();
    let in_the_way = in_the_way.to_string_lossy().into_owned();
    let msg = c.fails("run_residual", json!({"run": run, "out": in_the_way}));
    assert!(msg.contains("is a file"), "{msg}");
    let (job, _) = c.ok("run_residual", json!({"run": run}));
    let res = job["out"].as_str().unwrap().to_string();
    assert!(res.ends_with("residual-2"), "{job}");
    assert!(job["note"].as_str().unwrap().contains("run_fit"), "{job}");
    assert!(job["note"].as_str().unwrap().contains("run_lra"), "{job}");
    // divide, the default: stage 2 reads the run's own files against the
    // residual's folder
    assert_eq!(
        job["fit"]["flags"],
        json!({"coord-column-names": "x,y", "offset": res}),
        "{job}"
    );
    assert_eq!(job["fit"]["data"], json!([data.to_string_lossy()]), "{job}");
    // The job reads an empty data file; let it end before the server does.
    let id = job["job"].as_str().unwrap().to_string();
    c.ok("job_wait", json!({"id": id, "seconds": 120}));
}
