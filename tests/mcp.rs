//! `pinto mcp` driven as an agent's client drives it: JSON-RPC lines on
//! its stdin, answers on its stdout, over a run written to disk.

#![cfg(feature = "view")]

mod common;

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    id: u64,
}

impl Client {
    fn start(dir: &std::path::Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_pinto"))
            .arg("mcp")
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
    // The view's own, then what the protocol adds (the records' notebook).
    assert_eq!(
        names[..5],
        [
            "view_open",
            "view_map",
            "view_markers",
            "view_cell",
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

    // The agent's next move shows, with its call.
    c.ok("view_map", json!({"layer": "C1"}));
    t.wait_screen("layer  C1");
    t.wait_screen("view_map");

    // A zoom of the person's own pauses following; F takes it up again.
    t.type_(b"z");
    t.wait_screen("paused");
    c.ok("view_map", json!({"layer": "argmax"}));
    std::thread::sleep(Duration::from_millis(800));
    assert!(t.screen().contains("layer  C1"), "{}", t.screen());
    t.type_(b"F");
    t.wait_screen("layer  argmax");

    // The server gone, its last view stays.
    drop(c);
    t.wait_screen("ended");
    assert!(t.screen().contains("layer  argmax"));
    t.type_(b"q");
    assert_eq!(t.exit_code(), 0);
}
