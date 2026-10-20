//! `pinto mcp`: the Model Context Protocol on stdin/stdout, one JSON-RPC
//! message per line each way, answering what a server of tools is asked:
//! `initialize`, `ping`, `tools/list` and `tools/call`.
//!
//! The API is legume-mcp's, which takes this module's place once it is
//! published: a [`Handler`] lists its tools and answers calls, holding
//! whatever it keeps between them. A tool that fails answers with
//! `isError` and its message; JSON-RPC errors are for protocol faults.
//! Nothing but the protocol goes to stdout; logs go to stderr.

use base64::Engine;
use serde_json::{json, Value};
use std::io::{BufRead, Write};

/// Protocol versions spoken, oldest first: a client asking for one of
/// them gets it, any other the newest.
const VERSIONS: [&str; 4] = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"];

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

/// What `initialize` says of the server.
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    /// For the agent: what the server is for and the order of calls.
    pub instructions: Option<String>,
}

/// One tool as `tools/list` gives it.
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema of the call's arguments.
    pub input_schema: Value,
}

impl ToolSpec {
    pub fn new(name: &str, description: &str, input_schema: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// A piece of a tool's result.
pub enum Content {
    Text(String),
    /// The value as text; the result's first object is also its
    /// `structuredContent`.
    Json(Value),
    /// An image, base64 encoded.
    Png(Vec<u8>),
}

pub trait Handler {
    fn tools(&self) -> Vec<ToolSpec>;
    fn call(&mut self, name: &str, args: Value, cx: &Call) -> anyhow::Result<Vec<Content>>;
}

/// The call being answered: progress goes to the client, when it asked
/// for it.
pub struct Call<'a> {
    token: Option<Value>,
    /// Progress told so far: each notice counts one more.
    step: std::cell::Cell<u64>,
    send: &'a dyn Fn(&Value),
}

impl Call<'_> {
    /// Tell the client where the call has got to.
    pub fn progress(&self, msg: &str) {
        log::info!("{msg}");
        if let Some(token) = &self.token {
            let step = self.step.get() + 1;
            self.step.set(step);
            (self.send)(&json!({
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {"progressToken": token, "progress": step, "message": msg},
            }));
        }
    }
}

/// Answer the client on stdin/stdout until it closes stdin.
pub fn serve(info: ServerInfo, handler: impl Handler) -> anyhow::Result<()> {
    serve_on(
        std::io::stdin().lock(),
        std::io::stdout().lock(),
        info,
        handler,
    )
}

/// Answer every message read from `input` on `output`.
pub fn serve_on(
    input: impl BufRead,
    output: impl Write,
    info: ServerInfo,
    mut handler: impl Handler,
) -> anyhow::Result<()> {
    let output = std::cell::RefCell::new(output);
    let send = |msg: &Value| {
        let mut out = output.borrow_mut();
        // A client that has gone leaves nothing to tell; stdin ends next.
        let _ = serde_json::to_writer(&mut *out, msg)
            .map_err(std::io::Error::from)
            .and_then(|()| out.write_all(b"\n"))
            .and_then(|()| out.flush());
    };
    for line in input.split(b'\n') {
        let line = line?;
        if line.trim_ascii().is_empty() {
            continue;
        }
        // Bytes that are not UTF-8 are answered as any other bad JSON.
        let reply = match serde_json::from_slice::<Value>(&line) {
            Ok(msg) => answer(&info, &mut handler, &msg, &send),
            Err(e) => Some(failure(Value::Null, PARSE_ERROR, &e.to_string())),
        };
        if let Some(reply) = reply {
            send(&reply);
        }
    }
    Ok(())
}

/// The reply to `msg`; none to a notification or a response.
fn answer(
    info: &ServerInfo,
    handler: &mut impl Handler,
    msg: &Value,
    send: &dyn Fn(&Value),
) -> Option<Value> {
    let Some(obj) = msg.as_object() else {
        // A batch among them: the protocol since 2025-06-18 has none.
        return Some(failure(
            Value::Null,
            INVALID_REQUEST,
            "one request per message",
        ));
    };
    // Without an id, a notification; without a method, a response to
    // nothing asked. Neither is answered.
    let id = obj.get("id")?.clone();
    let method = obj.get("method")?;
    let Some(method) = method.as_str() else {
        return Some(failure(id, INVALID_REQUEST, "method is not a string"));
    };
    let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));
    let result = match method {
        "initialize" => Ok(initialize(info, &params)),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({
            "tools": handler.tools().into_iter().map(|t| json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            })).collect::<Vec<_>>(),
        })),
        "tools/call" => call_tool(handler, &params, send),
        _ => Err((METHOD_NOT_FOUND, format!("no method {method:?}"))),
    };
    Some(match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err((code, msg)) => failure(id, code, &msg),
    })
}

fn initialize(info: &ServerInfo, params: &Value) -> Value {
    let asked = params.get("protocolVersion").and_then(Value::as_str);
    let version = asked
        .filter(|v| VERSIONS.contains(v))
        .unwrap_or(VERSIONS[VERSIONS.len() - 1]);
    json!({
        "protocolVersion": version,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": info.name, "version": info.version},
        "instructions": info.instructions,
    })
}

fn call_tool(
    handler: &mut impl Handler,
    params: &Value,
    send: &dyn Fn(&Value),
) -> Result<Value, (i64, String)> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or((INVALID_PARAMS, "tools/call without a tool name".to_string()))?;
    if !handler.tools().iter().any(|t| t.name == name) {
        return Err((INVALID_PARAMS, format!("no tool {name:?}")));
    }
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let cx = Call {
        token: params.pointer("/_meta/progressToken").cloned(),
        step: Default::default(),
        send,
    };
    // A tool that panics fails as a tool: the server, and what it holds
    // open, carry on.
    let called = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handler.call(name, args, &cx)
    }))
    .unwrap_or_else(|panic| {
        let what = panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        Err(anyhow::anyhow!("{name} panicked: {what}"))
    });
    Ok(match called {
        Ok(contents) => {
            let structured = contents.iter().find_map(|c| match c {
                Content::Json(v) if v.is_object() => Some(v.clone()),
                _ => None,
            });
            let mut result = json!({
                "content": contents.into_iter().map(content).collect::<Vec<_>>(),
                "isError": false,
            });
            if let Some(v) = structured {
                result["structuredContent"] = v;
            }
            result
        }
        Err(e) => json!({
            "content": [{"type": "text", "text": format!("{e:#}")}],
            "isError": true,
        }),
    })
}

fn content(c: Content) -> Value {
    match c {
        Content::Text(text) => json!({"type": "text", "text": text}),
        Content::Json(v) => json!({"type": "text", "text": format!("{v:#}")}),
        Content::Png(bytes) => json!({
            "type": "image",
            "data": base64::engine::general_purpose::STANDARD.encode(bytes),
            "mimeType": "image/png",
        }),
    }
}

fn failure(id: Value, code: i64, msg: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": msg}})
}
