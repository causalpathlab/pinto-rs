//! `pinto run`'s tools for `pinto mcp`: the methods and their flags, and
//! a fit started as `pinto run` starts one: the exact command, checked as
//! pinto parses it, run as a child `pinto` in the project's records, in
//! the background when the agent asks (a fit takes minutes to hours).

use super::tui::form::{check, Kind, Method};
use super::tui::script;
use super::tui::METHODS;
use legume_mcp::{Call, Content, ToolSpec};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::path::Path;

/// The run tools, and pinto's command line to read the methods from.
pub struct Tools {
    cli: clap::Command,
}

impl Tools {
    /// `cli` built (`Command::build`), so its subcommands are complete.
    pub fn new(cli: clap::Command) -> Self {
        Tools { cli }
    }

    pub fn call(&mut self, name: &str, args: Value, cx: &Call) -> anyhow::Result<Vec<Content>> {
        match name {
            "run_methods" => self.methods(parse(args)?),
            "run_fit" => self.fit(parse(args)?, cx),
            _ => unreachable!("the server answers unknown tools itself"),
        }
    }

    fn methods(&self, a: MethodsArgs) -> anyhow::Result<Vec<Content>> {
        let Some(name) = a.method else {
            let methods = METHODS
                .iter()
                .map(|&m| {
                    let method = Method::new(&self.cli, m)?;
                    Ok(json!({
                        "method": m,
                        "about": method.about,
                        "coords": method.takes_coords(),
                        "batches": method.takes_batches(),
                    }))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            return Ok(vec![Content::Json(json!({ "methods": methods }))]);
        };
        let method = self.method(&name)?;
        let flags: Vec<Value> = method
            .fields
            .iter()
            .map(|f| {
                let mut flag = json!({
                    "flag": format!("--{}", f.long),
                    "help": f.help,
                    "default": f.default,
                });
                match &f.kind {
                    Kind::Flag { .. } => flag["switch"] = true.into(),
                    Kind::Choice(values) => {
                        flag["choices"] = values.iter().filter(|v| !v.is_empty()).cloned().collect()
                    }
                    Kind::Text => {}
                }
                if f.required {
                    flag["required"] = true.into();
                }
                if f.advanced {
                    flag["advanced"] = true.into();
                }
                flag
            })
            .collect();
        Ok(vec![Content::Json(json!({
            "method": name,
            "about": method.about,
            "coords": method.takes_coords(),
            "batches": method.takes_batches(),
            "flags": flags,
        }))])
    }

    fn method(&self, name: &str) -> anyhow::Result<Method> {
        anyhow::ensure!(
            METHODS.contains(&name),
            "method {name:?}: one of {}",
            METHODS.join(", ")
        );
        Method::new(&self.cli, name)
    }

    fn fit(&self, a: FitArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
        let mut method = self.method(&a.method)?;
        anyhow::ensure!(!a.data.is_empty(), "give the data files");
        anyhow::ensure!(
            a.coords.is_empty() || method.takes_coords(),
            "{} takes no coordinate files",
            a.method
        );
        anyhow::ensure!(
            a.batches.is_empty() || method.takes_batches(),
            "{} takes no batch files",
            a.method
        );
        for (flag, value) in &a.flags {
            let long = flag.trim_start_matches('-');
            let field = method
                .fields
                .iter_mut()
                .find(|f| f.long == long)
                .ok_or_else(|| {
                    anyhow::anyhow!("{} has no flag --{long}: run_methods lists them", a.method)
                })?;
            field.value = match value {
                Value::String(s) => s.clone(),
                Value::Array(vs) => vs.iter().map(text).collect::<Vec<_>>().join(","),
                v => text(v),
            };
        }
        // `--out res/` names the run for its method, as pinto does.
        let out = match a.out.strip_suffix('/') {
            Some(_) => format!("{}{}", a.out, a.method),
            None => a.out.clone(),
        };
        let manifest = format!("{out}.pinto.json");
        anyhow::ensure!(
            !Path::new(&manifest).exists(),
            "{} exists: a run is not written over; give another out",
            shown_path(&manifest)
        );
        let argv = method.argv(&a.data, &a.coords, &a.batches, &out);
        check(&self.cli, &argv).map_err(|e| anyhow::anyhow!("pinto {}: {e}", a.method))?;

        let command = format!("pinto {}", script::line(&argv));
        let summary = format!("{} fit of {}", a.method, a.data.join(", "));
        let written = vec![shown_path(&manifest)];
        let cmd = legume_mcp::own_command(&argv)?;
        let outputs = [manifest.clone().into()];
        if a.background {
            let job = cx.spawn_job(cmd, &outputs, &summary)?;
            return Ok(vec![Content::Json(json!({
                "job": job,
                "written": written,
                "command": command,
                "note": format!("job_wait (id {job}) or job_status until it ends, then view_open {out}"),
            }))]);
        }
        let log = cx.run_making(cmd, &outputs, &summary)?;
        Ok(vec![Content::Json(json!({
            "written": written,
            "command": command,
            "log": tail(&log, 20),
            "note": format!("view_open {out} to look at it"),
        }))])
    }
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct MethodsArgs {
    method: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FitArgs {
    method: String,
    data: Vec<String>,
    #[serde(default)]
    coords: Vec<String>,
    #[serde(default)]
    batches: Vec<String>,
    out: String,
    #[serde(default)]
    flags: Map<String, Value>,
    #[serde(default)]
    background: bool,
}

/// The arguments of a call, or what is wrong with them.
fn parse<T: serde::de::DeserializeOwned>(args: Value) -> anyhow::Result<T> {
    serde_json::from_value(args).map_err(|e| anyhow::anyhow!("arguments: {e}"))
}

/// A flag's value as typed on the command line.
fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

/// The last `n` lines of `log`.
fn tail(log: &str, n: usize) -> String {
    let lines: Vec<&str> = log.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// `p` as results give paths: from the root.
fn shown_path(p: &str) -> String {
    std::path::absolute(p).map_or_else(|_| p.to_string(), |a| a.to_string_lossy().into_owned())
}

pub fn specs() -> Vec<ToolSpec> {
    let methods =
        json!({"enum": METHODS, "description": "lc (link community, recommended), cage, or dsvd"});
    vec![
        ToolSpec::new(
            "run_methods",
            "pinto's fitting methods (what pinto run sets up); with a method, its flags: \
             name, help, default, choices, and whether it is a switch.",
            json!({
                "type": "object",
                "properties": { "method": methods },
                "additionalProperties": false,
            }),
        ),
        ToolSpec::new(
            "run_fit",
            "Fit one method on data files, as pinto run would: the exact command is checked \
             as pinto parses it, then run as a child pinto and kept in the project's records. \
             A fit takes minutes to hours: with background, it returns a job at once (follow it \
             with job_status / job_wait / job_cancel). A run is never written over. When it \
             ends, view_open its out.",
            json!({
                "type": "object",
                "properties": {
                    "method": methods,
                    "data": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Expression data files (.zarr, .zarr.zip, .h5)"},
                    "coords": {"type": "array", "items": {"type": "string"}, "description": "Coordinate files, one per data file (barcode, x, y, …)"},
                    "batches": {"type": "array", "items": {"type": "string"}, "description": "Batch label files, one per data file"},
                    "out": {"type": "string", "description": "Output prefix (-o); ending in / names the run for its method (res/ → res/lc)"},
                    "flags": {"type": "object", "description": "The method's flags by name (run_methods lists them): {\"n-communities\": 12, \"verbose\": true}"},
                    "background": {"type": "boolean", "description": "Return a job at once instead of waiting for the fit [default: false]"},
                },
                "required": ["method", "data", "out"],
                "additionalProperties": false,
            }),
        ),
    ]
}
