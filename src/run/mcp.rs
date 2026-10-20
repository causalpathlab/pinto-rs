//! `pinto run`'s tools for `pinto mcp`: the methods and their flags, and
//! a fit set up as `pinto run` sets one up: the command built and checked
//! as pinto parses it, never over a run, run as a child `pinto` in the
//! project's records, as a background job unless asked otherwise (a fit
//! takes minutes to hours, and a call that waited would hold the server).

use super::form::{blamed, check, Kind, Method};
use super::script::{self, is_header, outputs, under};
use super::METHODS;
use crate::mcp::{parse, shown_path};
use legume_mcp::{Call, Content, Handler, ToolSpec};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::path::Path;

/// The run tools, and pinto's command line to read the methods from.
pub struct Tools {
    cli: clap::Command,
}

impl Default for Tools {
    fn default() -> Self {
        use clap::CommandFactory;
        let mut cli = crate::Cli::command();
        cli.build();
        Tools { cli }
    }
}

impl Handler for Tools {
    fn tools(&self) -> Vec<ToolSpec> {
        specs()
    }

    fn call(&mut self, name: &str, args: Value, cx: &Call) -> anyhow::Result<Vec<Content>> {
        match name {
            "run_methods" => self.methods(parse(args)?),
            "run_fit" => self.fit(parse(args)?, cx),
            _ => unreachable!("the server answers unknown tools itself"),
        }
    }
}

/// A method in a line: what it is, and the files it takes.
fn summary(name: &str, method: &Method) -> Value {
    json!({
        "method": name,
        "about": method.about,
        "coords": method.takes_coords(),
        "batches": method.takes_batches(),
    })
}

impl Tools {
    fn methods(&self, a: MethodsArgs) -> anyhow::Result<Vec<Content>> {
        let Some(name) = a.method else {
            let methods = METHODS
                .iter()
                .map(|&m| Ok(summary(m, &Method::new(&self.cli, m)?)))
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
        let mut out = summary(&name, &method);
        out["flags"] = flags.into();
        Ok(vec![Content::Json(out)])
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
        anyhow::ensure!(!a.out.trim().is_empty(), "give out, the output prefix");
        for (what, takes, files) in [
            ("coordinate", method.takes_coords(), &a.coords),
            ("batch", method.takes_batches(), &a.batches),
        ] {
            anyhow::ensure!(
                files.is_empty() || takes,
                "{} takes no {what} files",
                a.method
            );
            anyhow::ensure!(
                files.is_empty() || files.len() == a.data.len(),
                "{what} files: one per data file ({} for {})",
                files.len(),
                a.data.len()
            );
        }
        for (flag, value) in &a.flags {
            let long = flag.trim_start_matches('-');
            let field = method
                .fields
                .iter_mut()
                .find(|f| f.long == long)
                .ok_or_else(|| {
                    anyhow::anyhow!("{} has no flag --{long}: run_methods lists them", a.method)
                })?;
            field.value = match (&field.kind, value) {
                (Kind::Flag { .. }, Value::Bool(on)) => on.to_string(),
                (Kind::Flag { .. }, v) => {
                    anyhow::bail!("--{long} is a switch: true or false, not {v}")
                }
                (_, Value::Array(vs)) => vs.iter().map(text).collect::<Vec<_>>().join(","),
                (_, v) => text(v),
            };
        }
        // An `out` naming a folder, or a header, is named for the method,
        // as `pinto run` names it; `.` is this folder.
        let out = if a.out == "." {
            "./".into()
        } else {
            a.out.clone()
        };
        let out = if is_header(&out) {
            under(&out, &a.method)
        } else {
            out
        };
        // Paths from the root: the fit, and the records' replay of it from
        // the project, read the same files wherever the server started.
        let rooted = |paths: &[String]| -> Vec<String> {
            paths.iter().map(|p| shown_path(Path::new(p))).collect()
        };
        let (data, coords, batches) = (rooted(&a.data), rooted(&a.coords), rooted(&a.batches));
        let out = shown_path(Path::new(&out));
        let made = outputs(Path::new(""), &out);
        if let Some(there) = made.iter().find(|p| p.exists()) {
            anyhow::bail!(
                "{} exists: a run is not written over; give another out",
                shown_path(there)
            );
        }
        let argv = method.argv(&data, &coords, &batches, &out);
        check(&self.cli, &argv).map_err(|e| match blamed(&e, &method.fields) {
            Some(flag) => anyhow::anyhow!("pinto {} --{flag}: {e}", a.method),
            None => anyhow::anyhow!("pinto {}: {e}", a.method),
        })?;

        let manifest = made[0].clone();
        let mut reply = json!({ "command": format!("pinto {}", script::line(&argv)) });
        let summary = format!("{} fit of {}", a.method, a.data.join(", "));
        let mut cmd = legume_mcp::own_command(&argv)?;
        // The log level `pinto run` starts a fit with, so its progress
        // reaches job_status as it does the run screen.
        if std::env::var_os("RUST_LOG").is_none() {
            cmd.env("RUST_LOG", script::LOG_LEVEL);
        }
        let paths = vec![shown_path(&manifest)];
        if a.background {
            let job = cx.spawn_job(cmd, &[manifest], &summary)?;
            reply["writes"] = paths.into();
            reply["note"] =
                format!("job_wait (id {job}) or job_status until it ends, then view_open {out}")
                    .into();
            reply["job"] = job.into();
        } else {
            cx.run_making(cmd, &[manifest], &summary)?;
            reply["written"] = paths.into();
            reply["note"] = format!("view_open {out} to look at it").into();
        }
        Ok(vec![Content::Json(reply)])
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
    #[serde(default = "in_background")]
    background: bool,
}

fn in_background() -> bool {
    true
}

/// A flag's value as typed on the command line.
fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

fn specs() -> Vec<ToolSpec> {
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
            "Fit one method on data files, with the command pinto run would build: checked as \
             pinto parses it, then run as a child pinto and kept in the project's records. A fit \
             takes minutes to hours, so it runs as a background job: the call returns the job at \
             once, to follow with job_status / job_wait / job_cancel (background: false waits for \
             it, for a small data set). A run is never written over. When it ends, view_open its \
             out.",
            json!({
                "type": "object",
                "properties": {
                    "method": methods,
                    "data": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Expression data files (.zarr, .zarr.zip, .h5)"},
                    "coords": {"type": "array", "items": {"type": "string"}, "description": "Coordinate files, one per data file (barcode, x, y, …)"},
                    "batches": {"type": "array", "items": {"type": "string"}, "description": "Batch label files, one per data file"},
                    "out": {"type": "string", "description": "Output prefix (-o); ending in /, _, - or . names the run for its method (res/ → res/lc)"},
                    "flags": {"type": "object", "description": "The method's flags by name, as run_methods lists them (a switch takes true or false): {\"n-communities\": 12}"},
                    "background": {"type": "boolean", "description": "Run as a background job and return at once [default: true]; false waits for the fit"},
                },
                "required": ["method", "data", "out"],
                "additionalProperties": false,
            }),
        ),
    ]
}
