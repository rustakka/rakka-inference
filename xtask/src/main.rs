//! `xtask` — developer tooling for the rakka-inference workspace.
//!
//! Modeled on the rakka workspace's xtask. Subcommands:
//!
//! - `build`           — `cargo build` across the documented feature matrix.
//! - `test`            — `cargo test` across the documented feature matrix.
//! - `remote-only`     — build the `inference-cli` binary with no GPU deps.
//! - `verify`          — the 1.0-rc gate (build + test + clippy + audit + remote-only).
//! - `audit`           — count anti-pattern sentinels per crate, optionally vs baseline.
//! - `bump`            — bump workspace version + internal path-dep pins; refresh Cargo.lock.
//! - `release-checklist` — print which crates are publishable vs gated.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "help".into());
    match cmd.as_str() {
        "build" => build_matrix(),
        "test" => test_matrix(),
        "remote-only" => remote_only_build(),
        "verify" => verify(),
        "audit" => audit(args.collect()),
        "bump" => bump(args.collect()),
        "release-checklist" => release_checklist(),
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(anyhow!("unknown xtask subcommand: {other}")),
    }
}

fn print_help() {
    println!("rakka-inference xtask");
    println!();
    println!("USAGE:");
    println!("  cargo xtask <subcommand>");
    println!();
    println!("SUBCOMMANDS:");
    println!("  build               cargo build across the documented feature matrix");
    println!("  test                cargo test across the documented feature matrix");
    println!("  remote-only         build inference-cli with no GPU/Python deps");
    println!("  verify              1.0-rc gate (build + test + clippy + audit + remote-only)");
    println!("  audit [--check] [--json <out>]");
    println!("                      count anti-pattern sentinels per crate");
    println!("  bump <patch|minor|major|--pre <id>|--set <ver>>");
    println!("                      bump workspace version + internal pins, refresh Cargo.lock");
    println!("  release-checklist   list publishable vs gated crates");
    println!("  help                print this help");
}

// --- existing matrix subcommands -----------------------------------------

fn build_matrix() -> Result<()> {
    cargo(&["build", "--workspace"])?;
    cargo(&["build", "-p", "inference", "--features", "remote-only"])?;
    cargo(&["build", "-p", "inference", "--features", "candle,openai,anthropic,pipeline"])?;
    Ok(())
}

fn test_matrix() -> Result<()> {
    cargo(&["test", "--workspace"])?;
    cargo(&["test", "-p", "inference-remote-core"])?;
    Ok(())
}

fn remote_only_build() -> Result<()> {
    cargo(&[
        "build",
        "-p",
        "inference-cli",
        "--no-default-features",
        "--features",
        "remote-only",
    ])
}

fn cargo(args: &[&str]) -> Result<()> {
    let status = Command::new(env!("CARGO"))
        .args(args)
        .status()
        .with_context(|| format!("spawning `cargo {}`", args.join(" ")))?;
    if !status.success() {
        bail!("`cargo {}` failed: {status}", args.join(" "));
    }
    Ok(())
}

// --- verify (1.0-rc gate) ------------------------------------------------

fn verify() -> Result<()> {
    let cargo_bin = env!("CARGO");
    let steps: Vec<(&str, &[&str])> = vec![
        ("cargo build --workspace", &["build", "--workspace"]),
        ("cargo test --workspace --quiet", &["test", "--workspace", "--quiet"]),
        (
            "cargo clippy --workspace --all-targets -- -D warnings",
            &["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"],
        ),
        (
            "cargo build -p inference --no-default-features --features remote-only",
            &[
                "build",
                "-p",
                "inference",
                "--no-default-features",
                "--features",
                "remote-only",
            ],
        ),
    ];
    for (label, args) in &steps {
        println!("==> {label}");
        let status = Command::new(cargo_bin)
            .args(args.iter())
            .status()
            .with_context(|| format!("spawning `{label}`"))?;
        if !status.success() {
            return Err(anyhow!("{label} failed: {status}"));
        }
    }
    println!("==> cargo xtask audit --check");
    audit(vec!["--check".into()])?;

    // Remote-only invariant: cargo tree must show zero GPU deps.
    println!("==> remote-only invariant: cargo tree | grep -Ec 'cudarc|rakka-accel|candle|pyo3' == 0");
    let output = Command::new(cargo_bin)
        .args([
            "tree",
            "-p",
            "inference",
            "--no-default-features",
            "--features",
            "remote-only",
        ])
        .output()
        .context("spawning cargo tree")?;
    if !output.status.success() {
        return Err(anyhow!("cargo tree failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let tree = String::from_utf8_lossy(&output.stdout);
    let leaks: Vec<&str> = tree
        .lines()
        .filter(|l| {
            l.contains("cudarc") || l.contains("rakka-accel") || l.contains("candle") || l.contains("pyo3")
        })
        .collect();
    if !leaks.is_empty() {
        eprintln!("\nremote-only invariant violated — GPU deps in dep graph:");
        for l in &leaks {
            eprintln!("  {l}");
        }
        return Err(anyhow!("remote-only build leaked {} GPU dep line(s)", leaks.len()));
    }

    println!("\nverify: OK");
    Ok(())
}

// --- bump ----------------------------------------------------------------

#[derive(Debug)]
enum BumpKind {
    Patch,
    Minor,
    Major,
    Pre(String),
}

fn bump(args: Vec<String>) -> Result<()> {
    let mut iter = args.into_iter();
    let arg = iter
        .next()
        .ok_or_else(|| anyhow!("usage: bump <patch|minor|major> | bump --pre <id> | bump --set <version>"))?;
    let cargo_toml = Path::new("Cargo.toml");
    let current = read_workspace_version(cargo_toml)?;
    let next = match arg.as_str() {
        "patch" => semver_bump(&current, BumpKind::Patch)?,
        "minor" => semver_bump(&current, BumpKind::Minor)?,
        "major" => semver_bump(&current, BumpKind::Major)?,
        "--pre" => {
            let id = iter.next().ok_or_else(|| anyhow!("--pre requires <id>"))?;
            semver_bump(&current, BumpKind::Pre(id))?
        }
        "--set" => iter.next().ok_or_else(|| anyhow!("--set requires <version>"))?,
        other => return Err(anyhow!("unknown bump arg: {other}")),
    };
    println!("{} -> {}", current, next);
    write_workspace_version(cargo_toml, &next)?;
    write_workspace_deps_versions(cargo_toml, &current, &next)?;

    // Refresh Cargo.lock — `cargo update --workspace` is enough.
    let _ = Command::new(env!("CARGO")).args(["update", "--workspace"]).status();

    println!("RAKKA_INFERENCE_NEW_VERSION={next}");
    Ok(())
}

fn semver_bump(current: &str, kind: BumpKind) -> Result<String> {
    let (core, _pre) = match current.split_once('-') {
        Some((c, p)) => (c, Some(p)),
        None => (current, None),
    };
    let parts: Vec<&str> = core.split('.').collect();
    if parts.len() != 3 {
        return Err(anyhow!("version `{current}` is not MAJOR.MINOR.PATCH"));
    }
    let mut major: u64 = parts[0].parse().context("major")?;
    let mut minor: u64 = parts[1].parse().context("minor")?;
    let mut patch: u64 = parts[2].parse().context("patch")?;
    let next = match kind {
        BumpKind::Patch => {
            patch += 1;
            format!("{major}.{minor}.{patch}")
        }
        BumpKind::Minor => {
            minor += 1;
            patch = 0;
            format!("{major}.{minor}.{patch}")
        }
        BumpKind::Major => {
            major += 1;
            minor = 0;
            patch = 0;
            format!("{major}.{minor}.{patch}")
        }
        BumpKind::Pre(id) => format!("{major}.{minor}.{patch}-{id}"),
    };
    Ok(next)
}

fn read_workspace_version(path: &Path) -> Result<String> {
    let text = fs::read_to_string(path)?;
    let block_start = text
        .find("[workspace.package]")
        .ok_or_else(|| anyhow!("no [workspace.package] block in {}", path.display()))?;
    let block_end = text[block_start..]
        .find("\n[")
        .map(|i| block_start + i)
        .unwrap_or(text.len());
    let block = &text[block_start..block_end];
    for line in block.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("version") {
            let after_eq = rest.split_once('=').map(|(_, v)| v.trim()).unwrap_or("");
            let value = after_eq.trim_matches('"').trim_matches('\'');
            return Ok(value.to_string());
        }
    }
    Err(anyhow!("no version key in [workspace.package]"))
}

fn write_workspace_version(path: &Path, version: &str) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let block_start =
        text.find("[workspace.package]").ok_or_else(|| anyhow!("no [workspace.package] block"))?;
    let after_block = &text[block_start..];
    let local_idx = after_block.find("version").ok_or_else(|| anyhow!("no version line"))?;
    let abs = block_start + local_idx;
    let line_end = text[abs..].find('\n').map(|i| abs + i).unwrap_or(text.len());
    let new_line = format!("version       = \"{version}\"");
    let mut out = String::with_capacity(text.len() + new_line.len());
    out.push_str(&text[..abs]);
    out.push_str(&new_line);
    out.push_str(&text[line_end..]);
    fs::write(path, out)?;
    Ok(())
}

/// Bumps the `version = "<prev>"` pin on every internal path-dep line
/// inside `[workspace.dependencies]`. The release pipeline rejects a
/// crate whose internal deps still resolve to an older version, so this
/// must move in lockstep with the workspace version.
fn write_workspace_deps_versions(path: &Path, prev: &str, next: &str) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let block_start = match text.find("[workspace.dependencies]") {
        Some(i) => i,
        None => return Ok(()),
    };
    let after = &text[block_start + "[workspace.dependencies]".len()..];
    let block_len = after.find("\n[").map(|i| i + 1).unwrap_or(after.len());
    let head = &text[..block_start];
    let block = &text[block_start..block_start + "[workspace.dependencies]".len() + block_len];
    let tail = &text[block_start + "[workspace.dependencies]".len() + block_len..];

    let needle = format!("version = \"{prev}\"");
    let replacement = format!("version = \"{next}\"");
    let mut new_block = String::with_capacity(block.len());
    for line in block.split_inclusive('\n') {
        if line.contains("path = \"crates/") && line.contains(&needle) {
            new_block.push_str(&line.replace(&needle, &replacement));
        } else {
            new_block.push_str(line);
        }
    }
    let mut out = String::with_capacity(text.len());
    out.push_str(head);
    out.push_str(&new_block);
    out.push_str(tail);
    fs::write(path, out)?;
    Ok(())
}

// --- release-checklist ---------------------------------------------------

fn release_checklist() -> Result<()> {
    let cargo_toml = Path::new("Cargo.toml");
    let current = read_workspace_version(cargo_toml)?;
    println!("rakka-inference release checklist (workspace version: {current})\n");

    let publishable_now: &[&str] = &[
        "inference-core",
        "inference-remote-core",
        "inference-runtime-openai",
        "inference-runtime-anthropic",
        "inference-runtime-gemini",
        "inference-runtime-litellm",
    ];
    let gated_until_upstream: &[(&str, &str)] = &[
        ("inference-runtime", "depends on rakka-* crates which are not yet on crates.io"),
        ("inference-python-bridge", "depends on rakka-accel crates (when feature `python` is on, also pyo3)"),
        ("inference-runtime-vllm", "depends on rakka-accel + python-bridge"),
        ("inference-runtime-tensorrt", "depends on rakka-accel"),
        ("inference-runtime-ort", "depends on rakka-accel"),
        ("inference-runtime-candle", "depends on rakka-accel"),
        ("inference-runtime-cudarc", "depends on rakka-accel"),
        ("inference-runtime-mistralrs", "depends on rakka-accel"),
        ("inference-pipeline", "depends on rakka-streams; promote when rakka publishes"),
        ("inference-testkit", "depends on rakka-testkit"),
        ("inference-cli", "depends on rakka + inference-runtime"),
        ("inference", "rollup; promote after every member it re-exports is publishable"),
    ];

    println!("Publishable now ({}):", publishable_now.len());
    for c in publishable_now {
        println!("  - {c}");
    }
    println!();
    println!("Gated until upstream publishes ({}):", gated_until_upstream.len());
    for (c, reason) in gated_until_upstream {
        println!("  - {c}  — {reason}");
    }
    println!();
    println!(
        "release.yml uses RAKKA_INFERENCE_PUBLISH_ALLOWLIST (repo var) to control the\n\
         publish set. Default = the 'publishable now' list above. Once `rakka` and\n\
         `rakka-accel` ship to crates.io, set RAKKA_INFERENCE_PUBLISH_ALLOWLIST=\"\" to\n\
         publish the full workspace in dep order."
    );
    Ok(())
}

// --- audit ---------------------------------------------------------------

#[derive(Default, Clone)]
struct CrateCounts {
    files: usize,
    loc: usize,
    unwrap_used: usize,
    expect_used: usize,
    panic_macro: usize,
    todo_macro: usize,
    unimplemented_macro: usize,
    box_dyn_any: usize,
    placeholder_marker: usize,
    stub_comment: usize,
    placeholder_comment: usize,
    println_macro: usize,
    eprintln_macro: usize,
    dbg_macro: usize,
}

impl CrateCounts {
    fn add(&mut self, other: &CrateCounts) {
        self.files += other.files;
        self.loc += other.loc;
        self.unwrap_used += other.unwrap_used;
        self.expect_used += other.expect_used;
        self.panic_macro += other.panic_macro;
        self.todo_macro += other.todo_macro;
        self.unimplemented_macro += other.unimplemented_macro;
        self.box_dyn_any += other.box_dyn_any;
        self.placeholder_marker += other.placeholder_marker;
        self.stub_comment += other.stub_comment;
        self.placeholder_comment += other.placeholder_comment;
        self.println_macro += other.println_macro;
        self.eprintln_macro += other.eprintln_macro;
        self.dbg_macro += other.dbg_macro;
    }
}

fn audit(args: Vec<String>) -> Result<()> {
    let mut check_mode = false;
    let mut json_out: Option<PathBuf> = None;
    let mut iter = args.into_iter();
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--" => continue,
            "--check" => check_mode = true,
            "--json" => {
                json_out = Some(PathBuf::from(
                    iter.next().ok_or_else(|| anyhow!("--json requires a path argument"))?,
                ));
            }
            other => return Err(anyhow!("unknown audit flag: {other}")),
        }
    }

    let crates_dir = Path::new("crates");
    if !crates_dir.is_dir() {
        return Err(anyhow!("no crates/ directory found (cwd must be workspace root)"));
    }

    let mut per_crate: BTreeMap<String, CrateCounts> = BTreeMap::new();
    for (name, dir) in collect_crate_dirs(crates_dir)? {
        let counts = audit_crate(&dir)?;
        per_crate.insert(name, counts);
    }

    let mut total = CrateCounts::default();
    for c in per_crate.values() {
        total.add(c);
    }

    print_audit_table(&per_crate, &total);

    if let Some(path) = json_out.as_deref() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, audit_json(&per_crate, &total))?;
        println!("\nwrote {}", path.display());
    }

    if check_mode {
        let baseline_path = Path::new("docs/reports/audit-baseline.json");
        if !baseline_path.exists() {
            return Err(anyhow!(
                "--check requires {}; run `cargo xtask audit --json {}` first",
                baseline_path.display(),
                baseline_path.display()
            ));
        }
        let baseline_text = fs::read_to_string(baseline_path)?;
        let mut regressions = Vec::new();
        for (name, counts) in &per_crate {
            let baseline = parse_json_crate(&baseline_text, name).unwrap_or_default();
            check_metric(&mut regressions, name, "unwrap_used", counts.unwrap_used, baseline.unwrap_used);
            check_metric(&mut regressions, name, "expect_used", counts.expect_used, baseline.expect_used);
            check_metric(&mut regressions, name, "panic_macro", counts.panic_macro, baseline.panic_macro);
            check_metric(&mut regressions, name, "todo_macro", counts.todo_macro, baseline.todo_macro);
            check_metric(
                &mut regressions,
                name,
                "unimplemented_macro",
                counts.unimplemented_macro,
                baseline.unimplemented_macro,
            );
            check_metric(&mut regressions, name, "box_dyn_any", counts.box_dyn_any, baseline.box_dyn_any);
            check_metric(
                &mut regressions,
                name,
                "placeholder_marker",
                counts.placeholder_marker,
                baseline.placeholder_marker,
            );
            check_metric(&mut regressions, name, "stub_comment", counts.stub_comment, baseline.stub_comment);
            check_metric(
                &mut regressions,
                name,
                "placeholder_comment",
                counts.placeholder_comment,
                baseline.placeholder_comment,
            );
            check_metric(&mut regressions, name, "println_macro", counts.println_macro, baseline.println_macro);
            check_metric(&mut regressions, name, "eprintln_macro", counts.eprintln_macro, baseline.eprintln_macro);
            check_metric(&mut regressions, name, "dbg_macro", counts.dbg_macro, baseline.dbg_macro);
        }
        if !regressions.is_empty() {
            eprintln!("\naudit regressions vs baseline:");
            for r in &regressions {
                eprintln!("  {r}");
            }
            return Err(anyhow!("{} audit regression(s)", regressions.len()));
        }
        println!("\naudit: no regressions vs {}", baseline_path.display());
    }

    Ok(())
}

fn check_metric(out: &mut Vec<String>, crate_name: &str, metric: &str, current: usize, baseline: usize) {
    if current > baseline {
        out.push(format!("{crate_name}: {metric} {baseline} -> {current} (+{})", current - baseline));
    }
}

fn collect_crate_dirs(crates_dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(crates_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        out.push((name, path));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn audit_crate(crate_dir: &Path) -> Result<CrateCounts> {
    let src = crate_dir.join("src");
    if !src.is_dir() {
        return Ok(CrateCounts::default());
    }
    let mut counts = CrateCounts::default();
    walk_rs(&src, &mut counts)?;
    Ok(counts)
}

fn walk_rs(dir: &Path, counts: &mut CrateCounts) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_rs(&path, counts)?;
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let text = fs::read_to_string(&path)?;
        scan_file(&text, counts);
    }
    Ok(())
}

fn scan_file(text: &str, counts: &mut CrateCounts) {
    counts.files += 1;
    let mut in_test_module = false;
    let mut depth = 0i32;
    let mut test_depth_start = i32::MAX;
    for raw_line in text.lines() {
        counts.loc += 1;
        let line = raw_line.trim_start();
        let is_comment = line.starts_with("//");

        if line.starts_with("#[cfg(test)]") {
            test_depth_start = depth;
            in_test_module = true;
        }
        for ch in raw_line.chars() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if in_test_module && depth <= test_depth_start {
                        in_test_module = false;
                        test_depth_start = i32::MAX;
                    }
                }
                _ => {}
            }
        }

        if !is_comment && !in_test_module {
            if line.contains(".unwrap(") {
                counts.unwrap_used += 1;
            }
            if line.contains(".expect(") {
                counts.expect_used += 1;
            }
            if contains_macro(line, "panic!") {
                counts.panic_macro += 1;
            }
            if contains_macro(line, "todo!") {
                counts.todo_macro += 1;
            }
            if contains_macro(line, "unimplemented!") {
                counts.unimplemented_macro += 1;
            }
            if line.contains("Box<dyn Any") {
                counts.box_dyn_any += 1;
            }
            if contains_macro(line, "println!") {
                counts.println_macro += 1;
            }
            if contains_macro(line, "eprintln!") {
                counts.eprintln_macro += 1;
            }
            if contains_macro(line, "dbg!") {
                counts.dbg_macro += 1;
            }
        }

        if line.contains("__placeholder__") {
            counts.placeholder_marker += 1;
        }
        let lower = line.to_ascii_lowercase();
        if is_comment && lower.contains("// stub") {
            counts.stub_comment += 1;
        }
        if is_comment && lower.contains("// placeholder") {
            counts.placeholder_comment += 1;
        }
    }
}

fn contains_macro(line: &str, needle: &str) -> bool {
    if let Some(idx) = line.find(needle) {
        let before = line[..idx].chars().last();
        match before {
            None => true,
            Some(c) => !c.is_alphanumeric() && c != '_',
        }
    } else {
        false
    }
}

fn print_audit_table(per_crate: &BTreeMap<String, CrateCounts>, total: &CrateCounts) {
    let header = [
        "crate", "files", "LOC", "unwrap", "expect", "panic", "todo", "unimpl", "Box<Any", "PHldr", "stub//",
        "phldr//", "println", "eprint", "dbg",
    ];
    println!(
        "{:<32} {:>5} {:>6} {:>6} {:>6} {:>5} {:>4} {:>6} {:>7} {:>5} {:>6} {:>7} {:>7} {:>6} {:>4}",
        header[0],
        header[1],
        header[2],
        header[3],
        header[4],
        header[5],
        header[6],
        header[7],
        header[8],
        header[9],
        header[10],
        header[11],
        header[12],
        header[13],
        header[14],
    );
    for (name, c) in per_crate {
        println!(
            "{:<32} {:>5} {:>6} {:>6} {:>6} {:>5} {:>4} {:>6} {:>7} {:>5} {:>6} {:>7} {:>7} {:>6} {:>4}",
            name,
            c.files,
            c.loc,
            c.unwrap_used,
            c.expect_used,
            c.panic_macro,
            c.todo_macro,
            c.unimplemented_macro,
            c.box_dyn_any,
            c.placeholder_marker,
            c.stub_comment,
            c.placeholder_comment,
            c.println_macro,
            c.eprintln_macro,
            c.dbg_macro,
        );
    }
    println!(
        "{:<32} {:>5} {:>6} {:>6} {:>6} {:>5} {:>4} {:>6} {:>7} {:>5} {:>6} {:>7} {:>7} {:>6} {:>4}",
        "TOTAL",
        total.files,
        total.loc,
        total.unwrap_used,
        total.expect_used,
        total.panic_macro,
        total.todo_macro,
        total.unimplemented_macro,
        total.box_dyn_any,
        total.placeholder_marker,
        total.stub_comment,
        total.placeholder_comment,
        total.println_macro,
        total.eprintln_macro,
        total.dbg_macro,
    );
}

fn audit_json(per_crate: &BTreeMap<String, CrateCounts>, total: &CrateCounts) -> String {
    let mut s = String::from("{\n  \"crates\": {");
    for (i, (name, c)) in per_crate.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\n    \"{name}\": {{"));
        write_counts(&mut s, c, "      ");
        s.push_str("\n    }");
    }
    s.push_str("\n  },\n  \"total\": {");
    write_counts(&mut s, total, "    ");
    s.push_str("\n  }\n}\n");
    s
}

fn write_counts(s: &mut String, c: &CrateCounts, indent: &str) {
    let kvs = [
        ("files", c.files),
        ("loc", c.loc),
        ("unwrap_used", c.unwrap_used),
        ("expect_used", c.expect_used),
        ("panic_macro", c.panic_macro),
        ("todo_macro", c.todo_macro),
        ("unimplemented_macro", c.unimplemented_macro),
        ("box_dyn_any", c.box_dyn_any),
        ("placeholder_marker", c.placeholder_marker),
        ("stub_comment", c.stub_comment),
        ("placeholder_comment", c.placeholder_comment),
        ("println_macro", c.println_macro),
        ("eprintln_macro", c.eprintln_macro),
        ("dbg_macro", c.dbg_macro),
    ];
    for (i, (k, v)) in kvs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\n{indent}\"{k}\": {v}"));
    }
}

fn parse_json_crate(text: &str, crate_name: &str) -> Option<CrateCounts> {
    let key = format!("\"{crate_name}\":");
    let start = text.find(&key)?;
    let after = &text[start + key.len()..];
    let open = after.find('{')?;
    let close = after[open..].find('}')?;
    let body = &after[open + 1..open + close];
    let mut c = CrateCounts::default();
    for token in body.split(',') {
        let token = token.trim();
        if let Some((k, v)) = token.split_once(':') {
            let k = k.trim().trim_matches('"');
            let v: usize = v.trim().parse().ok()?;
            match k {
                "files" => c.files = v,
                "loc" => c.loc = v,
                "unwrap_used" => c.unwrap_used = v,
                "expect_used" => c.expect_used = v,
                "panic_macro" => c.panic_macro = v,
                "todo_macro" => c.todo_macro = v,
                "unimplemented_macro" => c.unimplemented_macro = v,
                "box_dyn_any" => c.box_dyn_any = v,
                "placeholder_marker" => c.placeholder_marker = v,
                "stub_comment" => c.stub_comment = v,
                "placeholder_comment" => c.placeholder_comment = v,
                "println_macro" => c.println_macro = v,
                "eprintln_macro" => c.eprintln_macro = v,
                "dbg_macro" => c.dbg_macro = v,
                _ => {}
            }
        }
    }
    Some(c)
}
