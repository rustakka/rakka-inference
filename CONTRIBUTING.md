# Contributing to rakka-inference

Welcome. This guide covers what we expect of contributors and how the
release pipeline reacts to your work.

## Quick start

```sh
git clone <fork>
cd rakka-inference
cargo xtask verify         # runs the same gate as CI
```

`cargo xtask verify` is the single source of truth for "is my change
ready to ship?". It runs:

1. `cargo build --workspace`
2. `cargo test --workspace --quiet`
3. `cargo clippy --workspace --all-targets -- -D warnings`
4. `cargo build -p inference --no-default-features --features remote-only`
5. `cargo xtask audit --check` (vs `docs/reports/audit-baseline.json`)
6. The **remote-only invariant** — the dep tree of an `inference
   --features remote-only` build must contain zero `cudarc`,
   `rakka-cuda`, `candle`, or `pyo3` references.

If any of these fail locally, CI will fail too.

## Conventional Commits — the release lever

The version-bump workflow reads your commit subjects to decide
whether to tag a new release. **Your commit message changes what
ships.**

| Subject prefix         | Effect                          |
|------------------------|---------------------------------|
| `feat: ...`            | next tag is a **minor** bump.   |
| `fix: ...` / `perf: ...` / `revert: ...` | next tag is a **patch** bump. |
| `BREAKING CHANGE` body or `!:` after type | next tag is a **major** bump. |
| `chore:` / `docs:` / `ci:` / `test:` / `refactor:` / `style:` / `build:` only | no release. |

Add `Release-As: x.y.z` in the footer to override (e.g. cutting
`1.0.0-rc.1` from a stretch of `chore:` PRs).

Examples:

```text
feat(remote-core): add cluster-distributed strict rate limiter

A new RateLimiterActor variant runs as a cluster singleton for
deployments where 429 budgets are hard caps.
```

```text
fix(openai): honor `Retry-After` header on 503 (not just 429)

Some Azure deployments return 503 with `Retry-After`; the retry
engine was using the policy delay instead.
```

```text
refactor(runtime): rename `Worker` -> `WorkerActor` for clarity

BREAKING CHANGE: `inference_runtime::Worker` is now `WorkerActor`.
```

## What the workspace looks like

18 crates. See the root [`README.md`](README.md) and the
[feature matrix](docs/feature-matrix.md) for the layered story.

The architectural invariant: `inference --features remote-only`
compiles **zero** GPU dependencies. Anywhere you add a
`rakka-cuda`-bearing dependency, gate it behind a feature so this
invariant keeps holding.

## Developer surface (`cargo xtask`)

```text
build               cargo build across the documented feature matrix
test                cargo test across the documented feature matrix
remote-only         build inference-cli with no GPU/Python deps
verify              1.0-rc gate (build + test + clippy + audit + remote-only)
audit [--check] [--json <out>]
                    count anti-pattern sentinels per crate
bump <patch|minor|major|--pre <id>|--set <ver>>
                    bump workspace version + internal pins, refresh Cargo.lock
release-checklist   list publishable vs gated crates
help                print this help
```

`cargo xtask audit` tracks `unwrap` / `expect` / `panic` / `todo` /
`unimplemented` / `Box<dyn Any>` / `println` / `eprintln` / `dbg` and
stub markers per crate. CI fails on regression versus
`docs/reports/audit-baseline.json`. If you intentionally add an
allowed instance, regenerate the baseline:

```sh
cargo xtask audit --json docs/reports/audit-baseline.json
git add docs/reports/audit-baseline.json
git commit -m "chore(audit): refresh baseline after <reason>"
```

## Adding a new runtime backend

1. Write a crate that depends only on `inference-core` (and
   `inference-remote-core` if you're adding a remote provider, or
   `rakka-cuda` if you're adding a local GPU runtime).
2. Implement `inference_core::ModelRunner`.
3. Add the crate to `Cargo.toml` and to the rollup
   (`crates/inference/Cargo.toml`) behind a feature flag.
4. Slot it into the publish dep-order list in `RELEASING.md`.
5. Add a per-crate `README.md` following the pattern of the existing
   ones (value-first opening, build profiles table, code sample).
6. Land via `feat:` commit.

The `inference-core` README has a copy-paste skeleton.

## Code style

- `cargo fmt --all` — runs in CI as a hard gate.
- `cargo clippy --workspace --all-targets -- -D warnings` — runs in
  CI as a hard gate.
- No `unwrap()` in non-test code; use `?` or surface a typed error.
- No `panic!`/`todo!`/`unimplemented!` in non-test code; prefer a
  documented `InferenceError::Internal("...")` so callers see why.
- Keep `inference-core` free of `tokio` / `rakka` / `rakka-cuda` /
  `pyo3` / HTTP clients. The dep budget is what makes
  `remote-only` work.

## Filing issues / PRs

Pull requests run the full CI matrix and `cargo-semver-checks` on
the publishable subset. If you're touching a public API on a
publishable crate (`inference-core`, `inference-remote-core`,
`inference-runtime-{openai,anthropic,gemini,litellm}`), the
semver-checks output appears as a PR comment. At `0.x` it's
warn-only; once we hit `1.0`, breaking changes will hard-fail.

## License

By contributing, you agree your changes ship under the workspace
license (Apache-2.0).
