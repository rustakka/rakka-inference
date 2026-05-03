# Releasing rakka-inference

The release pipeline is fully automated. Day-to-day, the only thing a
contributor does is **write a Conventional-Commit subject** when they
land a change on `main`. The pipeline does the rest.

```
Conventional-Commit on main
        │
        ▼
.github/workflows/version-bump.yml
        │  decides patch / minor / major / skip
        │  bumps Cargo.toml + Cargo.lock
        │  commits `chore(release): vX.Y.Z`
        │  tags `vX.Y.Z`
        │  pushes
        ▼
.github/workflows/release.yml   (fires on tag push)
        │  cargo xtask verify           ← 1.0-rc gate
        │  build-binaries (5 targets)
        │  github-release               ← release notes + tarballs
        │  publish-crates               ← dep order, allowlist-gated
        ▼
   crates.io + GitHub Release
```

---

## Conventional-Commit conventions

| Commit subject prefix       | Bump kind |
|-----------------------------|-----------|
| `feat: ...`                 | minor     |
| `fix: ...` / `perf: ...` / `revert: ...` | patch |
| Anything with `BREAKING CHANGE` body or `!:` after type | major |
| `chore:` / `docs:` / `ci:` / `test:` / `refactor:` / `style:` / `build:` only | skip |

A footer `Release-As: x.y.z` overrides the auto-decision and pins the
exact version (useful for `1.0.0-rc.1`, `0.2.0` after a long stretch
of skipped chores, etc).

```text
fix(remote-core): handle 503 retry-after correctly

The retry engine was honoring the policy delay instead of the
server-supplied Retry-After header on 503s.

Release-As: 0.1.4
```

---

## What the version-bump workflow does

1. Skips if the head commit message starts with `chore(release):`
   (prevents the bot from re-bumping its own commits).
2. Reads commits since the previous tag, picks the bump kind via the
   table above, or honors a `Release-As:` footer.
3. Calls `cargo xtask bump <kind>` (or `bump --set <ver>`) which
   - updates `[workspace.package].version` in `Cargo.toml`
   - updates every internal pin in `[workspace.dependencies]` that
     has `path = "crates/..."` (so each crate's resolved version
     stays in sync with the workspace)
   - runs `cargo update --workspace` to refresh `Cargo.lock`.
4. Commits + tags + pushes with `--follow-tags`. The tag push fires
   `release.yml`.

You can dry-run the decision via the GitHub UI:
**Actions → Version bump + tag → Run workflow → dry_run: true**.

---

## What the release workflow does

Five jobs, in order:

1. **verify** — `cargo xtask verify` runs:
   - `cargo build --workspace`
   - `cargo test --workspace --quiet`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo build -p inference --features remote-only`
   - `cargo xtask audit --check`
   - **the remote-only invariant**: `cargo tree -p inference
     --features remote-only` must contain zero `cudarc` /
     `rakka-cuda` / `candle` / `pyo3` lines. This is the
     architectural invariant — pull-requests that violate it fail
     CI before they reach a tag.

2. **build-binaries** — cross-platform `rakka` (the `inference-cli`
   binary) builds for:
   - `x86_64-unknown-linux-gnu`
   - `aarch64-unknown-linux-gnu` (via `cross`)
   - `x86_64-apple-darwin`
   - `aarch64-apple-darwin`
   - `x86_64-pc-windows-msvc`

   Each is built with `--no-default-features --features remote-only`
   so the released binary is the no-GPU-deps router by default. Local
   GPU users build from source with their preferred feature set.

3. **github-release** — auto-generates release notes from
   `git log --no-merges --pretty=format:'- %s (%h)' <prev-tag>..HEAD`
   and uploads the binary archives. `softprops/action-gh-release@v2`.

4. **publish-crates** — walks every `inference-*` crate in dependency
   order and runs `cargo publish`. The publish loop:
   - reads the **allowlist** from the repo variable
     `RAKKA_INFERENCE_PUBLISH_ALLOWLIST` (default below). Crates
     outside the allowlist are logged-and-skipped.
   - retries on `429 / Too Many Requests` with exponential backoff
     (90s × attempt, capped at 6 attempts).
   - treats `already uploaded` as success → re-tagging the same
     version is cheap.
   - sleeps 30 s between successful publishes to pace crates.io's
     "new crates per period" rolling-window limit.

5. **publish-pypi** — scaffolded but **disabled** (`if: false`) until
   `inference-py-bindings` ships a real surface. Re-enable by
   removing the guard and configuring trusted publishing on PyPI.

---

## The crates allowlist

`rakka-inference` declares path dependencies on the sibling
`rakka` and `rakka-cuda` workspaces. Until those workspaces publish
to crates.io, `cargo publish` for any inference-* crate that
transitively depends on them fails. The allowlist mechanism handles
this:

- **Default allowlist** (in `release.yml`'s `DEFAULT_PUBLISH_ALLOWLIST`
  env var):
  ```
  inference-core
  inference-remote-core
  inference-runtime-openai
  inference-runtime-anthropic
  inference-runtime-gemini
  inference-runtime-litellm
  ```
  These six crates have no `rakka` / `rakka-cuda` dependency in
  their published surface and can ship today.

- **Override** via repo variable
  `RAKKA_INFERENCE_PUBLISH_ALLOWLIST`. Set on
  *Settings → Secrets and variables → Actions → Variables → New repository variable*.

- **Full publish** — once `rakka` and `rakka-cuda` ship their stable
  versions to crates.io, set `RAKKA_INFERENCE_PUBLISH_ALLOWLIST=""`
  (empty) and the next tag will publish every member crate in dep
  order.

`cargo xtask release-checklist` prints the current state of every
crate so you can see what's gated and why.

---

## Required secrets and variables

| Type    | Name                                    | Purpose                                                                |
|---------|-----------------------------------------|------------------------------------------------------------------------|
| Secret  | `CRATES_IO_TOKEN`                       | crates.io API token with publish rights for every inference-* crate.   |
| Secret  | `GITHUB_TOKEN` (default)                | Used by `softprops/action-gh-release` and the bump push. No extra setup. |
| Variable| `RAKKA_INFERENCE_WORKSPACE_VERSION`     | Optional. `cargo-semver-checks` flips from warn → hard-fail when this starts with `1.`. Default `0.`. |
| Variable| `RAKKA_INFERENCE_PUBLISH_ALLOWLIST`     | Optional. Space-separated list of crates that may publish. Empty = publish all. |

For PyPI (when `publish-pypi` re-enables), additionally configure
trusted publishing on PyPI's project settings (preferred) or set
`PYPI_API_TOKEN`.

---

## Manual / emergency releases

`workflow_dispatch` triggers exist on **release.yml** and
**version-bump.yml** for one-off operations.

### Dry-run a release

> Actions → Release → Run workflow → `dry_run: true`

Runs `cargo publish --dry-run` for the allowlisted subset and skips
the GitHub-Release upload. Useful before flipping the allowlist or
shipping a controversial change.

### Force a specific bump kind

> Actions → Version bump + tag → Run workflow → `force: minor`

Bumps regardless of whether the commits since the last tag warrant
it. Use sparingly — usually `Release-As:` in a real commit footer is
the better path.

### Yanked / botched release

If `cargo publish` partially succeeded:

1. **Re-tag the same version** — `publish-crates` is idempotent,
   so already-uploaded crates skip. Only the missing ones re-attempt.
2. If a crate published with a critical bug:
   `cargo yank --vers X.Y.Z <crate>` from a maintainer machine, then
   land a `fix:` commit and let the next auto-tag happen.

---

## Local sanity checks

```sh
# Verify the workspace builds, tests, lints, and the remote-only
# invariant holds. Equivalent to what release.yml's verify job runs.
cargo xtask verify

# Print the audit table; --check fails on regression vs baseline.
cargo xtask audit
cargo xtask audit --check

# Re-generate the audit baseline (only after intentional changes).
cargo xtask audit --json docs/reports/audit-baseline.json

# See the publishable / gated split.
cargo xtask release-checklist

# Bump locally without committing (useful for testing the bump body).
cargo xtask bump patch          # 0.1.0 -> 0.1.1
cargo xtask bump minor          # 0.1.0 -> 0.2.0
cargo xtask bump --set 0.5.0    # explicit
cargo xtask bump --pre rc.1     # 0.1.0 -> 0.1.0-rc.1
```

---

## Crate publish dep-order

The publish loop in `release.yml` walks crates in this order. A crate
appears after every crate it depends on:

```
inference-core                   ← leaf
inference-runtime                ← rakka-* + inference-core
inference-python-bridge          ← inference-core (pyo3 optional)
inference-remote-core            ← inference-core + inference-runtime
inference-runtime-openai         ← + inference-remote-core
inference-runtime-anthropic
inference-runtime-gemini
inference-runtime-litellm        ← + inference-runtime-openai
inference-runtime-vllm           ← + inference-python-bridge
inference-runtime-tensorrt
inference-runtime-ort
inference-runtime-candle
inference-runtime-cudarc
inference-runtime-mistralrs
inference-pipeline               ← rakka-streams + inference-runtime
inference-testkit                ← rakka-testkit + remote-core
inference-cli                    ← rakka + inference-runtime
inference                        ← rollup; everything above
```

When you add a new crate to the workspace: **slot it into this list at
the earliest layer where all its dependencies are already listed**.
The publish loop sleeps 30 s between successful publishes, so a full
cold publish of all 18 crates takes ~10 minutes.

---

## What's intentionally not automated

- **CHANGELOG.md** — release notes are auto-generated from
  `git log` between tags. We don't keep a hand-curated CHANGELOG.
  If you want to summarise a release, prepend prose to `RELEASE_NOTES.md`
  in a `chore:` PR before tagging.
- **PyPI** — gated until `inference-py-bindings` is real (see
  `release.yml` job `publish-pypi`).
- **`semver-checks` hard-fail** — warn-only at `0.x`. Flip
  `RAKKA_INFERENCE_WORKSPACE_VERSION` to `1.` to arm.
- **Coordinated cross-workspace releases** with `rakka` / `rakka-cuda`
  — handled today by the allowlist; flip to full when those workspaces
  publish.
