# ai-skills/

Skills for AI coding assistants working on **projects that depend on
atomr-infer** — not for editing atomr-infer itself. They follow
the standard `SKILL.md` + frontmatter convention used by Claude Code,
Claude Agent SDK, and other agentic tools.

These skills are deliberately separate from the repo's own dev tooling
(`xtask/`, release workflows) so that distributing them to consumers
does not entangle this repo's internal development workflow.

## What's here

| Skill | Use when… |
|---|---|
| `atomr-infer-quickstart` | Standing up your first deployment — `Deployment` value object, `runtime` field, `atomr-infer serve --config`, the rollup feature flags |
| `atomr-infer-runtimes` | Choosing a backend — local Rust-native (Candle / cudarc / mistralrs), Python (vLLM), FFI (TensorRT / ORT), or remote (OpenAI / Anthropic / Gemini / LiteLLM) |
| `atomr-infer-remote-providers` | Wiring a remote provider — credentials, rate limits, circuit breakers, retries, cost estimation, fallback chains |
| `atomr-infer-pipelines` | Composing multi-runtime pipelines — `DynamicBatchingServer`, `InferenceCascade`, hybrid local→remote escalation, fallback on `RateLimitExceeded` / `CircuitOpen` |
| `atomr-infer-deployment` | Deploying to a cluster — feature-flag matrix, the `remote-only` invariant, `atomr-infer serve --config`, the project-file TOML schema, hot-swap & credential rotation |
| `atomr-infer-troubleshooting` | Debugging — typed `InferenceError` triage, 429 storms, circuit-breaker state, content-filter refusals, sticky CUDA-context recovery |
| `atomr-infer-extending` | Adding a new backend — implementing `ModelRunner`, plugging into the rollup, where to slot a new crate in publish dep-order |

Each `SKILL.md` is a thin router: it points at canonical docs in this
repo (`README.md`, `docs/feature-matrix.md`, `docs/rustakka-inference-architecture-v4.md`,
the per-crate READMEs) and at the relevant crate's API. It does **not**
restate API surfaces that belong in rustdoc, because those drift faster
than docs.

## Installing

Pick the path that matches your assistant. The skills themselves are
vendor-neutral `SKILL.md` files — only the install mechanism differs.

### Claude Code (recommended: marketplace)

If you use Claude Code, install via the plugin marketplace — this
keeps the skills updated as atomr-infer releases, with no manual
copy step:

```text
/plugin marketplace add rustakka/atomr-infer
/plugin install atomr-infer-ai-skills@atomr-infer
```

You can also install from a local checkout (useful when developing
against a atomr-infer fork):

```text
/plugin marketplace add /path/to/atomr-infer
/plugin install atomr-infer-ai-skills@atomr-infer
```

Skills auto-activate based on the `description` frontmatter — no need
to invoke them explicitly.

### Claude Agent SDK / project-local `.claude/skills/`

For SDK-based agents or project-local Claude Code setups that read
from `.claude/skills/`, copy or symlink the skills in:

```bash
# copy (snapshot)
cp -r /path/to/atomr-infer/ai-skills/skills/* .claude/skills/

# symlink (track upstream)
ln -s /path/to/atomr-infer/ai-skills/skills/atomr-infer-quickstart \
      .claude/skills/atomr-infer-quickstart
```

### Cursor

Cursor reads project rules from `.cursor/rules/`. Copy the skills in
as `.mdc` rules; Cursor will treat the frontmatter `description` as
the activation hint:

```bash
mkdir -p .cursor/rules
for s in /path/to/atomr-infer/ai-skills/skills/*/SKILL.md; do
  name=$(basename "$(dirname "$s")")
  cp "$s" ".cursor/rules/${name}.mdc"
done
```

### OpenAI Codex CLI

Codex CLI reads `AGENTS.md` (project-level) and `~/.codex/AGENTS.md`
(user-level). It does not have a SKILL.md loader, so reference the
skills from `AGENTS.md` and let the model pull them in on demand:

```markdown
<!-- in AGENTS.md -->
## atomr-infer skills
When working on atomr-infer, consult the matching skill in
`ai-skills/skills/<name>/SKILL.md`:
- first deployment / Deployment object / atomr-infer serve  → atomr-infer-quickstart
- choosing a backend / local vs remote / FFI vs API   → atomr-infer-runtimes
- API keys / rate limits / circuit breakers / costs   → atomr-infer-remote-providers
- batching / cascade / hybrid local→remote / fallback → atomr-infer-pipelines
- cluster rollout / feature flags / credential rotate → atomr-infer-deployment
- typed errors / 429 storms / sticky CUDA recovery    → atomr-infer-troubleshooting
- new ModelRunner / new crate / publish dep-order     → atomr-infer-extending
```

### Gemini CLI

Gemini CLI reads `GEMINI.md` and supports custom commands under
`.gemini/commands/`. Point Gemini at the skills the same way:

```markdown
<!-- in GEMINI.md -->
For atomr-infer work, load the relevant skill from
`ai-skills/skills/<name>/SKILL.md` before editing.
```

Optionally wrap each skill as a slash command in
`.gemini/commands/atomr-infer-<name>.toml` whose `prompt` includes
`@ai-skills/skills/<name>/SKILL.md`.

### Other harnesses (Aider, Continue, Zed, etc.)

Any tool that accepts a system prompt or rules file can load these
skills — `SKILL.md` is plain Markdown with YAML frontmatter. Either
include the file directly in the system prompt, or reference its path
from your tool's rules file (`.aider.conf.yml`, `.continue/`, etc.).

### Reading by hand

Each `SKILL.md` is a normal Markdown file — humans can read them
directly to learn the architecture without invoking an AI.

## Authoring conventions

- **One job per skill.** A skill is a router into the right docs +
  examples for one task. If a skill is trying to teach two things,
  it should be two skills (or it should defer to docs).
- **Defer to source-of-truth docs.** Link to `docs/*.md`,
  `crates/*/README.md`, and `examples/*` rather than restating them.
  Skills go stale; docs travel with the code.
- **Vendor-neutral.** No references to a specific assistant,
  harness, or tool. Describe atomr-infer, not the runtime loading
  the skill.
- **Frontmatter.** Each skill begins with `---` frontmatter
  containing `name` and `description`. The description is a
  one-line activation hint — what the user is doing when this
  skill should kick in.

## Versioning

These skills version with the repo. If a release changes a public
API covered by a skill, update the skill in the same PR. The skills
are not separately published.

## Companion skills

- [Sibling rakka skills](https://github.com/rustakka/atomr/tree/main/ai-skills)
  — actor design, supervision, persistence, clustering, Python bindings.
- [Sibling rakka-accel skills](https://github.com/rustakka/atomr-accel/tree/main/ai-skills)
  — DeviceActor, kernel selection, two-tier supervision, Python bindings,
  backend choice (CUDA today; ROCm/Metal/oneAPI/Vulkan on the roadmap).

Install all three together when you're building a service that uses
rakka primitives, rakka-accel GPU acceleration, **and** atomr-infer
runtimes.

## License

Same as the atomr-infer workspace: Apache-2.0.
