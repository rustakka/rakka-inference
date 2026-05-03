# ai-skills/

Skills for AI coding assistants working on **projects that depend on
rakka-inference** — not for editing rakka-inference itself. They follow
the standard `SKILL.md` + frontmatter convention used by Claude Code,
Claude Agent SDK, and other agentic tools.

These skills are deliberately separate from the repo's own dev tooling
(`xtask/`, release workflows) so that distributing them to consumers
does not entangle this repo's internal development workflow.

## What's here

| Skill | Use when… |
|---|---|
| `rakka-inference-quickstart` | Standing up your first deployment — `Deployment` value object, `runtime` field, `rakka serve --config`, the rollup feature flags |
| `rakka-inference-runtimes` | Choosing a backend — local Rust-native (Candle / cudarc / mistralrs), Python (vLLM), FFI (TensorRT / ORT), or remote (OpenAI / Anthropic / Gemini / LiteLLM) |
| `rakka-inference-remote-providers` | Wiring a remote provider — credentials, rate limits, circuit breakers, retries, cost estimation, fallback chains |
| `rakka-inference-pipelines` | Composing multi-runtime pipelines — `DynamicBatchingServer`, `InferenceCascade`, hybrid local→remote escalation, fallback on `RateLimitExceeded` / `CircuitOpen` |
| `rakka-inference-deployment` | Deploying to a cluster — feature-flag matrix, the `remote-only` invariant, `rakka serve --config`, the project-file TOML schema, hot-swap & credential rotation |
| `rakka-inference-troubleshooting` | Debugging — typed `InferenceError` triage, 429 storms, circuit-breaker state, content-filter refusals, sticky CUDA-context recovery |
| `rakka-inference-extending` | Adding a new backend — implementing `ModelRunner`, plugging into the rollup, where to slot a new crate in publish dep-order |

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
keeps the skills updated as rakka-inference releases, with no manual
copy step:

```text
/plugin marketplace add rustakka/rakka-inference
/plugin install rakka-inference-ai-skills@rakka-inference
```

You can also install from a local checkout (useful when developing
against a rakka-inference fork):

```text
/plugin marketplace add /path/to/rakka-inference
/plugin install rakka-inference-ai-skills@rakka-inference
```

Skills auto-activate based on the `description` frontmatter — no need
to invoke them explicitly.

### Claude Agent SDK / project-local `.claude/skills/`

For SDK-based agents or project-local Claude Code setups that read
from `.claude/skills/`, copy or symlink the skills in:

```bash
# copy (snapshot)
cp -r /path/to/rakka-inference/ai-skills/skills/* .claude/skills/

# symlink (track upstream)
ln -s /path/to/rakka-inference/ai-skills/skills/rakka-inference-quickstart \
      .claude/skills/rakka-inference-quickstart
```

### Cursor

Cursor reads project rules from `.cursor/rules/`. Copy the skills in
as `.mdc` rules; Cursor will treat the frontmatter `description` as
the activation hint:

```bash
mkdir -p .cursor/rules
for s in /path/to/rakka-inference/ai-skills/skills/*/SKILL.md; do
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
## rakka-inference skills
When working on rakka-inference, consult the matching skill in
`ai-skills/skills/<name>/SKILL.md`:
- first deployment / Deployment object / rakka serve  → rakka-inference-quickstart
- choosing a backend / local vs remote / FFI vs API   → rakka-inference-runtimes
- API keys / rate limits / circuit breakers / costs   → rakka-inference-remote-providers
- batching / cascade / hybrid local→remote / fallback → rakka-inference-pipelines
- cluster rollout / feature flags / credential rotate → rakka-inference-deployment
- typed errors / 429 storms / sticky CUDA recovery    → rakka-inference-troubleshooting
- new ModelRunner / new crate / publish dep-order     → rakka-inference-extending
```

### Gemini CLI

Gemini CLI reads `GEMINI.md` and supports custom commands under
`.gemini/commands/`. Point Gemini at the skills the same way:

```markdown
<!-- in GEMINI.md -->
For rakka-inference work, load the relevant skill from
`ai-skills/skills/<name>/SKILL.md` before editing.
```

Optionally wrap each skill as a slash command in
`.gemini/commands/rakka-inference-<name>.toml` whose `prompt` includes
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
  harness, or tool. Describe rakka-inference, not the runtime loading
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

- [Sibling rakka skills](https://github.com/rustakka/rakka/tree/main/ai-skills)
  — actor design, supervision, persistence, clustering, Python bindings.
- [Sibling rakka-accel skills](https://github.com/rustakka/rakka-accel/tree/main/ai-skills)
  — DeviceActor, kernel selection, two-tier supervision, Python bindings,
  backend choice (CUDA today; ROCm/Metal/oneAPI/Vulkan on the roadmap).

Install all three together when you're building a service that uses
rakka primitives, rakka-accel GPU acceleration, **and** rakka-inference
runtimes.

## License

Same as the rakka-inference workspace: Apache-2.0.
