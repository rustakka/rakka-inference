# ai-skills/

Skills for AI coding assistants working on **projects that depend on
rakka-inference** — not for editing rakka-inference itself. They follow
the standard `SKILL.md` + frontmatter convention used by Claude Code,
Claude Agent SDK, and other agentic tools.

These skills are deliberately separate from the repo's own dev tooling
(`.claude/`, `xtask/`, etc.) so that distributing them to consumers
does not entangle this repo's internal development workflow.

## What's here

| Skill | Use when… |
|---|---|
| [`rakka-inference-quickstart`](skills/rakka-inference-quickstart/SKILL.md) | Standing up your first deployment — `Deployment` value object, `runtime` field, `rakka serve --config`, the rollup feature flags |
| [`rakka-inference-runtimes`](skills/rakka-inference-runtimes/SKILL.md) | Choosing a backend — local Rust-native (Candle / cudarc / mistralrs), Python (vLLM), FFI (TensorRT / ORT), or remote (OpenAI / Anthropic / Gemini / LiteLLM) |
| [`rakka-inference-remote-providers`](skills/rakka-inference-remote-providers/SKILL.md) | Wiring a remote provider — credentials, rate limits, circuit breakers, retries, cost estimation, fallback chains |
| [`rakka-inference-pipelines`](skills/rakka-inference-pipelines/SKILL.md) | Composing multi-runtime pipelines — `DynamicBatchingServer`, `InferenceCascade`, hybrid local→remote escalation, fallback on `RateLimitExceeded` / `CircuitOpen` |
| [`rakka-inference-deployment`](skills/rakka-inference-deployment/SKILL.md) | Deploying to a cluster — feature-flag matrix, the `remote-only` invariant, `rakka serve --config`, the project-file TOML schema, hot-swap & credential rotation |
| [`rakka-inference-troubleshooting`](skills/rakka-inference-troubleshooting/SKILL.md) | Debugging — typed `InferenceError` triage, 429 storms, circuit-breaker state, content-filter refusals, sticky CUDA-context recovery |
| [`rakka-inference-extending`](skills/rakka-inference-extending/SKILL.md) | Adding a new backend — implementing `ModelRunner`, plugging into the rollup, where to slot a new crate in publish dep-order |

Each `SKILL.md` is a thin router: it points at canonical docs in this
repo (`README.md`, `docs/feature-matrix.md`, `docs/rustakka-inference-architecture-v4.md`,
the per-crate READMEs) and at the relevant crate's API. It does **not**
restate API surfaces that belong in rustdoc, because those drift faster
than docs.

## Installing

### Claude Code (plugin install)

```sh
/plugin install /path/to/rakka-inference/ai-skills
```

Claude Code reads `plugin.json` to find the skills directory and
auto-loads each `SKILL.md`. Skills become eligible based on their
frontmatter `description`'s trigger phrases.

### Other agent runtimes

Most agent tools accept a folder of `SKILL.md` files via a plugin
manifest. Point your tool at this folder; the skills in `skills/`
will be picked up automatically.

### Reading by hand

Each `SKILL.md` is a normal Markdown file — humans can read them
directly to learn the architecture without invoking an AI.

## Companion skills

The sibling [rakka workspace](https://github.com/rustakka/rakka) ships
its own [ai-skills bundle](https://github.com/rustakka/rakka/tree/main/ai-skills)
with skills for actor design, supervision, persistence, clustering,
and Python bindings. Install both bundles together when you're
building a service that uses both rakka primitives and rakka-inference
runtimes.

## License

Same as the rakka-inference workspace: Apache-2.0.
