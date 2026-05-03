# inference-runtime-anthropic

> Anthropic Messages API runtime. Same shape as
> [`inference-runtime-openai`](../inference-runtime-openai/);
> per-provider differences live in `wire.rs` and `error.rs`.

## What's different from OpenAI

- Auth header is `x-api-key` (not `Authorization: Bearer`).
- API version pinned via the `anthropic-version` header
  (`AnthropicConfig::anthropic_version`, defaults to `2023-06-01`).
- SSE event types are richer:
  `message_start`, `content_block_delta`, `message_delta`,
  `message_stop`, `ping`, `error`. The runner translates each to the
  canonical `TokenChunk` shape.
- System messages are extracted into the top-level `system` field
  (Anthropic doesn't accept `role: "system"` inline).
- Vision input is a content block with `type: "image"` carrying base64
  source data — fully supported via `ContentPart::ImageBase64`.
- Tool-use is round-tripped (input deltas surface as
  `tool_call_delta`).

## Pricing

`AnthropicPricing::published()` covers Opus 4, Sonnet 4, 3.5 Sonnet,
3.5 Haiku, 3 Haiku. Operators override per deployment.

## Quick start

```rust
use inference_runtime_anthropic::{AnthropicConfig, AnthropicRunner};
use inference_runtime_anthropic::config::SecretRef;

let cfg = AnthropicConfig::defaults(SecretRef::Env { name: "ANTHROPIC_API_KEY".into() });
let runner = AnthropicRunner::new(cfg, session_snapshot)?;
```
