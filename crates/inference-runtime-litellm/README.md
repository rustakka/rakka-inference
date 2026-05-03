# inference-runtime-litellm

> Thin LiteLLM-proxy adapter on top of
> [`inference-runtime-openai`](../inference-runtime-openai/).
> ~110 LOC.

LiteLLM exposes an OpenAI-compatible HTTP surface fronting any backend
(OpenAI, Anthropic, Bedrock, Azure, Cohere, …) and applies its own
caching / fallback / retry policies. The `LiteLlmRunner` is a newtype
around `OpenAiRunner` that:

- Points at the LiteLLM proxy URL instead of `api.openai.com`.
- Lowers the default `max_retries` to 1 — LiteLLM does its own
  retries, so client-side retries would compound.
- Preserves `runtime_kind() == LiteLlm` and
  `transport_kind().provider == LiteLlm` so dashboards and routing
  can distinguish "via LiteLLM" from "direct to OpenAI" even when the
  wire format is identical.

## Quick start

```rust
use inference_runtime_litellm::{LiteLlmConfig, LiteLlmRunner};
use inference_runtime_litellm::SecretRef;

let cfg = LiteLlmConfig {
    endpoint: url::Url::parse("http://litellm.internal:4000/v1/")?,
    api_key: SecretRef::Env { name: "LITELLM_KEY".into() },
    ..Default::default()
};
let openai_cfg = cfg.into_openai(/* matching openai SecretRef */);
let runner = LiteLlmRunner::new(openai_cfg, session_snapshot)?;
```

## When to choose this over `inference-runtime-openai`

- Your team already runs LiteLLM as the central provider gateway and
  wants observability tagged with the proxy hop.
- You want LiteLLM's fallback chains (Anthropic → Bedrock → OpenAI on
  failure) and we should stay out of the way.
- You're consolidating spend tracking through the proxy, so
  per-deployment cost in the inference-side `MetricsActor` is
  intentionally a downstream concern.
