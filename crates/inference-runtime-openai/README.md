# atomr-infer-runtime-openai

> OpenAI Chat Completions + Azure OpenAI runtime. ~600 LOC; everything
> heavy is in [`atomr-infer-remote-core`](../atomr-infer-remote-core/).

## Highlights

- **Both variants in one config.** `OpenAiVariant::Direct { endpoint }`
  for `api.openai.com` (or any OpenAI-compatible proxy);
  `OpenAiVariant::Azure { resource, deployment, api_version }` for
  Azure OpenAI's URL shape.
- **SSE streaming first-class.** `RunHandle::streaming(...)` over
  `reqwest::Response::bytes_stream` → `eventsource-stream` →
  `TokenChunk`s.
- **Provider-specific error refinement.** A 400 with
  `error.code == "context_length_exceeded"` is upgraded to
  `InferenceError::ContextLengthExceeded`; a 400 of type
  `content_filter` is upgraded to `InferenceError::ContentFiltered`
  and skips retries.
- **Pricing baked in.** `OpenAiPricing::published()` carries the
  current list rates for `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`,
  `o1-preview`, `o1-mini`. Operators override per deployment.

## Quick start

```rust
use inference_runtime_openai::{OpenAiConfig, OpenAiRunner};
use inference_runtime_openai::config::SecretRef;

let cfg = OpenAiConfig::defaults_for_openai(
    SecretRef::Env { name: "OPENAI_API_KEY".into() },
);
let runner = OpenAiRunner::new(cfg, session_snapshot)?;
```

`session_snapshot: Arc<ArcSwap<SessionSnapshot>>` comes from
`inference_remote_core::session::RemoteSessionActor::bootstrap(...)`.
The double-indirection (`Arc<ArcSwap<…>>`) means rotating the API key
swaps the snapshot in-place — in-flight requests complete with the
old credential, new ones use the rotated value, with no traffic
dropped.

## Azure example

```rust
use inference_runtime_openai::{OpenAiConfig, OpenAiVariant};

let cfg = OpenAiConfig {
    variant: OpenAiVariant::Azure {
        resource: "my-azure-resource".into(),
        deployment: "gpt-4o-deployment".into(),
        api_version: "2024-08-01-preview".into(),
    },
    api_key: SecretRef::Env { name: "AZURE_OPENAI_KEY".into() },
    /* defaults for rate_limits / retry / circuit_breaker / timeouts */
    ..OpenAiConfig::defaults_for_openai(SecretRef::Env { name: "_unused".into() })
};
```

## Cost estimation

```rust
use inference_runtime_openai::OpenAiPricing;
let p = OpenAiPricing::published().get("gpt-4o-mini").unwrap();
// p.input_per_mtok_usd, p.output_per_mtok_usd
```

The rollup's `inference::core::cost::from_rates(...)` lifts these into
a `CostEstimate` per `ExecuteBatch`. `MetricsActor` aggregates the
real numbers reported by the API in response headers / usage fields,
so dashboards see live spend, not just predictions.
