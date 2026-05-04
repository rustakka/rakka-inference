# atomr-infer-runtime-gemini

> Google Gemini runtime — both **AI Studio** (API key in query string)
> and **Vertex AI** (OAuth2 access token via the operator's preferred
> credential source).

## Two variants, one runner

```rust
use inference_runtime_gemini::{GeminiConfig, GeminiRunner, GeminiVariant};
use inference_runtime_gemini::config::SecretRef;

// AI Studio
let cfg = GeminiConfig {
    variant: GeminiVariant::AiStudio { /* default endpoint */ ..Default::default() },
    credential: SecretRef::Env { name: "GOOGLE_API_KEY".into() },
    safety: vec![],
    ..GeminiConfig::default_for_aistudio()
};

// Vertex
let cfg = GeminiConfig {
    variant: GeminiVariant::Vertex {
        project: "my-gcp-project".into(),
        region: "us-central1".into(),
    },
    credential: SecretRef::Adc,                // Application Default Credentials
    safety: vec![],
    ..Default::default()
};
```

## OAuth2 — pluggable, not bundled

We don't pull a full `oauth2` stack into the workspace root. Instead,
`inference_remote_core::session::CredentialProvider` is a trait with
one async method:

```rust
async fn token(&self) -> InferenceResult<SecretString>;
```

For Vertex, supply a provider that calls `gcloud auth
print-access-token` (or your preferred token source) and refreshes on
a timer. For AI Studio, `StaticApiKey` is enough — the runner appends
it as a `?key=...` query param.

## SSE format

Gemini's `streamGenerateContent?alt=sse` emits the same JSON schema
as the unary `generateContent`, one chunk per SSE `data:` line. The
runner reuses `inference_remote_core::sse::decode_sse_stream` and
deserializes each chunk via `serde_json::from_str::<GenerateContentResponse>`.

## Safety settings

`GeminiConfig::safety: Vec<SafetySetting>` is forwarded verbatim. The
error classifier upgrades a `FAILED_PRECONDITION` whose message
mentions "safety" or "blocked" to `InferenceError::ContentFiltered`,
so the `RequestActor` knows not to retry.

## Pricing

`GeminiPricing::published()` covers `gemini-2.0-pro`, `gemini-1.5-pro`,
`gemini-1.5-flash`, `gemini-2.0-flash`. Override per deployment for
custom Vertex pricing tiers.
