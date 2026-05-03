//! End-to-end demo for the remote-only architecture path.
//!
//! Stands up a `wiremock` server speaking the OpenAI Chat Completions
//! wire format, points an `OpenAiRunner` at it via a real session
//! snapshot, and exercises:
//!
//! 1. Happy path — single request returns streamed tokens.
//! 2. 429 handling — wiremock returns one `429 Retry-After: 1`; the
//!    `RetryEngine` backs off and the request succeeds on retry.
//! 3. Circuit breaker — wiremock returns a burst of 503s; the
//!    `CircuitBreakerHandle` opens; the next call short-circuits
//!    with `InferenceError::CircuitOpen`.
//!
//! Satisfies the doc §13 Phase-1/2c exit criteria for the remote
//! path.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use arc_swap::ArcSwap;
use futures::StreamExt;
use secrecy::SecretString;
use tracing_subscriber::EnvFilter;
use url::Url;

use inference_core::batch::{ExecuteBatch, Message, MessageContent, Role, SamplingParams};
use inference_core::deployment::{RateLimits, RetryPolicy, Timeouts};
use inference_core::error::InferenceError;
use inference_core::runner::ModelRunner;
use inference_core::runtime::{CircuitBreakerConfig, JitterKind, ProviderKind};

use inference_remote_core::circuit_breaker::CircuitBreakerHandle;
use inference_remote_core::http::build_client;
use inference_remote_core::retry::{Attempt, RetryDecision, RetryEngine};
use inference_remote_core::session::{CredentialProvider, SessionConfig, SessionSnapshot, StaticApiKey};

use inference_runtime_openai::config::SecretRef;
use inference_runtime_openai::{OpenAiConfig, OpenAiRunner, OpenAiVariant};

use inference_testkit::mock_openai::{inject_429_once, inject_5xx_once, mount_chat_happy_path, MockOpenAi};

fn batch(prompt: &str, stream: bool) -> ExecuteBatch {
    ExecuteBatch {
        request_id: format!("req-{}", now_ms()),
        model: "gpt-4o-mini".into(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Text(prompt.into()),
        }],
        sampling: SamplingParams {
            max_tokens: Some(64),
            ..Default::default()
        },
        stream,
        estimated_tokens: 16,
    }
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    // ---------- 1. happy path ---------------------------------------------
    let happy = MockOpenAi::start().await;
    mount_chat_happy_path(&happy.server, "Hello from the mock!").await;
    let runner = build_runner(&happy.url()).await?;
    println!("\n== happy path ==");
    drain_one(runner, batch("hi", true)).await?;

    // ---------- 2. 429 retry ----------------------------------------------
    let throttle = MockOpenAi::start().await;
    inject_429_once(&throttle.server).await;
    mount_chat_happy_path(&throttle.server, "succeeded after retry").await;
    let runner = build_runner(&throttle.url()).await?;
    let policy = RetryPolicy {
        max_retries: 3,
        initial_backoff: Duration::from_millis(50),
        max_backoff: Duration::from_secs(2),
        backoff_multiplier: 2.0,
        jitter: JitterKind::None,
        respect_retry_after: true,
    };
    let retry_engine = Arc::new(RetryEngine::new(policy, true));
    println!("\n== 429 retry ==");
    drive_with_retries(runner, batch("hi", true), retry_engine).await?;

    // ---------- 3. circuit breaker open after 5xxs -------------------------
    let burst = MockOpenAi::start().await;
    inject_5xx_once(&burst.server, 5).await;
    mount_chat_happy_path(&burst.server, "ignored — breaker is open").await;
    let runner = build_runner(&burst.url()).await?;
    let breaker = CircuitBreakerHandle::new(
        ProviderKind::OpenAi,
        CircuitBreakerConfig {
            failure_threshold: 3,
            open_duration: Duration::from_secs(5),
            half_open_max_probes: 1,
        },
    );
    println!("\n== circuit breaker ==");
    drive_with_breaker(runner, batch("hi", true), breaker).await?;

    Ok(())
}

async fn build_runner(base_url: &str) -> Result<OpenAiRunner> {
    let credential: Arc<dyn CredentialProvider> =
        Arc::new(StaticApiKey(SecretString::from("sk-mock".to_string())));
    let session_cfg = SessionConfig {
        user_agent: "rakka-inference-demo/0.1.0".into(),
        timeouts: Timeouts::default(),
        credential: credential.clone(),
    };
    let client = build_client(&session_cfg.timeouts, &session_cfg.user_agent)?;
    let token = credential.token().await?;
    let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
        client,
        credential: token,
    }));

    let endpoint = Url::parse(&format!("{}/v1/", base_url.trim_end_matches('/')))?;
    let cfg = OpenAiConfig {
        variant: OpenAiVariant::Direct { endpoint },
        api_key: SecretRef::Inline {
            value: "sk-mock".into(),
        },
        organization: None,
        project: None,
        rate_limits: RateLimits::default(),
        retry: RetryPolicy::default(),
        circuit_breaker: CircuitBreakerConfig::default(),
        timeouts: Timeouts::default(),
    };
    Ok(OpenAiRunner::new(cfg, snap)?)
}

async fn drain_one(mut runner: OpenAiRunner, b: ExecuteBatch) -> Result<()> {
    let handle = runner.execute(b).await?;
    let mut s = handle.into_stream();
    while let Some(item) = s.next().await {
        match item {
            Ok(c) => print!("{}", c.text_delta),
            Err(e) => println!("\nerr: {e}"),
        }
    }
    println!();
    Ok(())
}

async fn drive_with_retries(
    mut runner: OpenAiRunner,
    b: ExecuteBatch,
    engine: Arc<RetryEngine>,
) -> Result<()> {
    let mut attempt = Attempt::zero();
    loop {
        let res = runner.execute(b.clone()).await;
        match res {
            Ok(handle) => {
                let mut s = handle.into_stream();
                while let Some(item) = s.next().await {
                    match item {
                        Ok(c) => print!("{}", c.text_delta),
                        Err(e) => println!("\nstream-err: {e}"),
                    }
                }
                println!();
                return Ok(());
            }
            Err(e) => {
                println!("attempt {} → {e}", attempt.0);
                match engine.decide(attempt, &e) {
                    RetryDecision::Retry { after } => {
                        tokio::time::sleep(after).await;
                        attempt.0 += 1;
                        continue;
                    }
                    RetryDecision::GiveUp => return Err(anyhow::anyhow!(e)),
                }
            }
        }
    }
}

async fn drive_with_breaker(
    mut runner: OpenAiRunner,
    b: ExecuteBatch,
    breaker: Arc<CircuitBreakerHandle>,
) -> Result<()> {
    for i in 0..4 {
        let result: Result<(), InferenceError> = breaker
            .run(|| async { runner.execute(b.clone()).await.map(|_| ()) })
            .await;
        println!("call {i}: {result:?}  state={:?}", breaker.state());
    }
    println!("circuit final state: {:?}", breaker.state());
    Ok(())
}
