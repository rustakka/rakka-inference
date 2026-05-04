//! `RemoteSessionActor` — analog of the local `ContextActor` (CUDA §5.11).
//!
//! Owns the credential and HTTP-client lifecycle for a remote
//! deployment. Restartable by its parent `RemoteEngineCoreActor` when:
//!
//! - sustained 401s suggest the API key has rotated,
//! - configuration change (endpoint URL, timeouts) requires a fresh
//!   client,
//! - operator triggers `cluster.deployment(...).rebuild_session()`.
//!
//! In-flight requests held by `RemoteWorkerActor` complete with the
//! pre-rebuild client; new requests pick up the rebuilt one.

use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use atomr_core::actor::{Actor, Context};
use tokio::sync::oneshot;

use atomr_infer_core::deployment::Timeouts;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::SessionRebuildCause;
use atomr_infer_core::SecretString;

use crate::http::{build_client, HttpClient};

/// Configuration the session needs to (re)build its client.
#[derive(Clone)]
pub struct SessionConfig {
    pub user_agent: String,
    pub timeouts: Timeouts,
    /// Bearer / api-key credential. Cloned on rebuild so rotation
    /// requires the secret source to have changed.
    pub credential: Arc<dyn CredentialProvider>,
}

#[async_trait]
pub trait CredentialProvider: Send + Sync {
    async fn token(&self) -> InferenceResult<SecretString>;
}

/// Static API key — most common case.
pub struct StaticApiKey(pub SecretString);

#[async_trait]
impl CredentialProvider for StaticApiKey {
    async fn token(&self) -> InferenceResult<SecretString> {
        // SecretString isn't Clone; we re-create from the underlying
        // `&str` exposure — secrecy zeroizes on drop, which is fine.
        use atomr_infer_core::ExposeSecret;
        Ok(SecretString::from(self.0.expose_secret().to_string()))
    }
}

/// Snapshot held by every `RemoteWorkerActor`. Shared via `ArcSwap` so
/// rebuilds are lock-free for readers.
pub struct SessionSnapshot {
    pub client: HttpClient,
    pub credential: SecretString,
}

pub struct SessionRebuildRequest {
    pub cause: SessionRebuildCause,
    pub reply: oneshot::Sender<InferenceResult<()>>,
}

pub struct RemoteSessionActor {
    config: SessionConfig,
    snapshot: Arc<ArcSwap<SessionSnapshot>>,
}

impl RemoteSessionActor {
    /// Build the initial snapshot. Call before spawning the actor so
    /// callers can wire `snapshot()` into worker constructors.
    pub async fn bootstrap(config: SessionConfig) -> InferenceResult<Self> {
        let snapshot = Self::build_snapshot(&config).await?;
        Ok(Self {
            config,
            snapshot: Arc::new(ArcSwap::from_pointee(snapshot)),
        })
    }

    pub fn snapshot(&self) -> Arc<ArcSwap<SessionSnapshot>> {
        self.snapshot.clone()
    }

    async fn build_snapshot(config: &SessionConfig) -> InferenceResult<SessionSnapshot> {
        let client = build_client(&config.timeouts, &config.user_agent)
            .map_err(|e| InferenceError::Internal(format!("build http client: {e}")))?;
        let credential = config.credential.token().await?;
        Ok(SessionSnapshot { client, credential })
    }

    async fn rebuild(&mut self, cause: SessionRebuildCause) -> InferenceResult<()> {
        tracing::info!(?cause, "rebuilding remote session");
        let snap = Self::build_snapshot(&self.config).await?;
        self.snapshot.store(Arc::new(snap));
        Ok(())
    }
}

#[async_trait]
impl Actor for RemoteSessionActor {
    type Msg = SessionRebuildRequest;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        let res = self.rebuild(msg.cause).await;
        let _ = msg.reply.send(res);
    }
}
