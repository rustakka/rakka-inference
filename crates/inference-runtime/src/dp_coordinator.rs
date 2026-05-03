//! `DpCoordinatorActor` — one cluster-singleton per model. Doc §4, §6.1.
//!
//! Holds the routing CRDT (deployment-name → engine-core endpoints
//! with load scores) and answers `RouteTo` asks from `RequestActor`s.
//! The implementation is a thin in-process map for v0; a real cluster
//! deployment registers this actor with
//! `rakka_cluster_tools::ClusterSingletonManager` so there's exactly
//! one per (model, cluster).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use rakka_core::actor::{Actor, Context, UntypedActorRef};
use tokio::sync::oneshot;

use inference_core::error::InferenceError;

#[derive(Clone)]
pub struct RouteTarget {
    /// Untyped because the engine-core actor type differs by runtime
    /// (local vs remote) but the routing layer doesn't care.
    pub engine: UntypedActorRef,
    /// Best-effort load score (lower = less loaded). Filled by
    /// engine-cores via `ReportLoad`.
    pub load: f64,
}

impl std::fmt::Debug for RouteTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouteTarget")
            .field("load", &self.load)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Default)]
struct CoordinatorState {
    routes: HashMap<String, Vec<RouteTarget>>,
}

pub enum DpCoordinatorMsg {
    Register {
        deployment: String,
        target: RouteTarget,
    },
    Deregister {
        deployment: String,
        engine_path: rakka_core::actor::ActorPath,
    },
    ReportLoad {
        deployment: String,
        engine_path: rakka_core::actor::ActorPath,
        load: f64,
    },
    RouteTo {
        deployment: String,
        reply: oneshot::Sender<Result<RouteTarget, InferenceError>>,
    },
}

pub struct DpCoordinatorActor {
    state: Arc<RwLock<CoordinatorState>>,
}

impl Default for DpCoordinatorActor {
    fn default() -> Self {
        Self {
            state: Arc::new(RwLock::new(CoordinatorState::default())),
        }
    }
}

impl DpCoordinatorActor {
    pub fn new() -> Self {
        Self::default()
    }

    fn register(&self, deployment: String, target: RouteTarget) {
        self.state
            .write()
            .routes
            .entry(deployment)
            .or_default()
            .push(target);
    }

    fn deregister(&self, deployment: &str, path: &rakka_core::actor::ActorPath) {
        if let Some(v) = self.state.write().routes.get_mut(deployment) {
            v.retain(|t| t.engine.path() != path);
        }
    }

    fn report_load(&self, deployment: &str, path: &rakka_core::actor::ActorPath, load: f64) {
        if let Some(v) = self.state.write().routes.get_mut(deployment) {
            for t in v.iter_mut() {
                if t.engine.path() == path {
                    t.load = load;
                }
            }
        }
    }

    fn pick(&self, deployment: &str) -> Result<RouteTarget, InferenceError> {
        let st = self.state.read();
        let candidates = st
            .routes
            .get(deployment)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| InferenceError::Internal(format!("no engine for deployment `{deployment}`")))?;
        // Lowest load wins.
        let pick = candidates
            .iter()
            .min_by(|a, b| a.load.partial_cmp(&b.load).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .ok_or_else(|| InferenceError::Internal("empty candidate set".into()))?;
        Ok(pick)
    }
}

#[async_trait]
impl Actor for DpCoordinatorActor {
    type Msg = DpCoordinatorMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            DpCoordinatorMsg::Register { deployment, target } => self.register(deployment, target),
            DpCoordinatorMsg::Deregister {
                deployment,
                engine_path,
            } => self.deregister(&deployment, &engine_path),
            DpCoordinatorMsg::ReportLoad {
                deployment,
                engine_path,
                load,
            } => self.report_load(&deployment, &engine_path, load),
            DpCoordinatorMsg::RouteTo { deployment, reply } => {
                let _ = reply.send(self.pick(&deployment));
            }
        }
    }
}
