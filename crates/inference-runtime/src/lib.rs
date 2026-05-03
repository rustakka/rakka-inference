//! # inference-runtime
//!
//! Runtime-agnostic actor implementations on top of `rakka-core`.
//!
//! Per architecture doc §4 these are the actors whose logic doesn't
//! depend on whether the underlying backend is local-GPU or
//! remote-network: the gateway, the per-request lifecycle actor, the
//! coordinator, the deployment manager, placement, metrics. Local-GPU
//! specifics (`WorkerActor` with `ContextActor` two-tier supervision)
//! also live here because the *shape* of two-tier supervision is
//! shared infrastructure even though the per-runtime rebuild logic is
//! contributed by per-runtime crates.
//!
//! Remote-network engine cores live in `inference-remote-core`.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod deployment_manager;
pub mod dp_coordinator;
pub mod engine_core;
pub mod gateway;
pub mod metrics;
pub mod placement;
pub mod request;
pub mod worker;

pub use deployment_manager::{
    DeploymentManagerActor, DeploymentManagerMsg, DeploymentRecord, DeploymentState,
};
pub use dp_coordinator::{DpCoordinatorActor, DpCoordinatorMsg, RouteTarget};
pub use engine_core::{AddRequest, EngineCoreActor, EngineCoreMsg, LocalEngineConfig};
pub use gateway::{spawn_gateway, ApiGatewayActor, ApiGatewayMsg, GatewayConfig};
pub use metrics::{DeploymentMetrics, FailureKind, MetricsActor, MetricsMsg, MetricsSnapshot};
pub use placement::{
    DeploymentPlacementActor, NodeAssignment, PlacementConstraints, PlacementError, PlacementMsg,
    PlacementResult,
};
pub use request::{RequestActor, RequestMsg, Route, StreamingResponse};
pub use worker::{ContextActor, ContextMsg, WorkerActor, WorkerMsg, WorkerSlot};
