//! `DeploymentPlacementActor` — picks nodes for new deployments. Doc §7.2.
//!
//! Reads `transport_kind()` to decide whether the deployment needs a
//! GPU node (`LocalGpu`) or any egress-capable node
//! (`RemoteNetwork { provider }`). This actor operates at the
//! deployment → node level; once a node is chosen for a `LocalGpu`
//! deployment, the *which-GPU-on-this-node* decision is delegated to
//! the upstream `rakka_accel::cuda::placement::PlacementActor` (under the
//! `local-gpu` feature). Renamed from the doc's plain `PlacementActor`
//! to make the abstraction-level distinction visible at the call site.
//!
//! Without the `local-gpu` feature this actor still runs but the GPU
//! ordinals it returns are a naïve `0..gpus_per_replica` slice — fine
//! for remote-only builds and for tests that don't actually allocate
//! CUDA contexts. Operators wanting topology-aware GPU choice (NVLink
//! islands, MIG slicing, etc.) enable the feature and the
//! deployment-level placement starts asking the upstream actor for
//! per-node device choices.

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context};
use tokio::sync::oneshot;

use atomr_infer_core::deployment::Deployment;
use atomr_infer_core::runtime::{ProviderKind, TransportKind};

#[derive(Debug, Clone)]
pub struct PlacementConstraints {
    /// Available GPU-bearing nodes.
    pub gpu_nodes: Vec<String>,
    /// Available egress-capable nodes (CPU-only OK).
    pub egress_nodes: Vec<String>,
    /// Per-node hosting count for shared `(provider, api_key)`
    /// preferences; populated externally as deployments are added.
    pub egress_preference: std::collections::HashMap<(ProviderKind, String), Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct PlacementResult {
    pub deployment: String,
    pub assignments: Vec<NodeAssignment>,
}

#[derive(Debug, Clone)]
pub struct NodeAssignment {
    pub replica_index: u32,
    pub node: String,
    pub gpus: Vec<u32>,
}

#[derive(Debug, thiserror::Error)]
pub enum PlacementError {
    #[error("no GPU nodes available")]
    NoGpuNodes,
    #[error("no egress nodes available for provider {0:?}")]
    NoEgressNodes(ProviderKind),
    #[error("deployment requires {requested} GPUs but no node has that many free")]
    NotEnoughGpus { requested: u32 },
}

pub enum PlacementMsg {
    Place {
        deployment: Deployment,
        constraints: PlacementConstraints,
        reply: oneshot::Sender<Result<PlacementResult, PlacementError>>,
    },
}

pub struct DeploymentPlacementActor;

impl Default for DeploymentPlacementActor {
    fn default() -> Self {
        Self
    }
}

impl DeploymentPlacementActor {
    pub fn new() -> Self {
        Self
    }

    fn place(
        &self,
        deployment: &Deployment,
        constraints: &PlacementConstraints,
    ) -> Result<PlacementResult, PlacementError> {
        let kind = deployment.effective_runtime();
        let transport: TransportKind = (&kind).into();
        let mut assignments = Vec::with_capacity(deployment.replicas as usize);
        match transport {
            TransportKind::LocalGpu => {
                if constraints.gpu_nodes.is_empty() {
                    return Err(PlacementError::NoGpuNodes);
                }
                let want = deployment.gpus.unwrap_or(1);
                for i in 0..deployment.replicas {
                    let node = constraints.gpu_nodes[(i as usize) % constraints.gpu_nodes.len()].clone();
                    // Per-node GPU choice is the upstream
                    // `rakka_accel::cuda::placement::PlacementActor`'s job —
                    // topology constraints (NVLink islands, MIG, P2P
                    // groups) live there. v0 hands a contiguous range
                    // here and lets the per-node placement actor refine
                    // when it's wired up under the `local-gpu` feature.
                    let gpus = (0..want).collect();
                    assignments.push(NodeAssignment {
                        replica_index: i,
                        node,
                        gpus,
                    });
                }
            }
            TransportKind::RemoteNetwork { provider } => {
                if constraints.egress_nodes.is_empty() {
                    return Err(PlacementError::NoEgressNodes(provider));
                }
                for i in 0..deployment.replicas {
                    let node =
                        constraints.egress_nodes[(i as usize) % constraints.egress_nodes.len()].clone();
                    assignments.push(NodeAssignment {
                        replica_index: i,
                        node,
                        gpus: vec![],
                    });
                }
            }
        }
        Ok(PlacementResult {
            deployment: deployment.name.clone(),
            assignments,
        })
    }
}

#[async_trait]
impl Actor for DeploymentPlacementActor {
    type Msg = PlacementMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            PlacementMsg::Place {
                deployment,
                constraints,
                reply,
            } => {
                let _ = reply.send(self.place(&deployment, &constraints));
            }
        }
    }
}
