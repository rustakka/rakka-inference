//! `rakka serve` runtime — boot the actor system, register every
//! `[[deployment]]`, mount the gateway, wait for shutdown.

use std::sync::Arc;

use anyhow::Result;
use rakka_config::Config;
use rakka_core::actor::{ActorSystem, Props};

use inference_runtime::{
    spawn_gateway, DeploymentManagerActor, DeploymentManagerMsg, DpCoordinatorActor, GatewayConfig,
    MetricsActor,
};

use crate::config::ProjectFile;

pub async fn run_server(project: ProjectFile) -> Result<()> {
    let sys = ActorSystem::create(project.cluster.name.clone(), Config::reference())
        .await
        .map_err(|e| anyhow::anyhow!("create actor system: {e}"))?;
    tracing::info!(name = %sys.name(), "actor system started");

    // Cluster-wide singletons (in v0 — single-process — they're just
    // top-level actors; cluster registration is added when we wire up
    // `rakka_cluster_tools::ClusterSingletonManager`).
    let dp = sys
        .actor_of(Props::create(DpCoordinatorActor::new), "dp-coordinator")
        .map_err(|e| anyhow::anyhow!("spawn coordinator: {e}"))?;
    let mgr = sys
        .actor_of(
            Props::create(DeploymentManagerActor::default),
            "deployment-manager",
        )
        .map_err(|e| anyhow::anyhow!("spawn manager: {e}"))?;
    let _metrics = sys
        .actor_of(Props::create(MetricsActor::default), "metrics")
        .map_err(|e| anyhow::anyhow!("spawn metrics: {e}"))?;

    // Apply each deployment from the project file.
    for d in project.deployments {
        let name = d.name.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        mgr.tell(DeploymentManagerMsg::Apply {
            deployment: d,
            reply: tx,
        });
        match rx.await {
            Ok(Ok(())) => tracing::info!(deployment = %name, "applied"),
            Ok(Err(e)) => tracing::error!(deployment = %name, ?e, "deployment validation failed"),
            Err(_) => tracing::error!(deployment = %name, "manager dropped reply"),
        }
    }

    // Mount the gateway.
    let gateway_cfg = GatewayConfig {
        bind: project.cluster.bind,
    };
    let _gateway = spawn_gateway(&sys, gateway_cfg, dp).map_err(|e| anyhow::anyhow!("spawn gateway: {e}"))?;
    tracing::info!(bind = %project.cluster.bind, "gateway mounted");

    // Block until ctrl-c.
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown requested");
    sys.terminate().await;
    let _ = Arc::new(()); // keep arc-swap dep referenced where applicable
    Ok(())
}
