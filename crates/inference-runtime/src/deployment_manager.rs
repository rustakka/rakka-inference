//! `DeploymentManagerActor` — cluster-singleton owner of the deployment
//! catalog. Doc §4. Manages create/update/delete and surfaces the
//! current set to the gateway and `DpCoordinatorActor`.

use std::collections::HashMap;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context};
use tokio::sync::oneshot;

use atomr_infer_core::deployment::Deployment;
use atomr_infer_core::error::InferenceError;

#[derive(Debug, Clone)]
pub struct DeploymentRecord {
    pub deployment: Deployment,
    pub state: DeploymentState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeploymentState {
    Pending,
    Serving,
    Draining,
    Failed,
}

pub enum DeploymentManagerMsg {
    Apply {
        deployment: Deployment,
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },
    Remove {
        name: String,
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },
    List {
        reply: oneshot::Sender<Vec<DeploymentRecord>>,
    },
    Get {
        name: String,
        reply: oneshot::Sender<Option<DeploymentRecord>>,
    },
}

#[derive(Default)]
pub struct DeploymentManagerActor {
    records: HashMap<String, DeploymentRecord>,
}

impl DeploymentManagerActor {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Actor for DeploymentManagerActor {
    type Msg = DeploymentManagerMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            DeploymentManagerMsg::Apply { deployment, reply } => {
                let res = match deployment.validate() {
                    Ok(()) => {
                        let name = deployment.name.clone();
                        self.records.insert(
                            name,
                            DeploymentRecord {
                                deployment,
                                state: DeploymentState::Pending,
                            },
                        );
                        Ok(())
                    }
                    Err(e) => Err(InferenceError::Internal(e.to_string())),
                };
                let _ = reply.send(res);
            }
            DeploymentManagerMsg::Remove { name, reply } => {
                self.records.remove(&name);
                let _ = reply.send(Ok(()));
            }
            DeploymentManagerMsg::List { reply } => {
                let _ = reply.send(self.records.values().cloned().collect());
            }
            DeploymentManagerMsg::Get { name, reply } => {
                let _ = reply.send(self.records.get(&name).cloned());
            }
        }
    }
}
