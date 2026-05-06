//! `atomr-infer` CLI binary. Subcommands per doc §11.3 / §11.6.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use atomr_infer_cli::{run_server, ProjectFile};

#[derive(Parser)]
#[command(name = "atomr-infer", version, about = "atomr-infer CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Boot the actor system and serve all deployments declared in the
    /// supplied project file.
    Serve {
        #[arg(short, long, value_name = "PATH")]
        config: std::path::PathBuf,
    },
    /// Print the deployments in the project file (parse + validate).
    Status {
        #[arg(short, long, value_name = "PATH")]
        config: std::path::PathBuf,
    },
    /// Print per-deployment cost estimates. Doc §12.4.
    CostReport,
    /// Trigger a credential rebuild on a deployment. Doc §11.5.
    RotateCredentials { deployment: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve { config } => {
            let project = ProjectFile::from_path(&config)?;
            run_server(project).await
        }
        Cmd::Status { config } => {
            let project = ProjectFile::from_path(&config)?;
            for d in &project.deployments {
                let runtime = d.effective_runtime();
                println!(
                    "{:<32} model={:<48} runtime={:?} replicas={}",
                    d.name, d.model, runtime, d.replicas
                );
            }
            Ok(())
        }
        Cmd::CostReport => {
            // TODO(doc §12.4): query the running MetricsActor over IPC.
            // For v0 we just emit the placeholder so operators see the
            // command exists.
            println!("cost-report: not yet implemented (see doc §13 Phase 6)");
            Ok(())
        }
        Cmd::RotateCredentials { deployment } => {
            // TODO(doc §11.5): tell the named deployment's
            // RemoteSessionActor to rebuild.
            println!("rotate-credentials {deployment}: not yet implemented (see doc §13 Phase 6)");
            Ok(())
        }
    }
}
