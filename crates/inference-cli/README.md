# inference-cli

> The `rakka serve` binary. Boots an actor system, applies every
> `[[deployment]]` in your project file, mounts the gateway.

## Quick start

```sh
cargo run -p inference-cli --features all-remote -- \
    serve --config examples/remote_only_demo/demo.toml
```

…and `curl http://127.0.0.1:8080/v1/chat/completions` against it.

## Subcommands

| Subcommand                         | What it does                                                                  |
|------------------------------------|-------------------------------------------------------------------------------|
| `rakka serve --config <path>`      | Parse the project file, build the actor system, register every deployment, mount the gateway, wait for `Ctrl+C`. |
| `rakka status --config <path>`     | Print the deployments in the project file (validate without running).        |
| `rakka cost-report`                | Per-deployment cost — talks to a running `MetricsActor`. *(Phase 6 stub.)* |
| `rakka rotate-credentials <name>`  | Triggers `RemoteSessionActor::rebuild` on the named deployment. *(Phase 6 stub.)* |

## Project file (TOML)

```toml
[cluster]
name = "production"
bind = "0.0.0.0:8080"

[[deployment]]
name     = "gpt-4o-mini"
model    = "gpt-4o-mini"
runtime  = "open_ai"
replicas = 2

[deployment.serving]
max_concurrent        = 50
on_capacity_exhausted = "queue"     # queue | reject | fallback

[[deployment]]
name     = "tinyllama-local"
model    = "TinyLlama-1.1B-Chat-Q4_0"
runtime  = "candle"
gpus     = 1
replicas = 1
```

## Build profiles

| Build                                                                                  | Use case                                          |
|----------------------------------------------------------------------------------------|---------------------------------------------------|
| `cargo build -p inference-cli --no-default-features --features remote-only`            | Pure-remote router; no GPU deps in the binary.    |
| `cargo build -p inference-cli --features all-remote`                                   | All four remote providers + pipeline.             |
| `cargo build -p inference-cli --features default-prod`                                 | The doc's recommended production preset.          |
