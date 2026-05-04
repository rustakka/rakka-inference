# atomr-infer-py-bindings

> PyO3 bindings — declare `Deployment`s and connect to a `Cluster`
> from Python.

## Build

```sh
# Compile the cdylib (importable as a Python extension module).
cargo build -p atomr-infer-py-bindings --features python
# Or, for distribution:
maturin develop --features python
```

Default-features-off the crate compiles to an empty rlib so the
workspace builds without a Python venv.

## Surface (v0)

```python
from inference import Cluster, Deployment

cluster = Cluster.connect("rakka://prod:7355")

cluster.deploy(
    Deployment(
        name="gpt-4o",
        model="gpt-4o",
        replicas=2,
    )
)
```

## Roadmap

The doc's §11 lists six layers; this crate currently covers Layer 1
(declarative `Deployment` + `Cluster`). Layers 2–5 (decorator macros
like `@inference_actor`, escape-hatch `ActorRef` access, hybrid agent
helpers) follow once the Rust surface stabilises and we can guarantee
backwards-compat.
