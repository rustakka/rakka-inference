"""atomr-infer — Python entry point for the Rust actor-based inference runtime.

The native PyO3 extension lives in :mod:`atomr_infer._native` and is
split into per-domain submodules — ``core``, ``runtime``, ``config``,
``errors``, and ``cluster`` — mirroring the upstream ``atomr`` Python
binding layout.

Common imports::

    from atomr_infer import Cluster, Deployment, ExecuteBatch, Message, Role
    from atomr_infer import RuntimeKind, RuntimeConfig, SamplingParams
    from atomr_infer.errors import InferenceError, BadRequest

Async execute::

    import asyncio
    from atomr_infer import Cluster, Deployment, ExecuteBatch, Message, Role, \\
        RuntimeKind, SamplingParams

    cluster = Cluster.connect("inproc://test")
    cluster.deploy(Deployment(name="d", model="m", runtime=RuntimeKind.custom("mock")))
    batch = ExecuteBatch(
        request_id="r1", model="m",
        messages=[Message(Role("user"), "hello")],
        sampling=SamplingParams(max_tokens=16),
    )
    tokens = asyncio.run(cluster.execute("d", batch))
    print(tokens.text)
"""

from importlib import metadata as _metadata

from . import _native, config, core, errors, runtime

# pyo3 0.22 attaches submodules as attributes on the parent extension
# module rather than as importable packages, so re-export via attribute
# access instead of `from ._native.X import Y`.
Cluster = _native.cluster.Cluster
TokenStream = _native.cluster.TokenStream

ContentPart = _native.core.ContentPart
CostEstimate = _native.core.CostEstimate
Deployment = _native.core.Deployment
ExecuteBatch = _native.core.ExecuteBatch
FinishReason = _native.core.FinishReason
Message = _native.core.Message
MessageContent = _native.core.MessageContent
Replica = _native.core.Replica
Role = _native.core.Role
SamplingParams = _native.core.SamplingParams
TokenChunk = _native.core.TokenChunk
TokenUsage = _native.core.TokenUsage
Tokens = _native.core.Tokens

CircuitBreakerConfig = _native.runtime.CircuitBreakerConfig
JitterKind = _native.runtime.JitterKind
ProviderKind = _native.runtime.ProviderKind
RuntimeConfig = _native.runtime.RuntimeConfig
RuntimeKind = _native.runtime.RuntimeKind
TransportKind = _native.runtime.TransportKind

try:
    __version__ = _metadata.version("atomr-infer")
except _metadata.PackageNotFoundError:  # editable installs / running from source
    __version__ = "0.0.0+unknown"

__all__ = [
    # Cluster + streaming
    "Cluster",
    "TokenStream",
    # core data types
    "ContentPart",
    "CostEstimate",
    "Deployment",
    "ExecuteBatch",
    "FinishReason",
    "Message",
    "MessageContent",
    "Replica",
    "Role",
    "SamplingParams",
    "TokenChunk",
    "TokenUsage",
    "Tokens",
    # runtime taxonomy
    "CircuitBreakerConfig",
    "JitterKind",
    "ProviderKind",
    "RuntimeConfig",
    "RuntimeKind",
    "TransportKind",
    # submodule facades
    "config",
    "core",
    "errors",
    "runtime",
]
