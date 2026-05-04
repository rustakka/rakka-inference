"""atomr-infer — Python entry point for the Rust actor-based inference runtime.

The native PyO3 extension lives in :mod:`atomr_infer._native`; this
package re-exports the public surface so users write::

    from atomr_infer import Cluster, Deployment

instead of touching the underscore-prefixed extension module directly.

The current 0.3.x surface is intentionally narrow — `Deployment` value
objects and a `Cluster.connect(...).deploy(...)` shape — and tracks the
RFC v4 architecture document. Additional bindings (decorators, escape
hatches into the cluster's actor refs) land as the underlying Rust
surface stabilises.
"""

from ._native import Cluster, Deployment

__all__ = ["Cluster", "Deployment"]
__version__ = "0.3.0"
