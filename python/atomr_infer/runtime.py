"""``atomr_infer.runtime`` — facade over ``atomr_infer._native.runtime``.

Exposes the runtime taxonomy: ``RuntimeKind``, ``RuntimeConfig``,
``ProviderKind``, ``TransportKind``, ``CircuitBreakerConfig``,
``JitterKind``.
"""

from ._native import runtime as _sub

globals().update({k: getattr(_sub, k) for k in dir(_sub) if not k.startswith("_")})
__all__ = [k for k in dir(_sub) if not k.startswith("_")]
