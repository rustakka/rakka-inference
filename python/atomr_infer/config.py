"""``atomr_infer.config`` — facade over ``atomr_infer._native.config``.

Exposes per-deployment serving config: ``Serving``, ``CapacityPolicy``,
``RateLimits``, ``RetryPolicy``, ``Timeouts``, ``Budget``,
``BudgetAction``.
"""

from ._native import config as _sub

globals().update({k: getattr(_sub, k) for k in dir(_sub) if not k.startswith("_")})
__all__ = [k for k in dir(_sub) if not k.startswith("_")]
