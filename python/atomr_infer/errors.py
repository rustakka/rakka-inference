"""``atomr_infer.errors`` — facade over ``atomr_infer._native.errors``.

Exception hierarchy mirroring ``inference_core::error::InferenceError``.
``InferenceError`` is the base; every concrete variant (``RateLimited``,
``CircuitOpen``, ``BadRequest``, …) subclasses it, so::

    try:
        await cluster.execute("d", batch)
    except errors.InferenceError as e:
        ...

catches every runtime-typed failure.
"""

from ._native import errors as _sub

globals().update({k: getattr(_sub, k) for k in dir(_sub) if not k.startswith("_")})
__all__ = [k for k in dir(_sub) if not k.startswith("_")]
