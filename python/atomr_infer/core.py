"""``atomr_infer.core`` — facade over ``atomr_infer._native.core``.

Exposes the user-facing data types: ``Deployment``, ``ExecuteBatch``,
``Message``, ``MessageContent``, ``Role``, ``ContentPart``,
``SamplingParams``, ``TokenChunk``, ``Tokens``, ``TokenUsage``,
``FinishReason``, ``CostEstimate``, ``Replica``.
"""

from ._native import core as _sub

globals().update({k: getattr(_sub, k) for k in dir(_sub) if not k.startswith("_")})
__all__ = [k for k in dir(_sub) if not k.startswith("_")]
