"""Exception-hierarchy parity: every concrete variant subclasses the
``InferenceError`` base, and looking up by name from the facade module
returns the same type as the native module."""

import pytest

from atomr_infer import errors


CONCRETE_VARIANTS = [
    "RateLimited",
    "CircuitOpen",
    "ContentFiltered",
    "ContextLengthExceeded",
    "BadRequest",
    "Unauthorized",
    "Forbidden",
    "Backpressure",
    "BudgetExceeded",
    "NetworkError",
    "ServerError",
    "Timeout",
    "CudaContextPoisoned",
    "Internal",
]


def test_base_is_exposed() -> None:
    assert hasattr(errors, "InferenceError")
    assert issubclass(errors.InferenceError, Exception)


@pytest.mark.parametrize("name", CONCRETE_VARIANTS)
def test_variant_subclasses_base(name: str) -> None:
    cls = getattr(errors, name)
    assert issubclass(cls, errors.InferenceError)


def test_raise_and_catch_via_base() -> None:
    with pytest.raises(errors.InferenceError):
        raise errors.BadRequest("malformed input")


def test_raise_concrete_keeps_concrete_type() -> None:
    with pytest.raises(errors.RateLimited) as exc_info:
        raise errors.RateLimited("hit 429")
    assert isinstance(exc_info.value, errors.InferenceError)
