"""End-to-end async execute over the testkit MockRunner.

Demonstrates the full pipeline: ``Cluster.deploy`` → registry → async
``execute`` (drains stream into ``Tokens``) and ``execute_stream``
(yields ``TokenChunk`` per ``__anext__``). Uses ``pytest.mark.asyncio``
which the maturin/CI Python image already pins via ``pytest-asyncio``.
"""

import asyncio

import pytest

from atomr_infer import (
    Cluster,
    Deployment,
    ExecuteBatch,
    Message,
    Role,
    RuntimeConfig,
    RuntimeKind,
    SamplingParams,
)
from atomr_infer.errors import InferenceError


def _build_batch() -> ExecuteBatch:
    return ExecuteBatch(
        request_id="r1",
        model="mock",
        messages=[Message(Role("user"), "hello")],
        sampling=SamplingParams(temperature=0.0, max_tokens=8),
        stream=False,
        estimated_tokens=4,
    )


def _deploy_mock(cluster: Cluster, name: str = "d", chunks: list[str] | None = None) -> None:
    config = {"chunks": chunks} if chunks is not None else {}
    dep = Deployment(
        name=name,
        model="mock",
        replicas=1,
        runtime=RuntimeKind.custom("mock"),
        runtime_config=RuntimeConfig.custom("mock", config),
    )
    cluster.deploy(dep)


@pytest.mark.asyncio
async def test_execute_aggregates_mock_chunks_into_tokens() -> None:
    cluster = Cluster.connect("inproc://test")
    _deploy_mock(cluster, chunks=["hello ", "world"])

    tokens = await cluster.execute("d", _build_batch())

    assert tokens.text == "hello world"
    assert tokens.finish_reason is not None
    assert tokens.finish_reason.name == "stop"
    # MockRunner reports input=1, output=len(chunks).
    assert tokens.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_execute_default_chunks_when_no_config() -> None:
    cluster = Cluster.connect("inproc://test")
    dep = Deployment(name="d", model="mock", runtime=RuntimeKind.custom("mock"))
    cluster.deploy(dep)

    tokens = await cluster.execute("d", _build_batch())
    assert tokens.text  # cluster supplies a default mock chunk


@pytest.mark.asyncio
async def test_execute_stream_yields_chunks() -> None:
    cluster = Cluster.connect("inproc://test")
    _deploy_mock(cluster, chunks=["foo", "bar", "baz"])

    pieces: list[str] = []
    async for chunk in cluster.execute_stream("d", _build_batch()):
        pieces.append(chunk.text_delta)

    assert pieces == ["foo", "bar", "baz"]


@pytest.mark.asyncio
async def test_execute_unknown_deployment_raises_inference_error() -> None:
    cluster = Cluster.connect("inproc://test")
    with pytest.raises(InferenceError):
        await cluster.execute("missing", _build_batch())


def test_deploy_rejects_local_gpu_runtime() -> None:
    cluster = Cluster.connect("inproc://test")
    dep = Deployment(name="d", model="m", runtime=RuntimeKind.vllm())
    with pytest.raises(InferenceError):
        cluster.deploy(dep)


def test_deploy_rejects_remote_runtime_until_session_wiring() -> None:
    cluster = Cluster.connect("inproc://test")
    dep = Deployment(name="d", model="gpt-4o", runtime=RuntimeKind.openai())
    with pytest.raises(InferenceError):
        cluster.deploy(dep)


def test_deploy_runs_validate_on_input() -> None:
    cluster = Cluster.connect("inproc://test")
    # Empty name fails Deployment::validate; surfaces as InferenceError.
    dep = Deployment(name="", model="m", runtime=RuntimeKind.custom("mock"))
    with pytest.raises(InferenceError):
        cluster.deploy(dep)


def test_event_loop_runs_with_asyncio_run() -> None:
    """``asyncio.run`` round-trip — confirms ``future_into_py`` is
    runtime-agnostic. We have to construct the awaitable inside the
    coroutine because ``future_into_py`` captures the running loop at
    call time."""
    cluster = Cluster.connect("inproc://test")
    _deploy_mock(cluster, name="r", chunks=["x"])

    async def _go() -> str:
        tokens = await cluster.execute("r", _build_batch())
        return tokens.text

    assert asyncio.run(_go()) == "x"
