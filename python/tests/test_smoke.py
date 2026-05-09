"""Smoke tests — confirm the wheel imports and the data-types parity surface
is reachable.

Runs after ``maturin develop`` or against an installed wheel; CI exercises
them as part of the release pipeline.
"""

import atomr_infer
from atomr_infer import (
    Cluster,
    ContentPart,
    Deployment,
    ExecuteBatch,
    FinishReason,
    Message,
    MessageContent,
    Role,
    RuntimeKind,
    SamplingParams,
    TokenUsage,
)


def test_module_version_is_set() -> None:
    assert isinstance(atomr_infer.__version__, str)
    assert atomr_infer.__version__


def test_deployment_full_field_round_trip() -> None:
    dep = Deployment(name="gpt-4o-mini", model="gpt-4o-mini", replicas=2, gpus=None)
    assert dep.name == "gpt-4o-mini"
    assert dep.model == "gpt-4o-mini"
    assert dep.replicas == 2
    assert dep.gpus is None
    assert dep.idempotent is True

    dep.replicas = 4
    dep.runtime = RuntimeKind.openai()
    assert dep.replicas == 4
    assert dep.runtime is not None
    assert dep.runtime.name == "openai"


def test_role_round_trip() -> None:
    r = Role("user")
    assert r.name == "user"
    assert Role("user") == Role("user")


def test_finish_reason_round_trip() -> None:
    assert FinishReason("stop").name == "stop"
    assert FinishReason("tool_calls").name == "tool_calls"


def test_runtime_kind_static_constructors_and_tag() -> None:
    assert RuntimeKind.openai().name == "openai"
    assert RuntimeKind.openai().is_remote() is True
    assert RuntimeKind.vllm().is_remote() is False
    custom = RuntimeKind.custom("mock")
    assert custom.name == "custom"
    assert custom.tag == "mock"


def test_message_accepts_str_or_parts() -> None:
    m = Message(Role("user"), "hello")
    assert m.role.name == "user"
    assert m.content.kind == "text"

    parts = [ContentPart.text("hi"), ContentPart.image_url("https://example.com/x.png")]
    m2 = Message(Role("user"), parts)
    assert m2.content.kind == "parts"


def test_message_content_factories() -> None:
    mc = MessageContent.text("hi")
    assert mc.kind == "text"
    assert MessageContent.parts([ContentPart.text("a")]).kind == "parts"


def test_execute_batch_constructor() -> None:
    batch = ExecuteBatch(
        request_id="r1",
        model="m",
        messages=[Message(Role("user"), "hi")],
        sampling=SamplingParams(temperature=0.5, max_tokens=8),
        stream=False,
        estimated_tokens=4,
    )
    assert batch.request_id == "r1"
    assert batch.model == "m"
    assert len(batch.messages) == 1
    assert batch.sampling.temperature == 0.5
    assert batch.sampling.max_tokens == 8
    assert batch.estimated_tokens == 4
    assert batch.stream is False


def test_token_usage_round_trip() -> None:
    u = TokenUsage(input_tokens=3, output_tokens=5, reasoning_tokens=1, cached_tokens=0)
    assert u.input_tokens == 3
    assert u.output_tokens == 5
    assert u.reasoning_tokens == 1


def test_cluster_connect_returns_handle() -> None:
    cluster = Cluster.connect("inproc://test")
    assert cluster.endpoint() == "inproc://test"
    assert cluster.deployments() == []


def test_cluster_deploy_mock_runtime_succeeds() -> None:
    cluster = Cluster.connect("inproc://test")
    dep = Deployment(name="d", model="m", replicas=1, runtime=RuntimeKind.custom("mock"))
    cluster.deploy(dep)
    assert "d" in cluster.deployments()
