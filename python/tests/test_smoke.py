"""Smoke tests — confirm the wheel imports and the public surface is reachable.

These are deliberately tiny because the 0.3.x PyO3 surface is itself
narrow (see RFC v4 §11.1). They run after `maturin develop` or against
an installed wheel; CI exercises them as part of the release pipeline.
"""

import atomr_infer


def test_module_version_is_set() -> None:
    assert isinstance(atomr_infer.__version__, str)
    assert atomr_infer.__version__


def test_deployment_round_trips_basic_fields() -> None:
    dep = atomr_infer.Deployment(
        name="gpt-4o-mini",
        model="gpt-4o-mini",
        replicas=1,
    )
    assert dep.name() == "gpt-4o-mini"
    assert dep.model() == "gpt-4o-mini"


def test_cluster_connect_returns_handle() -> None:
    cluster = atomr_infer.Cluster.connect("inproc://test")
    assert cluster.endpoint() == "inproc://test"


def test_cluster_deploy_accepts_deployment() -> None:
    cluster = atomr_infer.Cluster.connect("inproc://test")
    dep = atomr_infer.Deployment(name="d", model="m", replicas=2, gpus=1)
    cluster.deploy(dep)
