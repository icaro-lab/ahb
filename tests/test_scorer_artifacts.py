from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from inspect_ai.dataset import MemoryDataset, Sample

from ahb_inspect import tasks
from ahb_inspect.scorer_artifacts import (
    scorer_config_snapshot,
    scorer_deployment_snapshot,
    validate_scorer_artifacts,
)

ARTIFACTS = Path(__file__).parents[1] / "src" / "ahb_inspect" / "artifacts"
CONFIG_PATH = ARTIFACTS / "scorer-config-v1.json"
DEPLOYMENT_PATH = ARTIFACTS / "deployment-v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_canonical_scorer_artifacts_match_runtime_configuration():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    deployment = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    assert config == scorer_config_snapshot()
    assert deployment == scorer_deployment_snapshot()
    assert validate_scorer_artifacts(
        CONFIG_PATH,
        DEPLOYMENT_PATH,
        _digest(CONFIG_PATH),
        _digest(DEPLOYMENT_PATH),
    ) == {
        "config_sha256": _digest(CONFIG_PATH),
        "deployment_sha256": _digest(DEPLOYMENT_PATH),
    }


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    [
        ("config", lambda value: value.update(rubric_sha256="f" * 64)),
        (
            "deployment",
            lambda value: value["provider_policy"].update(zdr=False),
        ),
        (
            "deployment",
            lambda value: value.update(resolved_model="typesafe/jev-other"),
        ),
    ],
)
def test_scorer_artifact_content_drift_is_rejected_even_with_new_digest(
    tmp_path, artifact, mutation
):
    config_path = tmp_path / "config.json"
    deployment_path = tmp_path / "deployment.json"
    config = scorer_config_snapshot()
    deployment = scorer_deployment_snapshot()
    config_digest = _write_json(config_path, config)
    deployment_digest = _write_json(deployment_path, deployment)
    if artifact == "config":
        mutation(config)
        config_digest = _write_json(config_path, config)
    else:
        mutation(deployment)
        deployment_digest = _write_json(deployment_path, deployment)

    with pytest.raises(ValueError, match="does not match"):
        validate_scorer_artifacts(
            config_path,
            deployment_path,
            config_digest,
            deployment_digest,
        )


def test_scorer_artifacts_reject_hash_mismatch_and_partial_pair(tmp_path):
    with pytest.raises(ValueError, match="must be bound as a pair"):
        validate_scorer_artifacts(CONFIG_PATH, None, _digest(CONFIG_PATH), None)

    with pytest.raises(ValueError, match="digest mismatch"):
        validate_scorer_artifacts(
            CONFIG_PATH,
            DEPLOYMENT_PATH,
            "0" * 64,
            _digest(DEPLOYMENT_PATH),
        )


def test_legacy_jev_task_without_artifacts_remains_supported(monkeypatch):
    monkeypatch.setattr(
        tasks,
        "load_ahb_dataset",
        lambda **kwargs: MemoryDataset([Sample(id="synthetic-1", input="synthetic")]),
    )
    task = tasks.ahb_jev(dataset_limit=1, use_system_prompt=False)
    assert task.name == "ahb_jev"
    assert task.metadata["scorer_profile"]["scorer_artifact_digests"] is None


def test_task_forwards_bound_local_dataset_to_loader(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        tasks,
        "load_ahb_dataset",
        lambda **kwargs: (
            calls.update(kwargs)
            or MemoryDataset([Sample(id="synthetic-1", input="synthetic")])
        ),
    )
    task = tasks.ahb_jev(
        dataset_limit=3600,
        use_system_prompt=False,
        dataset_artifact_path="/private/artifacts/ahb.jsonl",
        dataset_artifact_sha256="a" * 64,
    )
    assert calls["artifact_path"] == "/private/artifacts/ahb.jsonl"
    assert calls["artifact_sha256"] == "a" * 64
    assert task.metadata["dataset_artifact_format"] == "ahb-full-split-jsonl-v1"


def test_packaged_task_consumes_artifacts_and_records_only_digests(monkeypatch):
    monkeypatch.setattr(
        tasks,
        "load_ahb_dataset",
        lambda **kwargs: MemoryDataset([Sample(id="synthetic-1", input="synthetic")]),
    )
    config_digest = _digest(CONFIG_PATH)
    deployment_digest = _digest(DEPLOYMENT_PATH)
    task = tasks.ahb_jev(
        dataset_limit=1,
        use_system_prompt=False,
        scorer_config_path=str(CONFIG_PATH),
        scorer_deployment_path=str(DEPLOYMENT_PATH),
        scorer_config_sha256=config_digest,
        scorer_deployment_sha256=deployment_digest,
    )
    assert task.metadata["scorer_profile"]["scorer_artifact_digests"] == {
        "config_sha256": config_digest,
        "deployment_sha256": deployment_digest,
    }
