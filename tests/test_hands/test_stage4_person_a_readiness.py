import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from zpds.hands.config import WilorConfig
from zpds.hands.estimator_factory import (
    EstimatorRuntime,
    EstimatorUnavailableError,
    _validate_wilor_joint_mapping_contract,
    create_hand_estimator,
    validate_estimator_runtime,
)
from zpds.hands.orchestration import InferenceWriterBundle
from zpds.hands.testing import (
    FakeBBoxWriter,
    FakeFrameStatusWriter,
    FakeHandEstimator,
)
from zpds.hands.wilor_preflight import check_wilor_assets


def _write_asset(path: Path, content: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _wilor_fixture(tmp_path: Path) -> tuple[WilorConfig, Path]:
    root = tmp_path / "models" / "wilor"
    detector = root / "detector.pt"
    checkpoint = root / "wilor_final.ckpt"
    model_config = root / "model_config.yaml"
    mano_model = root / "mano_data" / "mano" / "MANO_RIGHT.pkl"
    mean_params = root / "mano_data" / "mano_mean_params.npz"

    detector_meta = _write_asset(detector, b"detector")
    checkpoint_meta = _write_asset(checkpoint, b"checkpoint")
    model_config_meta = _write_asset(model_config, b"model-config")
    mano_meta = _write_asset(mano_model, b"mano-model")
    mean_meta = _write_asset(mean_params, b"mean-params")

    revision = "model-revision-123"
    source_commit = "a" * 40
    manifest = root / "download_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_repository": {"revision": revision},
                "files": [
                    {"name": "detector.pt", **detector_meta},
                    {"name": "wilor_final.ckpt", **checkpoint_meta},
                    {"name": "model_config.yaml", **model_config_meta},
                ],
                "mano": {
                    "right_hand_model": mano_meta,
                    "mean_parameters": mean_meta,
                },
            }
        ),
        encoding="utf-8",
    )
    config = WilorConfig(
        enabled=True,
        upstream_commit=source_commit,
        model_revision=revision,
        detector_path=str(detector),
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=str(checkpoint_meta["sha256"]),
        model_config_path=str(model_config),
        mano_model_path=str(mano_model),
        mano_mean_params_path=str(mean_params),
        asset_manifest_path=str(manifest),
    )
    return config, checkpoint


def test_wilor_preflight_accepts_complete_pinned_assets(
    tmp_path: Path,
) -> None:
    config, _checkpoint = _wilor_fixture(tmp_path)

    report = check_wilor_assets(config)

    assert report.ready is True
    assert report.errors == ()
    assert len(report.assets) == 5
    assert all(asset.ok for asset in report.assets)
    assert report.model_revision == config.model_revision


def test_wilor_preflight_rejects_tampered_checkpoint(
    tmp_path: Path,
) -> None:
    config, checkpoint = _wilor_fixture(tmp_path)
    checkpoint.write_bytes(b"tampered-checkpoint")

    report = check_wilor_assets(config)

    assert report.ready is False
    assert any(asset.name == "checkpoint" and not asset.ok for asset in report.assets)
    assert any("checkpoint" in error for error in report.errors)


def test_wilor_runtime_metadata_must_match_frozen_config(
    tmp_path: Path,
) -> None:
    config, _checkpoint = _wilor_fixture(tmp_path)
    runtime = EstimatorRuntime(
        estimator=FakeHandEstimator([]),
        model_name="wilor",
        model_version="test",
        checkpoint_sha256="wrong-sha",
        upstream_git_commit=config.upstream_commit,
        active_backend="wilor",
    )

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_estimator_runtime(
            runtime,
            SimpleNamespace(wilor=config),  # type: ignore[arg-type]
        )


def test_writer_bundle_enforces_person_c_contract() -> None:
    bundle = InferenceWriterBundle(
        frame_status=FakeFrameStatusWriter(),
        bbox=FakeBBoxWriter(),
    )
    assert bundle.frame_status is not None
    assert bundle.bbox is not None

    with pytest.raises(TypeError, match="frame_status Writer"):
        InferenceWriterBundle(
            frame_status=object(),  # type: ignore[arg-type]
            bbox=FakeBBoxWriter(),
        )


def test_person_a_accepts_person_b_runtime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "wilor-source"
    (source / "wilor").mkdir(parents=True)
    (source / ".git").mkdir()
    commit = "b" * 40
    (source / ".git" / "HEAD").write_text(commit, encoding="utf-8")
    wilor = WilorConfig(
        enabled=True,
        source_path=str(source),
        upstream_commit=commit,
        upstream_license_checked=True,
    )

    class FakeBackend:
        def __init__(self, config: object) -> None:
            self.model_info = SimpleNamespace(
                model_version=wilor.model_version,
                checkpoint_sha256=wilor.checkpoint_sha256,
                upstream_git_commit=wilor.upstream_commit,
                device=wilor.device,
                precision=wilor.precision,
            )

    monkeypatch.setattr(
        "zpds.hands.backends.wilor.WiLoRBackend",
        FakeBackend,
    )
    monkeypatch.setattr(
        "zpds.hands.wilor_adapter.WiLoRAdapter",
        lambda backend: object(),
    )
    monkeypatch.setattr(
        "zpds.hands.wilor_estimator.WiLoRHandEstimator",
        lambda **kwargs: FakeHandEstimator([]),
    )

    runtime = create_hand_estimator(
        SimpleNamespace(wilor=wilor),  # type: ignore[arg-type]
    )

    assert runtime.active_backend == "wilor"
    assert runtime.run_meta["joint_mapping_version"] == (
        "wilor-to-hands-v1-v1"
    )


@pytest.mark.parametrize(
    ("required_version", "mapping", "message"),
    [
        ("unexpected-version", tuple(range(21)), "Unsupported"),
        (None, tuple(range(20)), "exactly 21"),
        (None, tuple(range(20)) + (19,), "duplicate"),
        (None, tuple(range(20)) + (21,), "range"),
    ],
)
def test_person_a_rejects_invalid_person_b_joint_mapping_contract(
    required_version: str | None,
    mapping: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(EstimatorUnavailableError, match=message):
        _validate_wilor_joint_mapping_contract(
            required_version=required_version,
            mapping_version="wilor-to-hands-v1-v1",
            mapping=mapping,
        )
