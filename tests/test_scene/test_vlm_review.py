"""人员 B：VLM 复核器（OpenAI 兼容 API）测试。"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from zpds.scene.config import VLMConfig
from zpds.scene.schemas import SceneProposal, VLMReviewResult
from zpds.scene.vlm_review import (
    OpenAICompatibleVLMReviewer,
    SceneLabels,
    VLMUnavailableError,
    _post_json,
    load_scene_labels,
    select_review_queue,
)


def _make_scene(scene_id: str = "scene_000001") -> SceneProposal:
    return SceneProposal(
        scene_id=scene_id,
        start_ns=0,
        end_ns=10_000_000_000,
        confidence=0.9,
        sources=("dino",),
        boundary_scores={"dino": 0.8},
        config_hash="hash-a",
    )


def _make_config() -> VLMConfig:
    return VLMConfig(
        enabled=True,
        base_url="http://vlm.test/v1",
        model="qwen-test",
        api_key_env="TEST_VLM_API_KEY",
        labels_path="",
        timeout_s=5.0,
        review_confidence_threshold=0.6,
    )


def _make_labels() -> SceneLabels:
    return SceneLabels(
        scene_labels=("kitchen", "office"),
        task_labels=("cooking", "writing"),
    )


def _response(parsed: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(parsed, ensure_ascii=False)
                }
            }
        ]
    }


def _make_frames() -> tuple[object, object, object]:
    import numpy as np

    return (
        np.full((16, 16, 3), 20, dtype=np.uint8),
        np.full((16, 16, 3), 120, dtype=np.uint8),
        np.full((16, 16, 3), 220, dtype=np.uint8),
    )


class TestReviewer:
    def test_valid_review_and_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_VLM_API_KEY", "secret")
        calls = 0

        def fake_post(url, headers, payload, *, timeout_s):
            nonlocal calls
            calls += 1
            assert "Authorization" in headers
            assert headers["Authorization"].startswith("Bearer ")
            assert "secret" not in payload  # 密钥不进入 payload
            return _response(
                {
                    "scene_label": "kitchen",
                    "task_label": "cooking",
                    "decision": "consistent",
                    "confidence": 0.9,
                    "reasons": "画面与厨房烹饪一致",
                }
            )

        monkeypatch.setattr(
            "zpds.scene.vlm_review._post_json",
            fake_post,
        )
        reviewer = OpenAICompatibleVLMReviewer(
            _make_config(),
            labels=_make_labels(),
            config_hash="hash-a",
        )
        scene = _make_scene()
        result = reviewer.review(scene, _make_frames())
        second = reviewer.review(scene, _make_frames())

        assert isinstance(result, VLMReviewResult)
        assert result.scene_label == "kitchen"
        assert result.task_label == "cooking"
        assert result.decision == "consistent"
        assert result.confidence == 0.9
        assert result.config_hash == "hash-a"
        assert second is result
        assert calls == 1  # 第二次命中缓存

    def test_out_of_set_label_becomes_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_VLM_API_KEY", "secret")
        monkeypatch.setattr(
            "zpds.scene.vlm_review._post_json",
            lambda *args, **kwargs: _response(
                {
                    "scene_label": "bathroom",
                    "task_label": "cooking",
                    "decision": "consistent",
                    "confidence": 0.9,
                    "reasons": "越界标签",
                }
            ),
        )
        reviewer = OpenAICompatibleVLMReviewer(
            _make_config(),
            labels=_make_labels(),
            config_hash="hash-a",
        )
        result = reviewer.review(_make_scene(), _make_frames())

        assert result.decision == "unknown"
        assert result.scene_label == "unknown"
        assert "不在闭集" in result.reasons

    def test_missing_api_key_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_VLM_API_KEY", raising=False)
        reviewer = OpenAICompatibleVLMReviewer(
            _make_config(),
            config_hash="hash-a",
        )
        with pytest.raises(VLMUnavailableError, match="TEST_VLM_API_KEY"):
            reviewer.review(_make_scene(), _make_frames())

    def test_missing_model_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_VLM_API_KEY", "secret")
        monkeypatch.delenv("VLM_MODEL", raising=False)
        reviewer = OpenAICompatibleVLMReviewer(
            VLMConfig(
                enabled=True,
                base_url="http://vlm.test/v1",
                model="",
                api_key_env="TEST_VLM_API_KEY",
            ),
            config_hash="hash-a",
        )
        with pytest.raises(VLMUnavailableError, match="VLM_MODEL"):
            reviewer.review(_make_scene(), _make_frames())

    def test_invalid_decision_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_VLM_API_KEY", "secret")
        monkeypatch.setattr(
            "zpds.scene.vlm_review._post_json",
            lambda *args, **kwargs: _response(
                {
                    "scene_label": "kitchen",
                    "task_label": "cooking",
                    "decision": "maybe",
                    "confidence": 0.9,
                    "reasons": "非法决策",
                }
            ),
        )
        reviewer = OpenAICompatibleVLMReviewer(
            _make_config(),
            labels=_make_labels(),
            config_hash="hash-a",
        )
        with pytest.raises(VLMUnavailableError, match="decision"):
            reviewer.review(_make_scene(), _make_frames())

    def test_api_failure_is_wrapped_as_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_urlopen(request, timeout, context=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(
            "zpds.scene.vlm_review.urllib.request.urlopen",
            fake_urlopen,
        )
        with pytest.raises(VLMUnavailableError, match="VLM API 请求失败"):
            _post_json(
                "http://vlm.test/v1/chat/completions",
                {},
                {"model": "m"},
                timeout_s=5.0,
            )


class TestSceneLabels:
    def test_load_labels(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.yaml"
        path.write_text(
            "scene_labels:\n  - kitchen\n  - office\ntask_labels:\n  - cooking\n",
            encoding="utf-8",
        )
        labels = load_scene_labels(path)
        assert labels.scene_labels == ("kitchen", "office")
        assert labels.task_labels == ("cooking",)

    def test_duplicate_labels_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.yaml"
        path.write_text(
            "scene_labels:\n  - kitchen\n  - kitchen\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="重复"):
            load_scene_labels(path)


class TestReviewQueue:
    def test_inconsistent_and_low_confidence_enter_queue(self) -> None:
        results = [
            VLMReviewResult(
                scene_id="s1",
                scene_label="kitchen",
                task_label="cooking",
                decision="consistent",
                confidence=0.9,
                reasons="ok",
            ),
            VLMReviewResult(
                scene_id="s2",
                scene_label="office",
                task_label="cooking",
                decision="inconsistent",
                confidence=0.95,
                reasons="冲突",
            ),
            VLMReviewResult(
                scene_id="s3",
                scene_label="kitchen",
                task_label="cooking",
                decision="consistent",
                confidence=0.4,
                reasons="低置信",
            ),
        ]
        queue = select_review_queue(results, confidence_threshold=0.6)
        assert [item.scene_id for item in queue] == ["s2", "s3"]
