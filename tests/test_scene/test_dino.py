from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np
import pytest

from zpds.scene.backends.dino import (
    DINO_SMALL_EMBEDDING_DIMENSION,
    SCENE_EXTRA_ERROR,
    DinoV2SmallEmbedder,
)
from zpds.scene.config import SceneConfig
from zpds.scene.contracts import SemanticEmbedder


@pytest.fixture(scope="module")
def config() -> SceneConfig:
    return SceneConfig.load("configs/scene/default.yaml")


def _semantic_embeddings(frames_rgb) -> np.ndarray:
    embeddings = np.zeros(
        (len(frames_rgb), DINO_SMALL_EMBEDDING_DIMENSION),
        dtype=np.float32,
    )
    for index, frame in enumerate(frames_rgb):
        dimension = 0 if float(np.mean(frame)) < 128.0 else 1
        embeddings[index, dimension] = 1.0
    return embeddings


def test_import_scene_does_not_import_torch_or_transformers() -> None:
    command = (
        "import sys; import zpds; import zpds.scene; "
        "assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_scene_extra_has_actionable_error(
    config: SceneConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = importlib.import_module

    def missing_runtime(name: str, package: str | None = None):
        if name in {"torch", "transformers"}:
            raise ModuleNotFoundError(name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_runtime)
    embedder = DinoV2SmallEmbedder(config.stage_b)

    assert embedder.runtime_loaded is False
    with pytest.raises(RuntimeError, match=r"pip install -e") as error:
        embedder.embed([np.zeros((8, 8, 3), dtype=np.uint8)])
    assert SCENE_EXTRA_ERROR in str(error.value)


def test_fake_embedder_satisfies_protocol_and_normalises(config: SceneConfig) -> None:
    embedder = DinoV2SmallEmbedder(
        config.stage_b,
        embedding_function=lambda frames: np.full(
            (len(frames), DINO_SMALL_EMBEDDING_DIMENSION),
            2.0,
            dtype=np.float32,
        ),
    )

    result = embedder.embed([np.zeros((8, 8, 3), dtype=np.uint8)])

    assert isinstance(embedder, SemanticEmbedder)
    assert result.shape == (1, 384)
    assert np.linalg.norm(result[0]) == pytest.approx(1.0)
    assert embedder.runtime_loaded is False


def test_embedding_dimension_and_zero_vectors_are_rejected(config: SceneConfig) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    wrong_dimension = DinoV2SmallEmbedder(
        config.stage_b,
        embedding_function=lambda frames: np.ones(
            (len(frames), 383),
            dtype=np.float32,
        ),
    )
    with pytest.raises(ValueError, match="形状"):
        wrong_dimension.embed([frame])

    zero_vector = DinoV2SmallEmbedder(
        config.stage_b,
        embedding_function=lambda frames: np.zeros(
            (len(frames), 384),
            dtype=np.float32,
        ),
    )
    with pytest.raises(ValueError, match="零向量"):
        zero_vector.embed([frame])


def test_candidate_sampling_uses_one_fps_and_two_second_context(
    config: SceneConfig,
) -> None:
    embedder = DinoV2SmallEmbedder(
        config.stage_b,
        embedding_function=_semantic_embeddings,
    )

    full = embedder.sample_frame_indices(frame_count=100, fps=10.0)
    candidate = embedder.sample_frame_indices(
        frame_count=100,
        fps=10.0,
        candidate_frame_indices=[50],
    )

    assert full == (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
    assert candidate == (30, 40, 50, 60, 70)


def test_cosine_change_and_local_z_score_find_semantic_boundary(
    config: SceneConfig,
) -> None:
    embedder = DinoV2SmallEmbedder(
        config.stage_b,
        embedding_function=_semantic_embeddings,
    )
    frames = [
        np.full((4, 4, 3), 0 if index < 35 else 255, dtype=np.uint8)
        for index in range(70)
    ]

    boundaries = embedder.score_boundaries(
        frames,
        frame_indices=list(range(70)),
        timestamps_ns=[index * 1_000_000_000 for index in range(70)],
    )

    assert len(boundaries) == 1
    assert boundaries[0].frame_index == 35
    assert boundaries[0].timestamp_ns == 35_000_000_000
    assert boundaries[0].score == pytest.approx(1.0)
    assert boundaries[0].z_score > config.stage_b.z_score_threshold


def test_detect_scans_bgr_video_and_preserves_original_frame_index(
    config: SceneConfig,
) -> None:
    embedder = DinoV2SmallEmbedder(
        config.stage_b,
        embedding_function=_semantic_embeddings,
    )
    frames = [
        np.full((4, 4, 3), 0 if index < 350 else 255, dtype=np.uint8)
        for index in range(700)
    ]

    boundaries = embedder.detect(
        frames,
        fps=10.0,
        start_timestamp_ns=123,
    )

    assert len(boundaries) == 1
    assert boundaries[0].frame_index == 350
    assert boundaries[0].timestamp_ns == 35_000_000_123


def test_score_boundaries_rejects_misaligned_metadata(config: SceneConfig) -> None:
    embedder = DinoV2SmallEmbedder(
        config.stage_b,
        embedding_function=_semantic_embeddings,
    )
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * 2

    with pytest.raises(ValueError, match="长度"):
        embedder.score_boundaries(
            frames,
            frame_indices=[0],
            timestamps_ns=[0, 1],
        )
