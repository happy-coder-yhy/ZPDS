"""DINOv2-Small 语义 embedding 与相邻关键帧边界评分。"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any

import cv2
import numpy as np

from zpds.scene.backends.common import timestamp_ns, to_bgr, validate_frames
from zpds.scene.config import DinoConfig
from zpds.scene.schemas import BoundaryScore

DINO_SMALL_MODEL_ID = "facebook/dinov2-small"
DINO_SMALL_EMBEDDING_DIMENSION = 384
SCENE_EXTRA_ERROR = (
    'DINOv2 backend requires the scene extra: pip install -e ".[scene]"'
)
EmbeddingFunction = Callable[[Sequence[np.ndarray]], np.ndarray]


class DinoV2SmallEmbedder:
    """唯一的 v1 语义后端，运行时依赖在首次推理时才导入。"""

    def __init__(
        self,
        config: DinoConfig,
        *,
        embedding_function: EmbeddingFunction | None = None,
    ) -> None:
        if config.model != DINO_SMALL_MODEL_ID:
            raise ValueError(f"DINO v1 仅支持 {DINO_SMALL_MODEL_ID}")
        self.config = config
        self._embedding_function = embedding_function
        self._torch: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None

    @property
    def embedding_dimension(self) -> int:
        return DINO_SMALL_EMBEDDING_DIMENSION

    @property
    def runtime_loaded(self) -> bool:
        return self._model is not None

    def _load_runtime(self) -> None:
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(SCENE_EXTRA_ERROR) from error

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"DINOv2 配置请求 {self.config.device}，但 CUDA 不可用"
            )
        try:
            processor_class = transformers.AutoImageProcessor
            model_class = transformers.AutoModel
        except AttributeError as error:
            raise RuntimeError(
                "transformers 缺少 AutoImageProcessor/AutoModel；"
                f"{SCENE_EXTRA_ERROR}"
            ) from error

        self._processor = processor_class.from_pretrained(self.config.model)
        self._model = model_class.from_pretrained(self.config.model)
        self._model.eval()
        self._model.to(self.config.device)
        self._torch = torch

    def _transformers_embed(
        self,
        frames_rgb: Sequence[np.ndarray],
    ) -> np.ndarray:
        self._load_runtime()
        assert self._torch is not None
        assert self._processor is not None
        assert self._model is not None
        batches: list[np.ndarray] = []
        for start in range(0, len(frames_rgb), self.config.batch_size):
            batch = list(frames_rgb[start : start + self.config.batch_size])
            inputs = self._processor(images=batch, return_tensors="pt")
            inputs = {
                name: tensor.to(self.config.device)
                for name, tensor in inputs.items()
            }
            with self._torch.inference_mode():
                output = self._model(**inputs)
                cls_embedding = output.last_hidden_state[:, 0, :]
            batches.append(cls_embedding.detach().cpu().numpy())
        return np.concatenate(batches, axis=0)

    def _validate_and_normalise_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        expected_rows: int,
    ) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        expected_shape = (expected_rows, self.embedding_dimension)
        if values.shape != expected_shape:
            raise ValueError(
                f"DINOv2 embedding 形状必须是 {expected_shape}，实际为 {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("DINOv2 embedding 包含 NaN 或无穷值")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(np.float32).eps):
            raise ValueError("DINOv2 embedding 不能包含零向量")
        return values / norms

    def embed(self, frames_rgb: Sequence[np.ndarray]) -> np.ndarray:
        for index, frame in enumerate(frames_rgb):
            if not isinstance(frame, np.ndarray):
                raise TypeError(f"frames_rgb[{index}] 必须是 numpy.ndarray")
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
                raise ValueError(f"frames_rgb[{index}] 必须是非空 RGB 图像")
        if not frames_rgb:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)
        if self._embedding_function is None:
            embeddings = self._transformers_embed(frames_rgb)
        else:
            embeddings = self._embedding_function(frames_rgb)
        return self._validate_and_normalise_embeddings(
            embeddings,
            expected_rows=len(frames_rgb),
        )

    def local_z_scores(
        self,
        changes: Sequence[float] | np.ndarray,
    ) -> tuple[float, ...]:
        """使用当前样本两侧各 ``z_score_window`` 个邻居计算 z-score。"""

        values = np.asarray(changes, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("changes 必须是一维有限数值序列")
        radius = self.config.z_score_window
        z_scores = np.zeros(len(values), dtype=np.float64)
        epsilon = np.finfo(np.float64).eps
        for index, value in enumerate(values):
            start = max(0, index - radius)
            end = min(len(values), index + radius + 1)
            neighbors = np.concatenate((values[start:index], values[index + 1 : end]))
            if len(neighbors) < self.config.min_z_score_samples:
                continue
            mean = float(np.mean(neighbors))
            deviation = float(np.std(neighbors))
            difference = float(value - mean)
            if deviation <= epsilon:
                z_scores[index] = difference / epsilon if difference > 0.0 else 0.0
            else:
                z_scores[index] = difference / deviation
        return tuple(float(value) for value in z_scores)

    def score_boundaries(
        self,
        frames_rgb: Sequence[np.ndarray],
        *,
        frame_indices: Sequence[int],
        timestamps_ns: Sequence[int],
    ) -> list[BoundaryScore]:
        if not (
            len(frames_rgb) == len(frame_indices) == len(timestamps_ns)
        ):
            raise ValueError("frames_rgb、frame_indices、timestamps_ns 长度必须一致")
        if any(
            current <= previous
            for previous, current in pairwise(frame_indices)
        ):
            raise ValueError("frame_indices 必须严格递增")
        if any(
            current <= previous
            for previous, current in pairwise(timestamps_ns)
        ):
            raise ValueError("timestamps_ns 必须严格递增")
        if len(frames_rgb) < 2:
            return []

        embeddings = self.embed(frames_rgb)
        similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        changes = np.zeros(len(frames_rgb), dtype=np.float64)
        changes[1:] = np.clip(1.0 - similarities, 0.0, 1.0)
        z_scores = self.local_z_scores(changes)
        return [
            BoundaryScore(
                frame_index=int(frame_indices[index]),
                timestamp_ns=int(timestamps_ns[index]),
                score=float(changes[index]),
                z_score=z_scores[index],
            )
            for index in range(1, len(frames_rgb))
            if z_scores[index] > self.config.z_score_threshold
        ]

    def sample_frame_indices(
        self,
        *,
        frame_count: int,
        fps: float,
        candidate_frame_indices: Sequence[int] | None = None,
    ) -> tuple[int, ...]:
        if isinstance(frame_count, bool) or frame_count < 0:
            raise ValueError("frame_count 必须是非负整数")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps 必须是大于 0 的有限数值")
        if frame_count == 0:
            return ()
        if self.config.sample_fps > fps:
            raise ValueError("scene.stage_b.sample_fps 不能高于视频 fps")

        if candidate_frame_indices is None:
            intervals = [(0, frame_count - 1)]
        else:
            context = round(self.config.candidate_context_s * fps)
            intervals = []
            for candidate in sorted(set(candidate_frame_indices)):
                if isinstance(candidate, bool) or not 0 <= candidate < frame_count:
                    raise ValueError(f"候选帧号超出视频范围: {candidate}")
                intervals.append(
                    (max(0, candidate - context), min(frame_count - 1, candidate + context))
                )

        stride = fps / self.config.sample_fps
        sampled: set[int] = set()
        for start, end in intervals:
            position = float(start)
            while position <= end + np.finfo(np.float64).eps:
                sampled.add(min(end, round(position)))
                position += stride
        return tuple(sorted(sampled))

    def detect(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        fps: float,
        start_timestamp_ns: int = 0,
        candidate_frame_indices: Sequence[int] | None = None,
    ) -> list[BoundaryScore]:
        """按 1 FPS 扫描全视频，或扫描候选点前后各 2 秒。"""

        validate_frames(frames_bgr, fps=fps)
        indices = self.sample_frame_indices(
            frame_count=len(frames_bgr),
            fps=fps,
            candidate_frame_indices=candidate_frame_indices,
        )
        frames_rgb = [
            cv2.cvtColor(to_bgr(frames_bgr[index]), cv2.COLOR_BGR2RGB)
            for index in indices
        ]
        timestamps = [
            timestamp_ns(
                index,
                fps=fps,
                start_timestamp_ns=start_timestamp_ns,
            )
            for index in indices
        ]
        return self.score_boundaries(
            frames_rgb,
            frame_indices=indices,
            timestamps_ns=timestamps,
        )


__all__ = [
    "DINO_SMALL_EMBEDDING_DIMENSION",
    "DINO_SMALL_MODEL_ID",
    "SCENE_EXTRA_ERROR",
    "DinoV2SmallEmbedder",
    "EmbeddingFunction",
]
