"""
Tasks Hand Landmarker 后端（新版 API）。

使用 mp.tasks.vision.HandLandmarker + .task 模型文件，
要求 VIDEO 模式 + 严格递增时间戳。

适用环境：Linux / Windows / 非沙箱 macOS。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


class TasksHandLandmarkerBackend:
    """MediaPipe Tasks API 手部检测后端。

    使用 detect_for_video() 进行帧间跟踪推理。

    用法::

        backend = TasksHandLandmarkerBackend(
            model_path="models/mediapipe/hand_landmarker.task",
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            delegate="cpu",
        )
        raw = backend.infer(frame_rgb, timestamp_ms=33)
        backend.close()
    """

    name = "tasks_hand_landmarker"

    def __init__(
        self,
        model_path: str | Path = "models/mediapipe/hand_landmarker.task",
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        delegate: str = "cpu",
    ):
        """初始化 Tasks Hand Landmarker。

        Args:
            model_path: .task 模型文件路径。
            num_hands: 最大检测手数。
            min_hand_detection_confidence: 手掌检测最低置信度。
            min_hand_presence_confidence: 手部存在最低置信度（仅 Tasks 支持）。
            min_tracking_confidence: 跟踪最低置信度。
            delegate: 推理设备，"cpu" 或 "gpu"。

        Raises:
            FileNotFoundError: 模型文件不存在。
            RuntimeError: Tasks API 初始化失败（可能触发 fallback）。
        """
        self._model_path = Path(model_path)
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"找不到 MediaPipe 模型文件: {self._model_path.resolve()}"
            )

        import mediapipe as mp

        self._mp = mp

        # 解析 delegate
        delegate_map = {
            "cpu": mp.tasks.BaseOptions.Delegate.CPU,
            "gpu": mp.tasks.BaseOptions.Delegate.GPU,
        }
        delegate_enum = delegate_map.get(
            delegate.lower(), mp.tasks.BaseOptions.Delegate.CPU
        )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self._model_path.resolve()),
            delegate=delegate_enum,
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        try:
            self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                f"Tasks Hand Landmarker 初始化失败: {exc}"
            ) from exc

    # ---- 核心推理 ----

    def infer(self, frame_rgb, timestamp_ms: int):
        """执行单帧推理。

        Args:
            frame_rgb: RGB uint8 图像 (H, W, 3)。
            timestamp_ms: 帧时间戳（毫秒），必须严格递增。

        Returns:
            MediaPipe HandLandmarkerResult 原始对象。
        """
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        return self._detector.detect_for_video(mp_image, timestamp_ms)

    # ---- 原始结果提取 ----

    @staticmethod
    def extract_landmarks(result) -> list:
        """从 Tasks 结果提取关键点列表。"""
        if result is None:
            return []
        return list(result.hand_landmarks) if result.hand_landmarks else []

    @staticmethod
    def extract_handedness(result) -> list[list]:
        """从 Tasks 结果提取左右手列表。

        Returns:
            [[Category, ...], [Category, ...]] — 每只手一个列表。
        """
        if result is None:
            return []
        return list(result.handedness) if result.handedness else []

    # ---- 资源释放 ----

    def close(self):
        """释放 MediaPipe 资源。"""
        if hasattr(self, "_detector") and self._detector is not None:
            self._detector.close()
            self._detector = None  # type: ignore[assignment]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
