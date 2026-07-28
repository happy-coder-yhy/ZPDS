"""
MediaPipe Hand Landmarker 适配器。

将 MediaPipe Tasks API 的输出统一转换为 RawHandResult，
提供配置驱动的初始化、单帧推理和资源管理。

用法:
    from zpds.hands.mediapipe_adapter import MediaPipeHandEstimator

    estimator = MediaPipeHandEstimator(model_path="models/mediapipe/hand_landmarker.task")
    results = estimator.estimate(frame_rgb, timestamp_ms=0)
    estimator.close()
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from zpds.hands.base import RawHandResult


# ---- 配置 ----

@dataclass
class HandEstimatorConfig:
    """手部估计器配置。

    Attributes:
        model: 模型后端名称（"mediapipe"）。
        model_path: .task 模型文件路径。
        num_hands: 最大检测手数。
        min_hand_detection_confidence: 手掌检测最低置信度。
        min_hand_presence_confidence: 手部存在最低置信度。
        min_tracking_confidence: 手部跟踪最低置信度。
        bbox_padding_ratio: BBox 边距扩展比例。
    """

    model: str = "mediapipe"
    model_path: str = "models/mediapipe/hand_landmarker.task"
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    bbox_padding_ratio: float = 0.10

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "HandEstimatorConfig":
        """从 YAML 配置文件加载。期望顶层 ``hands:`` 键。"""
        import yaml

        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {path.resolve()}")

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if data is None:
            raise ValueError(f"配置文件为空: {path.resolve()}")

        hands_cfg = data.get("hands", data)
        return cls(
            model=hands_cfg.get("model", "mediapipe"),
            model_path=hands_cfg.get("model_path", "models/mediapipe/hand_landmarker.task"),
            num_hands=int(hands_cfg.get("num_hands", 2)),
            min_hand_detection_confidence=float(
                hands_cfg.get("min_hand_detection_confidence", 0.5)
            ),
            min_hand_presence_confidence=float(
                hands_cfg.get("min_hand_presence_confidence", 0.5)
            ),
            min_tracking_confidence=float(
                hands_cfg.get("min_tracking_confidence", 0.5)
            ),
            bbox_padding_ratio=float(hands_cfg.get("bbox_padding_ratio", 0.10)),
        )


# ---- 推理耗时统计 ----

@dataclass
class InferenceTiming:
    """单帧推理耗时分解（毫秒）。"""

    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def fps(self) -> float:
        """等效 FPS。"""
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0


# ---- MediaPipe 适配器 ----

class MediaPipeHandEstimator:
    """MediaPipe Hand Landmarker VIDEO 模式封装。

    使用 VIDEO 模式（detect_for_video）利用帧间跟踪，
    比 IMAGE 模式逐帧独立检测更快更稳定。

    用法::

        estimator = MediaPipeHandEstimator(
            model_path="models/mediapipe/hand_landmarker.task",
            num_hands=2,
        )
        results = estimator.estimate(frame_rgb, timestamp_ms=33)
        # ... 继续推理 ...
        estimator.close()

        # 或用作上下文管理器：
        with MediaPipeHandEstimator() as estimator:
            results = estimator.estimate(frame, 0)
    """

    def __init__(
        self,
        model_path: str | Path = "models/mediapipe/hand_landmarker.task",
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        bbox_padding_ratio: float = 0.10,
    ):
        """初始化 MediaPipe Hand Landmarker。

        Args:
            model_path: .task 模型文件路径。
            num_hands: 最大检测手数。
            min_hand_detection_confidence: 手掌检测最低置信度 [0, 1]。
            min_hand_presence_confidence: 手部存在最低置信度 [0, 1]。
            min_tracking_confidence: 跟踪最低置信度 [0, 1]。
            bbox_padding_ratio: BBox 边距比例。

        Raises:
            FileNotFoundError: 模型文件不存在。
            RuntimeError: MediaPipe 初始化失败。
        """
        self._config = HandEstimatorConfig(
            model="mediapipe",
            model_path=str(model_path),
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            bbox_padding_ratio=bbox_padding_ratio,
        )

        self._model_path = Path(model_path)
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"找不到 MediaPipe 模型文件: {self._model_path.resolve()}"
            )

        import mediapipe as mp

        self._mp = mp

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self._model_path.resolve()),
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms: int = -1
        self._timing_history: list[InferenceTiming] = []

    # ---- 上下文管理器 ----

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """释放 MediaPipe 资源。"""
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None  # type: ignore[assignment]

    # ---- 属性 ----

    @property
    def config(self) -> HandEstimatorConfig:
        return self._config

    @property
    def timing_history(self) -> list[InferenceTiming]:
        return list(self._timing_history)

    @property
    def average_timing(self) -> InferenceTiming:
        """累计推理耗时均值。"""
        if not self._timing_history:
            return InferenceTiming()
        n = len(self._timing_history)
        return InferenceTiming(
            preprocess_ms=sum(t.preprocess_ms for t in self._timing_history) / n,
            inference_ms=sum(t.inference_ms for t in self._timing_history) / n,
            postprocess_ms=sum(t.postprocess_ms for t in self._timing_history) / n,
            total_ms=sum(t.total_ms for t in self._timing_history) / n,
        )

    # ---- 核心推理 ----

    def estimate(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> list[RawHandResult]:
        """对单帧 RGB 图像运行手部检测。

        Args:
            frame_rgb: RGB uint8 图像 (H, W, 3)。
            timestamp_ms: 帧时间戳（毫秒），VIDEO 模式下必须严格递增。

        Returns:
            RawHandResult 列表，每只手一个。无检测时为空列表。

        Raises:
            ValueError: 输入帧为空、形状不正确或时间戳未递增。
        """
        t_start = time.perf_counter()

        # ---- 输入校验 ----
        self._validate_input(frame_rgb, timestamp_ms)

        # ---- 预处理 ----
        t_pre = time.perf_counter()
        mp_image = self._preprocess(frame_rgb)
        t_pre_end = time.perf_counter()

        # ---- MediaPipe 推理 ----
        t_inf_start = time.perf_counter()

        detection_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        t_inf_end = time.perf_counter()

        # ---- 后处理：转换 → RawHandResult ----
        results = self._postprocess(detection_result, frame_rgb)

        t_end = time.perf_counter()

        # ---- 记录耗时 ----
        timing = InferenceTiming(
            preprocess_ms=(t_pre_end - t_pre) * 1000,
            inference_ms=(t_inf_end - t_inf_start) * 1000,
            postprocess_ms=(t_end - t_inf_end) * 1000,
            total_ms=(t_end - t_start) * 1000,
        )
        self._timing_history.append(timing)

        self._last_timestamp_ms = timestamp_ms

        return results

    # ---- 内部方法 ----

    def _validate_input(self, frame_rgb: np.ndarray, timestamp_ms: int):
        """校验输入帧和时间戳。"""
        # 空帧检测
        if frame_rgb is None:
            raise ValueError("输入帧为 None")
        if not isinstance(frame_rgb, np.ndarray):
            raise ValueError(f"输入帧不是 numpy 数组，类型: {type(frame_rgb)}")
        if frame_rgb.size == 0:
            raise ValueError("输入帧为空（size=0）")
        if frame_rgb.ndim != 3:
            raise ValueError(f"输入帧应为 3 维 (H,W,3)，实际 {frame_rgb.ndim} 维，形状 {frame_rgb.shape}")
        if frame_rgb.shape[2] != 3:
            raise ValueError(f"输入帧通道数应为 3 (RGB)，实际 {frame_rgb.shape[2]}，形状 {frame_rgb.shape}")

        # 时间戳单调递增校验
        if timestamp_ms <= self._last_timestamp_ms:
            raise ValueError(
                f"VIDEO 模式时间戳必须严格递增: "
                f"当前 {timestamp_ms}ms ≤ 上一次 {self._last_timestamp_ms}ms"
            )

    def _preprocess(self, frame_rgb: np.ndarray):
        """预处理：numpy → MediaPipe Image。"""
        # MediaPipe Image 创建本身会做格式转换如果需要
        return self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

    def _postprocess(
        self,
        detection_result,
        frame_rgb: np.ndarray,
    ) -> list[RawHandResult]:
        """将 MediaPipe 检测结果转换为 RawHandResult 列表。

        处理：
        - 无检测 → 空列表
        - 关键点越界 → 裁剪到 [0, 1]
        - 超过配置手数 → 截断到 num_hands
        """
        if detection_result is None:
            return []

        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness

        if not hand_landmarks_list:
            return []

        h, w = frame_rgb.shape[:2]
        max_hands = self._config.num_hands

        results: list[RawHandResult] = []

        for i in range(min(len(hand_landmarks_list), max_hands)):
            landmarks = hand_landmarks_list[i]
            handedness = handedness_list[i][0]  # 每只手一个 Category，取 score 最高的

            # 关键点数量校验 + 越界裁剪
            self._validate_keypoints(landmarks, i)

            result = RawHandResult.from_mediapipe(
                hand_landmarks=landmarks,
                handedness=handedness,
                image_width=w,
                image_height=h,
                bbox_padding_ratio=self._config.bbox_padding_ratio,
                hand_index=i,
            )
            results.append(result)

        return results

    @staticmethod
    def _validate_keypoints(landmarks, hand_index: int):
        """校验关键点数量和合法性。"""
        if len(landmarks) != 21:
            # 不抛异常 — 记录并跳过越界关键点在后处理中处理
            print(
                f"[MediaPipeHandEstimator] 警告: 手 #{hand_index} "
                f"关键点数={len(landmarks)}（期望 21）",
                file=sys.stderr,
            )

    @classmethod
    def from_config(cls, config: HandEstimatorConfig) -> "MediaPipeHandEstimator":
        """从 HandEstimatorConfig 实例构造。"""
        return cls(
            model_path=config.model_path,
            num_hands=config.num_hands,
            min_hand_detection_confidence=config.min_hand_detection_confidence,
            min_hand_presence_confidence=config.min_hand_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            bbox_padding_ratio=config.bbox_padding_ratio,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "MediaPipeHandEstimator":
        """从 YAML 配置文件构造。"""
        config = HandEstimatorConfig.from_yaml(yaml_path)
        return cls.from_config(config)
