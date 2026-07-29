"""
MediaPipe 手部检测统一适配层。

根据配置选择后端（Tasks / Solutions），对上层提供单一接口：

    estimator.estimate(frame_rgb, timestamp_ms) → list[RawHandResult]

支持三种后端模式：
- tasks_hand_landmarker: 强制使用新版 Tasks API
- solutions_hands: 强制使用经典 legacy API
- auto: 优先 Tasks，初始化失败时自动回退到 Solutions

用法:
    from zpds.hands.mediapipe_adapter import MediaPipeHandEstimator

    estimator = MediaPipeHandEstimator.from_yaml("config.yaml")
    results = estimator.estimate(frame_rgb, timestamp_ms=0)
    print(estimator.backend_info)  # 查看实际使用的后端
    estimator.close()
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from zpds.hands.base import BackendInfo, ModelInfo, RawHandResult, SessionStats


# ---- 配置 ----

@dataclass
class HandEstimatorConfig:
    """手部估计器配置。

    Attributes:
        backend: 后端选择 ("auto" | "tasks_hand_landmarker" | "solutions_hands")。
        fallback_backend: auto 模式回退目标（当前仅 "solutions_hands"）。
        num_hands: 最大检测手数。
        min_hand_detection_confidence: 手掌检测最低置信度。
        min_hand_presence_confidence: 手部存在最低置信度（仅 Tasks 支持）。
        min_tracking_confidence: 跟踪最低置信度。
        bbox_padding_ratio: BBox 边距比例。
        tasks: Tasks 后端专属配置。
        solutions: Solutions 后端专属配置。
    """

    # ---- 后端选择 ----
    backend: str = "auto"
    fallback_backend: str = "solutions_hands"

    # ---- 通用参数 ----
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    bbox_padding_ratio: float = 0.10

    # ---- 后端专属 ----
    tasks: _TasksConfig = field(default_factory=lambda: _TasksConfig())
    solutions: _SolutionsConfig = field(default_factory=lambda: _SolutionsConfig())

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

        # 解析子配置
        tasks_cfg = hands_cfg.get("tasks", {})
        solutions_cfg = hands_cfg.get("solutions", {})

        return cls(
            backend=hands_cfg.get("backend", "auto"),
            fallback_backend=hands_cfg.get("fallback_backend", "solutions_hands"),
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
            tasks=_TasksConfig(
                model_path=tasks_cfg.get(
                    "model_path", "models/mediapipe/hand_landmarker.task"
                ),
                delegate=tasks_cfg.get("delegate", "cpu"),
            ),
            solutions=_SolutionsConfig(
                model_complexity=int(solutions_cfg.get("model_complexity", 1)),
                input_mirrored=bool(solutions_cfg.get("input_mirrored", False)),
            ),
        )


@dataclass
class _TasksConfig:
    """Tasks 后端专属配置。"""
    model_path: str = "models/mediapipe/hand_landmarker.task"
    delegate: str = "cpu"


@dataclass
class _SolutionsConfig:
    """Solutions 后端专属配置。"""
    model_complexity: int = 1
    input_mirrored: bool = False


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


# ---- 统一适配器 ----

class MediaPipeHandEstimator:
    """MediaPipe 手部检测统一适配层。

    根据配置自动选择/回退后端，对上层保持相同接口。

    用法::

        # 方式 1：构造函数
        estimator = MediaPipeHandEstimator(backend="solutions_hands")

        # 方式 2：YAML 配置
        estimator = MediaPipeHandEstimator.from_yaml("config.yaml")

        # 方式 3：Config 对象
        config = HandEstimatorConfig(backend="auto")
        estimator = MediaPipeHandEstimator.from_config(config)

        results = estimator.estimate(frame_rgb, timestamp_ms=0)
        estimator.close()
    """

    def __init__(
        self,
        # 后端选择
        backend: str = "auto",
        fallback_backend: str = "solutions_hands",
        # 通用参数
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        bbox_padding_ratio: float = 0.10,
        # Tasks 后端专属
        model_path: str | Path = "models/mediapipe/hand_landmarker.task",
        delegate: str = "cpu",
        # Solutions 后端专属
        model_complexity: int = 1,
        input_mirrored: bool = False,
    ):
        """初始化适配器。

        所有参数既可通过构造函数传入，也可通过 from_config / from_yaml 配置。

        Raises:
            ValueError: 后端名称不合法或配置冲突。
        """
        self._config = HandEstimatorConfig(
            backend=backend,
            fallback_backend=fallback_backend,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            bbox_padding_ratio=bbox_padding_ratio,
            tasks=_TasksConfig(model_path=str(model_path), delegate=delegate),
            solutions=_SolutionsConfig(
                model_complexity=model_complexity,
                input_mirrored=input_mirrored,
            ),
        )

        # 模型文件校验与记录
        t_init_start = time.perf_counter()
        self._model_info = ModelInfo.from_file(self._config.tasks.model_path)
        if not self._model_info.exists:
            print(
                f"[MediaPipeHandEstimator] 模型文件不存在: {self._model_info.path}",
                file=sys.stderr,
            )
            print(
                f"[MediaPipeHandEstimator] 下载命令: "
                f"curl -L \"{self._model_info.download_url}\" "
                f"-o \"{self._model_info.path}\"",
                file=sys.stderr,
            )

        self._backend_info: Optional[BackendInfo] = None
        self._last_timestamp_ms: int = -1
        self._timing_history: list[InferenceTiming] = []

        # 会话统计
        self._session_stats = SessionStats(model_info=self._model_info)

        # 初始化后端（含 fallback 逻辑）
        self._backend = self._create_backend()

        # 记录初始化耗时
        self._session_stats.init_time_ms = (time.perf_counter() - t_init_start) * 1000
        self._session_stats.backend_info = self._backend_info

    # ---- 后端创建与选择 ----

    def _create_backend(self):
        """根据配置创建后端实例，处理 auto fallback 逻辑。

        Fallback 策略：
        - 仅捕获 RuntimeError（环境/初始化失败），不吞代码错误。
        - 模型文件不存在直接抛出 FileNotFoundError，不触发 fallback。
        - backend 明确指定时不回退，直接报错。
        """
        cfg = self._config
        requested = cfg.backend
        fallback_reason = ""
        fallback_used = False

        # ---- 强制模式 ----
        if requested == "tasks_hand_landmarker":
            backend = self._init_tasks()
            active_backend_name = "tasks_hand_landmarker"

        elif requested == "solutions_hands":
            backend = self._init_solutions()
            active_backend_name = "solutions_hands"

        # ---- 自动模式 ----
        elif requested == "auto":
            backend, fallback_used, fallback_reason = self._try_auto()
            active_backend_name = (
                "solutions_hands" if fallback_used else "tasks_hand_landmarker"
            )

        else:
            raise ValueError(
                f"不支持的后端类型: {requested!r}，"
                f"可选: auto / tasks_hand_landmarker / solutions_hands"
            )

        # 记录后端信息
        self._backend_info = BackendInfo(
            requested_backend=requested,
            active_backend=active_backend_name,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            delegate=cfg.tasks.delegate
            if active_backend_name == "tasks_hand_landmarker"
            else "",
        )

        if fallback_used:
            print(
                f"[MediaPipeHandEstimator] 注意: {fallback_reason}",
                file=sys.stderr,
            )

        return backend

    def _init_tasks(self):
        """初始化 Tasks 后端。"""
        from zpds.hands.backends.tasks_hand_landmarker import (
            TasksHandLandmarkerBackend,
        )

        return TasksHandLandmarkerBackend(
            model_path=self._config.tasks.model_path,
            num_hands=self._config.num_hands,
            min_hand_detection_confidence=self._config.min_hand_detection_confidence,
            min_hand_presence_confidence=self._config.min_hand_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
            delegate=self._config.tasks.delegate,
        )

    def _init_solutions(self):
        """初始化 Solutions 后端。"""
        from zpds.hands.backends.solutions_hands import SolutionsHandsBackend

        return SolutionsHandsBackend(
            num_hands=self._config.num_hands,
            min_detection_confidence=self._config.min_hand_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
            model_complexity=self._config.solutions.model_complexity,
            input_mirrored=self._config.solutions.input_mirrored,
        )

    def _try_auto(self):
        """auto 模式：优先 Tasks，失败回退 Solutions。

        Returns:
            (backend_instance, fallback_used, fallback_reason)
        """
        # 1) 尝试 Tasks
        try:
            backend = self._init_tasks()
            return backend, False, ""
        except FileNotFoundError:
            raise  # 模型不存在不触发 fallback
        except RuntimeError as exc:
            tasks_error = str(exc)

        # 2) 回退到 Solutions
        try:
            backend = self._init_solutions()
            reason = (
                f"Tasks 初始化失败，自动回退到 solutions_hands。"
                f"Tasks 错误: {tasks_error}"
            )
            return backend, True, reason
        except Exception as exc:
            raise RuntimeError(
                f"Tasks 和 Solutions 后端均初始化失败。"
                f"Tasks 错误: {tasks_error}; Solutions 错误: {exc}"
            ) from exc

    # ---- 上下文管理器 ----

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """释放后端资源。"""
        if hasattr(self, "_backend") and self._backend is not None:
            self._backend.close()
            self._backend = None  # type: ignore[assignment]

    # ---- 属性 ----

    @property
    def config(self) -> HandEstimatorConfig:
        return self._config

    @property
    def model_info(self) -> ModelInfo:
        """模型文件信息（含 SHA-256）。"""
        return self._model_info

    @property
    def backend_info(self) -> Optional[BackendInfo]:
        """实际使用的后端信息（含 fallback 记录）。"""
        return self._backend_info

    @property
    def session_stats(self) -> SessionStats:
        """当前会话统计。"""
        return self._session_stats

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
            timestamp_ms: 帧时间戳（毫秒），必须严格递增。

        Returns:
            RawHandResult 列表，每只手一个。无检测时返回空列表。

        Raises:
            ValueError: 输入帧为空、形状不正确或时间戳未递增。
        """
        t_start = time.perf_counter()

        # ---- 会话统计 ----
        self._session_stats.total_frames += 1

        # ---- 输入校验 ----
        try:
            self._validate_input(frame_rgb, timestamp_ms)
        except ValueError:
            self._session_stats.empty_frames += 1
            self._session_stats.exception_frames += 1
            raise

        # ---- 预处理 ----
        t_pre = time.perf_counter()
        h, w = frame_rgb.shape[:2]
        t_pre_end = time.perf_counter()

        # ---- 后端推理 ----
        t_inf_start = time.perf_counter()
        try:
            raw_result = self._backend.infer(frame_rgb, timestamp_ms)
        except Exception:
            self._session_stats.exception_frames += 1
            raise
        t_inf_end = time.perf_counter()

        # ---- 后处理：统一转换为 RawHandResult ----
        results = self._convert_to_hand_results(raw_result, w, h)

        t_end = time.perf_counter()

        # ---- 无手检测统计 ----
        if len(results) == 0:
            self._session_stats.no_hand_frames += 1
        else:
            self._session_stats.hand_frames += 1

        # ---- 坐标校验（问题 4：确保像素坐标 / 归一化坐标不混淆） ----
        self._validate_coordinate_convention(results, w, h)

        # ---- 记录耗时 ----
        total_ms = (t_end - t_start) * 1000
        timing = InferenceTiming(
            preprocess_ms=(t_pre_end - t_pre) * 1000,
            inference_ms=(t_inf_end - t_inf_start) * 1000,
            postprocess_ms=(t_end - t_inf_end) * 1000,
            total_ms=total_ms,
        )
        self._timing_history.append(timing)

        # 累计推理耗时
        self._session_stats.total_inference_ms += total_ms
        if self._session_stats.total_frames > 0:
            self._session_stats.avg_inference_ms = (
                self._session_stats.total_inference_ms
                / self._session_stats.total_frames
            )

        self._last_timestamp_ms = timestamp_ms

        return results

    # ---- 输入校验 ----

    def _validate_input(self, frame_rgb: np.ndarray, timestamp_ms: int):
        """校验输入帧和时间戳。"""
        if frame_rgb is None:
            raise ValueError("输入帧为 None")
        if not isinstance(frame_rgb, np.ndarray):
            raise ValueError(f"输入帧不是 numpy 数组，类型: {type(frame_rgb)}")
        if frame_rgb.size == 0:
            raise ValueError("输入帧为空（size=0）")
        if frame_rgb.ndim != 3:
            raise ValueError(
                f"输入帧应为 3 维 (H,W,3)，实际 {frame_rgb.ndim} 维，"
                f"形状 {frame_rgb.shape}"
            )
        if frame_rgb.shape[2] != 3:
            raise ValueError(
                f"输入帧通道数应为 3 (RGB)，实际 {frame_rgb.shape[2]}，"
                f"形状 {frame_rgb.shape}"
            )

        # 时间戳单调递增校验
        if timestamp_ms <= self._last_timestamp_ms:
            raise ValueError(
                f"时间戳必须严格递增: "
                f"当前 {timestamp_ms}ms ≤ 上一次 {self._last_timestamp_ms}ms"
            )

    @staticmethod
    def _validate_coordinate_convention(
        results: list[RawHandResult],
        image_width: int,
        image_height: int,
    ):
        """校验坐标约定（问题 4）：确保像素/归一化坐标各司其职。

        - pixel 坐标必须在 [0, image_width) / [0, image_height) 范围内
        - normalized 坐标必须在 [0, 1] 范围内（允许微小浮点误差）
        - 禁止将归一化坐标误写入 pixel 字段

        校验失败时打印 stderr 警告，不抛异常。
        """
        for i, r in enumerate(results):
            kp = r.keypoints
            violation = False

            # 像素坐标应在图像范围内
            for j, (px, py) in enumerate(kp.pixel):
                if px < -1 or px > image_width + 1 or py < -1 or py > image_height + 1:
                    print(
                        f"[MediaPipeHandEstimator] 坐标异常: "
                        f"手#{i} kp[{j}] pixel=({px:.1f}, {py:.1f}) "
                        f"超出图像范围 ({image_width}x{image_height})",
                        file=sys.stderr,
                    )
                    violation = True

            # 归一化坐标应在 [0,1] 范围
            for j, (nx, ny, _nz) in enumerate(kp.normalized):
                if nx < -0.01 or nx > 1.01 or ny < -0.01 or ny > 1.01:
                    print(
                        f"[MediaPipeHandEstimator] 坐标异常: "
                        f"手#{i} kp[{j}] normalized=({nx:.4f}, {ny:.4f}) "
                        f"超出 [0,1] 范围",
                        file=sys.stderr,
                    )
                    violation = True

            # 检测是否误将归一化坐标写入 pixel
            pixel_values = [v for pt in kp.pixel for v in pt]
            all_in_01 = all(0 <= v <= 1 for v in pixel_values)
            if all_in_01 and (image_width > 2 and image_height > 2):
                print(
                    f"[MediaPipeHandEstimator] 坐标约定警告: "
                    f"手#{i} 所有 pixel 坐标都在 [0,1] 内，"
                    f"可能误将归一化坐标写入了 pixel 字段。",
                    file=sys.stderr,
                )

            # 检测是否误将像素坐标写入 normalized
            if (
                kp.normalized
                and abs(kp.normalized[0][0]) > 2.0
                and abs(kp.normalized[0][1]) > 2.0
            ):
                print(
                    f"[MediaPipeHandEstimator] 坐标约定警告: "
                    f"手#{i} normalized 坐标值较大，"
                    f"可能误将像素坐标写入了 normalized 字段。",
                    file=sys.stderr,
                )

    # ---- 统一结果转换 ----

    def _convert_to_hand_results(
        self,
        raw_result,
        image_width: int,
        image_height: int,
    ) -> list[RawHandResult]:
        """将后端原始输出统一转换为 RawHandResult 列表。

        处理两个后端的差异：
        - Tasks: result.hand_landmarks / result.handedness
        - Solutions: result.multi_hand_landmarks / result.multi_handedness
          + 根据 input_mirrored 纠正左右手
        """
        # 提取关键点和左右手（使用后端自身的静态方法）
        landmarks_list = self._backend.extract_landmarks(raw_result)
        handedness_list = self._backend.extract_handedness(raw_result)

        if not landmarks_list:
            return []

        max_hands = self._config.num_hands
        results: list[RawHandResult] = []

        for i in range(min(len(landmarks_list), max_hands)):
            landmarks = landmarks_list[i]

            # 获取 handedness
            if i < len(handedness_list) and handedness_list[i]:
                hc = handedness_list[i][0]  # 取最高分分类
                raw_hand_label = hc.category_name if hc.category_name else "Unknown"
                hand_score = float(hc.score) if hc.score else 0.0
            else:
                raw_hand_label = "Unknown"
                hand_score = 0.0

            # Solutions 后端：根据 input_mirrored 纠正左右手
            if self._backend.name == "solutions_hands":
                hand_label = self._backend.normalize_handedness(
                    raw_hand_label,
                    self._config.solutions.input_mirrored,
                )
            else:
                hand_label = raw_hand_label

            result = RawHandResult.from_mediapipe(
                hand_landmarks=landmarks,
                handedness=_FakeHandedness(hand_label, hand_score),
                image_width=image_width,
                image_height=image_height,
                bbox_padding_ratio=self._config.bbox_padding_ratio,
                hand_index=i,
            )
            results.append(result)

        return results

    # ---- 工厂方法 ----

    @classmethod
    def from_config(cls, config: HandEstimatorConfig) -> "MediaPipeHandEstimator":
        """从 HandEstimatorConfig 实例构造。"""
        return cls(
            backend=config.backend,
            fallback_backend=config.fallback_backend,
            num_hands=config.num_hands,
            min_hand_detection_confidence=config.min_hand_detection_confidence,
            min_hand_presence_confidence=config.min_hand_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            bbox_padding_ratio=config.bbox_padding_ratio,
            model_path=config.tasks.model_path,
            delegate=config.tasks.delegate,
            model_complexity=config.solutions.model_complexity,
            input_mirrored=config.solutions.input_mirrored,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "MediaPipeHandEstimator":
        """从 YAML 配置文件构造。"""
        config = HandEstimatorConfig.from_yaml(yaml_path)
        return cls.from_config(config)


# ---- 内部辅助 ----

class _FakeHandedness:
    """模拟 MediaPipe Category 对象，统一转换用。"""

    def __init__(self, category_name: str, score: float):
        self.category_name = category_name
        self.score = score
