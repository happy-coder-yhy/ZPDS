"""WiLoR 手部估计器（阶段 3：帧状态、失败记录和回退）。

本模块是 WiLoR 面向 Pipeline 的统一入口，负责：
- 每帧产生明确的 :class:`HandFrameResult`（不通过空列表掩盖失败）
- 按 :class:`WiLoRFallbackPolicy` 决定是否调用 MediaPipe 回退
- 支持对照模式（compare_with_mediapipe）
- 记录完整帧统计，对齐 sample map

职责边界：
    - Backend：模型加载 + 原始推理
    - Adapter：坐标逆变换 + BBox 校验
    - **Estimator（本模块）：帧状态、失败记录、回退调度**
    - Router（后续）：primary/fallback 对外统一入口
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from zpds.hands.schemas import (
    HandFrameResult,
    ModelAttemptResult,
    RawHandResult,
)
from zpds.hands.wilor_joint_mapping import (
    WILOR_TO_HANDS_V1_V1,
    convert_wilor_to_raw_hand_result,
)
from zpds.hands.wilor_schema import (
    FRAME_LEVEL_ERRORS,
    RUN_LEVEL_ERRORS,
    WiLoRError,
    WiLoRFallbackPolicy,
    WiLoRModelInfo,
    WiLoRRunReport,
    WiLoRRunThresholds,
)


# ════════════════════════════════════════════════════════════════════
# Fallback estimator 最小接口（鸭子类型）
# ════════════════════════════════════════════════════════════════════


class FallbackEstimator(Protocol):
    """回退估计器所需的接口。

    MediaPipeHandEstimator 天然满足此协议。
    """

    def estimate(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> list[RawHandResult]:
        ...

    @property
    def config(self) -> object: ...

    def close(self) -> None: ...


# ════════════════════════════════════════════════════════════════════
# WiLoR 运行配置
# ════════════════════════════════════════════════════════════════════


@dataclass
class WiLoREstimatorConfig:
    """WiLoRHandEstimator 的初始化配置。"""

    fallback_policy: WiLoRFallbackPolicy = field(
        default_factory=WiLoRFallbackPolicy,
    )
    model_name: str = "wilor"
    model_version: str = ""


# ════════════════════════════════════════════════════════════════════
# 帧统计
# ════════════════════════════════════════════════════════════════════


@dataclass
class WiLoRFrameStats:
    """WiLoR 单 session 的逐帧累计统计。

    满足恒等式::

        total_frames
        = detected + no_hand + failed + skipped_invalid_input
    """

    total_frames: int = 0
    detected: int = 0
    no_hand: int = 0
    failed: int = 0
    skipped_invalid_input: int = 0

    fallback_attempted: int = 0
    fallback_used: int = 0

    total_inference_ms: float = 0.0

    @property
    def avg_inference_ms(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.total_inference_ms / self.total_frames

    @property
    def failure_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.failed / self.total_frames


# ════════════════════════════════════════════════════════════════════
# 估计器
# ════════════════════════════════════════════════════════════════════


class WiLoRHandEstimator:
    """WiLoR 手部估计器。

    封装 WiLoRAdapter → 帧状态管理 → 回退调度 → HandFrameResult。

    用法::

        estimator = WiLoRHandEstimator(
            adapter=wilor_adapter,
            model_info=backend.model_info,
            fallback_estimator=mediapipe_estimator,
        )
        frame_result = estimator.estimate_frame(frame_rgb, timestamp_ms=0)
        print(frame_result.primary.status)  # detected / no_hand / failed / ...
    """

    def __init__(
        self,
        *,
        adapter: object,  # WiLoRAdapter（避免顶层导入）
        model_info: WiLoRModelInfo,
        fallback_estimator: FallbackEstimator | None = None,
        config: WiLoREstimatorConfig | None = None,
        thresholds: WiLoRRunThresholds | None = None,
    ) -> None:
        """初始化估计器。

        Args:
            adapter: WiLoRAdapter 实例（有 detect() 方法）。
            model_info: 从 WiLoRBackend 采集的运行时元信息。
            fallback_estimator: MediaPipe 回退估计器（可选，None 时不回退）。
            config: 回退策略等配置。
            thresholds: 运行级失败阈值。为 None 时使用默认值。
        """
        self._adapter = adapter
        self._model_info = model_info
        self._fallback = fallback_estimator
        self._config = config or WiLoREstimatorConfig()
        self._thresholds = thresholds or WiLoRRunThresholds()
        self._stats = WiLoRFrameStats()
        self._last_timestamp_ms: int = -1

        # 运行级状态
        self._consecutive_failures: int = 0
        self._run_aborted: bool = False
        self._abort_reason: str | None = None
        self._run_errors: list[dict] = []
        self._inference_times: list[float] = []

        # 对照模式累计（与回退独立）
        self._comparison_runs: list[ModelAttemptResult] = []

        # 初始化计时器列表
        self._init_time_ms = 0.0

    # ---- 属性 ----

    @property
    def model_info(self) -> WiLoRModelInfo:
        return self._model_info

    @property
    def frame_stats(self) -> WiLoRFrameStats:
        return self._stats

    @property
    def config(self) -> WiLoREstimatorConfig:
        return self._config

    # ---- 主入口 ----

    def estimate_frame(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> HandFrameResult:
        """对单帧执行 WiLoR 检测，返回完整帧状态。

        与返回 ``list[RawHandResult]`` 不同，本方法始终产生明确的
        :class:`HandFrameResult`，确保 failure / no_hand / invalid_input
        不会全部被空列表掩盖。

        Args:
            frame_rgb: RGB uint8 图像 ``(H, W, 3)``。
            timestamp_ms: 帧时间戳（毫秒），必须严格递增。

        Returns:
            完整帧状态，不含糊。
        """
        self._stats.total_frames += 1

        # ---- 运行级终止检查 ----
        if self._run_aborted:
            primary = ModelAttemptResult(
                model_name="wilor",
                backend_name="wilor",
                status="not_run",
                hands=[],
                inference_ms=0.0,
                failure_reason=f"Run aborted: {self._abort_reason}",
                model_version=self._model_info.model_version,
                checkpoint_sha256=self._model_info.checkpoint_sha256,
                device=self._model_info.device,
            )
            return HandFrameResult(
                timestamp_ms=timestamp_ms,
                requested_model="wilor",
                primary=primary,
                fallback=None,
                fallback_attempted=False,
                fallback_used=False,
                effective_model=None,
                effective_hands=[],
            )

        # ---- 输入校验 ----
        try:
            _validate_input(frame_rgb, timestamp_ms)
            if timestamp_ms <= self._last_timestamp_ms:
                raise ValueError(
                    f"timestamp_ms 必须严格递增: "
                    f"当前 {timestamp_ms} ≤ 上次 {self._last_timestamp_ms}"
                )
        except (TypeError, ValueError) as exc:
            self._stats.skipped_invalid_input += 1
            return self._invalid_input_result(
                timestamp_ms=timestamp_ms,
                reason=str(exc),
            )

        self._last_timestamp_ms = timestamp_ms

        # ---- WiLoR 主模型 ----
        primary = self._run_wilor(frame_rgb, timestamp_ms)

        # ---- 运行级终止检查 ----
        self.should_abort()

        # ---- 对照模式（可选） ----
        if (
            self._config.fallback_policy.compare_with_mediapipe
            and self._fallback is not None
        ):
            self._run_comparison(frame_rgb, timestamp_ms)

        # ---- 回退策略 ----
        return self._apply_fallback_policy(
            frame_rgb=frame_rgb,
            timestamp_ms=timestamp_ms,
            primary=primary,
        )

    def estimate(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> list[RawHandResult]:
        """兼容旧 :class:`HandEstimator` Protocol。

        委托给 :meth:`estimate_frame`，返回 ``effective_hands``。
        注意：在 21 点映射未验收前，effective_hands 恒为空。
        """
        result = self.estimate_frame(frame_rgb, timestamp_ms)
        return list(result.effective_hands)

    # ---- 内部 ----

    def _run_wilor(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> ModelAttemptResult:
        """执行 WiLoR 检测并返回 ModelAttemptResult。

        分类异常：运行级异常直接抛出，单帧级异常记录后继续。
        """
        t_start = time.perf_counter()

        try:
            detections = self._adapter.detect(frame_rgb, timestamp_ms)
            inference_ms = (time.perf_counter() - t_start) * 1000

            self._consecutive_failures = 0
            self._inference_times.append(inference_ms)
            self._stats.total_inference_ms += inference_ms

            if detections:
                self._stats.detected += 1
                status = "detected"
                hands: list[RawHandResult] = []
                for det in detections:
                    try:
                        iw = det.transform.original_width if det.transform else frame_rgb.shape[1]
                        ih = det.transform.original_height if det.transform else frame_rgb.shape[0]
                        raw = convert_wilor_to_raw_hand_result(
                            det,
                            mapping=WILOR_TO_HANDS_V1_V1,
                            image_width=iw,
                            image_height=ih,
                        )
                        if raw is not None:
                            hands.append(raw)
                    except Exception:
                        # 单个 detection 转换失败不拖垮整帧
                        pass
            else:
                self._stats.no_hand += 1
                status = "no_hand"
                hands = []

            return ModelAttemptResult(
                model_name="wilor",
                backend_name="wilor",
                status=status,  # type: ignore[arg-type]
                hands=hands,
                inference_ms=inference_ms,
                failure_reason=None,
                model_version=self._model_info.model_version,
                checkpoint_sha256=self._model_info.checkpoint_sha256,
                device=self._model_info.device,
            )

        except NotImplementedError:
            inference_ms = (time.perf_counter() - t_start) * 1000
            return self._record_frame_failure(
                inference_ms, "WiLoR 推理尚未实现（阶段 3 占位）"
            )

        except Exception as exc:
            inference_ms = (time.perf_counter() - t_start) * 1000
            level = self.classify_error(exc)

            if level == "run_level":
                self._run_aborted = True
                self._abort_reason = f"{type(exc).__name__}: {exc}"
                self._run_errors.append({
                    "failure_type": type(exc).__name__,
                    "failure_reason": str(exc),
                    "level": "run_level",
                })
                raise  # 运行级异常直接抛出

            return self._record_frame_failure(
                inference_ms,
                f"{type(exc).__name__}: {exc}",
            )

    def _record_frame_failure(
        self,
        inference_ms: float,
        reason: str,
    ) -> ModelAttemptResult:
        """记录单帧失败，更新连续失败计数。"""
        self._stats.failed += 1
        self._consecutive_failures += 1
        self._inference_times.append(inference_ms)
        self._stats.total_inference_ms += inference_ms

        self._run_errors.append({
            "failure_type": "WiLoRInferenceError",
            "failure_reason": reason,
            "level": "frame_level",
            "consecutive_failures": self._consecutive_failures,
        })

        return ModelAttemptResult(
            model_name="wilor",
            backend_name="wilor",
            status="failed",
            hands=[],
            inference_ms=inference_ms,
            failure_reason=reason,
            model_version=self._model_info.model_version,
            checkpoint_sha256=self._model_info.checkpoint_sha256,
            device=self._model_info.device,
        )

    def _run_mediapipe_fallback(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> ModelAttemptResult:
        """调用 MediaPipe 回退估计器。"""
        if self._fallback is None:
            return ModelAttemptResult(
                model_name="mediapipe",
                backend_name="none",
                status="not_run",
                hands=[],
                inference_ms=0.0,
                failure_reason="未配置 fallback estimator",
                model_version="",
                checkpoint_sha256=None,
                device="",
            )

        t_start = time.perf_counter()

        try:
            results = self._fallback.estimate(frame_rgb, timestamp_ms)
            inference_ms = (time.perf_counter() - t_start) * 1000

            status = "detected" if results else "no_hand"
            return ModelAttemptResult(
                model_name="mediapipe",
                backend_name="mediapipe",
                status=status,  # type: ignore[arg-type]
                hands=list(results),
                inference_ms=inference_ms,
                failure_reason=None,
                model_version="hand_landmarker_v1",
                checkpoint_sha256=None,
                device="cpu",
            )

        except Exception as exc:
            inference_ms = (time.perf_counter() - t_start) * 1000
            return ModelAttemptResult(
                model_name="mediapipe",
                backend_name="mediapipe",
                status="failed",
                hands=[],
                inference_ms=inference_ms,
                failure_reason=f"{type(exc).__name__}: {exc}",
                model_version="hand_landmarker_v1",
                checkpoint_sha256=None,
                device="cpu",
            )

    def _should_fallback(self, primary: ModelAttemptResult) -> bool:
        """判断是否需要对当前主模型结果触发回退。"""
        policy = self._config.fallback_policy

        if self._fallback is None:
            return False  # 未配置回退

        if policy.compare_with_mediapipe:
            return False  # 对照模式，不回退

        if primary.status == "failed":
            return policy.on_wilor_frame_failure

        if primary.status == "no_hand":
            return policy.on_wilor_no_hand

        if primary.status == "skipped_invalid_input":
            return policy.on_invalid_input

        return False

    def _apply_fallback_policy(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
        primary: ModelAttemptResult,
    ) -> HandFrameResult:
        """应用回退策略，组装 HandFrameResult。"""
        if not self._should_fallback(primary):
            return HandFrameResult(
                timestamp_ms=timestamp_ms,
                requested_model="wilor",
                primary=primary,
                fallback=None,
                fallback_attempted=False,
                fallback_used=False,
                fallback_reason=None,
                effective_model=(
                    "wilor"
                    if primary.status in {"detected", "no_hand"}
                    else None
                ),
                effective_hands=list(primary.hands),
            )

        # ---- 执行回退 ----
        fallback = self._run_mediapipe_fallback(frame_rgb, timestamp_ms)
        self._stats.fallback_attempted += 1

        fallback_succeeded = fallback.status in {"detected", "no_hand"}

        if fallback_succeeded:
            self._stats.fallback_used += 1

        return HandFrameResult(
            timestamp_ms=timestamp_ms,
            requested_model="wilor",
            primary=primary,
            fallback=fallback,
            fallback_attempted=True,
            fallback_used=fallback_succeeded,
            fallback_reason=primary.failure_reason,
            effective_model=(
                "mediapipe" if fallback_succeeded else None
            ),
            effective_hands=(
                list(fallback.hands) if fallback_succeeded else []
            ),
        )

    def _run_comparison(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> None:
        """对照模式：无论 WiLoR 结果如何，同时运行 MediaPipe。"""
        comparison = self._run_mediapipe_fallback(frame_rgb, timestamp_ms)
        self._comparison_runs.append(comparison)

    def _invalid_input_result(
        self,
        timestamp_ms: int,
        reason: str,
    ) -> HandFrameResult:
        """构造 skipped_invalid_input 的 HandFrameResult。

        timestamp_ms 可能为负数（这是被判定为 invalid 的原因），
        写入 HandFrameResult 时做保护处理。
        """
        return HandFrameResult(
            timestamp_ms=max(timestamp_ms, 0),
            requested_model="wilor",
            primary=ModelAttemptResult(
                model_name="wilor",
                backend_name="wilor",
                status="skipped_invalid_input",
                hands=[],
                inference_ms=0.0,
                failure_reason=reason,
                model_version=self._model_info.model_version,
                checkpoint_sha256=self._model_info.checkpoint_sha256,
                device=self._model_info.device,
            ),
            fallback=None,
            fallback_attempted=False,
            fallback_used=False,
            fallback_reason=None,
            effective_model=None,
            effective_hands=[],
        )

    # ---- 运行级状态 ----

    @property
    def is_aborted(self) -> bool:
        """运行是否因连续失败或失败比例超标被终止。"""
        return self._run_aborted

    @property
    def abort_reason(self) -> str | None:
        return self._abort_reason

    def should_abort(self) -> bool:
        """检查是否达到运行级终止阈值。

        两种触发条件：
        1. 连续失败帧数 ≥ max_consecutive_frame_failures
        2. 失败比例 > max_failure_ratio

        仅在条件满足时标记 _run_aborted，后续帧跳过推理。
        """
        if self._run_aborted:
            return True

        if self._consecutive_failures >= self._thresholds.max_consecutive_frame_failures:
            self._run_aborted = True
            self._abort_reason = (
                f"连续失败 {self._consecutive_failures} 帧，"
                f"超过阈值 {self._thresholds.max_consecutive_frame_failures}"
            )
            return True

        if self._stats.failure_ratio > self._thresholds.max_failure_ratio:
            self._run_aborted = True
            self._abort_reason = (
                f"失败比例 {self._stats.failure_ratio:.2%}，"
                f"超过阈值 {self._thresholds.max_failure_ratio:.2%}"
            )
            return True

        return False

    def build_run_report(self) -> WiLoRRunReport:
        """基于 session 统计构造完整 Run Report。"""
        times = sorted(self._inference_times) if self._inference_times else [0.0]
        n = len(times)
        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)

        return WiLoRRunReport(
            requested_model="wilor",
            ego_bbox_every_frame=True,
            model=self._model_info,
            coverage={
                "decoded_frames": self._stats.total_frames,
                "sample_map_rows": self._stats.total_frames,
                "wilor_requests": self._stats.total_frames,
                "detected_frames": self._stats.detected,
                "no_hand_frames": self._stats.no_hand,
                "failed_frames": self._stats.failed,
                "invalid_input_frames": self._stats.skipped_invalid_input,
            },
            fallback={
                "configured": self._fallback is not None,
                "model": "mediapipe" if self._fallback is not None else "none",
                "attempted_frames": self._stats.fallback_attempted,
                "successful_frames": self._stats.fallback_used,
                "failed_frames": self._stats.fallback_attempted - self._stats.fallback_used,
            },
            effective_output={
                "wilor_frames": self._stats.detected + self._stats.no_hand,
                "mediapipe_fallback_frames": self._stats.fallback_used,
                "unresolved_frames": self._stats.failed - self._stats.fallback_used,
            },
            timing={
                "average_inference_ms": self._stats.avg_inference_ms,
                "p50_inference_ms": times[p50_idx] if times else 0.0,
                "p95_inference_ms": times[min(p95_idx, n - 1)] if times else 0.0,
                "max_inference_ms": max(times) if times else 0.0,
                "total_inference_ms": self._stats.total_inference_ms,
            },
            quality={
                "joint_mapping_version": "wilor-to-hands-v1-v1",
                "coordinate_frame": "model_camera",
                "scale_status": "uncalibrated",
            },
            completion={
                "wilor_full_frame_requirement_met": (
                    not self._run_aborted
                    and self._stats.failed == 0
                ),
                "reason": self._abort_reason or (
                    ""
                    if self._stats.failed == 0
                    else f"{self._stats.failed} WiLoR frame failures"
                ),
            },
            errors=list(self._run_errors),
        )

    @staticmethod
    def classify_error(exc: Exception) -> str:
        """判断异常是运行级还是单帧级。

        Returns:
            ``"run_level"`` 或 ``"frame_level"``。
        """
        if isinstance(exc, RUN_LEVEL_ERRORS):
            return "run_level"
        if isinstance(exc, FRAME_LEVEL_ERRORS):
            return "frame_level"
        # 未分类的异常按运行级处理（安全侧）
        return "run_level"

    def close(self) -> None:
        self._adapter.close()
        if self._fallback is not None:
            self._fallback.close()


# ════════════════════════════════════════════════════════════════════
# 输入校验（与 adapter 保持一致）
# ════════════════════════════════════════════════════════════════════


def _validate_input(frame_rgb: np.ndarray, timestamp_ms: int) -> None:
    if not isinstance(frame_rgb, np.ndarray):
        raise TypeError("frame_rgb 必须是 np.ndarray")
    if frame_rgb.size == 0:
        raise ValueError("frame_rgb 不能为空")
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(
            f"frame_rgb 必须是形状为 (H, W, 3) 的 RGB 图像，"
            f"实际形状 {frame_rgb.shape}"
        )
    if frame_rgb.dtype != np.uint8:
        raise TypeError(f"frame_rgb 必须是 uint8，实际 {frame_rgb.dtype}")
    if timestamp_ms < 0:
        raise ValueError(f"timestamp_ms 不能为负数，实际 {timestamp_ms}")


__all__ = [
    "WiLoREstimatorConfig",
    "WiLoRFrameStats",
    "WiLoRHandEstimator",
]
