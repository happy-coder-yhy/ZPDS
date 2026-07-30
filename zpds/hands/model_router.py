"""Hand Model Router — 主模型 / 回退调度。

负责：

- primary（WiLoR） + fallback（MediaPipe）的组织
- 回退策略：frame_failure / no_hand / invalid_input
- 对照模式：同时运行两个模型，独立记录
- 初始化失败时的运行级路由

不负责模型专属逻辑（坐标变换、关节映射等）。
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from zpds.hands.schemas import (
    HandFrameResult,
    ModelAttemptResult,
    RawHandResult,
)
from zpds.hands.wilor_schema import WiLoRFallbackPolicy


# ════════════════════════════════════════════════════════════════════
# 接口
# ════════════════════════════════════════════════════════════════════


class PrimaryEstimator(Protocol):
    """Router 要求的主模型接口。

    WiLoRHandEstimator 天然满足此协议。
    """

    @property
    def model_info(self) -> object: ...

    def estimate_frame(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> HandFrameResult: ...

    def estimate(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> list[RawHandResult]: ...

    def close(self) -> None: ...


class FallbackEstimator(Protocol):
    """Router 要求的回退模型接口。

    MediaPipeHandEstimator 天然满足此协议。
    """

    def estimate(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> list[RawHandResult]: ...

    def close(self) -> None: ...


# ════════════════════════════════════════════════════════════════════
# Router 配置
# ════════════════════════════════════════════════════════════════════


class HandModelRouter:
    """主模型 + 回退的调度 Router。

    职责：
    - 调用 primary（WiLoR）
    - 根据 fallback_policy 决定是否调用 fallback（MediaPipe）
    - 组装 HandFrameResult（primary 和 fallback 独立保留）
    - 支持对照模式（compare_with_mediapipe）

    用法::

        router = HandModelRouter(
            primary=wilor_estimator,
            fallback=mediapipe_estimator,
            fallback_policy=WiLoRFallbackPolicy(),
        )
        result = router.estimate_frame(frame_rgb, timestamp_ms=0)
        router.close()
    """

    def __init__(
        self,
        *,
        primary: PrimaryEstimator,
        fallback: FallbackEstimator | None = None,
        fallback_policy: WiLoRFallbackPolicy | None = None,
    ) -> None:
        """初始化 Router。

        Args:
            primary: WiLoR 主估计器。
            fallback: MediaPipe 回退估计器。为 None 时不回退。
            fallback_policy: 回退/对照策略。为 None 时使用默认（仅 frame_failure 回退）。
        """
        self._primary = primary
        self._fallback = fallback
        self._policy = fallback_policy or WiLoRFallbackPolicy()

        # 初始化状态记录
        self._primary_init_failed: bool = False
        self._primary_init_error: str | None = None
        self._run_active: bool = True

    # ---- 主入口 ----

    def estimate_frame(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> HandFrameResult:
        """对单帧执行 primary + 按需 fallback。

        Args:
            frame_rgb: RGB uint8 图像 ``(H, W, 3)``。
            timestamp_ms: 帧时间戳（毫秒）。

        Returns:
            完整 HandFrameResult，primary 和 fallback 独立记录。
        """
        # ---- 对照模式 ----
        if self._policy.compare_with_mediapipe:
            return self._run_comparison(frame_rgb, timestamp_ms)

        # ---- 正常回退模式 ----
        primary = self._run_primary(frame_rgb, timestamp_ms)

        if not self._should_fallback(primary):
            return self._wrap_primary_only(primary, timestamp_ms)

        # ---- 调用 fallback ----
        fallback = self._run_fallback(frame_rgb, timestamp_ms)

        fallback_succeeded = fallback.status in {"detected", "no_hand"}

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

    def estimate(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> list[RawHandResult]:
        """兼容旧 HandEstimator Protocol。返回 effective_hands。"""
        result = self.estimate_frame(frame_rgb, timestamp_ms)
        return list(result.effective_hands)

    # ---- 内部 ----

    def _run_primary(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> ModelAttemptResult:
        """调用主模型并包装为 ModelAttemptResult。"""
        try:
            # WiLoRHandEstimator 直接返回 HandFrameResult
            if hasattr(self._primary, "estimate_frame"):
                result = self._primary.estimate_frame(frame_rgb, timestamp_ms)
                return result.primary
        except Exception:
            pass

        # Fallback: 使用 estimate() 协议
        try:
            results = self._primary.estimate(frame_rgb, timestamp_ms)
            status = "detected" if results else "no_hand"
            return ModelAttemptResult(
                model_name="wilor",
                backend_name="wilor",
                status=status,  # type: ignore[arg-type]
                hands=list(results),
                inference_ms=0.0,
                failure_reason=None,
                model_version="unknown",
                checkpoint_sha256=None,
                device="unknown",
            )
        except Exception as exc:
            return ModelAttemptResult(
                model_name="wilor",
                backend_name="wilor",
                status="failed",
                hands=[],
                inference_ms=0.0,
                failure_reason=f"{type(exc).__name__}: {exc}",
                model_version="unknown",
                checkpoint_sha256=None,
                device="unknown",
            )

    def _run_fallback(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> ModelAttemptResult:
        """调用回退模型。"""
        if self._fallback is None:
            return ModelAttemptResult(
                model_name="mediapipe",
                backend_name="none",
                status="not_run",
                hands=[],
                inference_ms=0.0,
                failure_reason="未配置 fallback",
                model_version="unknown",
                checkpoint_sha256=None,
                device="unknown",
            )

        try:
            results = self._fallback.estimate(frame_rgb, timestamp_ms)
            status = "detected" if results else "no_hand"
            return ModelAttemptResult(
                model_name="mediapipe",
                backend_name="mediapipe",
                status=status,  # type: ignore[arg-type]
                hands=list(results),
                inference_ms=0.0,
                failure_reason=None,
                model_version="hand_landmarker_v1",
                checkpoint_sha256=None,
                device="cpu",
            )
        except Exception as exc:
            return ModelAttemptResult(
                model_name="mediapipe",
                backend_name="mediapipe",
                status="failed",
                hands=[],
                inference_ms=0.0,
                failure_reason=f"{type(exc).__name__}: {exc}",
                model_version="hand_landmarker_v1",
                checkpoint_sha256=None,
                device="cpu",
            )

    def _should_fallback(self, primary: ModelAttemptResult) -> bool:
        """判断是否回退。"""
        if self._fallback is None:
            return False

        if self._policy.compare_with_mediapipe:
            return False

        if primary.status == "failed":
            return self._policy.on_wilor_frame_failure
        if primary.status == "no_hand":
            return self._policy.on_wilor_no_hand
        if primary.status == "skipped_invalid_input":
            return self._policy.on_invalid_input

        return False

    def _run_comparison(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> HandFrameResult:
        """对照模式：WiLoR + MediaPipe 都运行，结果独立记录。"""
        primary = self._run_primary(frame_rgb, timestamp_ms)
        comparison = self._run_fallback(frame_rgb, timestamp_ms)

        # 对照模式下，effective 始终是 WiLoR 的结论
        return HandFrameResult(
            timestamp_ms=timestamp_ms,
            requested_model="wilor",
            primary=primary,
            fallback=comparison,  # 对比结果放在 fallback 槽
            fallback_attempted=True,
            fallback_used=False,   # 注意：对照不是回退
            fallback_reason=None,
            effective_model=(
                "wilor"
                if primary.status in {"detected", "no_hand"}
                else None
            ),
            effective_hands=list(primary.hands),
        )

    def _wrap_primary_only(
        self,
        primary: ModelAttemptResult,
        timestamp_ms: int,
    ) -> HandFrameResult:
        """仅用主模型结果组帧。"""
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

    def close(self) -> None:
        self._primary.close()
        if self._fallback is not None:
            self._fallback.close()


# ════════════════════════════════════════════════════════════════════
# 工厂函数 — 处理 WiLoR 初始化失败
# ════════════════════════════════════════════════════════════════════


def create_hand_model_router(
    *,
    wilor_config,  # WiLoRConfig
    wilor_adapter,  # WiLoRAdapter (pre-created)
    wilor_model_info,  # WiLoRModelInfo
    mediapipe_estimator: FallbackEstimator | None = None,
    fallback_policy: WiLoRFallbackPolicy | None = None,
    on_init_failure: str = "fallback",
) -> HandModelRouter:
    """创建 HandModelRouter，处理 WiLoR 初始化失败。

    初始化流程::

        尝试创建 WiLoRHandEstimator
            ↓
        成功 → 返回完整 Router (primary=WiLoR, fallback=MP)
            ↓
        失败 → 根据 on_init_failure:
              - "fallback": 创建仅 fallback 的 Router，记录 primary_init_failed
              - "raise": 抛出 WiLoRInitializationError

    Args:
        wilor_config: WiLoR 后端配置。
        wilor_adapter: 已创建的 WiLoR 适配器。
        wilor_model_info: WiLoR 运行时元信息。
        mediapipe_estimator: MediaPipe 回退估计器。
        fallback_policy: 回退策略。
        on_init_failure: ``"fallback"`` 或 ``"raise"``。

    Returns:
        HandModelRouter 实例。

    Raises:
        WiLoRInitializationError: on_init_failure="raise" 且 WiLoR 初始化失败。
    """
    from zpds.hands.wilor_estimator import WiLoRHandEstimator, WiLoREstimatorConfig
    from zpds.hands.wilor_schema import WiLoRInitializationError

    policy = fallback_policy or WiLoRFallbackPolicy()

    try:
        wilor_estimator = WiLoRHandEstimator(
            adapter=wilor_adapter,
            model_info=wilor_model_info,
            fallback_estimator=None,  # Router 自己管理 fallback
            config=WiLoREstimatorConfig(
                fallback_policy=WiLoRFallbackPolicy(
                    on_wilor_frame_failure=False,  # Router 做 fallback
                    compare_with_mediapipe=False,
                ),
            ),
        )
    except Exception as exc:
        if on_init_failure == "raise":
            raise WiLoRInitializationError(
                f"WiLoR 初始化失败: {exc}"
            ) from exc

        # fallback 模式
        if mediapipe_estimator is None:
            raise WiLoRInitializationError(
                "WiLoR 初始化失败且未配置 MediaPipe fallback"
            ) from exc

        router = HandModelRouter(
            primary=_DegradedPrimary(str(exc)),
            fallback=mediapipe_estimator,
            fallback_policy=WiLoRFallbackPolicy(
                on_wilor_frame_failure=True,
                on_wilor_no_hand=False,
                on_invalid_input=False,
            ),
        )
        router._primary_init_failed = True
        router._primary_init_error = str(exc)
        return router

    return HandModelRouter(
        primary=wilor_estimator,
        fallback=mediapipe_estimator,
        fallback_policy=policy,
    )


class _DegradedPrimary:
    """WiLoR 初始化失败时的降级主模型。

    所有帧返回 ``status="not_run"`` + 失败原因，触发 Router 回退。
    """

    def __init__(self, error: str) -> None:
        self._error = error

    @property
    def model_info(self) -> object:
        return type("Info", (), {"model_version": "degraded"})()

    def estimate_frame(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> HandFrameResult:
        primary = ModelAttemptResult(
            model_name="wilor",
            backend_name="wilor",
            status="failed",
            hands=[],
            inference_ms=0.0,
            failure_reason=f"WiLoR 初始化失败: {self._error}",
            model_version="degraded",
            checkpoint_sha256=None,
            device="",
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

    def estimate(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> list[RawHandResult]:
        return []

    def close(self) -> None:
        pass


__all__ = [
    "HandModelRouter",
    "create_hand_model_router",
]
