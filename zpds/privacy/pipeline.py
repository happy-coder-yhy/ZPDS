"""Privacy 脱敏流水线 — 帧循环编排。

对标 ``hands/pipeline.py``：读取帧 → 人脸检测 → 文本检测 → PII 分类 → 遮挡 → 产出。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from zpds.privacy.backend_router import PrivacyBackendPolicy
from zpds.privacy.config import PrivacyConfig
from zpds.privacy.contracts import (
    FaceDetector,
    FrameRedactionRecord,
    PIIClassifier,
    RedactionRunStatistics,
    TextDetector,
)
from zpds.privacy.estimator_factory import (
    EstimatorRuntime,
    PrivacyEstimatorError,
    PrivacyEstimatorFactory,
)
from zpds.privacy.propagation import KLTRegionPropagator
from zpds.privacy.redaction import FrameRedactor, TemporalSmoother
from zpds.privacy.schemas import (
    PIIClassification,
    PrivacyRunManifest,
    RedactionRegion,
)


class PrivacyPipelineError(RuntimeError):
    """脱敏流水线在某个特定帧失败。"""

    def __init__(self, message: str, *, frame_index: int, timestamp_ns: int) -> None:
        super().__init__(f"{message}: frame={frame_index}, ts={timestamp_ns}")
        self.frame_index = frame_index
        self.timestamp_ns = timestamp_ns


@dataclass(frozen=True)
class PipelineStats:
    """一次 Pipeline 运行的统计（对标 hands/PipelineStats）。"""

    frames_processed: int = 0
    frames_with_faces: int = 0
    frames_with_text: int = 0
    total_face_regions: int = 0
    total_text_regions: int = 0
    total_pii_masked: int = 0
    pii_categories_found: tuple[str, ...] = ()
    llm_available: bool = False
    elapsed_seconds: float = 0.0

    @property
    def average_fps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds


class PrivacyPipeline:
    """运行一次视频脱敏并产生 ``FrameRedactionRecord`` 流。

    视频读取和模型加载都是一次性的；如需重新运行，请创建新的 Pipeline 实例。
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        config: PrivacyConfig | None = None,
        policy: PrivacyBackendPolicy | None = None,
        profile: str = "",
        session_id: str = "",
        # 直接注入后端（优先于 factory）
        face_detector: FaceDetector | None = None,
        text_detector: TextDetector | None = None,
        pii_classifier: PIIClassifier | None = None,
        # 间隔采样（每 N 帧检测一次，中间帧用 KLT 光流传播检测结果）
        face_interval: int = 1,
        text_interval: int = 1,
        max_frames: int | None = None,
        # 强制检测帧（场景边界等画面布局剧变点）：检测 + 重置传播缓存
        reset_frames: set[int] | None = None,
    ) -> None:
        self._video_path = Path(video_path)
        if not self._video_path.is_file():
            raise FileNotFoundError(f"视频文件不存在: {self._video_path}")

        self._config = config or PrivacyConfig.defaults()
        self._policy = policy or PrivacyBackendPolicy.from_profile(profile or "guida_ego")
        self._session_id = session_id or self._video_path.stem
        self._max_frames = max_frames
        self._reset_frames = set(reset_frames or ())

        # 后端
        factory = PrivacyEstimatorFactory(self._config, self._policy)
        self._runtime = factory.build_runtime()
        self._face_detector = face_detector if face_detector is not None else factory.create_face_detector()
        self._text_detector = text_detector if text_detector is not None else factory.create_text_detector()
        self._pii_classifier = pii_classifier if pii_classifier is not None else factory.create_pii_classifier()

        # 遮挡器
        self._redactor = FrameRedactor(
            face_method=self._config.face_method,       # type: ignore[arg-type]
            text_method=self._config.redaction_text_method,  # type: ignore[arg-type]
            blur_ksize=self._config.face_blur_ksize,
            blur_sigma=self._config.face_blur_sigma,
            pixelate_blocks=self._config.face_pixelate_blocks,
        )
        self._face_smoother = TemporalSmoother(
            window_frames=self._config.redaction_smoothing_window,
            iou_threshold=self._config.redaction_smoothing_iou,
        ) if self._config.redaction_temporal_smoothing else None
        self._text_smoother = TemporalSmoother(
            window_frames=self._config.redaction_smoothing_window,
            iou_threshold=self._config.redaction_smoothing_iou,
        ) if self._config.redaction_temporal_smoothing else None

        self._face_interval = max(1, face_interval)
        self._text_interval = max(1, text_interval)

        self._stats = PipelineStats()
        self._started = False

    # ---- properties ----

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    @property
    def runtime(self) -> EstimatorRuntime:
        return self._runtime

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def config_hash(self) -> str:
        return self._config.config_hash

    # ---- run ----

    def run(self) -> Iterator[FrameRedactionRecord]:
        """逐帧运行脱敏流水线，产出 FrameRedactionRecord 流。"""
        if self._started:
            raise PrivacyPipelineError(
                "PrivacyPipeline 实例不能重复运行",
                frame_index=0,
                timestamp_ns=0,
            )
        self._started = True

        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise PrivacyPipelineError(
                f"无法打开视频: {self._video_path}",
                frame_index=0, timestamp_ns=0,
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        stat = RedactionRunStatistics()
        frames_processed = 0
        started_at = time.perf_counter()

        # KLT 传播器：检测帧之间逐帧传播遮挡区域（惰性创建，需要帧尺寸）
        propagator: KLTRegionPropagator | None = None
        prev_gray: np.ndarray | None = None

        try:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self._max_frames is not None and frames_processed >= self._max_frames:
                    break

                timestamp_ns = int(frame_index / fps * 1_000_000_000)
                t0 = time.perf_counter()

                if propagator is None:
                    h, w = frame.shape[:2]
                    propagator = KLTRegionPropagator(w, h)
                is_force = frame_index in self._reset_frames

                # ---- 传播：把上一帧的 track 跟到本帧（中间帧不跑模型） ----
                if is_force:
                    propagator.reset()  # 场景边界：布局剧变，旧 track 作废
                    prev_gray = None
                if propagator.track_count:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        propagator.step(prev_gray, gray)
                    prev_gray = gray
                else:
                    prev_gray = None

                # ---- 人脸检测（间隔采样 + 强制帧） ----
                face_ms = 0.0
                if self._face_detector is not None and (
                    frame_index % self._face_interval == 0 or is_force
                ):
                    t_face = time.perf_counter()
                    faces = self._face_detector.detect(frame, frame_index, timestamp_ns)
                    face_ms = (time.perf_counter() - t_face) * 1000
                    propagator.sync_faces(
                        faces,
                        face_method=self._config.face_method,  # type: ignore[arg-type]
                    )

                # ---- 文本检测（间隔采样 + 强制帧） ----
                text_ms = 0.0
                pii_ms = 0.0
                pii_results: list[PIIClassification] = []
                llm_available = False
                if self._text_detector is not None and (
                    frame_index % self._text_interval == 0 or is_force
                ):
                    t_text = time.perf_counter()
                    texts = self._text_detector.detect(frame, frame_index, timestamp_ns)
                    text_ms = (time.perf_counter() - t_text) * 1000

                    # ---- PII 分类（LLM，按 text hash 缓存；仅检测帧） ----
                    if texts and self._pii_classifier is not None:
                        t_pii = time.perf_counter()
                        try:
                            pii_results = self._pii_classifier.classify(list(texts))
                            llm_available = True
                        except Exception:
                            llm_available = False
                        pii_ms = (time.perf_counter() - t_pii) * 1000

                    # 仅 mask 的文本进入传播（keep 的不遮挡）
                    mask_texts = [
                        p.text for p in pii_results if p.decision == "mask"
                    ]
                    if mask_texts:
                        propagator.sync_texts(
                            mask_texts,
                            text_method=self._config.redaction_text_method,  # type: ignore[arg-type]
                            categories={
                                id(p.text): p.category
                                for p in pii_results
                                if p.decision == "mask"
                            },
                        )

                # 检测/同步后若新增 track 而本帧 gray 未记录（首帧/reset 帧），
                # 以本帧 gray 作为 KLT 传播链起点（points 必须与 prev 帧对齐）
                if propagator.track_count and prev_gray is None:
                    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ---- 当前帧的遮挡信息（检测帧与中间帧一致，来自传播器） ----
                faces = propagator.faces(
                    frame_index, timestamp_ns, self._runtime.face_backend
                )
                texts = propagator.texts(
                    frame_index, timestamp_ns, self._runtime.text_backend
                )
                regions = propagator.regions()

                # 跨帧平滑
                face_regions = [r for r in regions if r.kind == "face"]
                text_regions = [r for r in regions if r.kind == "text"]
                if self._face_smoother and face_regions:
                    face_regions = self._face_smoother.smooth(face_regions)
                if self._text_smoother and text_regions:
                    text_regions = self._text_smoother.smooth(text_regions)
                regions = face_regions + text_regions

                # ---- 遮挡 ----
                redact_ms = 0.0
                redacted = None
                if regions:
                    t_redact = time.perf_counter()
                    redacted = self._redactor.apply(frame, regions)
                    redact_ms = (time.perf_counter() - t_redact) * 1000

                # ---- 产出记录 ----
                record = FrameRedactionRecord(
                    frame_index=frame_index,
                    timestamp_ns=timestamp_ns,
                    faces=tuple(faces),
                    texts=tuple(texts),
                    pii_classifications=tuple(pii_results),
                    regions=tuple(regions),
                    redacted_frame=redacted,
                    face_detector_used=self._runtime.face_backend if self._face_detector else "",
                    text_detector_used=self._runtime.text_backend if self._text_detector else "",
                    pii_classifier_used=self._runtime.pii_backend,
                    llm_available=llm_available,
                    face_inference_ms=face_ms,
                    text_inference_ms=text_ms,
                    pii_classification_ms=pii_ms,
                    redaction_ms=redact_ms,
                )
                stat.add(record)
                frames_processed += 1
                frame_index += 1
                yield record

        finally:
            cap.release()
            elapsed = time.perf_counter() - started_at
            stat.elapsed_seconds = elapsed

        self._stats = PipelineStats(
            frames_processed=stat.frames_processed,
            frames_with_faces=stat.frames_with_faces,
            frames_with_text=stat.frames_with_text,
            total_face_regions=stat.total_face_regions,
            total_text_regions=stat.total_text_regions,
            total_pii_masked=stat.total_pii_masked,
            pii_categories_found=tuple(sorted(stat.pii_categories_found)),
            llm_available=stat.llm_available,
            elapsed_seconds=elapsed,
        )

    def run_to_list(self) -> list[FrameRedactionRecord]:
        """收集全部帧记录（小视频/测试用）。"""
        return list(self.run())

    def build_manifest(self) -> PrivacyRunManifest:
        """运行后构建 manifest。"""
        return PrivacyRunManifest(
            session_id=self._session_id,
            source_uri=str(self._video_path),
            profile=self._policy.face_applicability,  # 简化：用 face 适用性代表 profile
            producer="zpds.privacy",
            version="v1",
            config_hash=self._config.config_hash,
            face_model_hash="",
            llm_endpoint=self._config.pii_llm_url,
            total_frames=self._stats.frames_processed,
            frames_with_faces=self._stats.frames_with_faces,
            frames_with_text=self._stats.frames_with_text,
            total_face_regions=self._stats.total_face_regions,
            total_text_regions=self._stats.total_text_regions,
            pii_categories_found=self._stats.pii_categories_found,
            llm_available=self._stats.llm_available,
            elapsed_seconds=self._stats.elapsed_seconds,
        )


__all__ = [
    "PipelineStats",
    "PrivacyPipeline",
    "PrivacyPipelineError",
]
