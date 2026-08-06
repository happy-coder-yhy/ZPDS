"""稀疏检测帧之间的遮挡区域传播 — KLT 稀疏光流跟踪。

检测帧（每 interval 帧 + 场景边界帧）做完整检测；中间帧不跑模型，
用 ``cv2.calcOpticalFlowPyrLK`` 把检测帧 bbox 的 8 个网格点
（四角 + 四边中点）跟踪到当前帧，bbox 由有效点范围 + 膨胀系数重建。

为什么需要传播而不是直接复用检测帧 bbox（interval 缓存）：
自我中心（egocentric）视频里文字/人脸在世界中基本静止，但相机持续
运动，同一目标在画面中的位置逐帧漂移。KLT 跟踪点跟随光流即可逐帧
修正遮挡窗口位置；跟踪退化（有效点不足）时退化为"原框 + 膨胀"，
并在下个检测帧由完整检测纠正。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np

from zpds.privacy.schemas import (
    FaceDetection,
    RedactionMethod,
    RedactionRegion,
    TextDetection,
)

# bbox 网格点数：四角 + 四边中点
_GRID_POINTS = 8
# 有效跟踪点少于该数量时认为跟踪不可靠，改用"原框 + 膨胀"
_MIN_VALID_POINTS = 4
# 传播 bbox 每边外扩比例（补偿光流漂移/压缩伪影）
_DEFAULT_EXPANSION = 0.12
# 跟踪退化时 bbox 的额外膨胀比例
_STALE_DILATION = 0.30
# 检测帧里 track 与检测框的 IoU 匹配阈值
_MATCH_IOU = 0.3


def _points_for_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> np.ndarray:
    """归一化 bbox → 像素 8 网格点 (8, 2) float32。"""
    x1, y1, x2, y2 = bbox_xyxy
    px1, py1, px2, py2 = x1 * width, y1 * height, x2 * width, y2 * height
    cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
    return np.array(
        [
            [px1, py1], [px2, py1], [px2, py2], [px1, py2],  # 四角
            [cx, py1], [px2, cy], [cx, py2], [px1, cy],      # 四边中点
        ],
        dtype=np.float32,
    )


def _bbox_from_points(
    points: np.ndarray,
    width: int,
    height: int,
    expansion: float,
) -> tuple[float, float, float, float]:
    """有效点范围 + 膨胀 → 归一化 bbox。"""
    x1, y1 = float(points[:, 0].min()), float(points[:, 1].min())
    x2, y2 = float(points[:, 0].max()), float(points[:, 1].max())
    dx = (x2 - x1) * expansion
    dy = (y2 - y1) * expansion
    bx1 = max(0.0, (x1 - dx) / width)
    by1 = max(0.0, (y1 - dy) / height)
    bx2 = min(1.0, (x2 + dx) / width)
    by2 = min(1.0, (y2 + dy) / height)
    if bx2 <= bx1 or by2 <= by1:
        return (0.0, 0.0, 1.0, 1.0)  # 防御：不应发生
    return (bx1, by1, bx2, by2)


def _dilate_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    ratio: float,
) -> tuple[float, float, float, float]:
    """按比例外扩归一化 bbox（clamp 到 [0,1]）。"""
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    return (
        max(0.0, x1 - w * ratio),
        max(0.0, y1 - h * ratio),
        min(1.0, x2 + w * ratio),
        min(1.0, y2 + h * ratio),
    )


def _iou_norm(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """两个归一化 bbox 的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


@dataclass
class _Track:
    """一个正在传播的遮挡目标。"""

    kind: Literal["face", "text"]
    points: np.ndarray  # (N,2) float32 像素跟踪点
    bbox_xyxy: tuple[float, float, float, float]  # 归一化 0~1
    confidence: float
    method: RedactionMethod
    text: str = ""       # 文本 track 最后一次 OCR 内容
    category: str = ""   # 文本 PII 类别
    misses: int = 0      # 检测帧连续未匹配次数（超过阈值删除）
    stale_frames: int = 0  # 有效点不足的连续传播帧数


class KLTRegionPropagator:
    """KLT 光流传播器：检测帧之间逐帧更新遮挡 bbox。

    用法（配合 ``PrivacyPipeline`` 的帧循环）：
    1. 每帧先 ``step(prev_gray, gray)`` 传播上一检测帧的 track；
    2. 检测帧再 ``sync_faces`` / ``sync_texts`` 用完整检测结果匹配纠正；
    3. ``regions()`` / ``faces()`` / ``texts()`` 输出当前帧遮挡信息。
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        expansion: float = _DEFAULT_EXPANSION,
        match_iou: float = _MATCH_IOU,
        max_misses: int = 2,
        max_stale_frames: int = 10,
        lk_win_size: int = 21,
        lk_max_level: int = 2,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width/height 必须为正: {width}x{height}")
        self._w = width
        self._h = height
        self._expansion = expansion
        self._match_iou = match_iou
        self._max_misses = max_misses
        self._max_stale_frames = max_stale_frames
        self._lk_win = lk_win_size
        self._lk_max_level = lk_max_level
        self._tracks: list[_Track] = []

    # ---- 状态 ----

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """清空全部 track（场景边界帧：画面布局剧变，传播失效）。"""
        self._tracks.clear()

    # ---- 传播 ----

    def step(self, prev_gray: np.ndarray, gray: np.ndarray) -> None:
        """把全部 track 从上一帧传播到当前帧（KLT 稀疏光流）。"""
        if not self._tracks:
            return
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03,
        )
        dead: list[_Track] = []
        for tr in self._tracks:
            new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                tr.points,
                None,
                winSize=(self._lk_win, self._lk_win),
                maxLevel=self._lk_max_level,
                criteria=criteria,
            )
            valid = status.ravel() == 1
            pts = new_pts[valid]
            if pts.shape[0] < _MIN_VALID_POINTS:
                # 跟踪退化：原框 + 更大膨胀，等待下个检测帧纠正
                tr.stale_frames += 1
                tr.bbox_xyxy = _dilate_bbox(tr.bbox_xyxy, _STALE_DILATION)
                if tr.stale_frames > self._max_stale_frames:
                    dead.append(tr)
                continue
            tr.points = pts
            tr.stale_frames = 0
            tr.bbox_xyxy = _bbox_from_points(
                pts, self._w, self._h, self._expansion
            )
        for tr in dead:
            self._tracks.remove(tr)

    # ---- 检测帧同步 ----

    def sync_faces(
        self,
        faces: list[FaceDetection],
        *,
        face_method: RedactionMethod = "blur",
    ) -> None:
        """用检测帧的人脸结果匹配/新建/删除 face track。"""
        self._sync(
            faces,
            kind="face",
            method=face_method,
            text_of=lambda _det: "",
            category_of=lambda _det: "face",
        )

    def sync_texts(
        self,
        mask_entries: list[TextDetection],
        *,
        text_method: RedactionMethod = "black_rect",
        categories: Optional[dict[int, str]] = None,
    ) -> None:
        """用检测帧的**已判定 mask** 文本结果同步 text track。

        :param mask_entries: LLM 判定为需遮挡的 TextDetection 列表
            （``p.text`` 与 ``PIIClassification.text`` 为同一对象引用，
            因此 ``id(p.text)`` 可查 categories）
        :param categories: ``{id(TextDetection): PII 类别}``
        """
        categories = categories or {}
        self._sync(
            mask_entries,
            kind="text",
            method=text_method,
            text_of=lambda det: det.text,
            category_of=lambda det: categories.get(id(det), "unknown"),
        )

    def _sync(
        self,
        detections: list,
        *,
        kind: Literal["face", "text"],
        method: RedactionMethod,
        text_of,
        category_of,
    ) -> None:
        """通用同步：检测结果与 track 按 IoU 匹配，未匹配的旧 track 计数删除。"""
        matched_ids: set[int] = set()
        for det in detections:
            tr = self._match(det.bbox_xyxy, matched_ids)
            if tr is not None:
                tr.bbox_xyxy = det.bbox_xyxy
                tr.points = _points_for_bbox(det.bbox_xyxy, self._w, self._h)
                tr.confidence = det.confidence
                tr.misses = 0
                tr.stale_frames = 0
                if kind == "text":
                    tr.text = text_of(det)
                    tr.category = category_of(det)
                matched_ids.add(id(tr))
            else:
                new_track = _Track(
                    kind=kind,
                    points=_points_for_bbox(det.bbox_xyxy, self._w, self._h),
                    bbox_xyxy=det.bbox_xyxy,
                    confidence=det.confidence,
                    method=method,
                    text=text_of(det) if kind == "text" else "",
                    category=category_of(det) if kind == "text" else "",
                )
                self._tracks.append(new_track)
                # 新 track 本帧已"匹配"，不参与 misses 计数
                matched_ids.add(id(new_track))

        # 未匹配的同类旧 track：misses += 1，超限删除
        dead = [
            tr
            for tr in self._tracks
            if tr.kind == kind and id(tr) not in matched_ids
        ]
        for tr in dead:
            tr.misses += 1
            if tr.misses > self._max_misses:
                self._tracks.remove(tr)

    def _match(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        exclude_ids: set[int],
    ) -> Optional[_Track]:
        """与现有 track 找最大 IoU 匹配（>= 阈值），并排除已匹配的 track。"""
        best: Optional[_Track] = None
        best_iou = 0.0
        for tr in self._tracks:
            if id(tr) in exclude_ids:
                continue
            iou = _iou_norm(bbox_xyxy, tr.bbox_xyxy)
            if iou > best_iou:
                best = tr
                best_iou = iou
        return best if best_iou >= self._match_iou else None

    # ---- 输出 ----

    def regions(self) -> list[RedactionRegion]:
        """当前全部 track → 遮挡区域（每帧都可用）。"""
        return [
            RedactionRegion(
                kind=tr.kind,
                bbox_xyxy=tr.bbox_xyxy,
                method=tr.method,
                category=tr.category or tr.kind,
                confidence=tr.confidence,
            )
            for tr in self._tracks
        ]

    def faces(
        self,
        frame_index: int,
        timestamp_ns: int,
        backend: str = "yolo11n_face",
    ) -> list[FaceDetection]:
        """当前 face track → FaceDetection 记录。"""
        return [
            FaceDetection(
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                bbox_xyxy=tr.bbox_xyxy,
                confidence=tr.confidence,
                backend=backend,
            )
            for tr in self._tracks
            if tr.kind == "face"
        ]

    def texts(
        self,
        frame_index: int,
        timestamp_ns: int,
        detector: str = "easyocr",
    ) -> list[TextDetection]:
        """当前 text track → TextDetection 记录（text 为最后一次 OCR 内容）。"""
        return [
            TextDetection(
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                bbox_xyxy=tr.bbox_xyxy,
                text=tr.text,
                confidence=tr.confidence,
                detector=detector,
            )
            for tr in self._tracks
            if tr.kind == "text"
        ]


__all__ = ["KLTRegionPropagator"]
