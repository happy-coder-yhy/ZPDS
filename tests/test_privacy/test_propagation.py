"""测试 KLT 区域传播器与 pipeline 稀疏检测集成。"""

import cv2
import numpy as np
import pytest

from zpds.privacy.pipeline import PrivacyPipeline
from zpds.privacy.propagation import KLTRegionPropagator
from zpds.privacy.schemas import (
    FaceDetection,
    PIIClassification,
    TextDetection,
)


def _textured_base(h: int = 128, w: int = 160) -> np.ndarray:
    """平滑随机噪声（多尺度自然纹理，KLT 金字塔跟踪可靠）。

    注意：合成测试图必须用这种连续纹理——棋盘格高频图案会让
    KLT 金字塔混淆，纯色/大块平坦区则无法跟踪（aperture 问题）。
    """
    rng = np.random.default_rng(42)
    low = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(low, (9, 9), 3.0)


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


class TestKLTPropagator:
    def test_translation_follows_target(self):
        h, w = 128, 160
        base = _textured_base(h, w)
        prop = KLTRegionPropagator(w, h)
        initial = (0.3 * w, 0.3 * h)
        prop.sync_faces(
            [FaceDetection(0, 0, (0.2, 0.2, 0.4, 0.4), 0.9)],
            face_method="blur",
        )
        prev = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        for n in range(1, 6):
            m = np.float32([[1, 0, 2 * n], [0, 1, 1 * n]])
            frame = cv2.warpAffine(
                base, m, (w, h), borderMode=cv2.BORDER_REFLECT
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prop.step(prev, gray)
            prev = gray
            cx, cy = _center(prop.regions()[0].bbox_xyxy)
            expect = (initial[0] + 2 * n, initial[1] + 1 * n)
            # _center 返回归一化坐标，转像素再比较（KLT 逐帧误差 ~1px，累计放宽到 3px）
            assert abs(cx * w - expect[0]) < 3.0, f"frame {n}: cx={cx*w} expect={expect[0]}"
            assert abs(cy * h - expect[1]) < 3.0, f"frame {n}: cy={cy*h} expect={expect[1]}"

    def test_stale_dilates_bbox(self):
        h, w = 96, 128
        base = _textured_base(h, w)
        prop = KLTRegionPropagator(w, h, max_stale_frames=5)
        prop.sync_faces(
            [FaceDetection(0, 0, (0.3, 0.3, 0.5, 0.5), 0.9)],
            face_method="blur",
        )
        prev = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        # 全黑帧：KLT 无可跟纹理 → 跟踪退化 → 原框 + 膨胀
        black = np.zeros((h, w, 3), dtype=np.uint8)
        gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
        prop.step(prev, gray)
        assert prop.track_count == 1
        tr = prop._tracks[0]
        assert tr.stale_frames == 1
        # 0.3 - 0.2*0.3 = 0.24（膨胀后 x1 左移）
        assert prop.regions()[0].bbox_xyxy[0] < 0.27

    def test_stale_over_limit_drops_track(self):
        h, w = 96, 128
        base = _textured_base(h, w)
        # max_stale_frames=0：第一次跟踪退化即删除（验证 stale 计数路径）
        prop = KLTRegionPropagator(w, h, max_stale_frames=0)
        prop.sync_faces(
            [FaceDetection(0, 0, (0.3, 0.3, 0.5, 0.5), 0.9)],
            face_method="blur",
        )
        prev = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        black = np.zeros((h, w, 3), dtype=np.uint8)
        gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
        prop.step(prev, gray)
        assert prop.track_count == 0

    def test_sync_miss_drops_track(self):
        prop = KLTRegionPropagator(100, 100, max_misses=2)
        prop.sync_faces(
            [FaceDetection(0, 0, (0.1, 0.1, 0.3, 0.3), 0.9)],
            face_method="blur",
        )
        assert prop.track_count == 1
        prop.sync_faces([], face_method="blur")  # miss 1
        assert prop.track_count == 1
        prop.sync_faces([], face_method="blur")  # miss 2
        assert prop.track_count == 1
        prop.sync_faces([], face_method="blur")  # miss 3 > 2 → 删除
        assert prop.track_count == 0

    def test_sync_updates_matched_track(self):
        prop = KLTRegionPropagator(100, 100)
        prop.sync_faces(
            [FaceDetection(0, 0, (0.1, 0.1, 0.3, 0.3), 0.9)],
            face_method="blur",
        )
        # 检测帧 bbox 微移（IoU 匹配）→ track 更新为检测值
        prop.sync_faces(
            [FaceDetection(1, 1, (0.12, 0.12, 0.32, 0.32), 0.8)],
            face_method="blur",
        )
        assert prop.track_count == 1
        bbox = prop.regions()[0].bbox_xyxy
        assert abs(bbox[0] - 0.12) < 1e-6
        assert abs(bbox[2] - 0.32) < 1e-6

    def test_text_track_keeps_text_and_category(self):
        prop = KLTRegionPropagator(100, 100)
        td = TextDetection(0, 0, (0.1, 0.1, 0.4, 0.4), "张三", 0.9)
        prop.sync_texts(
            [td], text_method="black_rect", categories={id(td): "person_name"}
        )
        texts = prop.texts(1, 1)
        assert len(texts) == 1
        assert texts[0].text == "张三"
        assert texts[0].bbox_xyxy == (0.1, 0.1, 0.4, 0.4)
        regions = prop.regions()
        assert regions[0].kind == "text"
        assert regions[0].category == "person_name"
        assert regions[0].method == "black_rect"

    def test_reset_clears_tracks(self):
        prop = KLTRegionPropagator(100, 100)
        prop.sync_faces(
            [FaceDetection(0, 0, (0.1, 0.1, 0.3, 0.3), 0.9)],
            face_method="blur",
        )
        prop.reset()
        assert prop.track_count == 0
        assert prop.regions() == []


# ---------------------------------------------------------------------------
# pipeline 稀疏检测集成
# ---------------------------------------------------------------------------


class _FakeFaceDetector:
    """返回随帧号移动的人脸检测（模拟运动目标）。"""

    def __init__(self, start_bbox, dx_px: int, w: int, h: int):
        self._bbox = start_bbox
        self._dx = dx_px
        self._w, self._h = w, h

    def detect(self, frame, frame_index, timestamp_ns):
        dx = self._dx * frame_index / self._w
        x1, y1, x2, y2 = self._bbox
        return [
            FaceDetection(
                frame_index, timestamp_ns,
                (x1 + dx, y1, x2 + dx, y2), 0.9,
            )
        ]

    def close(self):
        pass


class _FakeFaceDetectorWithGap(_FakeFaceDetector):
    """第 10 帧目标消失（模拟场景切换后画面无目标）。"""

    def detect(self, frame, frame_index, timestamp_ns):
        if frame_index == 10:
            return []
        return super().detect(frame, frame_index, timestamp_ns)


class _FakeTextDetector:
    def detect(self, frame, frame_index, timestamp_ns):
        return [TextDetection(frame_index, timestamp_ns, (0.1, 0.6, 0.3, 0.7), "文字", 0.9)]

    def close(self):
        pass


class _FakeClassifier:
    def classify(self, texts):
        return [
            PIIClassification(t, "person_name", "mask", 0.9) for t in texts
        ]

    def close(self):
        pass


def _write_video(path, frames):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def _make_moving_video(n_frames: int = 24):
    h, w = 96, 128
    base = _textured_base(h, w)
    frames = []
    for n in range(n_frames):
        m = np.float32([[1, 0, 2 * n], [0, 1, 0]])  # 第 n 帧平移 2n px
        frames.append(
            cv2.warpAffine(base, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        )
    return frames


class TestPipelineSparseDetection:
    def test_intermediate_frames_propagated(self, tmp_path):
        frames = _make_moving_video(24)
        video = tmp_path / "prop.mp4"
        _write_video(video, frames)

        face_det = _FakeFaceDetector((0.25, 0.3, 0.45, 0.5), 2, 128, 96)
        pipeline = PrivacyPipeline(
            video,
            profile="guida",
            face_detector=face_det,
            text_detector=_FakeTextDetector(),
            pii_classifier=_FakeClassifier(),
            face_interval=3,
            text_interval=3,
        )
        records = pipeline.run_to_list()
        assert len(records) == 24

        # 人脸：检测帧 + 传播帧都存在，中心跟随目标（0.35w + 2n px）
        for n, record in enumerate(records):
            assert record.faces, f"frame {n}: 应有 face track"
            cx = _center(record.faces[0].bbox_xyxy)[0] * 128
            expect = 0.35 * 128 + 2 * n
            assert abs(cx - expect) < 3.0, f"frame {n}: cx={cx} expect={expect}"
        # 文本（静止）：全部帧存在；检测帧 sync 拉回固定位置，
        # 传播帧最多漂移 2 个间隔帧（≤4px），允许 6px 容差
        for n, record in enumerate(records):
            assert record.texts, f"frame {n}: 应有 text track"
            cx = _center(record.texts[0].bbox_xyxy)[0] * 128
            assert abs(cx - 0.2 * 128) < 6.0, f"frame {n}: text cx={cx}"
        # 每帧至少一个 mask 遮挡区域（face 或 text）
        assert all(record.regions for record in records)
        # 检测帧的 pii 分类存在，传播帧不重复分类
        detected = [n for n in range(24) if n % 3 == 0]
        for n, record in enumerate(records):
            if n in detected:
                assert record.pii_classifications, f"检测帧 {n} 应有分类"
                assert record.llm_available
            else:
                assert not record.pii_classifications, f"传播帧 {n} 不应分类"

    def test_reset_frames_force_detection(self, tmp_path):
        frames = _make_moving_video(16)
        video = tmp_path / "prop_reset.mp4"
        _write_video(video, frames)

        # 第 10 帧目标消失；reset_frames={10} 强制检测 → 立即清空 track
        face_det = _FakeFaceDetectorWithGap((0.25, 0.3, 0.45, 0.5), 2, 128, 96)
        pipeline = PrivacyPipeline(
            video,
            profile="guida",
            face_detector=face_det,
            text_detector=_FakeTextDetector(),
            pii_classifier=_FakeClassifier(),
            face_interval=5,
            text_interval=5,
            reset_frames={10},
        )
        records = pipeline.run_to_list()
        # 帧 10：reset + 检测（空）→ 无 face
        assert records[10].faces == ()
        # 后续传播帧也无 face（track 已清空，下一检测帧前不重建）
        assert records[11].faces == ()
        assert records[12].faces == ()  # 检测帧 15 前无检测

    def test_no_reset_frames_keeps_track_until_next_detection(self, tmp_path):
        frames = _make_moving_video(16)
        video = tmp_path / "prop_noreset.mp4"
        _write_video(video, frames)

        face_det = _FakeFaceDetectorWithGap((0.25, 0.3, 0.45, 0.5), 2, 128, 96)
        pipeline = PrivacyPipeline(
            video,
            profile="guida",
            face_detector=face_det,
            text_detector=_FakeTextDetector(),
            pii_classifier=_FakeClassifier(),
            face_interval=5,
            text_interval=5,
        )
        records = pipeline.run_to_list()
        # 无 reset：帧 10 检测（gap 空）→ 仅 miss+1，track 继续传播
        assert records[10].faces, "无 reset 帧时 track 应继续存在"
        assert records[11].faces
        # 检测帧 15：目标重新出现 → 检测到并更新 track
        assert records[15].faces
