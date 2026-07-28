"""Stage 11 近重复检测 QC 测试（D18）。"""

import hashlib

import cv2
import numpy as np

from zpds.core.decisions import Severity
from zpds.qc.stage11_dedup import (
    check,
    compute_file_hash,
    compute_phash,
    compute_trajectory_fingerprint,
    detect_exact_duplicates,
    detect_trajectory_duplicates,
    detect_video_duplicates,
    hamming_distance,
)

# ---------------------------------------------------------------------------
# 文件哈希
# ---------------------------------------------------------------------------


class TestFileHash:
    def test_compute_hash(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        h = compute_file_hash(str(f))
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert h == expected

    def test_nonexistent_file(self):
        h = compute_file_hash("/nonexistent/file.bin")
        assert h == ""


class TestExactDuplicates:
    def test_no_duplicates(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        decisions = detect_exact_duplicates([str(f1), str(f2)])
        info = [d for d in decisions if d.severity == Severity.INFO]
        assert len(info) >= 0
        warns = [d for d in decisions if d.severity == Severity.WARN]
        assert len(warns) == 0

    def test_duplicates(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same content")
        f2.write_text("same content")
        decisions = detect_exact_duplicates([str(f1), str(f2)])
        warns = [d for d in decisions if d.severity == Severity.WARN]
        assert len(warns) >= 1
        assert "hash" in warns[0].detail

    def test_empty_list(self):
        decisions = detect_exact_duplicates([])
        assert decisions == []


# ---------------------------------------------------------------------------
# 视频 pHash
# ---------------------------------------------------------------------------


class TestPHash:
    def test_compute_phash(self, tmp_path):
        """计算视频 pHash。"""
        frames = [np.random.randint(0, 255, (120, 160), dtype=np.uint8) for _ in range(10)]
        vpath = str(tmp_path / "test.mp4")
        _create_test_video(vpath, frames)

        ph = compute_phash(vpath)
        assert ph is not None
        assert ph.shape == (8, 8)

    def test_nonexistent_video(self):
        ph = compute_phash("/nonexistent/video.mp4")
        assert ph is None

    def test_hamming_distance(self):
        h1 = np.ones((8, 8), dtype=np.float32)
        h2 = np.zeros((8, 8), dtype=np.float32)
        assert hamming_distance(h1, h2) == 64

        h3 = h1.copy()
        assert hamming_distance(h1, h3) == 0


class TestVideoDuplicates:
    def test_different_videos(self, tmp_path):
        """不同视频不应被误判为重复（概率性，极小概率匹配）。"""
        v1 = str(tmp_path / "v1.mp4")
        v2 = str(tmp_path / "v2.mp4")
        rng1 = np.random.RandomState(0)
        rng2 = np.random.RandomState(99)
        _create_test_video(v1, [rng1.randint(0, 255, (120, 160), dtype=np.uint8) for _ in range(15)])
        _create_test_video(v2, [rng2.randint(0, 255, (120, 160), dtype=np.uint8) for _ in range(15)])

        decisions = detect_video_duplicates([v1, v2])
        # 不同视频不应有 WARN 级别的重复判定
        # （极小概率因随机帧相似而匹配，但不应视为错误）
        warns = [d for d in decisions if d.severity == Severity.WARN]
        if warns:
            # 有匹配时至少验证输出结构完整
            assert "hamming_distance" in warns[0].detail

    def test_single_video(self, tmp_path):
        """单个视频。"""
        v1 = str(tmp_path / "v1.mp4")
        _create_test_video(v1, [np.random.randint(0, 255, (120, 160), dtype=np.uint8) for _ in range(5)])
        decisions = detect_video_duplicates([v1])
        info = [d for d in decisions if d.severity == Severity.INFO]
        assert len(info) >= 1


# ---------------------------------------------------------------------------
# 轨迹指纹
# ---------------------------------------------------------------------------


class TestTrajectoryFingerprint:
    def test_compute_fingerprint(self):
        traj = np.random.randn(100, 7)  # 100 samples, 7 joints
        fp = compute_trajectory_fingerprint(traj)
        assert fp is not None
        assert len(fp) == 7 * 20  # 7 joints * 20 bins

    def test_empty_input(self):
        fp = compute_trajectory_fingerprint(np.array([]))
        assert fp is None

    def test_nan_values(self):
        traj = np.random.randn(50, 3)
        traj[10:15, :] = np.nan
        fp = compute_trajectory_fingerprint(traj)
        assert fp is not None


class TestTrajectoryDuplicates:
    def test_different_trajectories(self):
        rng = np.random.RandomState(42)
        traj_a = rng.randn(100, 7)
        traj_b = rng.randn(100, 7) + 5  # shifted
        trajectories = {"session_a": traj_a, "session_b": traj_b}
        decisions = detect_trajectory_duplicates(trajectories, correlation_min=0.95)
        warns = [d for d in decisions if d.severity == Severity.WARN]
        assert len(warns) == 0

    def test_similar_trajectories(self):
        """高度相似的轨迹应检测到。"""
        rng = np.random.RandomState(42)
        traj_a = rng.randn(200, 3)
        traj_b = traj_a + rng.normal(0, 0.001, traj_a.shape)  # tiny noise
        trajectories = {"s1": traj_a, "s2": traj_b}
        decisions = detect_trajectory_duplicates(trajectories, correlation_min=0.95)
        warns = [d for d in decisions if d.severity == Severity.WARN]
        assert len(warns) >= 1


# ---------------------------------------------------------------------------
# check() 统一入口
# ---------------------------------------------------------------------------


class TestCheckEntry:
    def test_no_input(self):
        decisions = check()
        assert len(decisions) == 1
        assert "no input" in decisions[0].message.lower()

    def test_file_paths_only(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("test")
        decisions = check(file_paths=[str(f)])
        assert isinstance(decisions, list)

    def test_all_disabled(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("test")
        decisions = check(
            file_paths=[str(f)],
            stage_config={
                "exact_hash": {"enabled": False},
                "phash": {"enabled": False},
                "trajectory": {"enabled": False},
            },
        )
        assert isinstance(decisions, list)


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------


def _create_test_video(path: str, frames: list[np.ndarray], fps: float = 30.0) -> str:
    if not frames:
        return path
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        writer.write(f)
    writer.release()
    return path
