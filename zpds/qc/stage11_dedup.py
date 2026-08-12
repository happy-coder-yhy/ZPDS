"""Stage 11: 跨 Session 近重复检测（D18）。

分级执行：
  1. 文件精确 Hash（SHA-256）
  2. 视频 pHash 或轻量视觉特征
  3. 机器人轨迹指纹

该功能只产生重复候选组和证据，不自动删除数据。
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from zpds.core.decisions import Decision, ReasonCode, Severity
from zpds.qc.cascade import register_stage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认阈值
# ---------------------------------------------------------------------------

DEFAULT_PHASH_HAMMING_MAX = 10        # pHash 汉明距离阈值（≤此值视为候选重复）
DEFAULT_TRAJECTORY_CORR_MIN = 0.95    # 轨迹相关性阈值（≥此值视为候选重复）
DEFAULT_PHASH_SAMPLE_COUNT = 10       # pHash 采样帧数


# ---------------------------------------------------------------------------
# 文件精确 Hash
# ---------------------------------------------------------------------------


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """计算文件的哈希值。

    Parameters
    ----------
    file_path : str
    algorithm : str
        "sha256", "md5", 等。

    Returns
    -------
    str
        十六进制哈希字符串，读取失败返回空字符串。
    """
    try:
        h = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        logger.warning("Failed to hash file: %s", file_path, exc_info=True)
        return ""


def detect_exact_duplicates(
    file_paths: list[str],
    *,
    algorithm: str = "sha256",
) -> list[Decision]:
    """通过文件精确哈希检测重复文件。

    Parameters
    ----------
    file_paths : list[str]
        待检查的文件路径列表。
    algorithm : str
        哈希算法。

    Returns
    -------
    list[Decision]
    """
    decisions: list[Decision] = []
    hash_to_paths: dict[str, list[str]] = {}

    for fp in file_paths:
        fhash = compute_file_hash(fp, algorithm=algorithm)
        if not fhash:
            continue
        hash_to_paths.setdefault(fhash, []).append(fp)

    duplicate_groups = {h: ps for h, ps in hash_to_paths.items() if len(ps) > 1}

    for fhash, paths in duplicate_groups.items():
        decisions.append(
            Decision(
                stage=11,
                reason=ReasonCode.NEAR_DUPLICATE,
                severity=Severity.WARN,
                message=(
                    f"Exact file hash duplicate group ({len(paths)} files): "
                    f"hash={fhash[:16]}..."
                ),
                detail={
                    "method": f"exact_{algorithm}",
                    "hash": fhash,
                    "file_count": len(paths),
                    "files": paths,
                    "recommended_action": "manual_review",
                },
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# 视频 pHash
# ---------------------------------------------------------------------------


def compute_phash(
    video_path: str,
    sample_count: int = DEFAULT_PHASH_SAMPLE_COUNT,
    hash_size: int = 8,
    frames: Sequence | None = None,
) -> np.ndarray | None:
    """计算视频的感知哈希（pHash）。

    均匀采样 N 帧，计算每帧的 DCT 感知哈希，取均值作为视频指纹。

    Parameters
    ----------
    video_path : str
    sample_count : int
        采样帧数。
    hash_size : int
        pHash 尺寸（8 → 64-bit hash）。
    frames : Optional[Sequence[np.ndarray]]
        共享帧源（BGR，下标 = 帧号）。提供时直接随机访问采样帧，
        不再打开 VideoCapture（mkv 的 POS_FRAMES 跳转是顺序 seek，
        实际解码量远大于采样数）。

    Returns
    -------
    Optional[np.ndarray]
        shape (hash_size, hash_size) 的浮点指纹，失败返回 None。
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for pHash computation")
        return None

    if frames is not None:
        total_frames = len(frames)
        if total_frames <= 0:
            return None
        cap = None
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

    # 均匀采样帧索引
    if total_frames <= sample_count:
        sample_indices = list(range(total_frames))
    else:
        step = total_frames / sample_count
        sample_indices = [int(i * step) for i in range(sample_count)]

    fingerprints: list[np.ndarray] = []

    for target_idx in sample_indices:
        if frames is not None:
            frame = frames[target_idx]
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if not ret:
                continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size * 4, hash_size * 4))

        # DCT
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:hash_size, :hash_size]
        mean_val = np.mean(dct_low)

        # 二值化
        phash = (dct_low > mean_val).astype(np.float32)
        fingerprints.append(phash)

    if cap is not None:
        cap.release()

    if not fingerprints:
        return None

    # 均值指纹
    return np.mean(np.stack(fingerprints, axis=0), axis=0)


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """计算两个浮点指纹的近似汉明距离（阈值二值化后比较）。"""
    binary1 = (hash1 > 0.5).astype(np.uint8)
    binary2 = (hash2 > 0.5).astype(np.uint8)
    return int(np.sum(binary1 != binary2))


def detect_video_duplicates(
    video_paths: list[str],
    *,
    hamming_max: int = DEFAULT_PHASH_HAMMING_MAX,
    sample_count: int = DEFAULT_PHASH_SAMPLE_COUNT,
    frames: Sequence | None = None,
) -> list[Decision]:
    """通过视频 pHash 检测候选重复视频。

    只产生候选组，不自动删除数据。

    Parameters
    ----------
    video_paths : list[str]
        视频文件路径列表。
    hamming_max : int
        判定重复的汉明距离上限。
    sample_count : int
        pHash 采样帧数。

    Returns
    -------
    list[Decision]
    """
    decisions: list[Decision] = []

    # 计算所有视频的 pHash（主视频路径匹配到共享帧源时直接随机访问采样）
    frame_source_path = None
    if frames is not None:
        frame_source_path = str(Path(getattr(frames, "video_path", "")).resolve())
    hashes: dict[str, np.ndarray | None] = {}
    for vp in video_paths:
        use_frames = None
        if frame_source_path and str(Path(vp).resolve()) == frame_source_path:
            use_frames = frames
        hashes[vp] = compute_phash(vp, sample_count=sample_count, frames=use_frames)

    valid = {vp: h for vp, h in hashes.items() if h is not None}
    paths = list(valid.keys())
    n = len(paths)

    # 两两比较
    checked: set[tuple] = set()
    for i in range(n):
        for j in range(i + 1, n):
            pair = (paths[i], paths[j])
            dist = hamming_distance(valid[paths[i]], valid[paths[j]])
            if dist <= hamming_max:
                decisions.append(
                    Decision(
                        stage=11,
                        reason=ReasonCode.NEAR_DUPLICATE,
                        severity=Severity.WARN,
                        message=(
                            f"Near-duplicate video pair (pHash Hamming={dist}): "
                            f"{Path(paths[i]).name} <-> {Path(paths[j]).name}"
                        ),
                        detail={
                            "method": "phash",
                            "hamming_distance": int(dist),
                            "hamming_threshold": hamming_max,
                            "file_a": paths[i],
                            "file_b": paths[j],
                            "recommended_action": "manual_review",
                        },
                    )
                )
                checked.add(pair)

    if not decisions:
        decisions.append(
            Decision(
                stage=11,
                reason=ReasonCode.NEAR_DUPLICATE,
                severity=Severity.INFO,
                message=f"No near-duplicate videos found among {n} file(s)",
                detail={
                    "method": "phash",
                    "checked_file_count": n,
                    "hamming_threshold": hamming_max,
                },
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# 机器人轨迹指纹
# ---------------------------------------------------------------------------


def compute_trajectory_fingerprint(
    joint_positions: np.ndarray,
    n_bins: int = 20,
) -> np.ndarray | None:
    """计算机器人关节轨迹的轻量指纹。

    对每条关节轨迹计算直方图分布，拼接为联合指纹。

    Parameters
    ----------
    joint_positions : np.ndarray
        shape (T, J) 的关节位置序列。
    n_bins : int
        直方图 bin 数。

    Returns
    -------
    Optional[np.ndarray]
        shape (J * n_bins,) 的指纹向量。
    """
    if joint_positions.size == 0 or joint_positions.ndim != 2:
        return None

    n_joints = joint_positions.shape[1]
    fingerprints = []

    for j in range(n_joints):
        col = joint_positions[:, j]
        finite = col[np.isfinite(col)]
        if len(finite) < 2:
            hist = np.zeros(n_bins)
        else:
            hist, _ = np.histogram(finite, bins=n_bins, density=True)
        fingerprints.append(hist)

    return np.concatenate(fingerprints)


def detect_trajectory_duplicates(
    trajectories: dict[str, np.ndarray],
    *,
    correlation_min: float = DEFAULT_TRAJECTORY_CORR_MIN,
    n_bins: int = 20,
) -> list[Decision]:
    """通过关节轨迹指纹检测候选重复 session。

    Parameters
    ----------
    trajectories : dict[str, np.ndarray]
        {session_id: joint_positions_array} 映射。
    correlation_min : float
        相关性阈值（≥此值视为候选重复）。
    n_bins : int

    Returns
    -------
    list[Decision]
    """
    decisions: list[Decision] = []

    # 计算所有轨迹指纹
    fingerprints: dict[str, np.ndarray | None] = {}
    for sid, traj in trajectories.items():
        fingerprints[sid] = compute_trajectory_fingerprint(traj, n_bins=n_bins)

    valid = {sid: fp for sid, fp in fingerprints.items() if fp is not None}
    sids = list(valid.keys())
    n = len(sids)

    checked = 0
    for i in range(n):
        for j in range(i + 1, n):
            fp_a = valid[sids[i]]
            fp_b = valid[sids[j]]
            # Pearson 相关系数
            corr = np.corrcoef(fp_a, fp_b)[0, 1]
            checked += 1
            if np.isfinite(corr) and corr >= correlation_min:
                decisions.append(
                    Decision(
                        stage=11,
                        reason=ReasonCode.NEAR_DUPLICATE,
                        severity=Severity.WARN,
                        message=(
                            f"Near-duplicate trajectory pair (corr={corr:.4f}): "
                            f"{sids[i]} <-> {sids[j]}"
                        ),
                        detail={
                            "method": "trajectory_fingerprint",
                            "correlation": round(float(corr), 6),
                            "correlation_threshold": correlation_min,
                            "session_a": sids[i],
                            "session_b": sids[j],
                            "recommended_action": "manual_review",
                        },
                    )
                )

    if not decisions and checked > 0:
        decisions.append(
            Decision(
                stage=11,
                reason=ReasonCode.NEAR_DUPLICATE,
                severity=Severity.INFO,
                message=f"No near-duplicate trajectories among {n} session(s)",
                detail={
                    "method": "trajectory_fingerprint",
                    "checked_pairs": checked,
                    "session_count": n,
                    "correlation_threshold": correlation_min,
                },
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# Stage 11 统一入口
# ---------------------------------------------------------------------------


def check(
    file_paths: list[str] | None = None,
    video_paths: list[str] | None = None,
    trajectories: dict[str, np.ndarray] | None = None,
    *,
    stage_config: dict | None = None,
    frames: Sequence | None = None,
) -> list[Decision]:
    """Stage 11 统一检查入口：跨 Session 近重复检测。

    分级执行：
      1. 文件精确哈希
      2. 视频 pHash
      3. 机器人轨迹指纹

    只产生候选组，不自动删除数据。

    Parameters
    ----------
    file_paths : Optional[list[str]]
        待检查的文件路径列表。
    video_paths : Optional[list[str]]
        视频文件路径列表。
    trajectories : Optional[dict[str, np.ndarray]]
        {session_id: joint_positions} 映射。
    stage_config : Optional[dict]
        阈值覆盖。
    frames : Optional[Sequence]
        共享帧源（当前 session 主视频）。提供时对匹配该源视频的
        路径直接用帧源随机访问采样，不再打开 VideoCapture。

    Returns
    -------
    list[Decision]
    """
    cfg = stage_config or {}
    decisions: list[Decision] = []

    # Level 1: 文件精确哈希
    if file_paths:
        hash_cfg = cfg.get("exact_hash", {})
        if hash_cfg.get("enabled", True):
            decisions.extend(
                detect_exact_duplicates(
                    file_paths,
                    algorithm=hash_cfg.get("algorithm", "sha256"),
                )
            )

    # Level 2: 视频 pHash
    if video_paths:
        phash_cfg = cfg.get("phash", {})
        if phash_cfg.get("enabled", True):
            decisions.extend(
                detect_video_duplicates(
                    video_paths,
                    hamming_max=phash_cfg.get(
                        "hamming_max", DEFAULT_PHASH_HAMMING_MAX
                    ),
                    sample_count=phash_cfg.get(
                        "sample_count", DEFAULT_PHASH_SAMPLE_COUNT
                    ),
                    frames=frames,
                )
            )

    # Level 3: 机器人轨迹指纹
    if trajectories:
        traj_cfg = cfg.get("trajectory", {})
        if traj_cfg.get("enabled", True):
            decisions.extend(
                detect_trajectory_duplicates(
                    trajectories,
                    correlation_min=traj_cfg.get(
                        "correlation_min", DEFAULT_TRAJECTORY_CORR_MIN
                    ),
                    n_bins=traj_cfg.get("n_bins", 20),
                )
            )

    if not file_paths and not video_paths and not trajectories:
        decisions.append(
            Decision(
                stage=11,
                reason=ReasonCode.NEAR_DUPLICATE,
                severity=Severity.INFO,
                message="Stage 11: no input data for dedup check",
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# QCCascade 注册入口
# ---------------------------------------------------------------------------


@register_stage(11)
def _check_stage11(context: dict) -> list[Decision]:
    """Stage 11 QCCascade 入口：从 context dict 提取参数并调用 check()。

    Stage 11 是跨流 / 跨 session 的去重检查，只需在整个级联中运行一次。
    使用 ``_stage11_done`` 上下文标记避免多 stream 重复执行。
    """
    if context.get("_stage11_done"):
        return []
    context["_stage11_done"] = True
    stage_config = context.get("stage_config", {})
    return check(
        file_paths=context.get("file_paths"),
        video_paths=context.get("video_paths"),
        trajectories=context.get("trajectories"),
        stage_config=stage_config,
        frames=context.get("frames"),
    )
