"""
坏帧检测器：扫描 MKV 中解码失败 / None 的帧。

顺带统计解码器 stderr 的 MJPEG 损坏宏块警告（decode_warning_count）。
ffmpeg 的 ``[mjpeg @ ...] error count: N`` 只写 C 层 stderr——句柄在进程
启动时缓存，进程内 os.dup2 / SetStdHandle 均捕获不到（已实证），必须由
子进程解码收集（CreateProcess 级标准句柄管道）。因此本检测器的解码工作
整体交给子进程（坏帧索引 + stderr 一次拿到），主进程只做解析，不重复解码。
"""

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import cv2

from zpds_prepare.decisions.issue_model import QualityIssue

# 地址无 0x 前缀（OpenCV 内置 ffmpeg 打印裸指针），两种格式都兼容
_MJPEG_ERROR_RE = re.compile(r"\[mjpeg @ 0?x?[0-9a-f]+\] error count: (\d+)")

# 子进程解码脚本：仅解码并打印坏帧索引 JSON；mjpeg 警告留在子进程 stderr，
# 由父进程从 proc.stderr 解析（无时序问题）。
_DECODE_PROBE_SRC = textwrap.dedent(
    """\
    import json
    import sys

    import cv2

    cap = cv2.VideoCapture(sys.argv[1])
    bad = []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            bad.append(n)
        n += 1
    cap.release()
    print(json.dumps({"frames": n, "bad": bad}), flush=True)
    """
)


def _mjpeg_error_count(stderr_text: str) -> int:
    """从子进程 stderr 中提取最大 MJPEG 错误累计值。

    ffmpeg 每次遇到损坏宏块打印当前累计计数（同实例可能多次出现同值），
    取最大值即该解码实例的最终错误数。
    """
    counts = [int(v) for v in _MJPEG_ERROR_RE.findall(stderr_text)]
    return max(counts) if counts else 0


def _decode_probe(video_path: str) -> tuple[list[int] | None, int, str]:
    """子进程解码一遍，返回 (坏帧索引, 总帧数, stderr 文本)。

    探测失败（子进程崩溃/超时）返回 (None, 0, "")，调用方回退主进程解码。
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _DECODE_PROBE_SRC, video_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return None, 0, ""
    if proc.returncode != 0:
        return None, 0, proc.stderr
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None, 0, proc.stderr
    return data["bad"], int(data["frames"]), proc.stderr


def _decode_inline(video_path: str) -> tuple[list[int], int]:
    """主进程解码兜底（探测子进程失败时）：只做坏帧检测，无 stderr 统计。"""
    cap = cv2.VideoCapture(video_path)
    bad: list[int] = []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            bad.append(n)
        n += 1
    cap.release()
    return bad, n


def detect_bad_frames(
    video_path: str,
    timestamps_ns: list[int],
    stream_id: str = "ego_rgb",
) -> list[QualityIssue]:
    """检测 MKV 中解码失败的帧。

    如果坏帧形成连续区间，以区间的 start/end 时间戳表示；
    如果坏帧零星分布，以整个 session 范围表示（方便标记）。

    顺带产出 ``decode_warning_count`` issue：解码器 stderr 报告的 MJPEG
    损坏宏块数（>0 时附加；帧本身可能已修复或解码成功，故为 warn 级）。

    Args:
        video_path: MKV 文件路径
        timestamps_ns: index.jsonl 帧时间戳列表
        stream_id: 数据流标识

    Returns:
        QualityIssue 列表（无坏帧时为空）
    """
    if not Path(video_path).exists():
        return []

    bad_indices, total_frames, stderr_text = _decode_probe(video_path)
    if bad_indices is None:
        # 子进程探测失败（崩溃/超时）→ 主进程解码兜底（仅坏帧，无 stderr 统计）
        bad_indices, total_frames = _decode_inline(video_path)
        stderr_text = ""

    issues: list[QualityIssue] = []
    decode_error_count = _mjpeg_error_count(stderr_text)
    if decode_error_count > 0:
        issues.append(QualityIssue(
            issue_type="decode_warning_count",
            stream_id=stream_id,
            start_ns=timestamps_ns[0] if timestamps_ns else 0,
            end_ns=timestamps_ns[-1] if timestamps_ns else 0,
            severity="warn",
            decision="keep_with_flag",
            details={
                "count": decode_error_count,
                "total_frames_scanned": total_frames,
                "source": "ffmpeg_mjpeg_stderr",
                "message": f"视频解码器报告 {decode_error_count} 个 MJPEG 损坏宏块",
            },
        ))

    if not bad_indices:
        return issues

    # 将坏帧索引映射到时间戳
    n = min(len(timestamps_ns), total_frames)

    # 合并连续坏帧区间（复用黑屏检测的区间合并思路）
    spans = _merge_consecutive(bad_indices, timestamps_ns[:n])

    for start_ns, end_ns, count in spans:
        issues.append(QualityIssue(
            issue_type="bad_frame",
            stream_id=stream_id,
            start_ns=start_ns,
            end_ns=end_ns,
            severity="error",
            decision="keep_with_flag",
            details={
                "bad_frame_count": count,
                "total_frames_scanned": total_frames,
                "bad_ratio": round(count / max(total_frames, 1), 4),
            },
        ))

    return issues


def _merge_consecutive(
    indices: list[int],
    timestamps_ns: list[int],
) -> list[tuple[int, int, int]]:
    """将连续索引合并为 (start_ns, end_ns, count)。"""
    if not indices or not timestamps_ns:
        return []

    spans = []
    start_idx = indices[0]
    prev_idx = indices[0]

    for i in range(1, len(indices)):
        current = indices[i]
        if current != prev_idx + 1:
            # 区间结束
            spans.append(_make_span(start_idx, prev_idx, timestamps_ns))
            start_idx = current
        prev_idx = current

    # 最后一个区间
    spans.append(_make_span(start_idx, prev_idx, timestamps_ns))
    return spans


def _make_span(
    start_idx: int,
    end_idx: int,
    timestamps_ns: list[int],
) -> tuple[int, int, int]:
    """将起止帧号转为 (start_ns, end_ns, frame_count)。"""
    start_ns = (
        timestamps_ns[start_idx]
        if start_idx < len(timestamps_ns) else 0
    )
    end_ns = (
        timestamps_ns[end_idx]
        if end_idx < len(timestamps_ns) else 0
    )
    # 加上最后一帧的近似持续
    if end_idx > 0 and end_idx < len(timestamps_ns):
        end_ns += timestamps_ns[end_idx] - timestamps_ns[end_idx - 1]
    count = end_idx - start_idx + 1
    return start_ns, end_ns, count
