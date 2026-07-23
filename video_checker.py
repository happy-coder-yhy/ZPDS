"""
视频完整性检查：基本信息、坏帧、黑屏。
"""

import os
import cv2


def check_video_info(video_path: str) -> dict:
    """读取视频基本信息。

    Returns:
        {"frame_count": int, "fps": float, "width": int, "height": int}
        文件不存在时返回全 0。
    """
    if not os.path.exists(video_path):
        return {"frame_count": 0, "fps": 0.0, "width": 0, "height": 0}

    cap = cv2.VideoCapture(video_path)
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def check_bad_frames(video_path: str) -> int:
    """统计解码失败 / None 帧数量。"""
    if not os.path.exists(video_path):
        return 0

    cap = cv2.VideoCapture(video_path)
    bad = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            bad += 1
    cap.release()
    return bad


def check_black_frames(
    video_path: str,
    threshold: float = 5.0,
    max_print: int = 10,
) -> dict:
    """检测黑屏帧（灰度均值低于阈值）。

    Returns:
        {"black_count": int, "black_list": [frame_idx, ...], "max_print": int}
    """
    if not os.path.exists(video_path):
        return {"black_count": 0, "black_list": [], "max_print": max_print}

    cap = cv2.VideoCapture(video_path)
    black_list = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.mean() < threshold:
            black_list.append(frame_idx)
        frame_idx += 1

    cap.release()
    return {
        "black_count": len(black_list),
        "black_list": black_list,
        "max_print": max_print,
    }
