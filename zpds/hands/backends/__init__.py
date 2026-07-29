"""MediaPipe 手部检测后端。

- TasksHandLandmarkerBackend: 新版 Tasks API（推荐，需 .task 模型文件）
- SolutionsHandsBackend: 经典 legacy API（兼容性好，不需额外模型文件）
"""

from __future__ import annotations

from zpds.hands.backends.tasks_hand_landmarker import TasksHandLandmarkerBackend
from zpds.hands.backends.solutions_hands import SolutionsHandsBackend

__all__ = [
    "TasksHandLandmarkerBackend",
    "SolutionsHandsBackend",
]
