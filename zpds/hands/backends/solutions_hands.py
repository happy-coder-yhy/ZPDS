"""
Solutions Hands 后端（经典 legacy API）。

使用 mp.solutions.hands.Hands + 内置 CPU 计算图，
不需要额外 .task 模型文件，兼容性最好。

适用环境：macOS sandbox / CI 无界面 / 任何 Tasks API 不可用的环境。
"""

from __future__ import annotations


class SolutionsHandsBackend:
    """MediaPipe 经典解决方案手部检测后端。

    使用 process() 进行推理，static_image_mode=False 启用帧间跟踪。

    注意：经典 Solutions 默认假设自拍画面的水平镜像输入。
    如果数据是未镜像的非自拍第一人称视频，需设置 input_mirrored=False，
    适配层会自动交换左右手标签。

    用法::

        backend = SolutionsHandsBackend(
            num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            input_mirrored=False,
        )
        raw = backend.infer(frame_rgb, timestamp_ms=33)
        backend.close()
    """

    name = "solutions_hands"

    def __init__(
        self,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        input_mirrored: bool = False,
    ):
        """初始化经典 Solutions Hands。

        Args:
            num_hands: 最大检测手数。
            min_detection_confidence: 手掌检测最低置信度 [0, 1]。
            min_tracking_confidence: 跟踪最低置信度 [0, 1]。
            model_complexity: 模型复杂度 (0=轻量, 1=完整)。
            input_mirrored: 输入是否已水平镜像。

        Raises:
            RuntimeError: 当前 mediapipe 版本不支持 solutions 模块。
                solutions 在 mediapipe>=0.10.19 中已移除。
                请降级到 mediapipe<=0.10.18:
                    pip install mediapipe==0.10.18
                或改用 tasks_hand_landmarker 后端。
        """
        import mediapipe as mp

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            raise RuntimeError(
                "当前 mediapipe 版本不支持 solutions 模块。"
                "solutions 在 mediapipe>=0.10.19 中已移除。"
                "请降级: pip install mediapipe==0.10.18，"
                "或改用 backend: tasks_hand_landmarker。"
            )

        self._mp = mp
        self._input_mirrored = input_mirrored

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,         # 启用帧间跟踪
            max_num_hands=num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ---- 核心推理 ----

    def infer(self, frame_rgb, timestamp_ms: int):
        """执行单帧推理。

        Solutions 后端不直接使用 timestamp_ms（由上层适配器校验），
        但保留参数以维持统一接口。

        Args:
            frame_rgb: RGB uint8 图像 (H, W, 3)。
            timestamp_ms: 帧时间戳（毫秒），仅用于上层校验，后端自身不使用。

        Returns:
            mp.solutions.hands.Hands 原始 process 结果。
        """
        return self._hands.process(frame_rgb)

    # ---- 原始结果提取 ----

    @staticmethod
    def extract_landmarks(result) -> list:
        """从 Solutions 结果提取关键点列表。"""
        if result is None:
            return []
        return list(result.multi_hand_landmarks) if result.multi_hand_landmarks else []

    @staticmethod
    def extract_handedness(result) -> list[list]:
        """从 Solutions 结果提取左右手列表。

        Returns:
            [[Category, ...], [Category, ...]] — 每只手一个列表。
            适配层随后根据 input_mirrored 决定是否交换。
        """
        if result is None:
            return []
        return (
            list(result.multi_handedness) if result.multi_handedness else []
        )

    # ---- 左右手纠正 ----

    @property
    def input_mirrored(self) -> bool:
        return self._input_mirrored

    @staticmethod
    def normalize_handedness(label: str, input_mirrored: bool) -> str:
        """根据输入是否镜像，纠正左右手标签。

        经典 Solutions 默认假设水平镜像的自拍画面。
        如果实际输入未镜像，需要交换 Left ↔ Right。

        Args:
            label: 原始标签 ("Left" 或 "Right")。
            input_mirrored: 输入是否已镜像。

        Returns:
            纠正后的标签。
        """
        if input_mirrored:
            return label

        if label == "Left":
            return "Right"
        if label == "Right":
            return "Left"
        return "Unknown"

    # ---- 资源释放 ----

    def close(self):
        """释放 MediaPipe 资源。"""
        if hasattr(self, "_hands") and self._hands is not None:
            self._hands.close()
            self._hands = None  # type: ignore[assignment]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
