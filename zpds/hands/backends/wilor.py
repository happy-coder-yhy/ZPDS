"""WiLoR 手部检测后端。

延迟导入：
    ``torch``、WiLoR 模型、YOLO 检测器等依赖均在首次调用
    ``infer_raw`` 时按需加载。仅 ``import zpds`` 不需要 PyTorch。

架构：
    - ``WiLoRBackend``：模型加载 + 原始推理
    - Adapter 层：坐标逆变换 + BBox 校验 + 输出 WiLoRDetection
    - Estimator 层：帧状态 + 回退调度
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from zpds.hands.wilor_schema import (
    WiLoRConfig,
    WiLoRModelInfo,
    WiLoRUnavailableError,
)


class WiLoRBackend:
    """WiLoR 手部检测后端。

    用法::

        config = WiLoRConfig(
            checkpoint_path="models/wilor/wilor_final.ckpt",
            wilor_source_path="e:/ZSPD/WiLoR",
            detector_path="models/wilor/detector.pt",
            model_config_path="models/wilor/model_config.yaml",
            device="cuda",
            model_version="wilor_cvpr2025",
        )
        backend = WiLoRBackend(config)
        raw = backend.infer_raw(frame_rgb)
        backend.close()
    """

    name = "wilor"

    def __init__(self, config: WiLoRConfig) -> None:
        self._config = config
        self._model: Any = None
        self._detector: Any = None
        self._model_cfg: Any = None
        self._device: str = config.device
        self._torch: Any = None
        self._deps_loaded: bool = False
        self._closed: bool = False

        t_start = time.perf_counter()

        self._model_info = WiLoRModelInfo.from_config(
            config, init_time_ms=0.0,
        )

        self._model_info.init_time_ms = (time.perf_counter() - t_start) * 1000

    @property
    def config(self) -> WiLoRConfig:
        return self._config

    @property
    def model_info(self) -> WiLoRModelInfo:
        return self._model_info

    @property
    def device(self) -> str:
        return self._device

    # ---- 核心推理 ----

    def infer_raw(self, frame_rgb: np.ndarray) -> dict[str, Any]:
        """执行 WiLoR 原始推理。

        返回 WiLoR 上游原始输出字典，包含:
            - pred_keypoints_3d: (1, 21, 3) MANO 顺序 3D 关节
            - pred_keypoints_2d: (1, 21, 2) 2D 关节（模型空间）
            - pred_vertices: (1, 778, 3) MANO 顶点
            - pred_cam / pred_cam_t: 相机参数
            - pred_mano_params: MANO pose/shape
            - focal_length: 焦距
            - detections: YOLO 检测结果列表

        Args:
            frame_rgb: RGB uint8 图像 ``(H, W, 3)``。

        Returns:
            包含推理结果、检测信息、批量上下文的字典。
        """
        if self._closed:
            raise RuntimeError("WiLoRBackend 已关闭，不能再调用 infer_raw")

        self._ensure_deps_loaded()

        return self._run_inference(frame_rgb)

    # ---- 资源释放 ----

    def close(self) -> None:
        self._closed = True
        if self._model is not None:
            del self._model
            self._model = None
        if self._detector is not None:
            del self._detector
            self._detector = None
        if self._torch is not None and hasattr(self._torch, "cuda"):
            try:
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:
                pass
        self._deps_loaded = False

    def __enter__(self) -> "WiLoRBackend":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False

    # ---- 内部 ----

    def _ensure_deps_loaded(self) -> None:
        """按需加载全量依赖。"""
        if self._deps_loaded:
            return

        # 1. PyTorch
        self._torch = _import_torch()

        # 2. pyrender stub（推理不需要渲染）
        _stub_pyrender()

        # 3. WiLoR 源码 + YOLO
        self._load_wilor_modules()

        # 4. 确定设备
        self._device = self._resolve_device()

        # 5. 补充元信息
        self._model_info.torch_version = getattr(self._torch, "__version__", "")
        self._model_info.cuda_version = self._detect_cuda_version()
        self._model_info.gpu_name = self._detect_gpu_name()

        self._deps_loaded = True

    def _load_wilor_modules(self) -> None:
        """加载 WiLoR 模型和 YOLO 检测器。"""
        import os as _os

        source = self._config.wilor_source_path
        if source:
            sys.path.insert(0, source)

        from wilor.models import load_wilor
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils import recursive_to
        from ultralytics import YOLO

        self._ViTDetDataset = ViTDetDataset
        self._recursive_to = recursive_to

        # WiLoR 内部使用相对路径（./mano_data/），切换到源码目录加载
        _prev_cwd = _os.getcwd()
        try:
            if source:
                _os.chdir(source)
            self._model, self._model_cfg = load_wilor(
                checkpoint_path=self._config.checkpoint_path,
                cfg_path=self._config.model_config_path,
            )
        finally:
            _os.chdir(_prev_cwd)

        # 加载检测器
        detector_path = self._config.detector_path
        if detector_path:
            self._detector = YOLO(detector_path)
        else:
            self._detector = YOLO("yolov8n.pt")  # fallback

        dev = self._torch.device(self._device)
        self._model = self._model.to(dev)
        self._model.eval()

        if self._device != "cpu":
            self._detector = self._detector.to(dev)

    def _resolve_device(self) -> str:
        if self._config.device == "cpu":
            return "cpu"
        if self._torch.cuda.is_available():
            if self._config.device == "cuda":
                return "cuda"
            return self._config.device
        print(
            f"[WiLoRBackend] CUDA 不可用，回退到 CPU",
            file=sys.stderr,
        )
        return "cpu"

    def _run_inference(self, frame_rgb: np.ndarray) -> dict[str, Any]:
        """执行完整推理流水线。"""

        # YOLO 检测
        detections = self._detector(frame_rgb, conf=0.3, verbose=False)[0]

        bboxes: list = []
        is_right: list = []
        box_confidences: list = []

        for det in detections:
            b = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(b[:4].tolist())
            box_confidences.append(float(b[4]))

        if len(bboxes) == 0:
            return {
                "detections": [],
                "pred_keypoints_3d": None,
                "pred_keypoints_2d": None,
                "pred_vertices": None,
                "pred_cam": None,
                "pred_cam_t": None,
                "pred_mano_params": None,
                "focal_length": None,
                "boxes": [],
                "is_right": [],
                "box_confidences": [],
                "box_center": [],
                "box_size": [],
                "scaled_focal_length": [],
            }

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # ViTDetDataset → DataLoader
        dataset = self._ViTDetDataset(
            self._model_cfg, frame_rgb, boxes, right, rescale_factor=2.0,
        )
        dataloader = self._torch.utils.data.DataLoader(
            dataset, batch_size=min(8, len(bboxes)),
            shuffle=False, num_workers=0,
        )

        all_kp3d = []
        all_kp2d = []
        all_verts = []
        all_cam = []
        all_cam_t = []
        all_focal = []
        all_mano = []
        all_box_center = []
        all_box_size = []
        all_right = []
        all_batch_info = []
        all_scaled_focal = []

        # WiLoR config constants for cam_crop_to_full
        _focal_length_cfg = self._model_cfg.EXTRA.FOCAL_LENGTH
        _image_size_cfg = self._model_cfg.MODEL.IMAGE_SIZE

        for batch in dataloader:
            batch = self._recursive_to(batch, self._device)

            with self._torch.no_grad():
                out = self._model(batch)

            bs = batch["img"].shape[0]
            for n in range(bs):
                all_kp3d.append(out["pred_keypoints_3d"][n].cpu().numpy())
                all_kp2d.append(out["pred_keypoints_2d"][n].cpu().numpy())
                all_verts.append(out["pred_vertices"][n].cpu().numpy())
                all_cam.append(out["pred_cam"][n].cpu().numpy())
                all_cam_t.append(out["pred_cam_t"][n].cpu().numpy())
                all_focal.append(out["focal_length"][n].cpu().numpy())
                all_mano.append({
                    "global_orient": out["pred_mano_params"]["global_orient"][n].cpu().numpy(),
                    "hand_pose": out["pred_mano_params"]["hand_pose"][n].cpu().numpy(),
                })

                # ---- batch context for cam_crop_to_full ----
                _bc = batch["box_center"][n].cpu().numpy() if "box_center" in batch else None
                _bsz = batch["box_size"][n].cpu().item() if "box_size" in batch else None
                _isz = batch["img_size"][n].cpu().numpy() if "img_size" in batch else None
                _sfl = float(
                    _focal_length_cfg / _image_size_cfg * _isz.max()
                ) if _isz is not None else None

                all_box_center.append(_bc)
                all_box_size.append(_bsz)
                all_scaled_focal.append(_sfl)

                all_right.append(batch["right"][n].cpu().numpy() if "right" in batch else None)
                all_batch_info.append({
                    "img_size": _isz,
                })

        return {
            "detections": bboxes,
            "pred_keypoints_3d": all_kp3d,
            "pred_keypoints_2d": all_kp2d,
            "pred_vertices": all_verts,
            "pred_cam": all_cam,
            "pred_cam_t": all_cam_t,
            "pred_mano_params": all_mano,
            "focal_length": all_focal,
            "boxes": boxes,
            "is_right": right,
            "box_confidences": box_confidences,
            "box_center": all_box_center,
            "box_size": all_box_size,
            "scaled_focal_length": all_scaled_focal,
            "batch_info": all_batch_info,
        }

    def _detect_cuda_version(self) -> str | None:
        try:
            if self._torch and self._torch.cuda.is_available():
                v = self._torch.version.cuda
                return str(v) if v is not None else None
        except Exception:
            pass
        return None

    def _detect_gpu_name(self) -> str | None:
        try:
            if self._torch and self._torch.cuda.is_available():
                return self._torch.cuda.get_device_name(0)
        except Exception:
            pass
        return None


# ════════════════════════════════════════════════════════════════════
# 模块级工具
# ════════════════════════════════════════════════════════════════════


def _import_torch() -> Any:
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError as exc:
        raise WiLoRUnavailableError(
            "当前环境未安装 PyTorch。请安装: pip install torch"
        ) from exc
    return torch


def _stub_pyrender() -> None:
    """绕过 pyrender/OpenGL 导入（推理不需要渲染器）。"""
    import types as _types

    if "pyrender" not in sys.modules:
        pr = _types.ModuleType("pyrender")
        for _name in [
            "Node", "Mesh", "Material", "Scene", "PerspectiveCamera",
            "OffscreenRenderer", "DirectionalLight", "MetallicRoughnessMaterial",
            "Primitive", "SpotLight", "PointLight", "Light", "camera",
        ]:
            setattr(pr, _name, type(_name, (), {}))
        pr.OffscreenRenderer = type("OffscreenRenderer", (), {
            "__init__": lambda s, *a, **kw: None,
            "render": lambda s, *a, **kw: None,
            "delete": lambda s: None,
        })
        pr.Mesh = type("Mesh", (), {
            "from_trimesh": classmethod(lambda c, *a, **kw: None),
        })
        sys.modules["pyrender"] = pr

    if "trimesh" not in sys.modules:
        tm = _types.ModuleType("trimesh")
        tm.Trimesh = type("Trimesh", (), {})
        sys.modules["trimesh"] = tm


__all__ = ["WiLoRBackend"]
