"""全局配置:模型路径、OCR 语言、LLM 接口与打码参数。

提供两层 API：
1. 模块级变量（向后兼容，现有 backends 使用）
2. ``PrivacyConfig`` dataclass（新流水线使用，YAML 驱动，对标 hands/config.py）
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# 项目根目录 (zpds/privacy/config.py → zpds/ → ZPDS/)
_PKG_DIR = Path(__file__).resolve().parent  # zpds/privacy/
ROOT = _PKG_DIR.parents[1]                   # ZPDS/ (project root)
_MODELS_DIR = _PKG_DIR / "models"


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载:KEY=VALUE,支持 # 注释与引号;已存在的环境变量优先。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


# 加载项目根目录下的 .env(API Key 等敏感配置)
_load_dotenv(ROOT / ".env")

# ============================================================================
# 模块级变量（向后兼容现有 backends）
# ============================================================================

YOLO_MODEL_PATH = Path(os.environ.get(
    "PRIVACY_YOLO_PATH",
    _MODELS_DIR / "yolo.pt",
))

FACE_MODEL_PATH = Path(os.environ.get(
    "PRIVACY_FACE_MODEL",
    _MODELS_DIR / "yolov11n-face.pt",
))

OCR_LANGS = ["ch_sim", "en"]
OCR_TEXT_THRESHOLD = 0.1
OCR_LOW_TEXT = 0.3

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = os.environ.get(
    "PRIVACY_LLM_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL = os.environ.get("PRIVACY_LLM_MODEL", "qwen-plus")

YOLO_CONF = 0.25
OCR_MIN_CONFIDENCE = 0.4
FACE_CONFIDENCE = 0.5
BLUR_STRENGTH = 0.35


# ============================================================================
# PrivacyConfig — YAML 驱动的配置 dataclass（对标 hands/config.py）
# ============================================================================

VALID_FACE_METHODS = frozenset({"blur", "pixelate"})
VALID_TEXT_METHODS = frozenset({"black_rect", "blur", "pixelate"})


def _config_sha256(document: dict[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PrivacyConfig:
    """经过校验、可追溯的 Privacy 运行配置。"""

    path: Path
    document: dict[str, Any]

    # ---- 开关 ----
    enabled: bool = True

    # ---- 人脸 ----
    face_enabled: bool = True
    face_backend: str = "yolo11n_face"
    face_model_path: str = ""
    face_confidence: float = 0.5
    face_method: str = "blur"
    face_blur_ksize: int = 41
    face_blur_sigma: int = 15
    face_pixelate_blocks: int = 10
    face_interval_frames: int = 1
    face_min_area_ratio: float = 0.002

    # ---- 文本检测 ----
    text_enabled: bool = True
    text_backend: str = "easyocr"
    text_ocr_langs: tuple[str, ...] = ("ch_sim", "en")
    text_yolo_model_path: str = ""
    text_confidence: float = 0.3
    text_interval_frames: int = 1

    # ---- PII 分类 ----
    pii_backend: str = "llm"
    pii_llm_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    pii_llm_model: str = "qwen-plus"
    pii_llm_timeout_s: int = 30
    pii_cache_enabled: bool = True

    # ---- 遮挡 ----
    redaction_text_method: str = "black_rect"
    redaction_temporal_smoothing: bool = True
    redaction_smoothing_window: int = 5
    redaction_smoothing_iou: float = 0.3

    # ---- 追溯 ----
    config_hash: str = ""

    @classmethod
    def load(cls, path: str | Path) -> PrivacyConfig:
        """从 YAML 文件加载并校验配置。"""
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Privacy 配置文件不存在: {config_path}")

        with config_path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if not isinstance(loaded, dict):
            raise TypeError(f"配置文件顶层必须是对象: {config_path}")

        document: dict[str, Any] = copy.deepcopy(loaded)
        cfg = document.get("privacy", document)

        face = cfg.get("face", {}) or {}
        text = cfg.get("text", {}) or {}
        pii = cfg.get("pii", {}) or {}
        redact = cfg.get("redaction", {}) or {}
        output = cfg.get("output", {}) or {}

        # 解析 model 路径（相对 → 绝对）
        def _resolve_path(raw: str, field_name: str) -> str:
            if not raw:
                return ""
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = config_path.parent / p
            return str(p.resolve())

        config = cls(
            path=config_path,
            document=document,

            enabled=bool(cfg.get("enabled", True)),

            face_enabled=bool(face.get("enabled", True)),
            face_backend=str(face.get("backend", "yolo11n_face")),
            face_model_path=_resolve_path(
                face.get("model_path", str(_MODELS_DIR / "yolov11n-face.pt")),
                "face.model_path",
            ),
            face_confidence=float(face.get("confidence_threshold", 0.5)),
            face_method=str(face.get("method", "blur")),
            face_blur_ksize=int(face.get("blur_ksize", 41)),
            face_blur_sigma=int(face.get("blur_sigma", 15)),
            face_pixelate_blocks=int(face.get("pixelate_blocks", 10)),
            face_interval_frames=int(face.get("interval_frames", 1)),
            face_min_area_ratio=float(face.get("min_area_ratio", 0.002)),

            text_enabled=bool(text.get("enabled", True)),
            text_backend=str(text.get("backend", "easyocr")),
            text_ocr_langs=tuple(text.get("ocr_langs", ["ch_sim", "en"])),
            text_yolo_model_path=_resolve_path(
                text.get("yolo_text_model_path", ""), "text.yolo_text_model_path",
            ),
            text_confidence=float(text.get("confidence_threshold", 0.3)),
            text_interval_frames=int(text.get("interval_frames", 1)),

            pii_backend=str(pii.get("backend", "llm")),
            pii_llm_url=str(pii.get("llm_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            pii_llm_model=str(pii.get("llm_model", "qwen-plus")),
            pii_llm_timeout_s=int(pii.get("llm_timeout_s", 30)),
            pii_cache_enabled=bool(pii.get("cache_enabled", True)),

            redaction_text_method=str(redact.get("text_method", "black_rect")),
            redaction_temporal_smoothing=bool(
                redact.get("temporal_smoothing", {}).get("enabled", True)
                if isinstance(redact.get("temporal_smoothing"), dict)
                else True
            ),
            redaction_smoothing_window=int(
                redact.get("temporal_smoothing", {}).get("window_frames", 5)
                if isinstance(redact.get("temporal_smoothing"), dict)
                else 5
            ),
            redaction_smoothing_iou=float(
                redact.get("temporal_smoothing", {}).get("iou_threshold", 0.3)
                if isinstance(redact.get("temporal_smoothing"), dict)
                else 0.3
            ),

            config_hash=_config_sha256(document),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.face_confidence < 0 or self.face_confidence > 1:
            raise ValueError("privacy.face.confidence_threshold 必须在 [0, 1] 范围内")
        if self.face_method not in VALID_FACE_METHODS:
            raise ValueError(
                f"privacy.face.method 必须是 {sorted(VALID_FACE_METHODS)}"
            )
        if self.face_interval_frames < 1:
            raise ValueError("privacy.face.interval_frames 必须 >= 1")
        if self.face_min_area_ratio < 0 or self.face_min_area_ratio > 1:
            raise ValueError("privacy.face.min_area_ratio 必须在 [0, 1] 范围内")
        if self.face_enabled and self.face_model_path and not Path(self.face_model_path).exists():
            raise FileNotFoundError(f"人脸模型不存在: {self.face_model_path}")

        if self.text_confidence < 0 or self.text_confidence > 1:
            raise ValueError("privacy.text.confidence_threshold 必须在 [0, 1] 范围内")
        if self.text_interval_frames < 1:
            raise ValueError("privacy.text.interval_frames 必须 >= 1")

        if self.pii_llm_timeout_s < 1:
            raise ValueError("privacy.pii.llm_timeout_s 必须 >= 1")

        if self.redaction_text_method not in VALID_TEXT_METHODS:
            raise ValueError(
                f"privacy.redaction.text_method 必须是 {sorted(VALID_TEXT_METHODS)}"
            )
        if self.redaction_smoothing_iou < 0 or self.redaction_smoothing_iou > 1:
            raise ValueError("privacy.redaction.temporal_smoothing.iou_threshold 必须在 [0, 1] 范围内")

    # ---- 便捷方法 ----

    @property
    def llm_api_key(self) -> str:
        return os.environ.get("DASHSCOPE_API_KEY", "")

    @property
    def effective_yolo_text_path(self) -> Path | None:
        """返回文本 YOLO 模型路径（配置了且存在时）。"""
        if not self.text_yolo_model_path:
            return None
        p = Path(self.text_yolo_model_path)
        return p if p.exists() else None

    @classmethod
    def defaults(cls) -> "PrivacyConfig":
        """返回全默认配置（用于冒烟测试）。"""
        return cls(
            path=Path("."),
            document={},
            face_model_path=str(_MODELS_DIR / "yolov11n-face.pt"),
            face_method="blur",
        )


__all__ = [
    "PrivacyConfig",
    "VALID_FACE_METHODS",
    "VALID_TEXT_METHODS",
]
