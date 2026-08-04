"""全局配置:模型路径、OCR 语言、LLM 接口与打码参数。

所有路径/参数均可通过环境变量覆盖,也可直接修改本文件。
"""
from __future__ import annotations

import os
from pathlib import Path

# 项目根目录 (zpds/privacy/config.py → zpds/ → ZPDS/)
_PKG_DIR = Path(__file__).resolve().parent  # zpds/privacy/
ROOT = _PKG_DIR.parents[1]                   # ZPDS/ (project root)


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

# ---- 模型路径(默认指向包内 models/ 目录,可用环境变量覆盖) ----
_MODELS_DIR = _PKG_DIR / "models"

YOLO_MODEL_PATH = Path(os.environ.get(
    "PRIVACY_YOLO_PATH",
    _MODELS_DIR / "yolo.pt",
))

# ---- 人脸检测模型(YOLOv11n-face,来自 akanametov/yolo-face) ----
FACE_MODEL_PATH = Path(os.environ.get(
    "PRIVACY_FACE_MODEL",
    _MODELS_DIR / "yolov11n-face.pt",
))

# ---- OCR:中英混合 ----
OCR_LANGS = ["ch_sim", "en"]
OCR_TEXT_THRESHOLD = 0.1
OCR_LOW_TEXT = 0.3

# ---- LLM:OpenAI 兼容接口,默认阿里云 DashScope ----
# 用 API 方式部署,无需本地显存。key 优先从环境变量读取。
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = os.environ.get(
    "PRIVACY_LLM_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL = os.environ.get("PRIVACY_LLM_MODEL", "qwen-plus")

# ---- 检测 / 打码参数 ----
YOLO_CONF = 0.25            # YOLO 文本检测置信度下限
OCR_MIN_CONFIDENCE = 0.4    # OCR 识别置信度下限(低于此值的文本丢弃)
FACE_CONFIDENCE = 0.5       # 人脸检测置信度下限
BLUR_STRENGTH = 0.35        # 高斯模糊核大小 = 区域短边 * 此比例(强制奇数)
