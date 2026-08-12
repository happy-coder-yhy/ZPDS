"""部署期模型预下载与校验。

目标：任务运行期零下载。服务器全新环境部署后先运行本脚本，将
EasyOCR 模型（craft 检测器 + 识别器）预下载到 ``~/.EasyOCR/model/``；
face / 文本提议 YOLO 检测模型已随仓库（``zpds/privacy/models/``），
本脚本仅校验存在性。

运行期 ``get_ocr_reader()`` 使用 ``download_enabled=False``：模型缺失时
直接报错（提示先运行本脚本），绝不静默联网下载——离线/受限服务器
任务运行期出现 Downloading 输出即为部署缺陷。

用法：
    python scripts/download_models.py              # 默认 ch_sim + en
    python scripts/download_models.py --force       # 强制重新下载
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# EasyOCR 缓存目录按 HOME/USERPROFILE 解析（与运行期一致）
def _easyocr_cache_dir() -> Path:
    return Path.home() / ".EasyOCR" / "model"

# 关键模型文件 -> 用途（名称按 EasyOCR 实际落盘名）
_EXPECTED_EASYOCR = [
    ("craft_mlt_25k.pth", "文本检测器"),
    ("zh_sim_g2.pth", "中文识别器 (ch_sim)"),
    ("english_g2.pth", "英文识别器 (en)"),
]

# 随仓库分发的检测模型
_REPO_MODELS = [
    ("zpds/privacy/models/yolov11n-face.pt", "人脸检测"),
    ("zpds/privacy/models/yolo.pt", "文本区域提议"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--langs", nargs="+", default=["ch_sim", "en"],
        help="EasyOCR 识别语言（默认: ch_sim en）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="即使模型已存在也强制重新下载",
    )
    args = parser.parse_args()

    cache = _easyocr_cache_dir()
    print(f"EasyOCR 模型缓存目录: {cache}")
    cache.mkdir(parents=True, exist_ok=True)

    # 1. 触发 EasyOCR 下载（download_enabled=True：缺失时自动下载）
    try:
        import easyocr
    except ImportError:
        print("错误: 未安装 easyocr（requirements.txt / requirements-server.txt）", file=sys.stderr)
        return 1
    print(f"预下载 EasyOCR 模型: {args.langs} ...")
    reader = easyocr.Reader(args.langs, download_enabled=True)
    del reader  # 释放资源（easyocr.Reader 无 close()）
    print("EasyOCR 模型加载完成")

    # 2. 校验 EasyOCR 关键文件
    missing = []
    for name, desc in _EXPECTED_EASYOCR:
        p = cache / name
        if p.exists() and p.stat().st_size > 0:
            print(f"  ✓ {name} ({desc}, {p.stat().st_size / 1e6:.1f} MB)")
        else:
            missing.append(name)
            print(f"  ✗ {name} ({desc}) 缺失")

    # 3. 校验随仓库分发的模型
    for rel, desc in _REPO_MODELS:
        p = ROOT / rel
        if p.exists() and p.stat().st_size > 0:
            print(f"  ✓ {rel} ({desc}, {p.stat().st_size / 1e6:.1f} MB)")
        else:
            missing.append(rel)
            print(f"  ✗ {rel} ({desc}) 缺失——仓库未包含该文件")

    if missing:
        print("\n校验未通过，缺失: " + ", ".join(missing), file=sys.stderr)
        return 1
    print("\n全部模型就绪，任务运行期将零下载。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
