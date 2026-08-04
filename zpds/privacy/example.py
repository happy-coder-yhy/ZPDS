"""命令行示例:python -m privacy_protection.example <图片或目录> [输出路径] [选项]

示例:
    python -m privacy_protection.example 图片.jpg
    python -m privacy_protection.example 图片.jpg 脱敏后.jpg
    python -m privacy_protection.example 图片.jpg --no-faces     # 只做文本脱敏
    python -m privacy_protection.example 图片目录 --no-text      # 只做人脸模糊
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .redactor import redact_file, redact_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="图像隐私脱敏:人脸模糊 + 私密文本模糊")
    parser.add_argument("input", help="输入图片或目录路径")
    parser.add_argument("output", nargs="?", default=None, help="输出路径(仅单图时有效)")
    parser.add_argument("--no-faces", action="store_true", help="跳过人脸模糊")
    parser.add_argument("--no-text", action="store_true", help="跳过私密文本模糊")
    args = parser.parse_args()

    path = Path(args.input)
    kwargs = {"redact_faces": not args.no_faces, "redact_text": not args.no_text}

    if path.is_dir():
        saved = redact_directory(path, **kwargs)
        print(f"已处理 {len(saved)} 张图片:")
        for p in saved:
            print(f"  {p}")
    elif path.is_file():
        out = redact_file(path, args.output, **kwargs)
        print(f"脱敏完成: {out}")
    else:
        print(f"路径不存在: {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
