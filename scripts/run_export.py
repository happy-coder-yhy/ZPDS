"""训练格式导出脚本。"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="ZPDS 训练格式导出")
    parser.add_argument("--release", required=True, help="Release ID")
    parser.add_argument("--format", choices=["lerobot", "rlds"], default="lerobot")
    parser.add_argument("--output", default="./exports", help="输出目录")
    args = parser.parse_args()

    print(f"Export — release={args.release}, format={args.format}")
    raise NotImplementedError("导出功能尚未实现")


if __name__ == "__main__":
    main()
