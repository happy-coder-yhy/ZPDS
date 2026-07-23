"""端到端流水线入口。"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="ZPDS 数据清洗流水线")
    parser.add_argument("--source", required=True, help="数据源路径")
    parser.add_argument("--profile", required=True, help="采集源 profile")
    parser.add_argument("--output", default="./output", help="输出目录")
    parser.add_argument("--stages", nargs="+", type=int, help="要运行的 QC 阶段")
    args = parser.parse_args()

    print(f"ZPDS Pipeline — profile={args.profile}, source={args.source}")
    # TODO: 实现完整流水线
    raise NotImplementedError("流水线尚未实现")


if __name__ == "__main__":
    main()
