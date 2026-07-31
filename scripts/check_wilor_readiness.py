"""在创建真实 WiLoR 后端前校验本地模型资产。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zpds.hands.config import HandsPipelineConfig
from zpds.hands.wilor_preflight import check_wilor_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="包含 hands.wilor 配置的 YAML 文件",
    )
    parser.add_argument(
        "--json-output",
        help="可选的预检 JSON 报告输出路径",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    runtime_config = HandsPipelineConfig.load(args.config)
    report = check_wilor_assets(runtime_config.wilor)
    document = report.to_dict()

    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)

    print(f"WiLoR assets ready: {report.ready}")
    print(f"Model revision: {report.model_revision}")
    for asset in report.assets:
        status = "OK" if asset.ok else "FAIL"
        print(f"[{status}] {asset.name}: {asset.path}")
        if asset.error:
            print(f"  {asset.error}")
    for error in report.errors:
        if error not in {asset.error for asset in report.assets}:
            print(f"[FAIL] {error}")
    return 0 if report.ready else 2


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
