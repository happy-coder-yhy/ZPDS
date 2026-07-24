"""管理五源 Gold Manifest 的资产哈希、单人审核和冻结状态。"""

import argparse
import hashlib
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

from zpds.utils.schema_validator import validate_with_schema

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "gold" / "five_source_manifest.yaml"
CHUNK_SIZE = 1024 * 1024


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML object: {path}")
    return value


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def collect_assets(manifest: dict, data_root: Path) -> tuple[list[str], list[str]]:
    """读取 Raw 文件并更新内存中的 hash/size，不修改 Raw。"""

    data_root = data_root.resolve(strict=True)
    updates: list[str] = []
    errors: list[str] = []
    for sample in manifest.get("samples", []):
        sample_id = sample.get("sample_id", "<unknown>")
        sample_changed = False
        for asset in sample.get("source_assets", []):
            relative_path = asset.get("relative_path")
            if not isinstance(relative_path, str):
                errors.append(f"{sample_id}: asset relative_path is missing")
                continue
            try:
                source_path = _resolve_under_root(data_root, relative_path)
            except ValueError as error:
                errors.append(f"{sample_id}: {error}")
                continue
            if not source_path.is_file():
                errors.append(f"{sample_id}: source asset not found: {relative_path}")
                continue
            sha256, size_bytes = hash_file(source_path)
            if asset.get("sha256") != sha256 or asset.get("size_bytes") != size_bytes:
                asset["sha256"] = sha256
                asset["size_bytes"] = size_bytes
                updates.append(f"{sample_id}:{asset.get('asset_id', relative_path)}")
                sample_changed = True
        if sample_changed and sample.get("review", {}).get("status") != "pending":
            previous_notes = sample["review"].get("notes", "")
            sample["review"] = {
                "status": "pending",
                "reviewers": [],
                "reviewed_at": None,
                "notes": (
                    f"资产内容变化，需要重新审核。此前备注：{previous_notes}"
                    if previous_notes
                    else "资产内容变化，需要重新审核。"
                ),
            }
    return updates, errors


def verify_assets(manifest: dict, data_root: Path) -> list[str]:
    """重新读取 Raw 并确认 Manifest 中记录的 hash/size 没有变化。"""

    candidate = deepcopy(manifest)
    updates, errors = collect_assets(candidate, data_root)
    errors.extend(f"recorded asset hash/size differs: {update}" for update in updates)
    return errors


def review_sample(
    manifest: dict,
    sample_id: str,
    reviewer: str,
    status: str,
    notes: str,
    reviewed_at: str | None = None,
) -> None:
    sample = next(
        (item for item in manifest.get("samples", []) if item.get("sample_id") == sample_id),
        None,
    )
    if sample is None:
        raise ValueError(f"Unknown Gold sample: {sample_id}")
    if status == "pending":
        sample["review"] = {
            "status": "pending",
            "reviewers": [],
            "reviewed_at": None,
            "notes": notes,
        }
        return
    designated_reviewer = manifest.get("designated_reviewer")
    if designated_reviewer and reviewer.strip() != designated_reviewer:
        raise ValueError(
            f"reviewer must match designated reviewer {designated_reviewer!r}"
        )
    if not reviewer.strip():
        raise ValueError("reviewer is required for approved or rejected review")
    timestamp = reviewed_at or datetime.now(timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )
    sample["review"] = {
        "status": status,
        "reviewers": [reviewer.strip()],
        "reviewed_at": timestamp,
        "notes": notes,
    }


def freeze_manifest(manifest: dict) -> list[str]:
    unapproved = [
        sample.get("sample_id", "<unknown>")
        for sample in manifest.get("samples", [])
        if sample.get("review", {}).get("status") != "approved"
    ]
    if unapproved:
        return [f"samples awaiting designated reviewer approval: {unapproved}"]
    candidate = deepcopy(manifest)
    candidate["status"] = "frozen"
    errors = validate_with_schema(candidate, "gold_manifest")
    if not errors:
        manifest["status"] = "frozen"
    return errors


def write_manifest(path: Path, manifest: dict) -> None:
    """在同一目录临时写入并原子替换 Manifest。"""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(
        manifest,
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()


def _resolve_under_root(data_root: Path, relative_path: str) -> Path:
    normalized = Path(relative_path.replace("/", os.sep))
    if normalized.is_absolute() or normalized.drive or ".." in normalized.parts:
        raise ValueError(f"unsafe relative path: {relative_path}")
    candidate = (data_root / normalized).resolve(strict=False)
    if not candidate.is_relative_to(data_root):
        raise ValueError(f"path escapes data root: {relative_path}")
    return candidate


def _print_status(manifest: dict) -> None:
    samples = manifest.get("samples", [])
    counts = {
        status: sum(
            1 for sample in samples if sample.get("review", {}).get("status") == status
        )
        for status in ("pending", "approved", "rejected")
    }
    print(
        f"manifest={manifest.get('status')} "
        f"designated_reviewer={manifest.get('designated_reviewer')} "
        f"samples={len(samples)} "
        f"approved={counts['approved']} pending={counts['pending']} "
        f"rejected={counts['rejected']}"
    )
    for sample in samples:
        review = sample.get("review", {})
        reviewer = ",".join(review.get("reviewers", [])) or "-"
        print(
            f"- {sample.get('sample_id')}: {review.get('status')} "
            f"reviewer={reviewer} assets={len(sample.get('source_assets', []))}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="只读显示审核状态")

    collect = subparsers.add_parser("collect", help="读取 Raw 并核对/收集哈希")
    collect.add_argument("--data-root", type=Path, required=True)
    collect.add_argument("--write", action="store_true", help="确认写回 Manifest")

    review = subparsers.add_parser("review", help="记录单个样例的人工审核")
    review.add_argument("--sample", required=True)
    review.add_argument(
        "--reviewer",
        help="默认使用 Manifest 中的 designated_reviewer",
    )
    review.add_argument(
        "--status",
        required=True,
        choices=("pending", "approved", "rejected"),
    )
    review.add_argument("--notes", default="")
    review.add_argument("--reviewed-at")
    review.add_argument("--write", action="store_true", help="确认写回 Manifest")

    freeze = subparsers.add_parser("freeze", help="全部审核通过后冻结 Manifest")
    freeze.add_argument("--data-root", type=Path, required=True)
    freeze.add_argument("--write", action="store_true", help="确认写回 Manifest")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "status":
        _print_status(manifest)
        return 0
    if args.command == "collect":
        updates, errors = collect_assets(manifest, args.data_root)
        for error in errors:
            print(f"ERROR: {error}")
        print(f"assets_changed={len(updates)}")
        if errors:
            return 1
        if args.write:
            write_manifest(args.manifest, manifest)
            print(f"updated={args.manifest}")
        else:
            print("dry-run: add --write to update the Manifest")
        return 0
    if args.command == "review":
        reviewer = args.reviewer or manifest.get("designated_reviewer", "")
        review_sample(
            manifest,
            sample_id=args.sample,
            reviewer=reviewer,
            status=args.status,
            notes=args.notes,
            reviewed_at=args.reviewed_at,
        )
        errors = validate_with_schema(manifest, "gold_manifest")
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        if args.write:
            write_manifest(args.manifest, manifest)
            print(f"reviewed={args.sample} status={args.status}")
        else:
            print("dry-run: add --write to record the review")
        return 0
    errors = verify_assets(manifest, args.data_root)
    if errors:
        print("Gold Manifest assets cannot be verified:")
        for error in errors:
            print(f"- {error}")
        return 1
    errors = freeze_manifest(manifest)
    if errors:
        print("Gold Manifest cannot be frozen:")
        for error in errors:
            print(f"- {error}")
        return 1
    if args.write:
        write_manifest(args.manifest, manifest)
        print(f"frozen={args.manifest}")
    else:
        print("dry-run: all checks passed; add --write to freeze")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
