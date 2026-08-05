"""在 Windows 上以独立、可写的临时目录运行 pytest。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import uuid


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / f".pytest-tmp-{os.getpid()}-{uuid.uuid4().hex}"
    run_root.mkdir(parents=False, exist_ok=False)
    system_temp = run_root / "system-temp"
    pytest_base = run_root / "pytest-base"
    cache_dir = run_root / "cache"
    yolo_config = run_root / "ultralytics"
    for path in (system_temp, pytest_base, cache_dir, yolo_config):
        path.mkdir()

    # 在启动 pytest 前验证目录确实可写，失败时立即给出准确原因。
    write_probe = system_temp / ".write-probe"
    write_probe.write_text("ok", encoding="ascii")
    write_probe.unlink()

    env = os.environ.copy()
    env.update(
        {
            "TEMP": str(system_temp),
            "TMP": str(system_temp),
            "TMPDIR": str(system_temp),
            "YOLO_CONFIG_DIR": str(yolo_config),
        }
    )

    command = [
        sys.executable,
        "-m",
        "pytest",
        *sys.argv[1:],
        "--basetemp",
        str(pytest_base),
        "-o",
        f"cache_dir={cache_dir}",
    ]
    completed = subprocess.run(command, cwd=repo_root, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
