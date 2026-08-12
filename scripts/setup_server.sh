#!/usr/bin/env bash
# 服务器环境一次性配置（对应问题 20「部署版本漂移同步」）：
#   1. 按 requirements-server.txt 安装/对齐依赖
#   2. 在 venv 的 site-packages 创建 sitecustomize.py（恢复 numpy 旧别名，chumpy 必需）
#   3. 校验关键版本：requirements-server.txt 全部锁定 == 逐项比对 + numpy<2 红线
#
# 用法（激活 venv 后）：
#   source .venv/bin/activate
#   bash scripts/setup_server.sh

set -euo pipefail

PY="${PYTHON:-python}"

# 1. 安装/对齐依赖（幂等：已满足的约束 pip 不重复下载）
echo "==> 安装依赖: requirements-server.txt"
"$PY" -m pip install -r requirements-server.txt

"$PY" - <<'PY'
import site
from pathlib import Path

code = """import numpy as np
# Restore deprecated numpy type aliases that chumpy needs
for _name, _target in [
    ('bool', np.bool_),
    ('int', np.int_),
    ('float', np.float64),
    ('complex', np.complex128),
    ('object', np.object_),
    ('unicode', np.str_),
    ('str', np.str_),
]:
    if not hasattr(np, _name):
        setattr(np, _name, _target)
"""

try:
    import sysconfig
    site_packages = sysconfig.get_paths()["purelib"]
    if not Path(site_packages).is_dir():
        raise FileNotFoundError(site_packages)
except (AttributeError, KeyError, FileNotFoundError):
    # 老版本 Python：退回 site.getsitepackages()（注意 [0] 可能是 venv 根，
    # 只取包含 site-packages 的条目）
    try:
        site_packages = next(
            p for p in site.getsitepackages()
            if p.endswith("site-packages")
        )
    except (StopIteration, AttributeError):
        import sysconfig
        site_packages = sysconfig.get_paths()["purelib"]

target = Path(site_packages) / "sitecustomize.py"
target.write_text(code, encoding="utf-8")
print(f"sitecustomize.py created: {target}")
PY

# 验证导入
"$PY" -c "import sitecustomize; import numpy as np; assert hasattr(np, 'bool'); print('chumpy numpy-alias patch OK')"

# 2b. chumpy ch.py patch（getargspec → getfullargspec，chumpy 0.70 兼容 Python 3.11）
# 幂等：已含 getfullargspec 时跳过；与本地 .venv 的 chumpy/ch.py 实际改动一致
"$PY" - <<'PY'
from pathlib import Path
import chumpy

ch_file = Path(chumpy.__file__).parent / "ch.py"
if "getargspec" not in ch_file.read_text(encoding="utf-8"):
    print("ch.py 已为 getfullargspec 版本，跳过 patch")
else:
    text = ch_file.read_text(encoding="utf-8").replace(
        "inspect.getargspec", "inspect.getfullargspec"
    )
    ch_file.write_text(text, encoding="utf-8")
    print(f"chumpy ch.py patched: {ch_file}")
PY

# 3. 依赖版本校验（部署漂移检测：环境必须与 requirements-server.txt 锁定一致）
"$PY" - <<'PY'
import importlib.metadata as md
import re
import sys
from pathlib import Path

def fail(msg: str):
    print(f"校验失败: {msg}", file=sys.stderr)
    raise SystemExit(1)

# 3a. requirements-server.txt 全部锁定 == 逐项比对
req_file = Path("requirements-server.txt")
constraints = []
for line in req_file.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    m = re.match(r"([a-zA-Z0-9_.-]+)\s*(==|<|<=|>|>=)\s*(\S+)", line)
    if m:
        constraints.append((m.group(1).lower().replace("_", "-"), m.group(2), m.group(3)))

def norm(v: str) -> tuple:
    """提取 major.minor.patch 三段用于比对。

    忽略 post/local 后缀差异（同包两种表示法：pip freeze 的
    ``0.1.1.post2209072238`` vs importlib.metadata 的 ``0.1.1-2209072238``）。
    """
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])

problems = []
for name, op, ver in constraints:
    if name == "chumpy":
        continue  # git 源行，跳过 == 比对（安装来源已保证）
    try:
        installed = md.version(name)
    except md.PackageNotFoundError:
        problems.append(f"{name}: 未安装（要求 {op}{ver}）")
        continue
    iv = norm(installed)
    if op == "==" and norm(installed) != norm(ver):
        problems.append(f"{name}: 已装 {installed}，要求 =={ver}")
    elif op == "<" and iv >= norm(ver):
        problems.append(f"{name}: 已装 {installed}，要求 <{ver}")

# 3b. 红线：numpy < 2（mediapipe/chumpy 兼容，独立于 freeze 快照检查）
try:
    numpy_ver = md.version("numpy")
except md.PackageNotFoundError:
    problems.append("numpy: 未安装（红线：<2）")
else:
    if tuple(int(x) for x in re.split(r"[.+]", numpy_ver)[:2]) >= (2, 0):
        problems.append(f"numpy: 已装 {numpy_ver}，红线要求 <2（mediapipe/chumpy 兼容）")

# 3c. opencv 多包共存隐患提示（contrib/python/headless 共享 cv2，互相覆盖）
opencv_pkgs = [c[0] for c in constraints if c[0].startswith("opencv")]
if len(opencv_pkgs) > 1:
    print(f"提示: 环境同时锁定 {len(opencv_pkgs)} 个 opencv 包 "
          f"({', '.join(opencv_pkgs)})——共享 cv2 文件会互相覆盖，"
          "如遇 cv2 行为异常请只保留 opencv-python-headless")

if problems:
    print("依赖漂移，共 " + str(len(problems)) + " 项:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    print("请修正后再运行（对齐 requirements-server.txt 或重新 pip install）", file=sys.stderr)
    raise SystemExit(1)

print(f"依赖校验通过：{len(constraints)} 个锁定约束全部满足，numpy<2 红线 OK")
PY
