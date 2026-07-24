"""受限子进程中的 primitive-only Pickle 摘要。"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from zpds.adapters.common import require_file

_CHILD_CODE = r"""
import builtins
import json
import pickle
import socket
import sys

socket.socket = lambda *args, **kwargs: (_ for _ in ()).throw(
    RuntimeError("network disabled")
)

ALLOWED = {
    ("builtins", "bool"),
    ("builtins", "bytes"),
    ("builtins", "dict"),
    ("builtins", "float"),
    ("builtins", "frozenset"),
    ("builtins", "int"),
    ("builtins", "list"),
    ("builtins", "set"),
    ("builtins", "str"),
    ("builtins", "tuple"),
}

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"global blocked: {module}.{name}")
        return getattr(builtins, name)

with open(sys.argv[1], "rb") as stream:
    value = RestrictedUnpickler(stream).load()

def summarize(item, depth=0):
    if depth >= 4:
        return {"type": type(item).__name__, "truncated": True}
    if isinstance(item, dict):
        return {
            "type": "dict",
            "length": len(item),
            "keys": [repr(key)[:120] for key in list(item)[:20]],
        }
    if isinstance(item, (list, tuple, set, frozenset)):
        return {"type": type(item).__name__, "length": len(item)}
    if isinstance(item, (str, bytes)):
        return {"type": type(item).__name__, "length": len(item)}
    return {"type": type(item).__name__, "value": item}

print(json.dumps(summarize(value), ensure_ascii=False, sort_keys=True))
"""


def summarize_primitive_pickle(
    path: str | Path,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """只允许 builtins 容器/标量；未知全局对象直接拒绝。"""

    source = require_file(path)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    with tempfile.TemporaryDirectory(prefix="zpds-pickle-") as working_directory:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _CHILD_CODE, str(source)],
            cwd=working_directory,
            env={},
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    if completed.returncode != 0:
        error = completed.stderr.strip().splitlines()
        summary = error[-1] if error else "isolated pickle process failed"
        raise ValueError(summary)
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise TypeError("isolated pickle summary is not an object")
    return value
