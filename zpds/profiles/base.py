"""BaseProfile — profile 基类。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaseProfile:
    """采集源 profile 基类。"""

    name: str
    description: str = ""
    version: str = "0.1.0"
    adapter_kind: str = ""
    required_globs: tuple[str, ...] = ()
    optional_globs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def discover(self, session_path: str | Path) -> dict[str, tuple[Path, ...]]:
        root = Path(session_path)
        if root.is_file():
            root = root.parent
        return {
            pattern: tuple(sorted(path for path in root.glob(pattern) if path.is_file()))
            for pattern in (*self.required_globs, *self.optional_globs)
        }

    def missing_required(self, session_path: str | Path) -> tuple[str, ...]:
        discovered = self.discover(session_path)
        return tuple(
            pattern for pattern in self.required_globs if not discovered[pattern]
        )
