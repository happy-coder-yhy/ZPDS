"""不反序列化内容的 Pickle opcode 静态检查。"""

import pickletools
from dataclasses import dataclass
from pathlib import Path

from zpds.adapters.common import require_file, sha256_file


@dataclass(frozen=True)
class PickleInspection:
    path: Path
    size_bytes: int
    sha256: str
    protocol: int | None
    opcode_count: int
    global_references: tuple[str, ...]
    unsafe_opcodes: tuple[str, ...]

    @property
    def requires_isolated_review(self) -> bool:
        return bool(self.global_references or self.unsafe_opcodes)


UNSAFE_OPCODES = {
    "EXT1",
    "EXT2",
    "EXT4",
    "GLOBAL",
    "INST",
    "NEWOBJ",
    "NEWOBJ_EX",
    "OBJ",
    "REDUCE",
    "STACK_GLOBAL",
}


def inspect_pickle(path: str | Path) -> PickleInspection:
    source = require_file(path)
    protocol: int | None = None
    opcode_count = 0
    globals_found: list[str] = []
    unsafe: list[str] = []
    with source.open("rb") as file:
        for opcode, argument, _ in pickletools.genops(file):
            opcode_count += 1
            if opcode.name == "PROTO" and isinstance(argument, int):
                protocol = int(argument)
            if opcode.name == "GLOBAL":
                globals_found.append(str(argument).replace("\n", "."))
            if opcode.name in UNSAFE_OPCODES:
                unsafe.append(opcode.name)
    return PickleInspection(
        path=source,
        size_bytes=source.stat().st_size,
        sha256=sha256_file(source),
        protocol=protocol,
        opcode_count=opcode_count,
        global_references=tuple(sorted(set(globals_found))),
        unsafe_opcodes=tuple(sorted(set(unsafe))),
    )
