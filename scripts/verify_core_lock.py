"""Verify that the active environment matches the resolved core lock."""

from __future__ import annotations

import importlib.metadata
import re
import sys
from pathlib import Path


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"lock entry is not exact: {line}")
        name, version = line.split("==", 1)
        expected[canonical(name)] = version
    return expected


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    expected = read_lock(
        repo_root / "requirements" / "core-py310-windows.lock.txt"
    )
    installed = {
        canonical(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    failures = [
        f"{name}: expected {version}, got {installed.get(name, 'MISSING')}"
        for name, version in sorted(expected.items())
        if installed.get(name) != version
    ]
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"core lock: OK ({len(expected)} distributions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
