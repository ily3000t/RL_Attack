"""Fail when core code acquires a dependency on source or paper repositories."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


FORBIDDEN_IMPORT_ROOTS = {
    "safe_rl",
    "wcdt",
    "WCDT_ACCVP_Attack",
    "third_party",
}
FORBIDDEN_TEXT = {
    "E:\\WCDT_ACCVP_Attack",
    "E:/WCDT_ACCVP_Attack",
}


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "src" / "rl_attack"
    failures: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        imports = imported_roots(path)
        forbidden = sorted(imports & FORBIDDEN_IMPORT_ROOTS)
        if forbidden:
            failures.append(f"{path}: forbidden imports {forbidden}")
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_TEXT:
            if marker in text:
                failures.append(f"{path}: forbidden absolute source path {marker!r}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"isolation: OK ({len(list(source_root.rglob('*.py')))} Python files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
