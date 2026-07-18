from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_core_has_no_wcdt_or_upstream_imports():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_isolation.py")],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
