from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_sumo_scenario_snapshot_matches_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    snapshot = root / "scenarios" / "highway_merge" / "v1"
    provenance = json.loads(
        (snapshot / "PROVENANCE.json").read_text(encoding="utf-8")
    )
    assert SHA1.fullmatch(provenance["source_commit"])
    assert provenance["contract_version"] == "sumo_merge_core_v1"
    for name, expected in provenance["files"].items():
        assert SHA256.fullmatch(expected)
        assert _sha256(snapshot / name) == expected


def test_upstream_repositories_are_locked_to_full_commits() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "third_party" / "upstream-lock.json").read_text(encoding="utf-8")
    )
    repositories = manifest["repositories"]
    assert len(repositories) == 6
    assert len({entry["name"] for entry in repositories}) == len(repositories)
    for entry in repositories:
        assert entry["url"].startswith("https://github.com/")
        assert SHA1.fullmatch(entry["commit"])
        assert entry["license"] in {"MIT", "UNKNOWN"}
        for submodule in entry["submodules"]:
            assert SHA1.fullmatch(submodule["commit"])


def test_core_dependency_lock_contains_only_exact_unique_versions() -> None:
    root = Path(__file__).resolve().parents[1]
    lock = root / "requirements" / "core-py310-windows.lock.txt"
    entries = [
        line.strip()
        for line in lock.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert entries
    assert all(line.count("==") == 1 for line in entries)
    names = [re.sub(r"[-_.]+", "-", line.split("==", 1)[0]).lower() for line in entries]
    assert len(names) == len(set(names))
