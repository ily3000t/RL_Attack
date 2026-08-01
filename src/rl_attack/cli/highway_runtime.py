"""Freeze and verify the audited Highway runtime manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rl_attack.core.artifacts import sha256_file
from rl_attack.envs.highway_manifest import (
    build_highway_runtime_manifest,
    find_git_repository_root,
    verify_highway_runtime_manifest,
    write_highway_runtime_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze or reproduce the exact highway-fast-v0 runtime used by "
            "P2/P4 SB3 checkpoints"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="probe and publish a manifest")
    freeze.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "runtime" / "highway_fast_v0.json",
    )
    freeze.add_argument("--repository-root", type=Path, default=Path.cwd())
    freeze.add_argument(
        "--dependency-lock",
        type=Path,
        default=Path("requirements") / "highway-runtime-py310-windows.lock.txt",
    )
    freeze.add_argument("--seed", type=int, default=40000)
    freeze.add_argument("--max-episode-steps", type=int, default=30)
    freeze.add_argument("--allow-dirty", action="store_true")
    freeze.add_argument("--overwrite", action="store_true")

    verify = subparsers.add_parser(
        "verify",
        help="reproduce and compare a frozen manifest",
    )
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--repository-root", type=Path, default=Path.cwd())
    verify.add_argument("--expected-file-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    repository_root = find_git_repository_root(args.repository_root)
    if args.command == "freeze":
        lock = args.dependency_lock
        if not lock.is_absolute():
            lock = repository_root / lock
        manifest = build_highway_runtime_manifest(
            repository_root=repository_root,
            dependency_lock=lock,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            allow_dirty=args.allow_dirty,
        )
        output = args.output
        if not output.is_absolute():
            output = repository_root / output
        published = write_highway_runtime_manifest(
            output,
            manifest,
            overwrite=args.overwrite,
        )
        result = {
            "status": "frozen",
            "manifest_path": str(published),
            "manifest_file_sha256": sha256_file(published),
            "payload_sha256": manifest["payload_sha256"],
            "formal_eligible": manifest["payload"]["formal_eligible"],
        }
    else:
        manifest_path = args.manifest
        if not manifest_path.is_absolute():
            manifest_path = repository_root / manifest_path
        result = verify_highway_runtime_manifest(
            manifest_path,
            repository_root=repository_root,
            expected_file_sha256=args.expected_file_sha256,
        )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
