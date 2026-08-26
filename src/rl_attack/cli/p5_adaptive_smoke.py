"""CLI for the P5 adaptive-attack engineering smoke."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

from rl_attack.experiments.p5_adaptive_smoke import (  # noqa: E402
    run_adaptive_smoke,
    verify_adaptive_smoke,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the test-scoped P5 adaptive-attack engineering smoke; "
            "this command never emits a defense-effectiveness claim"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute one fresh smoke bundle")
    run.add_argument("config", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify an immutable smoke bundle")
    verify.add_argument("run", type=Path)
    verify.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_adaptive_smoke(args.config, args.output_dir)
    else:
        result = verify_adaptive_smoke(
            args.run,
            expected_manifest_sha256=args.expected_manifest_sha256,
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
