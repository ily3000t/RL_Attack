"""Command-line entry point for the frozen-row P5 audit gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rl_attack.experiments.p5_audit import InvalidP5Audit, run_p5_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate a hash-pinned P5 RAPID-Guard evaluation matrix; "
            "existing outputs are never overwritten"
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "p5_rapid_guard_audit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        manifest = run_p5_audit(
            args.config,
            output_directory=args.output_dir,
        )
    except InvalidP5Audit as error:
        manifest = error.manifest or {
            "status": "invalid",
            "robust_summary_eligible": False,
            "invalid_reason": {
                "code": error.code,
                "message": str(error),
            },
        }
        print(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
    print(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
