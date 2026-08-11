from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rl_attack.experiments.p4_audit import InvalidP4Audit, run_p4_audit


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paired, hard-K P4 STFA audit gate"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "p4_stfa_audit",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--torch-threads",
        type=_positive_int,
        default=None,
        help="fix Torch intra-op threads to this count and inter-op threads to 1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        manifest = run_p4_audit(
            args.config,
            output_directory=args.output_dir,
            device=args.device,
            torch_threads=args.torch_threads,
            overwrite=args.overwrite,
        )
    except InvalidP4Audit as error:
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
