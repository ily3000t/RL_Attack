from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rl_attack.experiments.p4_effect_screening import (
    analyze_p4_effect_audit,
    prepare_p4_effect_screening,
    verify_p4_effect_screening,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, verify, or analyze the MergeLite9 P4 effect screen"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="train artifacts and emit frozen validation/final official P4 YAMLs",
    )
    prepare.add_argument("protocol", type=Path)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help=(
            "development-only escape hatch; source dirty state remains recorded "
            "and is not formal evidence"
        ),
    )

    verify = commands.add_parser(
        "verify",
        help="re-hash and official-loader-validate one complete preparation bundle",
    )
    verify.add_argument("preparation", type=Path)
    verify.add_argument("--device", default="cpu")

    analyze = commands.add_parser(
        "analyze",
        help="compute the fixed-bootstrap P4 effect gate from an official audit bundle",
    )
    analyze.add_argument("preparation", type=Path)
    analyze.add_argument("audit_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_p4_effect_screening(
            args.protocol,
            output_directory=args.output_dir,
            require_clean_source=not args.allow_dirty_source,
        )
    elif args.command == "verify":
        result = verify_p4_effect_screening(args.preparation, device=args.device)
    else:
        result = analyze_p4_effect_audit(
            args.preparation,
            args.audit_dir,
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
