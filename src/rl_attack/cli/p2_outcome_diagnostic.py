from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rl_attack.experiments.p2_outcome_diagnostic import (
    plan_outcome_diagnostic,
    run_outcome_diagnostic,
    verify_outcome_diagnostic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, run, or verify the post-hoc P2 outcome diagnostic gate"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("config", type=Path)
    plan.add_argument("--device", default="cpu")
    run = commands.add_parser("run")
    run.add_argument("config", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="cpu")
    verify = commands.add_parser("verify")
    verify.add_argument("output_dir", type=Path)
    verify.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        result = plan_outcome_diagnostic(args.config, device=args.device)
    elif args.command == "run":
        result = run_outcome_diagnostic(
            args.config, output_directory=args.output_dir, device=args.device
        )
    else:
        result = verify_outcome_diagnostic(args.output_dir, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
