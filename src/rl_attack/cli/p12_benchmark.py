from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rl_attack.experiments.p12_benchmark import (
    plan_benchmark,
    run_benchmark,
    verify_benchmark_output,
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, run, resume, or verify the frozen P1/P2 paired benchmark"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="validate inputs and print the frozen plan")
    plan.add_argument("config", type=Path)
    plan.add_argument("--device", default="cpu")

    run = commands.add_parser("run", help="execute the complete frozen matrix")
    run.add_argument("config", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--workers",
        type=_positive_integer,
        default=1,
        help=(
            "spawn this many victim evaluators "
            "(non-formal smoke/validation, CPU/default dependencies only)"
        ),
    )
    run.add_argument(
        "--worker-torch-threads",
        type=_positive_integer,
        default=1,
        help="Torch intra-op threads allocated to each spawned evaluator",
    )
    run.add_argument(
        "--max-new-shards",
        type=_positive_integer,
        help="pause after writing at most this many new complete shards",
    )

    verify = commands.add_parser("verify", help="verify an existing complete bundle")
    verify.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        result = plan_benchmark(args.config, device=args.device)
    elif args.command == "run":
        result = run_benchmark(
            args.config,
            output_directory=args.output_dir,
            device=args.device,
            resume=args.resume,
            max_new_shards=args.max_new_shards,
            workers=args.workers,
            worker_torch_threads=args.worker_torch_threads,
        )
    else:
        result = verify_benchmark_output(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
