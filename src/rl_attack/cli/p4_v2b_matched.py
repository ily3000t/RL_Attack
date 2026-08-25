from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _name in _THREAD_ENVIRONMENT:
    os.environ[_name] = "1"
_PRELOADED_SCIENTIFIC_MODULES = tuple(
    name for name in ("numpy", "torch", "gymnasium", "stable_baselines3") if name in sys.modules
)
os.environ["RL_ATTACK_P4_V2B_PREIMPORT_THREADS"] = (
    "1" if not _PRELOADED_SCIENTIFIC_MODULES else "0"
)
os.environ["RL_ATTACK_P4_V2B_PRELOADED_MODULES"] = ",".join(
    _PRELOADED_SCIENTIFIC_MODULES
)

from rl_attack.experiments.p4_v2b_matched import (  # noqa: E402
    run_p4_v2b_stage,
    verify_p4_v2b_stage_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one frozen P4-v2b development or matched B5 stage"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute one exact preregistered B5 stage")
    run.add_argument("preparation", type=Path)
    run.add_argument("--expected-manifest-sha256", required=True)
    run.add_argument(
        "--stage",
        required=True,
        choices=("development_validation", "matched_baseline"),
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--development-result", type=Path)
    run.add_argument("--expected-development-manifest-sha256")
    verify = commands.add_parser(
        "verify", help="independently recompute one production B5 result"
    )
    verify.add_argument("preparation", type=Path)
    verify.add_argument("--expected-manifest-sha256", required=True)
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--expected-run-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_p4_v2b_stage(
            args.preparation,
            expected_preparation_manifest_sha256=args.expected_manifest_sha256,
            stage=args.stage,
            output_directory=args.output_dir,
            development_result=args.development_result,
            expected_development_manifest_sha256=(
                args.expected_development_manifest_sha256
            ),
        )
    else:
        result = verify_p4_v2b_stage_run(
            args.preparation,
            expected_preparation_manifest_sha256=args.expected_manifest_sha256,
            run=args.run,
            expected_run_manifest_sha256=args.expected_run_manifest_sha256,
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
