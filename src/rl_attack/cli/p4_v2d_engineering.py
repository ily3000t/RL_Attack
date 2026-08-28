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
_PRELOADED = tuple(
    name for name in ("numpy", "torch", "gymnasium", "stable_baselines3") if name in sys.modules
)
os.environ["RL_ATTACK_P4_V2B_PREIMPORT_THREADS"] = "1" if not _PRELOADED else "0"
os.environ["RL_ATTACK_P4_V2B_PRELOADED_MODULES"] = ",".join(_PRELOADED)

from rl_attack.experiments.p4_v2d_engineering import (  # noqa: E402
    run_p4_v2d_engineering,
    verify_p4_v2d_engineering,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or verify P4-v2d engineering")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("config", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("config", type=Path)
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_p4_v2d_engineering(args.config, args.output_dir)
    else:
        result = verify_p4_v2d_engineering(
            args.config,
            args.run,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
