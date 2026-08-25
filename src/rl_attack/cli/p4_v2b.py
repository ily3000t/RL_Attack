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

from rl_attack.experiments.p4_v2b import (  # noqa: E402
    prepare_p4_v2b,
    verify_p4_v2b_preparation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or verify the frozen-victim MergeLite9 P4-v2b bundle"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare",
        help="collect trajectory-risk labels and train byte-pinned B2/B3 artifacts",
    )
    prepare.add_argument("protocol", type=Path)
    prepare.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser(
        "verify",
        help="reload every artifact and reconstruct the bound v2b runtime",
    )
    verify.add_argument("preparation", type=Path)
    verify.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_p4_v2b(args.protocol, output_directory=args.output_dir)
    else:
        result = verify_p4_v2b_preparation(
            args.preparation,
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
