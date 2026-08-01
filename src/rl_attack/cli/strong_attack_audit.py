from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rl_attack.experiments.p3_audit import InvalidAttackEvaluation, run_p3_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paired P3 reproduced strong-attack audit matrix"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "p3_reproduced_attacks",
    )
    parser.add_argument("--device", default="cpu")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume from complete, fingerprint-matched victim × attack × "
            "epsilon-ratio shards"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        manifest = run_p3_audit(
            args.config,
            output_directory=args.output_dir,
            device=args.device,
            overwrite=args.overwrite,
            resume=args.resume,
        )
    except InvalidAttackEvaluation as error:
        invalid_manifest = args.output_dir / "manifest.json"
        print(
            json.dumps(
                {
                    "status": "invalid",
                    "reason": str(error),
                    "manifest": str(invalid_manifest.resolve()),
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
