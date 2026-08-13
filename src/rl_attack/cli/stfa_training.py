"""Train pinned P4 STFA critic/director artifacts from fixed NPZ datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from rl_attack.training.stfa_director import (
    STFADirectorConfig,
    STFADirectorTrainConfig,
)
from rl_attack.training.stfa_pipeline import (
    load_critic_dataset,
    load_director_dataset,
    train_critic_from_npz,
    train_director_from_npz,
)
from rl_attack.training.stfa_safety_critic import STFASafetyCriticConfig


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--victim-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-victim-checkpoint-sha256",
        "--expected-victim-sha256",
        dest="expected_victim_checkpoint_sha256",
        required=True,
    )
    parser.add_argument("--dataset", "--input-npz", dest="dataset", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument(
        "--expected-dataset-manifest-sha256",
        "--expected-dataset-sidecar-sha256",
        dest="expected_dataset_manifest_sha256",
        required=True,
    )
    parser.add_argument("--expected-action-ontology-sha256", required=True)
    parser.add_argument(
        "--victim-action-mode",
        choices=("stochastic", "deterministic"),
        default="stochastic",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p4_training"))
    parser.add_argument("--run-name")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=(128, 128))
    parser.add_argument("--activation", choices=("relu", "tanh"), default="relu")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train auditable STFA artifacts from immutable NPZ inputs; "
            "this command does not collect rollouts or produce formal statistics"
        )
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    critic = subparsers.add_parser(
        "critic",
        help="train the clean-observation action-wise safety cost critic",
    )
    _add_common(critic)
    critic.add_argument("--gamma", type=float, default=0.99)
    critic.add_argument("--learning-rate", type=float, default=3.0e-4)
    critic.add_argument("--gradient-steps", type=int, default=200)
    critic.add_argument("--batch-size", type=int, default=64)
    critic.add_argument("--target-update-interval", type=int, default=10)
    critic.add_argument("--target-tau", type=float, default=0.05)
    critic.add_argument("--max-gradient-norm", type=float, default=10.0)

    director = subparsers.add_parser(
        "director",
        help="train the temporal selector and legal factor-target heads",
    )
    _add_common(director)
    director.add_argument("--critic-checkpoint", type=Path, required=True)
    director.add_argument(
        "--expected-critic-checkpoint-sha256",
        "--expected-critic-sha256",
        dest="expected_critic_checkpoint_sha256",
        required=True,
    )
    director.add_argument("--selection-threshold", type=float, default=0.5)
    director.add_argument("--stochastic-inference", action="store_true")
    director.add_argument("--gradient-steps", type=int, default=200)
    director.add_argument("--learning-rate", type=float, default=3.0e-4)
    director.add_argument("--selection-coefficient", type=float, default=1.0)
    director.add_argument("--lateral-coefficient", type=float, default=1.0)
    director.add_argument("--longitudinal-coefficient", type=float, default=1.0)
    director.add_argument("--max-gradient-norm", type=float, default=10.0)
    return parser


def _effective_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # Resolve eagerly so an invalid value fails before creating an output run.
    torch.device(requested)
    return requested


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    hidden_sizes = tuple(args.hidden_sizes)
    if not hidden_sizes or any(value <= 0 for value in hidden_sizes):
        raise ValueError("--hidden-sizes must contain positive integers")
    device = _effective_device(args.device)

    if args.stage == "critic":
        dataset = load_critic_dataset(
            args.dataset,
            expected_sha256=args.expected_dataset_sha256,
            expected_manifest_sha256=args.expected_dataset_manifest_sha256,
            expected_action_ontology_sha256=args.expected_action_ontology_sha256,
        )
        config = STFASafetyCriticConfig(
            observation_shape=tuple(dataset.transitions.observations.shape[1:]),
            n_actions=dataset.factorization.n_actions,
            hidden_sizes=hidden_sizes,
            activation=args.activation,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            gradient_steps=args.gradient_steps,
            batch_size=args.batch_size,
            target_update_interval=args.target_update_interval,
            target_tau=args.target_tau,
            max_gradient_norm=args.max_gradient_norm,
            seed=args.seed,
            device=device,
        )
        return train_critic_from_npz(
            victim_checkpoint=args.victim_checkpoint,
            expected_victim_checkpoint_sha256=(args.expected_victim_checkpoint_sha256),
            dataset_path=args.dataset,
            expected_dataset_sha256=args.expected_dataset_sha256,
            expected_dataset_manifest_sha256=(args.expected_dataset_manifest_sha256),
            expected_action_ontology_sha256=(args.expected_action_ontology_sha256),
            output_dir=args.output_dir,
            run_name=args.run_name,
            overwrite=args.overwrite,
            victim_action_mode=args.victim_action_mode,
            config=config,
        )

    if args.stage != "director":
        raise ValueError(f"unsupported STFA training stage: {args.stage!r}")
    dataset = load_director_dataset(
        args.dataset,
        expected_sha256=args.expected_dataset_sha256,
        expected_manifest_sha256=args.expected_dataset_manifest_sha256,
        expected_action_ontology_sha256=args.expected_action_ontology_sha256,
    )
    config = STFADirectorConfig(
        observation_shape=tuple(dataset.batch.observations.shape[1:]),
        n_actions=dataset.factorization.n_actions,
        hidden_sizes=hidden_sizes,
        activation=args.activation,
        selection_threshold=args.selection_threshold,
        stochastic_inference=args.stochastic_inference,
        reachable_top_k=(
            int(dataset.provenance["victim_probabilities"]["reachable_top_k"])
            if "victim_probabilities" in dataset.provenance
            else None
        ),
    )
    train_config = STFADirectorTrainConfig(
        gradient_steps=args.gradient_steps,
        learning_rate=args.learning_rate,
        selection_coefficient=args.selection_coefficient,
        lateral_coefficient=args.lateral_coefficient,
        longitudinal_coefficient=args.longitudinal_coefficient,
        max_gradient_norm=args.max_gradient_norm,
        seed=args.seed,
        device=device,
    )
    return train_director_from_npz(
        victim_checkpoint=args.victim_checkpoint,
        expected_victim_checkpoint_sha256=(args.expected_victim_checkpoint_sha256),
        critic_checkpoint=args.critic_checkpoint,
        expected_critic_checkpoint_sha256=(args.expected_critic_checkpoint_sha256),
        dataset_path=args.dataset,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_dataset_manifest_sha256=args.expected_dataset_manifest_sha256,
        expected_action_ontology_sha256=args.expected_action_ontology_sha256,
        output_dir=args.output_dir,
        run_name=args.run_name,
        overwrite=args.overwrite,
        victim_action_mode=args.victim_action_mode,
        config=config,
        train_config=train_config,
    )


def main(argv: Sequence[str] | None = None) -> None:
    manifest = run(_parser().parse_args(argv))
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
