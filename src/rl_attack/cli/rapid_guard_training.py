"""Train or verify a fixed-data P5 RAPID-Guard bundle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from rl_attack.defenses.rapid_guard.denoiser import (
    ResidualDenoiserConfig,
    ResidualDenoiserTrainConfig,
)
from rl_attack.defenses.rapid_guard.detector import FusionFitConfig
from rl_attack.training.rapid_guard_pipeline import (
    load_rapid_guard_bundle,
    load_rapid_guard_dataset,
    train_rapid_guard_from_npz,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train/verify RAPID-Guard artifacts from immutable raw NPZ cohorts; "
            "outputs are training-plumbing evidence, not formal robustness results"
        )
    )
    stages = parser.add_subparsers(dest="stage", required=True)
    train = stages.add_parser("train", help="train and calibrate one frozen bundle")
    train.add_argument("--victim-checkpoint", type=Path, required=True)
    train.add_argument("--expected-victim-checkpoint-sha256", required=True)
    train.add_argument("--fit-dataset", type=Path, required=True)
    train.add_argument("--expected-fit-dataset-sha256", required=True)
    train.add_argument("--expected-fit-manifest-sha256", required=True)
    train.add_argument("--calibration-dataset", type=Path, required=True)
    train.add_argument("--expected-calibration-dataset-sha256", required=True)
    train.add_argument("--expected-calibration-manifest-sha256", required=True)
    train.add_argument("--expected-action-ontology-sha256", required=True)
    train.add_argument("--expected-projector-contract-sha256", required=True)
    train.add_argument("--expected-environment-contract-sha256", required=True)
    train.add_argument("--expected-normalization-contract-sha256", required=True)
    train.add_argument("--expected-certificate-epsilon", type=float, required=True)
    train.add_argument("--expected-anchor-update-contract-sha256", required=True)
    train.add_argument("--expected-purifier-config-sha256", required=True)
    train.add_argument("--expected-fallback-config-sha256", required=True)
    train.add_argument("--output-dir", type=Path, default=Path("outputs/p5_training"))
    train.add_argument("--run-name")
    train.add_argument("--overwrite", action="store_true")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--alpha", type=float, default=0.1)
    train.add_argument("--device", default="cpu")
    train.add_argument(
        "--active-channels",
        nargs="+",
        default=(
            "temporal_innovation",
            "categorical_js",
            "ibp_margin_deficit",
        ),
    )
    train.add_argument("--fusion-gradient-steps", type=int, default=500)
    train.add_argument("--fusion-learning-rate", type=float, default=0.05)
    train.add_argument("--fusion-l2-penalty", type=float, default=1.0e-3)
    train.add_argument("--fusion-scale-floor", type=float, default=1.0e-8)
    train.add_argument("--denoiser-hidden-sizes", type=int, nargs="+", default=(128, 128))
    train.add_argument("--denoiser-activation", choices=("relu", "tanh"), default="relu")
    train.add_argument("--denoiser-gradient-steps", type=int, default=300)
    train.add_argument("--denoiser-learning-rate", type=float, default=3.0e-4)
    train.add_argument("--denoiser-mse-coefficient", type=float, default=1.0)
    train.add_argument(
        "--denoiser-policy-consistency-coefficient",
        type=float,
        default=0.0,
    )
    train.add_argument("--denoiser-max-gradient-norm", type=float, default=10.0)

    verify = stages.add_parser("verify", help="strictly load and verify one bundle")
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--expected-checkpoint-sha256", required=True)
    verify.add_argument("--expected-victim-checkpoint-sha256")
    verify.add_argument("--expected-victim-policy-state-sha256")
    verify.add_argument("--expected-environment-contract-sha256")
    verify.add_argument("--expected-observation-space-sha256")
    verify.add_argument("--expected-action-space-sha256")
    verify.add_argument("--expected-normalization-contract-sha256")
    verify.add_argument("--expected-action-ontology-sha256")
    verify.add_argument("--expected-projector-contract-sha256")
    verify.add_argument("--expected-certificate-epsilon", type=float)
    verify.add_argument("--expected-anchor-update-contract-sha256")
    verify.add_argument("--expected-purifier-config-sha256")
    verify.add_argument("--expected-fallback-config-sha256")
    verify.add_argument("--expected-fit-dataset-sha256")
    verify.add_argument("--expected-calibration-dataset-sha256")
    verify.add_argument("--expected-proposal-transform-sha256")
    verify.add_argument("--device", default="cpu")
    return parser


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(value)
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    if args.stage == "verify":
        loaded = load_rapid_guard_bundle(
            args.checkpoint,
            expected_sha256=args.expected_checkpoint_sha256,
            device=device,
            expected_victim_checkpoint_sha256=(
                args.expected_victim_checkpoint_sha256
            ),
            expected_victim_policy_state_sha256=(
                args.expected_victim_policy_state_sha256
            ),
            expected_environment_contract_sha256=(
                args.expected_environment_contract_sha256
            ),
            expected_observation_space_sha256=(
                args.expected_observation_space_sha256
            ),
            expected_action_space_sha256=args.expected_action_space_sha256,
            expected_normalization_contract_sha256=(
                args.expected_normalization_contract_sha256
            ),
            expected_action_ontology_sha256=(
                args.expected_action_ontology_sha256
            ),
            expected_projector_contract_sha256=(
                args.expected_projector_contract_sha256
            ),
            expected_certificate_epsilon=args.expected_certificate_epsilon,
            expected_anchor_update_contract_sha256=(
                args.expected_anchor_update_contract_sha256
            ),
            expected_purifier_config_sha256=(
                args.expected_purifier_config_sha256
            ),
            expected_fallback_config_sha256=(
                args.expected_fallback_config_sha256
            ),
            expected_fit_dataset_sha256=args.expected_fit_dataset_sha256,
            expected_calibration_dataset_sha256=(
                args.expected_calibration_dataset_sha256
            ),
            expected_proposal_transform_sha256=(
                args.expected_proposal_transform_sha256
            ),
        )
        return {
            "schema_version": "rl_attack.p5_rapid_guard_verify.v1",
            "evidence_scope": "artifact_integrity_and_binding_verification_only",
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "proposal_transform_hash": loaded.proposal_transform_hash,
            "detector_manifest": loaded.artifact.manifest,
            "formal_robustness_result": False,
            "empirical_robustness_result": False,
        }
    if args.stage != "train":
        raise ValueError(f"unsupported RAPID training stage: {args.stage!r}")
    hidden_sizes = tuple(args.denoiser_hidden_sizes)
    if not hidden_sizes or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in hidden_sizes
    ):
        raise ValueError("--denoiser-hidden-sizes must contain positive integers")
    fit_dataset = load_rapid_guard_dataset(
        args.fit_dataset,
        expected_sha256=args.expected_fit_dataset_sha256,
        expected_manifest_sha256=args.expected_fit_manifest_sha256,
        expected_role="fit",
    )
    return train_rapid_guard_from_npz(
        victim_checkpoint=args.victim_checkpoint,
        expected_victim_checkpoint_sha256=args.expected_victim_checkpoint_sha256,
        fit_dataset_path=args.fit_dataset,
        expected_fit_dataset_sha256=args.expected_fit_dataset_sha256,
        expected_fit_manifest_sha256=args.expected_fit_manifest_sha256,
        calibration_dataset_path=args.calibration_dataset,
        expected_calibration_dataset_sha256=(
            args.expected_calibration_dataset_sha256
        ),
        expected_calibration_manifest_sha256=(
            args.expected_calibration_manifest_sha256
        ),
        expected_action_ontology_sha256=args.expected_action_ontology_sha256,
        expected_projector_contract_sha256=(
            args.expected_projector_contract_sha256
        ),
        expected_environment_contract_sha256=(
            args.expected_environment_contract_sha256
        ),
        expected_normalization_contract_sha256=(
            args.expected_normalization_contract_sha256
        ),
        expected_certificate_epsilon=args.expected_certificate_epsilon,
        expected_anchor_update_contract_sha256=(
            args.expected_anchor_update_contract_sha256
        ),
        expected_purifier_config_sha256=args.expected_purifier_config_sha256,
        expected_fallback_config_sha256=args.expected_fallback_config_sha256,
        output_dir=args.output_dir,
        run_name=args.run_name,
        overwrite=args.overwrite,
        seed=args.seed,
        alpha=args.alpha,
        device=device,
        active_channels=tuple(args.active_channels),
        fusion_config=FusionFitConfig(
            gradient_steps=args.fusion_gradient_steps,
            learning_rate=args.fusion_learning_rate,
            l2_penalty=args.fusion_l2_penalty,
            scale_floor=args.fusion_scale_floor,
        ),
        denoiser_config=ResidualDenoiserConfig(
            observation_shape=fit_dataset.observation_shape,
            hidden_sizes=hidden_sizes,
            activation=args.denoiser_activation,
        ),
        denoiser_train_config=ResidualDenoiserTrainConfig(
            gradient_steps=args.denoiser_gradient_steps,
            learning_rate=args.denoiser_learning_rate,
            mse_coefficient=args.denoiser_mse_coefficient,
            policy_consistency_coefficient=(
                args.denoiser_policy_consistency_coefficient
            ),
            max_gradient_norm=args.denoiser_max_gradient_norm,
            seed=args.seed,
            device=device,
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    result = run(_parser().parse_args(argv))
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
