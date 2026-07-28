from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from rl_attack.attacks.observation.base import PerturbationBounds
from rl_attack.core.space_contract import (
    require_exact_box_space,
    require_exact_zero_based_discrete_space,
)
from rl_attack.training.pa_ad import (
    PAADTrainConfig,
    save_pa_ad_director,
    train_pa_ad_from_sb3,
)
from rl_attack.training.robust_sarsa import (
    RobustSarsaTrainConfig,
    robust_sarsa_manifest_path,
    save_robust_sarsa_checkpoint,
    sb3_policy_state_sha256,
    train_robust_sarsa_from_sb3,
)


_METHOD_CHECKPOINT_NAMES = {
    "robust-sarsa": "robust_sarsa.pt",
    "pa-ad": "pa_ad.pt",
}


@dataclass(frozen=True)
class _ObservationContract:
    raw_repr: str
    raw_shape: tuple[int, ...]
    raw_dtype: str
    agent_repr: str
    agent_shape: tuple[int, ...]
    agent_dtype: str
    requested_adapter: str
    resolved_adapter: str
    adapter_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_observation_space": {
                "repr": self.raw_repr,
                "shape": list(self.raw_shape),
                "dtype": self.raw_dtype,
            },
            "agent_observation_space": {
                "repr": self.agent_repr,
                "shape": list(self.agent_shape),
                "dtype": self.agent_dtype,
            },
            "observation_adapter": {
                "requested": self.requested_adapter,
                "name": self.resolved_adapter,
                "applied": self.adapter_applied,
                "order": "C",
                "layout": "row-major",
                "source_shape": list(self.raw_shape),
                "target_shape": list(self.agent_shape),
            },
        }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--victim-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-victim-sha256",
        help="Optional externally recorded SHA-256; mismatches fail before PPO loading.",
    )
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--observation-adapter",
        choices=("auto", "identity", "flatten-c"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "p3_reproduced_attack_training",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--overwrite", action="store_true")


def _add_network_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=(64, 64))
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a learned P3 strong attacker against one immutable SB3 PPO victim"
        )
    )
    subparsers = parser.add_subparsers(dest="method", required=True)

    robust = subparsers.add_parser(
        "robust-sarsa",
        help="Collect frozen-victim SARSA rollouts and fit a robust critic.",
    )
    _add_common_arguments(robust)
    _add_network_arguments(robust)
    robust.set_defaults(hidden_sizes=(128, 128))
    robust.add_argument("--rollout-steps", type=int, default=20_000)
    robust.add_argument("--gradient-steps", type=int, default=2_000)
    robust.add_argument("--batch-size", type=int, default=256)
    robust.add_argument("--robust-coefficient", type=float, default=0.1)
    robust.add_argument(
        "--state-epsilon",
        type=float,
        nargs="+",
        default=(0.05,),
        help=(
            "State-neighborhood L-infinity radius: one scalar or one value "
            "per flattened observation feature."
        ),
    )
    robust.add_argument("--action-epsilon", type=float, default=0.05)
    robust.add_argument("--action-robust-steps", type=int, default=5)
    robust.add_argument("--action-robust-restarts", type=int, default=1)
    robust.add_argument(
        "--state-robust-step-size",
        type=float,
        nargs="+",
        help=(
            "Optional state PGD step size: one scalar or one value per "
            "flattened observation feature."
        ),
    )
    robust.add_argument("--action-robust-step-size", type=float)
    robust.add_argument("--epsilon-warmup-fraction", type=float, default=0.75)
    robust.add_argument("--max-grad-norm", type=float, default=10.0)
    robust.add_argument(
        "--victim-action-mode",
        choices=("stochastic_sample", "deterministic_greedy"),
        default="stochastic_sample",
    )

    pa_ad = subparsers.add_parser(
        "pa-ad",
        help="Train the stochastic PA-AD PAMDP director with victim-negative reward.",
    )
    _add_common_arguments(pa_ad)
    _add_network_arguments(pa_ad)
    pa_ad.add_argument("--total-timesteps", type=int, default=2_048)
    pa_ad.add_argument("--rollout-steps", type=int, default=256)
    pa_ad.add_argument("--update-epochs", type=int, default=4)
    pa_ad.add_argument("--minibatch-size", type=int, default=64)
    pa_ad.add_argument("--gae-lambda", type=float, default=0.95)
    pa_ad.add_argument("--clip-range", type=float, default=0.2)
    pa_ad.add_argument("--value-coefficient", type=float, default=0.5)
    pa_ad.add_argument("--entropy-coefficient", type=float, default=0.0)
    pa_ad.add_argument("--max-gradient-norm", type=float, default=0.5)
    pa_ad.add_argument(
        "--normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    pa_ad.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=(0.02,),
        help=(
            "PA-AD training L-infinity radius: one scalar to broadcast or one "
            "value per flattened policy-input feature."
        ),
    )
    pa_ad.add_argument("--actor-steps", type=int, default=1)
    pa_ad.add_argument("--actor-step-size", type=float)
    pa_ad.add_argument("--alignment-weight", type=float, default=1.0)
    pa_ad.add_argument("--activation", choices=("tanh", "relu"), default="tanh")
    pa_ad.add_argument("--log-std-init", type=float, default=-0.5)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> str:
    if len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value.lower()


def _existing_checkpoint(path: Path) -> Path:
    expanded = path.expanduser().resolve()
    if expanded.is_file():
        return expanded
    zipped = expanded.with_suffix(".zip") if expanded.suffix != ".zip" else expanded
    if zipped.is_file():
        return zipped.resolve()
    raise FileNotFoundError(f"victim checkpoint does not exist: {path}")


def _same_file_path(left: Path, right: Path) -> bool:
    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    if os.path.normcase(str(left)) == os.path.normcase(str(right)):
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _run_directory(args: argparse.Namespace) -> Path:
    if args.run_name is not None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
            raise ValueError(
                "--run-name may contain only letters, digits, dot, underscore, and dash"
            )
        run_name = args.run_name
    else:
        env_component = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.env_id)
        run_name = f"{env_component}_{args.method}_seed{args.seed}"
    return args.output_dir.expanduser().resolve() / run_name


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not args.hidden_sizes or any(value <= 0 for value in args.hidden_sizes):
        raise ValueError("--hidden-sizes must contain positive widths")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if not 0 <= args.gamma <= 1:
        raise ValueError("--gamma must be within [0, 1]")
    for name in ("rollout_steps",):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.method == "robust-sarsa":
        for name in (
            "gradient_steps",
            "batch_size",
            "action_robust_steps",
            "action_robust_restarts",
        ):
            if int(getattr(args, name)) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
    else:
        for name in (
            "total_timesteps",
            "update_epochs",
            "minibatch_size",
            "actor_steps",
        ):
            if int(getattr(args, name)) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if not args.epsilon or any(
            not np.isfinite(value) or value < 0 for value in args.epsilon
        ):
            raise ValueError("--epsilon values must be finite and non-negative")
        if args.actor_step_size is not None and args.actor_step_size < 0:
            raise ValueError("--actor-step-size must be non-negative")


def _prepare_outputs(
    args: argparse.Namespace,
    victim_checkpoint: Path,
) -> tuple[Path, Path, Path]:
    run_dir = _run_directory(args)
    checkpoint = (run_dir / _METHOD_CHECKPOINT_NAMES[args.method]).resolve()
    sidecar = checkpoint.with_name(checkpoint.name + ".manifest.json")
    run_manifest = (run_dir / "manifest.json").resolve()
    for output in (checkpoint, sidecar, run_manifest):
        if _same_file_path(victim_checkpoint, output):
            raise ValueError(
                "victim input checkpoint and an output artifact resolve to the same file; "
                "--overwrite cannot replace the frozen victim"
            )
    existing = [path for path in (checkpoint, sidecar, run_manifest) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "output artifact already exists; pass --overwrite to replace the bundle: "
            + ", ".join(str(path) for path in existing)
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint, sidecar, run_manifest


def _make_training_env(
    env_id: str,
    victim: PPO,
    requested_adapter: str,
) -> tuple[gym.Env, _ObservationContract]:
    env = gym.make(env_id)
    try:
        raw = env.observation_space
        if not isinstance(raw, gym.spaces.Box):
            raise TypeError("learned strong-attacker training requires a Box observation space")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("learned strong-attacker training requires Discrete actions")
        if not isinstance(victim.observation_space, gym.spaces.Box):
            raise TypeError("victim checkpoint must use a Box observation space")
        if not isinstance(victim.action_space, gym.spaces.Discrete):
            raise TypeError("victim checkpoint must use Discrete actions")

        raw_shape = tuple(int(value) for value in raw.shape)
        victim_shape = tuple(int(value) for value in victim.observation_space.shape)
        should_flatten = requested_adapter == "flatten-c"
        if requested_adapter == "auto":
            should_flatten = (
                raw_shape != victim_shape
                and len(raw_shape) > 1
                and victim_shape == (int(np.prod(raw_shape)),)
            )
        if should_flatten:
            env = gym.wrappers.FlattenObservation(env)
            resolved_adapter = "gym.wrappers.FlattenObservation"
        else:
            resolved_adapter = "identity"

        agent = env.observation_space
        if not isinstance(agent, gym.spaces.Box):
            raise TypeError("policy-facing observation space must remain Box")
        agent_shape = tuple(int(value) for value in agent.shape)
        require_exact_box_space(
            agent,
            victim.observation_space,
            context="environment/victim after observation adapter resolution",
        )
        require_exact_zero_based_discrete_space(
            env.action_space,
            victim.action_space,
            context="environment/victim",
        )
        contract = _ObservationContract(
            raw_repr=repr(raw),
            raw_shape=raw_shape,
            raw_dtype=str(raw.dtype),
            agent_repr=repr(agent),
            agent_shape=agent_shape,
            agent_dtype=str(agent.dtype),
            requested_adapter=requested_adapter,
            resolved_adapter=resolved_adapter,
            adapter_applied=should_flatten,
        )
        return env, contract
    except Exception:
        env.close()
        raise


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _strict_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _robust_sarsa_config(args: argparse.Namespace) -> RobustSarsaTrainConfig:
    state_epsilon_values = tuple(float(value) for value in args.state_epsilon)
    state_epsilon: float | tuple[float, ...] = (
        state_epsilon_values[0]
        if len(state_epsilon_values) == 1
        else state_epsilon_values
    )
    state_robust_step_size = (
        None
        if args.state_robust_step_size is None
        else tuple(float(value) for value in args.state_robust_step_size)
    )
    if state_robust_step_size is not None and len(state_robust_step_size) == 1:
        state_robust_step_size = state_robust_step_size[0]
    return RobustSarsaTrainConfig(
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        gradient_steps=args.gradient_steps,
        batch_size=args.batch_size,
        hidden_sizes=tuple(args.hidden_sizes),
        robust_coefficient=args.robust_coefficient,
        state_epsilon=state_epsilon,
        action_epsilon=args.action_epsilon,
        action_robust_steps=args.action_robust_steps,
        action_robust_restarts=args.action_robust_restarts,
        state_robust_step_size=state_robust_step_size,
        action_robust_step_size=args.action_robust_step_size,
        epsilon_warmup_fraction=args.epsilon_warmup_fraction,
        max_grad_norm=args.max_grad_norm,
        victim_action_mode=args.victim_action_mode,
        seed=args.seed,
        device=args.device,
    )


def _pa_ad_config(
    args: argparse.Namespace,
    observation_shape: tuple[int, ...],
) -> PAADTrainConfig:
    actor_step_size = None
    if args.actor_step_size is not None:
        actor_step_size = np.full(
            observation_shape,
            args.actor_step_size,
            dtype=np.float32,
        ).tolist()
    return PAADTrainConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        max_gradient_norm=args.max_gradient_norm,
        normalize_advantage=args.normalize_advantage,
        actor_steps=args.actor_steps,
        actor_step_size=actor_step_size,  # type: ignore[arg-type]
        alignment_weight=args.alignment_weight,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation,
        log_std_init=args.log_std_init,
        seed=args.seed,
    )


def _pa_ad_epsilon(
    args: argparse.Namespace,
    observation_shape: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(args.epsilon, dtype=np.float32)
    feature_count = int(np.prod(observation_shape))
    if values.shape == (1,):
        return np.full(observation_shape, float(values[0]), dtype=np.float32)
    if values.shape != (feature_count,):
        raise ValueError(
            "--epsilon must contain one scalar or exactly one value per flattened "
            f"policy-input feature ({feature_count})"
        )
    return values.reshape(observation_shape).astype(np.float32, copy=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    victim_checkpoint = _existing_checkpoint(args.victim_checkpoint)
    victim_checkpoint_sha256 = _sha256(victim_checkpoint)
    if args.expected_victim_sha256 is not None:
        expected = _validate_sha256(
            args.expected_victim_sha256,
            name="--expected-victim-sha256",
        )
        if expected != victim_checkpoint_sha256:
            raise ValueError("victim checkpoint SHA-256 does not match the expected digest")
    checkpoint, sidecar, run_manifest_path = _prepare_outputs(
        args,
        victim_checkpoint,
    )

    victim = PPO.load(victim_checkpoint, device=args.device)
    policy_hash_before = sb3_policy_state_sha256(victim)
    env, observation_contract = _make_training_env(
        args.env_id,
        victim,
        args.observation_adapter,
    )
    try:
        if args.method == "robust-sarsa":
            config = _robust_sarsa_config(args)
            result = train_robust_sarsa_from_sb3(
                victim,
                env,
                victim_checkpoint_path=victim_checkpoint,
                expected_victim_checkpoint_sha256=victim_checkpoint_sha256,
                rollout_steps=args.rollout_steps,
                config=config,
            )
            checkpoint_sha256 = save_robust_sarsa_checkpoint(checkpoint, result)
            method_manifest = result.manifest
            victim_action_mode = config.victim_action_mode
            training_summary = {
                "rollout_steps": args.rollout_steps,
                "final_td_loss": result.final_td_loss,
                "final_robust_loss": result.final_robust_loss,
            }
        else:
            shape = tuple(int(value) for value in env.observation_space.shape)
            epsilon = _pa_ad_epsilon(args, shape)
            bounds = PerturbationBounds(
                epsilon=epsilon,
                lower=np.asarray(env.observation_space.low, dtype=np.float32),
                upper=np.asarray(env.observation_space.high, dtype=np.float32),
                mutable_mask=np.ones(shape, dtype=np.bool_),
            )
            config = _pa_ad_config(args, shape)
            result = train_pa_ad_from_sb3(
                victim,
                env,
                victim_checkpoint_path=victim_checkpoint,
                bounds=bounds,
                config=config,
            )
            method_manifest = save_pa_ad_director(
                result.director,
                checkpoint,
                victim_provenance=result.victim_provenance,
                trainer_manifest=result.trainer_manifest,
            )
            checkpoint_sha256 = _sha256(checkpoint)
            victim_action_mode = "stochastic"
            training_summary = {
                "collected_steps": result.collected_steps,
                "update_metrics": list(result.update_metrics),
                "perturbation_contract": result.trainer_manifest["run"][
                    "perturbation_contract"
                ],
            }

        policy_hash_after = sb3_policy_state_sha256(victim)
        if policy_hash_after != policy_hash_before:
            raise RuntimeError("frozen victim policy state changed during attacker training")
        if not sidecar.is_file():
            raise FileNotFoundError(f"training API did not create sidecar manifest: {sidecar}")
        if checkpoint_sha256 != _sha256(checkpoint):
            raise RuntimeError("saved attacker checkpoint hash changed before manifest writing")

        manifest: dict[str, Any] = {
            "schema_version": "rl_attack.p3_learned_attacker_training.v1",
            "status": "completed",
            "method": {
                "key": args.method.replace("-", "_"),
                "victim_action_mode": victim_action_mode,
                "learned_attacker": True,
            },
            "victim": {
                "requested_checkpoint": str(args.victim_checkpoint),
                "resolved_checkpoint": str(victim_checkpoint),
                "checkpoint_sha256": victim_checkpoint_sha256,
                "expected_checkpoint_sha256": args.expected_victim_sha256,
                "expected_digest_verified": args.expected_victim_sha256 is not None,
                "policy_state_sha256_before": policy_hash_before,
                "policy_state_sha256_after": policy_hash_after,
                "frozen": True,
                "eval_mode": not victim.policy.training,
                "all_parameters_require_grad_false": not any(
                    parameter.requires_grad for parameter in victim.policy.parameters()
                ),
            },
            "environment": {
                "id": args.env_id,
                **observation_contract.to_dict(),
                "action_space": repr(env.action_space),
            },
            "execution": {
                "seed": args.seed,
                "requested_device": args.device,
                "effective_device": str(victim.device),
            },
            "training": {
                "config": _jsonable(config),
                "summary": training_summary,
                "method_manifest": method_manifest,
            },
            "artifacts": {
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha256,
                },
                "checkpoint_manifest": {
                    "path": str(sidecar),
                    "sha256": _sha256(sidecar),
                },
                "run_manifest": {"path": str(run_manifest_path)},
            },
        }
        normalized_manifest = _jsonable(manifest)
        _strict_write(run_manifest_path, normalized_manifest)
        return normalized_manifest
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = run(args)
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
