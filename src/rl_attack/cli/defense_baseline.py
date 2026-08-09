from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from rl_attack.defenses.catalog import defense_method
from rl_attack.evaluation import evaluate_sb3_policy

_METHOD_TO_MODE = {
    "vanilla_ppo": "vanilla",
    "adv_ppo": "adv_ppo",
    "sa_ppo": "sa_ppo_style",
    "car_ppo": "car_ppo_style",
}

_METHOD_DEFAULTS = {
    "vanilla_ppo": {
        "epsilon": 0.0,
        "attack_steps": 10,
        "attack_step_size": None,
        "attack_random_start": False,
        "attack_restarts": 1,
        "epsilon_schedule_fraction": 0.0,
        "car_soft_lambda": 0.1,
    },
    "adv_ppo": {
        "epsilon": 0.02,
        "attack_steps": 10,
        "attack_step_size": 0.005,
        "attack_random_start": True,
        "attack_restarts": 1,
        "epsilon_schedule_fraction": 0.0,
        "car_soft_lambda": 0.1,
    },
    "sa_ppo": {
        "epsilon": 0.02,
        "attack_steps": 10,
        "attack_step_size": 0.005,
        "attack_random_start": True,
        "attack_restarts": 1,
        "epsilon_schedule_fraction": 0.75,
        "car_soft_lambda": 0.1,
    },
    "car_ppo": {
        "epsilon": 0.02,
        "attack_steps": 10,
        "attack_step_size": 0.005,
        "attack_random_start": True,
        "attack_restarts": 1,
        "epsilon_schedule_fraction": 0.75,
        "car_soft_lambda": 0.1,
    },
}

_AUDITED_HIGHWAY_ENV_ID = "highway-fast-v0"
_AUDITED_RUNTIME_ARGUMENTS = (
    "runtime_manifest",
    "runtime_manifest_sha256",
    "runtime_payload_sha256",
    "dependency_lock_sha256",
)


@dataclass(frozen=True)
class _TrainingAPI:
    DefenseMode: Any
    ObservationAttackKind: Any
    RobustPPOConfig: Any
    RobustPPO: Any


@dataclass(frozen=True)
class _ObservationContract:
    raw_repr: str
    raw_shape: tuple[int, ...]
    raw_dtype: str
    agent_repr: str
    agent_shape: tuple[int, ...]
    agent_dtype: str
    adapter: str
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
                "name": self.adapter,
                "applied": self.adapter_applied,
                "order": "C",
                "layout": "row-major",
                "source_shape": list(self.raw_shape),
                "target_shape": list(self.agent_shape),
            },
        }


@dataclass(frozen=True)
class _InputCheckpoint:
    requested_path: str
    resolved_path: Path
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_path": self.requested_path,
            "resolved_path": str(self.resolved_path),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class _AuditedHighwayRuntime:
    manifest_path: Path
    manifest_sha256: str
    payload_sha256: str
    dependency_lock_sha256: str
    max_episode_steps: int
    factory: str
    registry_key: str

    def manifest_record(self) -> dict[str, str]:
        return {
            "factory": self.factory,
            "registry_key": self.registry_key,
            "runtime_manifest_sha256": self.manifest_sha256,
            "runtime_payload_sha256": self.payload_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
        }


def _load_training_api() -> _TrainingAPI:
    """Delay the robust-training import so CLI metadata remains inspectable."""

    from rl_attack.defenses.training import (
        DefenseMode,
        ObservationAttackKind,
        RobustPPO,
        RobustPPOConfig,
    )

    return _TrainingAPI(
        DefenseMode=DefenseMode,
        ObservationAttackKind=ObservationAttackKind,
        RobustPPOConfig=RobustPPOConfig,
        RobustPPO=RobustPPO,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train or load a maintained SB3 PPO defense baseline and write an "
            "auditable model bundle"
        )
    )
    parser.add_argument(
        "--method",
        choices=sorted(_METHOD_TO_MODE),
        default="vanilla_ppo",
    )
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument(
        "--continue-timesteps",
        type=int,
        default=0,
        help="Additional learning steps after --load-model; zero performs load-only evaluation.",
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-start", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--load-model", type=Path)
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        help=(
            "Required for highway-fast-v0: frozen audited runtime manifest. "
            "Forbidden for other environments."
        ),
    )
    parser.add_argument("--runtime-manifest-sha256")
    parser.add_argument("--runtime-payload-sha256")
    parser.add_argument("--dependency-lock-sha256")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "p2_defenses",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--attack", choices=["none", "fgsm", "pgd"])
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--attack-steps", type=int)
    parser.add_argument("--attack-step-size", type=float)
    parser.add_argument("--attack-restarts", type=int)
    parser.add_argument("--epsilon-schedule-fraction", type=float)
    parser.add_argument("--car-soft-lambda", type=float)
    parser.add_argument(
        "--attack-random-start",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--adversarial-loss-coef", type=float)
    parser.add_argument("--policy-consistency-coef", type=float)
    parser.add_argument("--value-consistency-coef", type=float)

    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--n-steps", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("timesteps", "eval_episodes", "n_steps", "batch_size", "n_epochs"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.continue_timesteps < 0:
        raise ValueError("--continue-timesteps cannot be negative")
    if args.continue_timesteps and args.load_model is None:
        raise ValueError("--continue-timesteps requires --load-model")
    if args.epsilon is not None and args.epsilon < 0:
        raise ValueError("--epsilon cannot be negative")
    if args.attack_steps is not None and args.attack_steps <= 0:
        raise ValueError("--attack-steps must be positive")
    if args.attack_step_size is not None and args.attack_step_size <= 0:
        raise ValueError("--attack-step-size must be positive")
    if args.attack_restarts is not None and args.attack_restarts <= 0:
        raise ValueError("--attack-restarts must be positive")
    if (
        args.epsilon_schedule_fraction is not None
        and not 0 <= args.epsilon_schedule_fraction <= 1
    ):
        raise ValueError("--epsilon-schedule-fraction must be within [0, 1]")
    if args.car_soft_lambda is not None and args.car_soft_lambda <= 0:
        raise ValueError("--car-soft-lambda must be positive")
    for name in (
        "adversarial_loss_coef",
        "policy_consistency_coef",
        "value_consistency_coef",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if args.run_name is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        raise ValueError("--run-name may contain only letters, digits, dot, underscore, and dash")
    runtime_arguments = {
        name: getattr(args, name) for name in _AUDITED_RUNTIME_ARGUMENTS
    }
    supplied_runtime_arguments = {
        name for name, value in runtime_arguments.items() if value is not None
    }
    if args.env_id == _AUDITED_HIGHWAY_ENV_ID:
        missing = sorted(set(_AUDITED_RUNTIME_ARGUMENTS) - supplied_runtime_arguments)
        if missing:
            rendered = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            raise ValueError(
                "highway-fast-v0 requires all audited runtime pins; missing: "
                f"{rendered}"
            )
    elif supplied_runtime_arguments:
        rendered = ", ".join(
            f"--{name.replace('_', '-')}" for name in sorted(supplied_runtime_arguments)
        )
        raise ValueError(
            "audited Highway runtime arguments are forbidden for non-Highway "
            f"environments: {rendered}"
        )


def _enum_member(enum_type: Any, value: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        return value


def _build_robust_config(
    args: argparse.Namespace,
    api: _TrainingAPI,
) -> Any:
    method_defaults = _METHOD_DEFAULTS[args.method]
    kwargs: dict[str, Any] = {
        "mode": _enum_member(api.DefenseMode, _METHOD_TO_MODE[args.method]),
        "epsilon": (
            args.epsilon if args.epsilon is not None else method_defaults["epsilon"]
        ),
        "attack_steps": (
            args.attack_steps
            if args.attack_steps is not None
            else method_defaults["attack_steps"]
        ),
        "attack_random_start": (
            args.attack_random_start
            if args.attack_random_start is not None
            else method_defaults["attack_random_start"]
        ),
        "attack_restarts": (
            args.attack_restarts
            if args.attack_restarts is not None
            else method_defaults["attack_restarts"]
        ),
        "epsilon_schedule_fraction": (
            args.epsilon_schedule_fraction
            if args.epsilon_schedule_fraction is not None
            else method_defaults["epsilon_schedule_fraction"]
        ),
        "car_soft_lambda": (
            args.car_soft_lambda
            if args.car_soft_lambda is not None
            else method_defaults["car_soft_lambda"]
        ),
    }
    step_size = (
        args.attack_step_size
        if args.attack_step_size is not None
        else method_defaults["attack_step_size"]
    )
    if step_size is not None:
        kwargs["attack_step_size"] = step_size
    optional_values = {
        "attack": (
            _enum_member(api.ObservationAttackKind, args.attack)
            if args.attack is not None
            else None
        ),
        "adversarial_loss_coef": args.adversarial_loss_coef,
        "policy_consistency_coef": args.policy_consistency_coef,
        "value_consistency_coef": args.value_consistency_coef,
    }
    kwargs.update({key: value for key, value in optional_values.items() if value is not None})
    return api.RobustPPOConfig(**kwargs)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _model_path(model_stem: Path) -> Path:
    return model_stem if model_stem.suffix == ".zip" else model_stem.with_suffix(".zip")


def _existing_model_path(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    zipped = _model_path(path)
    if zipped.is_file():
        return zipped.resolve()
    raise FileNotFoundError(f"model checkpoint does not exist: {path}")


def _resolve_input_checkpoint(path: Path | None) -> _InputCheckpoint | None:
    if path is None:
        return None
    requested_path = str(path)
    resolved_path = _existing_model_path(path.expanduser().resolve())
    return _InputCheckpoint(
        requested_path=requested_path,
        resolved_path=resolved_path,
        sha256=_sha256(resolved_path),
    )


def _same_file_path(left: Path, right: Path) -> bool:
    left_resolved = left.expanduser().resolve()
    right_resolved = right.expanduser().resolve()
    if os.path.normcase(str(left_resolved)) == os.path.normcase(str(right_resolved)):
        return True
    try:
        return left_resolved.samefile(right_resolved)
    except OSError:
        return False


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _repository_provenance(repository_root: Path) -> dict[str, Any]:
    core_lock = repository_root / "requirements" / "core-py310-windows.lock.txt"
    upstream_lock = repository_root / "third_party" / "upstream-lock.json"
    for path in (core_lock, upstream_lock):
        if not path.is_file():
            raise FileNotFoundError(f"required provenance file does not exist: {path}")
    return {
        "repository": {
            "root": str(repository_root),
            "git_commit": _git_output(repository_root, "rev-parse", "HEAD"),
            "git_dirty": bool(
                _git_output(
                    repository_root,
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                )
            ),
        },
        "locks": {
            "core_requirements": {
                "path": str(core_lock.relative_to(repository_root)).replace("\\", "/"),
                "sha256": _sha256(core_lock),
            },
            "third_party_upstream": {
                "path": str(upstream_lock.relative_to(repository_root)).replace("\\", "/"),
                "sha256": _sha256(upstream_lock),
            },
        },
    }


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON mapping key: {key!r}")
        result[key] = value
    return result


def _strict_json_file(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_pairs,
        parse_constant=reject_constant,
    )


def _resolve_audited_highway_runtime(
    args: argparse.Namespace,
    repository_root: Path,
) -> _AuditedHighwayRuntime | None:
    if args.env_id != _AUDITED_HIGHWAY_ENV_ID:
        return None

    from rl_attack.core.artifacts import validate_sha256
    from rl_attack.envs.highway_manifest import (
        validate_highway_runtime_manifest,
        verify_highway_runtime_manifest,
    )
    from rl_attack.envs.highway_runtime import (
        HIGHWAY_RUNTIME_FACTORY,
        HIGHWAY_RUNTIME_REGISTRY_KEY,
    )

    manifest_path = args.runtime_manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"audited Highway runtime manifest does not exist: {manifest_path}"
        )
    expected_manifest_sha256 = validate_sha256(
        args.runtime_manifest_sha256,
        name="runtime_manifest_sha256",
    )
    expected_payload_sha256 = validate_sha256(
        args.runtime_payload_sha256,
        name="runtime_payload_sha256",
    )
    expected_dependency_lock_sha256 = validate_sha256(
        args.dependency_lock_sha256,
        name="dependency_lock_sha256",
    )
    actual_manifest_sha256 = _sha256(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("audited Highway runtime manifest file SHA-256 mismatch")

    manifest = validate_highway_runtime_manifest(_strict_json_file(manifest_path))
    if manifest["payload_sha256"] != expected_payload_sha256:
        raise ValueError("audited Highway runtime payload SHA-256 mismatch")
    payload = manifest["payload"]
    if not isinstance(payload, Mapping):  # pragma: no cover - validator guarantees this
        raise TypeError("audited Highway runtime payload must be a mapping")
    dependencies = payload.get("dependencies")
    environment = payload.get("environment")
    if not isinstance(dependencies, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("audited Highway runtime payload is structurally incomplete")
    identity = environment.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("audited Highway runtime identity is absent")
    if dependencies.get("lock_sha256") != expected_dependency_lock_sha256:
        raise ValueError("audited Highway dependency lock SHA-256 mismatch")
    expected_identity = {
        "id": _AUDITED_HIGHWAY_ENV_ID,
        "factory": HIGHWAY_RUNTIME_FACTORY,
        "registry_key": HIGHWAY_RUNTIME_REGISTRY_KEY,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"audited Highway runtime identity {key!r} mismatch")
    max_episode_steps = identity.get("max_episode_steps")
    if (
        isinstance(max_episode_steps, bool)
        or not isinstance(max_episode_steps, int)
        or max_episode_steps <= 0
    ):
        raise ValueError(
            "audited Highway runtime identity max_episode_steps must be positive"
        )

    lock_relative = dependencies.get("lock_path")
    if not isinstance(lock_relative, str) or not lock_relative:
        raise ValueError("audited Highway dependency lock_path must be a string")
    lock_path = (repository_root / lock_relative).resolve()
    try:
        lock_path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(
            "audited Highway dependency lock escapes the repository"
        ) from error
    if not lock_path.is_file():
        raise FileNotFoundError(f"audited Highway dependency lock is absent: {lock_path}")
    if _sha256(lock_path) != expected_dependency_lock_sha256:
        raise ValueError("audited Highway dependency lock file SHA-256 mismatch")

    evidence = verify_highway_runtime_manifest(
        manifest_path,
        repository_root=repository_root,
        expected_file_sha256=expected_manifest_sha256,
    )
    if evidence.get("manifest_file_sha256") != expected_manifest_sha256:
        raise RuntimeError("Highway runtime verifier returned a different manifest SHA-256")
    if evidence.get("payload_sha256") != expected_payload_sha256:
        raise RuntimeError("Highway runtime verifier returned a different payload SHA-256")
    return _AuditedHighwayRuntime(
        manifest_path=manifest_path,
        manifest_sha256=expected_manifest_sha256,
        payload_sha256=expected_payload_sha256,
        dependency_lock_sha256=expected_dependency_lock_sha256,
        max_episode_steps=max_episode_steps,
        factory=HIGHWAY_RUNTIME_FACTORY,
        registry_key=HIGHWAY_RUNTIME_REGISTRY_KEY,
    )


def _make_agent_env(
    env_id: str,
    *,
    highway_runtime: _AuditedHighwayRuntime | None = None,
) -> tuple[gym.Env, _ObservationContract]:
    if highway_runtime is not None:
        if env_id != _AUDITED_HIGHWAY_ENV_ID:
            raise ValueError("audited Highway runtime cannot construct another environment")
        from rl_attack.envs.highway_runtime import make_highway_fast_v0_audited

        env = make_highway_fast_v0_audited(
            max_episode_steps=highway_runtime.max_episode_steps
        )
        if not isinstance(env, gym.wrappers.FlattenObservation):
            env.close()
            raise TypeError("audited Highway factory must return FlattenObservation")
        raw_observation_space = env.env.observation_space
        adapter = "gym.wrappers.FlattenObservation"
        adapter_applied = True
    else:
        if env_id == _AUDITED_HIGHWAY_ENV_ID:
            raise ValueError("highway-fast-v0 requires an audited runtime manifest")
        env = gym.make(env_id)
        raw_observation_space = env.observation_space
        try:
            if not isinstance(raw_observation_space, gym.spaces.Box):
                raise TypeError("P2 defense baselines require a Box observation space")
            adapter_applied = len(raw_observation_space.shape) > 1
            if adapter_applied:
                env = gym.wrappers.FlattenObservation(env)
                adapter = "gym.wrappers.FlattenObservation"
            else:
                adapter = "identity"
        except Exception:
            env.close()
            raise
    try:
        if not isinstance(raw_observation_space, gym.spaces.Box):
            raise TypeError("P2 defense baselines require a Box observation space")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("P2 defense baselines require a Discrete action space")

        agent_observation_space = env.observation_space
        if not isinstance(agent_observation_space, gym.spaces.Box):
            raise TypeError("the policy-facing observation space must remain Box")
        contract = _ObservationContract(
            raw_repr=repr(raw_observation_space),
            raw_shape=tuple(int(size) for size in raw_observation_space.shape),
            raw_dtype=str(raw_observation_space.dtype),
            agent_repr=repr(agent_observation_space),
            agent_shape=tuple(int(size) for size in agent_observation_space.shape),
            agent_dtype=str(agent_observation_space.dtype),
            adapter=adapter,
            adapter_applied=adapter_applied,
        )
        return env, contract
    except Exception:
        env.close()
        raise


def _mode_value(config: Any) -> str | None:
    mode = getattr(config, "mode", None)
    if mode is None:
        return None
    return str(getattr(mode, "value", mode))


def _schedule_value(value: Any, progress_remaining: float) -> float:
    resolved = value(progress_remaining) if callable(value) else value
    return float(resolved)


def _requested_ppo_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
    }


def _effective_ppo_config(model: Any) -> dict[str, Any]:
    return {
        "learning_rate_config": _jsonable(model.learning_rate),
        "learning_rate_current": float(model.policy.optimizer.param_groups[0]["lr"]),
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "n_epochs": int(model.n_epochs),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "clip_range_initial": _schedule_value(model.clip_range, 1.0),
        "clip_range_current": _schedule_value(
            model.clip_range,
            float(model._current_progress_remaining),
        ),
        "ent_coef": float(model.ent_coef),
        "vf_coef": float(model.vf_coef),
        "max_grad_norm": float(model.max_grad_norm),
    }


def _clean_evaluation(
    model: Any,
    env_id: str,
    *,
    episode_seeds: Sequence[int],
    observation_contract: _ObservationContract,
    highway_runtime: _AuditedHighwayRuntime | None = None,
) -> dict[str, Any]:
    def env_factory() -> gym.Env:
        env, evaluation_contract = _make_agent_env(
            env_id,
            highway_runtime=highway_runtime,
        )
        if evaluation_contract != observation_contract:
            env.close()
            raise RuntimeError(
                "clean-evaluation observation adapter differs from the training adapter"
            )
        return env

    results = evaluate_sb3_policy(
        model,
        env_factory,
        episode_seeds=episode_seeds,
        deterministic=True,
    )
    returns = np.asarray([result.episode_return for result in results], dtype=np.float64)
    lengths = np.asarray([result.length for result in results], dtype=np.float64)
    return {
        "deterministic": True,
        "episode_seeds": [int(seed) for seed in episode_seeds],
        "episodes": len(results),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "return_median": float(np.median(returns)),
        "length_mean": float(lengths.mean()),
        "episode_results": [result.to_dict() for result in results],
    }


def _resolved_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name or f"{args.env_id}_{args.method}_seed{args.seed}"
    return args.output_dir.expanduser().resolve() / run_name


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    input_checkpoint = _resolve_input_checkpoint(args.load_model)
    run_dir = _resolved_run_dir(args)
    model_stem = run_dir / "model"
    output_model_path = _model_path(model_stem).resolve()
    if (
        input_checkpoint is not None
        and _same_file_path(input_checkpoint.resolved_path, output_model_path)
    ):
        raise ValueError(
            "input checkpoint and output model resolve to the same file; "
            "--overwrite cannot replace a model while it is being used as input"
        )

    repository_root = _repository_root()
    provenance = _repository_provenance(repository_root)
    highway_runtime = _resolve_audited_highway_runtime(args, repository_root)
    api = _load_training_api()
    requested_robust_config = _build_robust_config(args, api)
    requested_robust_config_data = _jsonable(requested_robust_config)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"run directory is not empty: {run_dir}; pass --overwrite to replace its bundle"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    env, observation_contract = _make_agent_env(
        args.env_id,
        highway_runtime=highway_runtime,
    )
    try:
        if input_checkpoint is not None:
            model = api.RobustPPO.load(
                input_checkpoint.resolved_path,
                env=env,
                device=args.device,
            )
            starting_timesteps = int(getattr(model, "num_timesteps", 0))
            loaded_config = getattr(model, "robust_config", None)
            if loaded_config is not None:
                loaded_mode = _mode_value(loaded_config)
                expected_mode = _METHOD_TO_MODE[args.method]
                if loaded_mode != expected_mode:
                    raise ValueError(
                        f"loaded checkpoint mode {loaded_mode!r} does not match "
                        f"--method {args.method!r} ({expected_mode!r})"
                    )
                effective_robust_config = loaded_config
            else:
                effective_robust_config = requested_robust_config
            if args.continue_timesteps:
                model.learn(
                    total_timesteps=args.continue_timesteps,
                    reset_num_timesteps=False,
                )
        else:
            model = api.RobustPPO(
                args.policy,
                env,
                robust_config=requested_robust_config,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                seed=args.seed,
                verbose=0,
                device=args.device,
            )
            starting_timesteps = int(getattr(model, "num_timesteps", 0))
            model.learn(total_timesteps=args.timesteps)
            effective_robust_config = getattr(
                model,
                "robust_config",
                requested_robust_config,
            )
        new_timesteps = int(getattr(model, "num_timesteps", 0)) - starting_timesteps

        model.save(model_stem)
        saved_model = _model_path(model_stem)
        if not saved_model.is_file():
            raise FileNotFoundError(
                f"model.save did not produce the expected artifact: {saved_model}"
            )
        saved_model = saved_model.resolve()
        saved_model_sha256 = _sha256(saved_model)

        episode_seeds = list(
            range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes)
        )
        clean_evaluation = _clean_evaluation(
            model,
            args.env_id,
            episode_seeds=episode_seeds,
            observation_contract=observation_contract,
            highway_runtime=highway_runtime,
        )
        effective_robust_config_data = _jsonable(effective_robust_config)
        method_spec = defense_method(args.method)
        manifest_path = run_dir / "manifest.json"
        observation_data = observation_contract.to_dict()
        effective_seed = _jsonable(getattr(model, "seed", None))
        effective_device = str(getattr(model, "device", args.device))
        environment_manifest: dict[str, Any] = {
            "id": args.env_id,
            **observation_data,
            "action_space": {
                "repr": repr(env.action_space),
                "type": type(env.action_space).__name__,
                "n": int(env.action_space.n),
                "start": int(env.action_space.start),
            },
        }
        if highway_runtime is not None:
            environment_manifest["audited_runtime"] = (
                highway_runtime.manifest_record()
            )
        manifest = {
            "schema_version": "rl_attack.defense_run.v2",
            "method": {
                "key": method_spec.key,
                "display_name": method_spec.display_name,
                "training_mode": _METHOD_TO_MODE[args.method],
                "reproduction_level": method_spec.reproduction_level.value,
                "training_objective": method_spec.training_objective,
                "limitations": method_spec.limitations,
                "reference_repository": method_spec.reference_repository,
                "paper_exact_reproduction": False,
                "upstream_runtime_dependency": False,
                "boundary": (
                    "Maintained clean-room SB3 implementation. Locked paper repositories "
                    "are isolated references and are not imported at runtime."
                ),
            },
            "environment": environment_manifest,
            "training": {
                "requested": {
                    "method": args.method,
                    "policy": args.policy,
                    "seed": args.seed,
                    "device": args.device,
                    "timesteps": args.timesteps,
                    "continue_timesteps": args.continue_timesteps,
                    "load_model": (
                        str(args.load_model) if args.load_model is not None else None
                    ),
                    "robust_config": requested_robust_config_data,
                    "ppo": _requested_ppo_config(args),
                },
                "effective": {
                    "loaded": input_checkpoint is not None,
                    "policy": model.policy.__class__.__name__,
                    "seed": effective_seed,
                    "device": effective_device,
                    "new_timesteps": new_timesteps,
                    "model_num_timesteps": int(
                        getattr(model, "num_timesteps", 0)
                    ),
                    "robust_config": effective_robust_config_data,
                    "ppo": _effective_ppo_config(model),
                    "last_train_metrics": _jsonable(
                        getattr(model, "last_train_metrics", {})
                    ),
                },
                "input_checkpoint": (
                    input_checkpoint.to_dict()
                    if input_checkpoint is not None
                    else None
                ),
            },
            "evaluation": {"clean": clean_evaluation},
            "artifacts": {
                "output_model": {
                    "requested_path": str(model_stem),
                    "resolved_path": str(saved_model),
                    "sha256": saved_model_sha256,
                },
                "manifest": {
                    "resolved_path": str(manifest_path.resolve()),
                },
            },
            "provenance": provenance,
            "runtime": {
                "python": platform.python_version(),
                "gymnasium": _version("gymnasium"),
                "stable_baselines3": _version("stable-baselines3"),
                "torch": _version("torch"),
                "device": {
                    "requested": args.device,
                    "effective": effective_device,
                },
            },
        }
        manifest = _jsonable(manifest)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = run(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
