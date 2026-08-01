"""Canonical provenance manifest for the audited Highway runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import gymnasium as gym
import numpy as np
import torch

from rl_attack.attacks.strong.stfa.action_factors import (
    HIGHWAY_CANONICAL_ACTION_INDEX_BY_NAME,
    highway_5_factorization,
)
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    publish_staged_files,
    sha256_file,
    validate_sha256,
)
from rl_attack.envs.highway_runtime import (
    HIGHWAY_FAST_V0_EFFECTIVE_CONFIG,
    HIGHWAY_INFO_SOURCES_KEY,
    HIGHWAY_KINEMATICS_FEATURES,
    HIGHWAY_ON_ROAD_SOURCE,
    HIGHWAY_RUNTIME_ENVIRONMENT_ID,
    HIGHWAY_RUNTIME_FACTORY,
    HIGHWAY_RUNTIME_REGISTRY_KEY,
    HIGHWAY_RUNTIME_TYPE,
    HIGHWAY_RUNTIME_VERSION,
    make_highway_fast_v0_audited,
)

HIGHWAY_RUNTIME_MANIFEST_SCHEMA = "rl_attack.highway_runtime_manifest.v1"
HIGHWAY_RUNTIME_PAYLOAD_SCHEMA = "rl_attack.highway_runtime_payload.v1"
HIGHWAY_RUNTIME_LOCK_COVERAGE = "runtime_critical_packages"
HIGHWAY_RUNTIME_REQUIRED_DISTRIBUTIONS = (
    "gymnasium",
    "highway-env",
    "numpy",
    "pyyaml",
    "stable-baselines3",
    "torch",
)
_PIN_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")


class InvalidHighwayRuntimeManifest(RuntimeError):
    """The manifest or current runtime differs from the frozen contract."""


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _qualified_type(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("manifest values cannot contain NaN or infinity")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not strict-JSON serializable: {type(value).__name__}")


def _strict_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def find_git_repository_root(start: str | Path) -> Path:
    candidate = Path(start).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory
    raise FileNotFoundError(f"no Git repository found above {candidate}")


def _run_command(arguments: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({' '.join(arguments)}): {detail}")
    return completed.stdout.strip()


def _git_record(repository_root: Path) -> dict[str, Any]:
    commit = _run_command(("git", "rev-parse", "HEAD"), cwd=repository_root)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Git HEAD is not a full lowercase commit id")
    branch = _run_command(
        ("git", "branch", "--show-current"),
        cwd=repository_root,
    )
    status = _run_command(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=repository_root,
    )
    try:
        origin = _run_command(
            ("git", "remote", "get-url", "origin"),
            cwd=repository_root,
        )
    except RuntimeError:
        origin = ""
    return {
        "repository_name": repository_root.name,
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "origin_url_sha256": (
            hashlib.sha256(origin.encode("utf-8")).hexdigest() if origin else None
        ),
    }


def _parse_runtime_lock(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(
                f"runtime lock line {line_number} must be an exact name==version pin"
            )
        name = _normalized_distribution(match.group(1))
        if name in result:
            raise ValueError(f"duplicate runtime lock distribution: {name}")
        result[name] = match.group(2)
    expected = set(HIGHWAY_RUNTIME_REQUIRED_DISTRIBUTIONS)
    if set(result) != expected:
        raise ValueError(
            "runtime lock must pin exactly the critical package set; "
            f"missing={sorted(expected - set(result))!r}, "
            f"extra={sorted(set(result) - expected)!r}"
        )
    return dict(sorted(result.items()))


def _dependency_record(lock_path: Path, repository_root: Path) -> dict[str, Any]:
    resolved_lock = lock_path.expanduser().resolve()
    try:
        relative_lock = resolved_lock.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise ValueError("dependency lock must be inside the Git repository") from error
    pins = _parse_runtime_lock(resolved_lock)
    installed: dict[str, str] = {}
    for name, expected_version in pins.items():
        try:
            actual_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required runtime distribution is absent: {name}") from error
        if actual_version != expected_version:
            raise RuntimeError(
                f"runtime distribution {name} differs from lock: "
                f"expected {expected_version}, got {actual_version}"
            )
        installed[name] = actual_version
    return {
        "lock_path": relative_lock,
        "lock_sha256": sha256_file(resolved_lock),
        "lock_coverage": HIGHWAY_RUNTIME_LOCK_COVERAGE,
        "pins": pins,
        "installed_versions": installed,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
        },
    }


def _cuda_record() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": [properties.major, properties.minor],
                    "total_memory_bytes": int(properties.total_memory),
                    "multi_processor_count": int(properties.multi_processor_count),
                }
            )
    driver_versions: list[str] | None = None
    executable = shutil.which("nvidia-smi")
    if executable is not None:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode == 0:
            driver_versions = [
                line.strip()
                for line in completed.stdout.splitlines()
                if line.strip()
            ]
    return {
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "devices": devices,
        "driver_versions": driver_versions,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def _array_record(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "c_contiguous": bool(array.flags.c_contiguous),
        "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _bound_json(value: np.ndarray) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if np.isnan(array).any():
        raise ValueError("Box bounds cannot contain NaN")
    encoded = np.empty(array.shape, dtype=object)
    finite = np.isfinite(array)
    encoded[finite] = array[finite]
    encoded[np.isposinf(array)] = "__positive_infinity__"
    encoded[np.isneginf(array)] = "__negative_infinity__"
    return encoded.tolist()


def _box_contract(space: gym.spaces.Box) -> str:
    return canonical_json_sha256(
        {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(np.dtype(space.dtype)),
            "low": _bound_json(space.low),
            "high": _bound_json(space.high),
        }
    )


def _box_record(space: gym.spaces.Box) -> dict[str, Any]:
    return {
        "type": "Box",
        "shape": list(space.shape),
        "dtype": str(np.dtype(space.dtype)),
        "low": _array_record(space.low),
        "high": _array_record(space.high),
        "contract_sha256": _box_contract(space),
    }


def _wrapper_chain(env: gym.Env) -> list[str]:
    result: list[str] = []
    cursor: gym.Env = env
    while True:
        result.append(_qualified_type(cursor))
        if not isinstance(cursor, gym.Wrapper):
            break
        cursor = cursor.env
    return result


def _info_shape(info: Mapping[str, Any]) -> dict[str, Any]:
    sources = info.get(HIGHWAY_INFO_SOURCES_KEY)
    if sources != {"on_road": HIGHWAY_ON_ROAD_SOURCE}:
        raise RuntimeError("Highway safety info source record is absent or malformed")
    if type(info.get("on_road")) is not bool:
        raise RuntimeError("Highway on_road info must be an exact bool")
    rewards = info.get("rewards")
    reward_keys = sorted(rewards) if isinstance(rewards, Mapping) else None
    return {
        "keys": sorted(info),
        "value_types": {
            key: _qualified_type(value) for key, value in sorted(info.items())
        },
        "reward_keys": reward_keys,
        "sources": dict(sources),
    }


def _environment_record(
    env: gym.Env,
    *,
    seed: int,
    max_episode_steps: int,
) -> dict[str, Any]:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("audited Highway policy space must be Box")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("audited Highway action space must be Discrete")
    policy_space = env.observation_space
    raw_space = env.env.observation_space
    if not isinstance(raw_space, gym.spaces.Box):
        raise TypeError("audited Highway raw space must be Box")

    unwrapped = env.unwrapped
    action_mapping = dict(unwrapped.action_type.actions_indexes)
    if action_mapping != dict(HIGHWAY_CANONICAL_ACTION_INDEX_BY_NAME):
        raise RuntimeError("Highway action mapping drifted after construction")
    factorization = highway_5_factorization()
    actions = [
        {
            "index": action.index,
            "name": next(
                name for name, index in action_mapping.items() if index == action.index
            ),
            "label": action.label,
            "lateral": action.lateral,
            "longitudinal": action.longitudinal,
            "globally_legal": action.available,
        }
        for action in factorization.actions
    ]
    action_contract = canonical_json_sha256(
        {
            "type": "Discrete",
            "n": int(env.action_space.n),
            "start": int(env.action_space.start),
            "dtype": str(np.dtype(env.action_space.dtype)),
            "factorization_contract_sha256": factorization.contract_hash,
        }
    )

    reset_observation, reset_info = env.reset(seed=seed)
    reset_available = [int(value) for value in unwrapped.get_available_actions()]
    next_observation, reward, terminated, truncated, step_info = env.step(1)
    step_available = [int(value) for value in unwrapped.get_available_actions()]
    reset_info_shape = _info_shape(reset_info)
    step_info_shape = _info_shape(step_info)
    info_contract = {
        "reset": reset_info_shape,
        "step": step_info_shape,
        "contract_sha256": canonical_json_sha256(
            {"reset": reset_info_shape, "step": step_info_shape}
        ),
    }

    observation_type = unwrapped.observation_type
    normalization = {
        "schema_version": "rl_attack.highway_kinematics_normalization.v1",
        "features": list(observation_type.features),
        "vehicles_count": int(observation_type.vehicles_count),
        "features_range": _jsonable(observation_type.features_range),
        "absolute": bool(observation_type.absolute),
        "order": observation_type.order,
        "normalize": bool(observation_type.normalize),
        "clip": bool(observation_type.clip),
        "see_behind": bool(observation_type.see_behind),
        "observe_intentions": bool(observation_type.observe_intentions),
        "include_obstacles": bool(observation_type.include_obstacles),
        "flatten_order": "C",
        "layout": "vehicle_rows_by_feature_columns",
    }
    normalization["contract_sha256"] = canonical_json_sha256(normalization)

    spec = env.spec
    if spec is None:
        raise RuntimeError("Highway runtime has no Gymnasium EnvSpec")
    effective_config = _jsonable(unwrapped.config)
    if effective_config != HIGHWAY_FAST_V0_EFFECTIVE_CONFIG:
        raise RuntimeError("Highway effective configuration drifted during probe")
    return {
        "identity": {
            "id": HIGHWAY_RUNTIME_ENVIRONMENT_ID,
            "registry_key": HIGHWAY_RUNTIME_REGISTRY_KEY,
            "factory": HIGHWAY_RUNTIME_FACTORY,
            "runtime_type": HIGHWAY_RUNTIME_TYPE,
            "runtime_version": HIGHWAY_RUNTIME_VERSION,
            "max_episode_steps": max_episode_steps,
            "env_spec": {
                "id": spec.id,
                "entry_point": str(spec.entry_point),
                "max_episode_steps": spec.max_episode_steps,
                "order_enforce": bool(spec.order_enforce),
                "disable_env_checker": bool(spec.disable_env_checker),
                "kwargs": _jsonable(spec.kwargs),
            },
            "wrapper_chain": _wrapper_chain(env),
        },
        "effective_config": effective_config,
        "effective_config_sha256": canonical_json_sha256(effective_config),
        "observation": {
            "raw": _box_record(raw_space),
            "policy": _box_record(policy_space),
            "normalization": normalization,
            "feature_names": list(HIGHWAY_KINEMATICS_FEATURES),
            "presence_feature": "presence",
            "flatten_order": "C",
            "layout": "row-major",
        },
        "action": {
            "type": "Discrete",
            "n": int(env.action_space.n),
            "start": int(env.action_space.start),
            "dtype": str(np.dtype(env.action_space.dtype)),
            "index_by_name": action_mapping,
            "actions": actions,
            "factorization_name": factorization.name,
            "factorization_version": factorization.version,
            "ontology_sha256": factorization.ontology_hash,
            "factorization_contract_sha256": factorization.contract_hash,
            "space_contract_sha256": action_contract,
            "availability_source": "env.unwrapped.get_available_actions()",
            "reset_available_indices": reset_available,
            "step_available_indices": step_available,
        },
        "safety_info": info_contract,
        "deterministic_probe": {
            "seed": seed,
            "action": 1,
            "action_name": "IDLE",
            "reset_observation": _array_record(reset_observation),
            "step_observation": _array_record(next_observation),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        },
    }


def build_highway_runtime_manifest(
    *,
    repository_root: str | Path,
    dependency_lock: str | Path,
    seed: int = 40000,
    max_episode_steps: int = 30,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Probe and freeze the exact runtime without writing an artifact."""

    seed = _strict_integer(seed, name="seed")
    max_episode_steps = _strict_integer(
        max_episode_steps,
        name="max_episode_steps",
        minimum=1,
    )
    if type(allow_dirty) is not bool:
        raise TypeError("allow_dirty must be bool")
    root = find_git_repository_root(repository_root)
    git = _git_record(root)
    if git["dirty"] and not allow_dirty:
        raise RuntimeError(
            "Git worktree is dirty; commit or clean it before a formal freeze "
            "(use allow_dirty only for non-formal diagnostics)"
        )
    dependencies = _dependency_record(Path(dependency_lock), root)
    env = make_highway_fast_v0_audited(max_episode_steps=max_episode_steps)
    try:
        environment = _environment_record(
            env,
            seed=seed,
            max_episode_steps=max_episode_steps,
        )
    finally:
        env.close()
    payload = {
        "schema_version": HIGHWAY_RUNTIME_PAYLOAD_SCHEMA,
        "formal_eligible": not git["dirty"],
        "formal_ineligibility_reasons": (
            [] if not git["dirty"] else ["git_worktree_dirty"]
        ),
        "repository": git,
        "dependencies": dependencies,
        "compute": {
            "python_executable_kind": "current_interpreter",
            "cuda": _cuda_record(),
        },
        "environment": environment,
    }
    payload = _jsonable(payload)
    return {
        "schema_version": HIGHWAY_RUNTIME_MANIFEST_SCHEMA,
        "payload": payload,
        "payload_sha256": canonical_json_sha256(payload),
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON mapping key: {key!r}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=reject_constant,
    )


def validate_highway_runtime_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Highway runtime manifest must be a mapping")
    manifest = dict(value)
    if set(manifest) != {"schema_version", "payload", "payload_sha256"}:
        raise ValueError("Highway runtime manifest has unknown or missing keys")
    if manifest["schema_version"] != HIGHWAY_RUNTIME_MANIFEST_SCHEMA:
        raise ValueError("Highway runtime manifest schema version mismatch")
    payload = manifest["payload"]
    if not isinstance(payload, Mapping):
        raise TypeError("Highway runtime manifest payload must be a mapping")
    payload = dict(payload)
    if payload.get("schema_version") != HIGHWAY_RUNTIME_PAYLOAD_SCHEMA:
        raise ValueError("Highway runtime payload schema version mismatch")
    expected = validate_sha256(
        manifest["payload_sha256"],
        name="payload_sha256",
    )
    if canonical_json_sha256(payload) != expected:
        raise ValueError("Highway runtime payload SHA-256 mismatch")
    canonical_json_sha256(manifest)
    return {
        "schema_version": HIGHWAY_RUNTIME_MANIFEST_SCHEMA,
        "payload": payload,
        "payload_sha256": expected,
    }


def verify_highway_runtime_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate bytes and reproduce the frozen runtime probe exactly."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    file_sha256 = sha256_file(path)
    if expected_file_sha256 is not None:
        expected = validate_sha256(
            expected_file_sha256,
            name="expected_file_sha256",
        )
        if file_sha256 != expected:
            raise InvalidHighwayRuntimeManifest("manifest file SHA-256 mismatch")
    manifest = validate_highway_runtime_manifest(_load_manifest(path))
    payload = manifest["payload"]
    try:
        dependencies = payload["dependencies"]
        environment = payload["environment"]
        probe = environment["deterministic_probe"]
        identity = environment["identity"]
        lock_relative = dependencies["lock_path"]
        seed = probe["seed"]
        max_episode_steps = identity["max_episode_steps"]
        stored_formal_eligible = payload["formal_eligible"]
    except (KeyError, TypeError) as error:
        raise InvalidHighwayRuntimeManifest(
            "manifest payload is structurally incomplete"
        ) from error
    if type(stored_formal_eligible) is not bool:
        raise InvalidHighwayRuntimeManifest("formal_eligible must be bool")
    root = find_git_repository_root(repository_root)
    lock_path = (root / str(lock_relative)).resolve()
    try:
        lock_path.relative_to(root)
    except ValueError as error:
        raise InvalidHighwayRuntimeManifest(
            "manifest dependency lock escapes repository"
        ) from error
    current = build_highway_runtime_manifest(
        repository_root=root,
        dependency_lock=lock_path,
        seed=_strict_integer(seed, name="manifest seed"),
        max_episode_steps=_strict_integer(
            max_episode_steps,
            name="manifest max_episode_steps",
            minimum=1,
        ),
        allow_dirty=not stored_formal_eligible,
    )
    if current["payload_sha256"] != manifest["payload_sha256"]:
        current_payload = current["payload"]
        differing_sections = sorted(
            key
            for key in set(payload) | set(current_payload)
            if payload.get(key) != current_payload.get(key)
        )
        raise InvalidHighwayRuntimeManifest(
            "current runtime differs from frozen payload in sections: "
            + ", ".join(differing_sections)
        )
    return {
        "status": "verified",
        "schema_version": HIGHWAY_RUNTIME_MANIFEST_SCHEMA,
        "manifest_path": str(path),
        "manifest_file_sha256": file_sha256,
        "payload_sha256": manifest["payload_sha256"],
        "formal_eligible": stored_formal_eligible,
        "git_commit": payload["repository"]["commit"],
        "environment_config_sha256": environment["effective_config_sha256"],
        "observation_contract_sha256": environment["observation"]["policy"][
            "contract_sha256"
        ],
        "action_ontology_sha256": environment["action"]["ontology_sha256"],
        "info_contract_sha256": environment["safety_info"]["contract_sha256"],
    }


def write_highway_runtime_manifest(
    output: str | Path,
    manifest: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Publish one validated canonical manifest transactionally."""

    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    normalized = validate_highway_runtime_manifest(manifest)
    destination = Path(output).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        staged.write_text(
            json.dumps(
                normalized,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        publish_staged_files({destination: staged}, overwrite=overwrite)
    finally:
        if staged.is_file():
            staged.unlink()
    return destination.resolve()


__all__ = [
    "HIGHWAY_RUNTIME_LOCK_COVERAGE",
    "HIGHWAY_RUNTIME_MANIFEST_SCHEMA",
    "HIGHWAY_RUNTIME_PAYLOAD_SCHEMA",
    "HIGHWAY_RUNTIME_REQUIRED_DISTRIBUTIONS",
    "InvalidHighwayRuntimeManifest",
    "build_highway_runtime_manifest",
    "find_git_repository_root",
    "validate_highway_runtime_manifest",
    "verify_highway_runtime_manifest",
    "write_highway_runtime_manifest",
]
