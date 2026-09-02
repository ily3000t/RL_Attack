"""Artifact-only importer for the frozen P4-v2c/v2d/v2e development bundle.

This module deliberately does not replay old code, load an old parent run, or
compare the bundle with the current Git tree.  The byte-level authority is the
published unified-development manifest and its closed 17-file inventory.  A
successful load additionally checks the schedule, episode, outcome, and query
ledgers before exposing deeply read-only JSON records to P4-v2f.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

GOLDEN_RELATIVE_ROOT = Path("outputs/p4_v2de_unified_development_20260901")
GOLDEN_MANIFEST_SHA256 = (
    "a883275fedf4920cc48e91d6a45ae37033ecf19f7bbbca3e54247214098c67aa"
)
GOLDEN_SCHEDULES_SHA256 = (
    "95052a41f9ccc0a51417c5e853bcbf9db786c3fe1c0313527248acc58878315f"
)
GOLDEN_STEPS_SHA256 = (
    "099bbcad53f4e19d28e06b1d4b65dd265de1bd51c88a259d83f0fe5f69c81beb"
)
GOLDEN_EPISODES_SHA256 = (
    "56018a3ae26fb35880db44e7033cf3be7c7f4ffd55587cbebe88cbebc0193b4a"
)
GOLDEN_SUMMARY_SHA256 = (
    "b739108f8284ad2330a5bf1def5d5073a90d551c16234d0e7f960e0a899acb6f"
)
GOLDEN_VICTIM_CHECKPOINT_SHA256 = (
    "109e89a0cf8227facf5a9c309b9db2bed7be299627f007e7e409d6de2e11de7e"
)
GOLDEN_VICTIM_POLICY_STATE_SHA256 = (
    "9b29eb2b873851daa4aade33957d6d811f47c722d4616e48dfc83836391bb881"
)

GOLDEN_EPISODE_SEEDS = tuple(range(556_000, 556_005))
GOLDEN_CONDITIONS = (
    "clean",
    "random_fixed_schedule",
    "fgsm_fixed_schedule",
    "pgd20x5_fixed_schedule",
    "mad20x5_fixed_schedule",
    "stfa_v2c_composite_on_v2e_schedule",
    "stfa_v2d_positive_part_on_v2e_schedule",
    "stfa_v2e_signed_return_fixed_timing",
)
GOLDEN_CLAIMS = MappingProxyType(
    {
        "causal_online_director_claimed": False,
        "effectiveness_claim_eligible": False,
        "formal_evaluation_eligible": False,
        "formal_summary_eligible": False,
        "statistical_significance_claimed": False,
        "sumo_effectiveness_claimed": False,
        "superiority_claim_eligible": False,
        "vanilla_problem_solved": False,
    }
)

_REQUIRED_FILES = frozenset(
    {
        "resolved_config.json",
        "v2d_trajectory_dataset.npz",
        "v2d_trajectory_dataset.npz.manifest.json",
        "stfa_v2d_return_critic.pt",
        "stfa_v2d_return_critic.pt.manifest.json",
        "v2e_signed_return_dataset.npz",
        "v2e_signed_return_dataset.npz.manifest.json",
        "stfa_v2e_signed_return_critic.pt",
        "stfa_v2e_signed_return_critic.pt.manifest.json",
        "schedules.json",
        "steps.json",
        "episodes.json",
        "summary.json",
        "comparison_table.json",
        "comparison_table.csv",
        "comparison_table.md",
        "manifest.json",
    }
)
_CORE_HASHES = MappingProxyType(
    {
        "manifest.json": GOLDEN_MANIFEST_SHA256,
        "schedules.json": GOLDEN_SCHEDULES_SHA256,
        "steps.json": GOLDEN_STEPS_SHA256,
        "episodes.json": GOLDEN_EPISODES_SHA256,
        "summary.json": GOLDEN_SUMMARY_SHA256,
    }
)
_SCHEDULE_SHA256_BY_SEED = MappingProxyType(
    {
        556_000: "3424c6025e5fb6a2e3b7562b37c6b1489eb098ab88ddbbacdce5612c31c68967",
        556_001: "b86b209ddfb74f7afdbcf16881bf0137a5d9ca5e56796aff06f352deadf311b9",
        556_002: "98d30093f2e3132fc0e827ab22dada800947cd83ff730894521c1938593334a2",
        556_003: "0cd1da619c9cc068edaf9a604bb38b61b140eebb599e5390e6a2a7e8282e49e9",
        556_004: "f671c1ba04bd5bf3dacdfa208c9b0ffa2a73d440c25108fd25a4027e16e4a8a4",
    }
)
_QUERY_KEYS = frozenset(
    {
        "observation_queries",
        "transform_queries",
        "gradient_queries",
        "projection_queries",
        "critic_queries",
        "director_queries",
        "total_queries",
    }
)
_OUTCOME_KEYS = frozenset(
    {
        "action_flips",
        "collision",
        "cumulative_safety_cost",
        "discounted_return",
        "discounted_safety_cost",
        "episode_length",
        "episode_return",
        "horizon_exhausted",
        "merge_failure",
        "merge_success",
        "minimum_gap",
        "minimum_ttc",
        "missed_merge",
        "near_miss",
        "near_miss_count",
        "nonzero_steps",
        "selected_steps",
        "terminated",
        "termination_reason",
        "truncated",
    }
)
_BOOL_OUTCOMES = frozenset(
    {
        "collision",
        "horizon_exhausted",
        "merge_failure",
        "merge_success",
        "missed_merge",
        "near_miss",
        "terminated",
        "truncated",
    }
)
_INT_OUTCOMES = frozenset(
    {
        "action_flips",
        "episode_length",
        "near_miss_count",
        "nonzero_steps",
        "selected_steps",
    }
)
_JSON_FILES = frozenset(name for name in _REQUIRED_FILES if name.endswith(".json"))


class InvalidP4V2FGolden(RuntimeError):
    """Raised when the frozen golden authority is absent or inconsistent."""


@dataclass(frozen=True, slots=True)
class GoldenFileEvidence:
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class P4V2FGoldenBundle:
    """Deeply read-only view of the verified historical development bundle."""

    root: Path
    manifest_sha256: str
    victim_checkpoint_sha256: str
    victim_policy_state_sha256: str
    episode_seeds: tuple[int, ...]
    conditions: tuple[str, ...]
    schedule_sha256_by_seed: Mapping[int, str]
    file_evidence: Mapping[str, GoldenFileEvidence]
    manifest: Mapping[str, Any]
    schedules: tuple[Mapping[str, Any], ...]
    steps: tuple[Mapping[str, Any], ...]
    episodes: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    comparison_table: Mapping[str, Any]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return True
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _strict_json_load_bytes(payload: bytes, *, name: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise InvalidP4V2FGolden(f"duplicate JSON key in {name}: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise InvalidP4V2FGolden(f"non-finite JSON constant in {name}: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidP4V2FGolden(f"invalid UTF-8 JSON: {name}") from error
    _reject_nonfinite(value, name=name)
    return value


def _reject_nonfinite(value: object, *, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidP4V2FGolden(f"non-finite JSON number in {name}")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_nonfinite(child, name=name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_nonfinite(child, name=name)


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise InvalidP4V2FGolden(f"{name} must be a JSON object")
    return dict(value)


def _sequence(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise InvalidP4V2FGolden(f"{name} must be a JSON array")
    return value


def _query_vector(value: object, *, name: str) -> dict[str, int]:
    record = _mapping(value, name=name)
    if set(record) != _QUERY_KEYS:
        raise InvalidP4V2FGolden(f"{name} query keys differ")
    if any(type(value) is not int or value < 0 for value in record.values()):
        raise InvalidP4V2FGolden(f"{name} queries must be non-negative integers")
    subtotal = sum(record[key] for key in _QUERY_KEYS if key != "total_queries")
    if record["total_queries"] != subtotal:
        raise InvalidP4V2FGolden(f"{name} total query ledger differs")
    return record


def _sum_queries(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int]:
    return {key: left[key] + right[key] for key in _QUERY_KEYS}


def _zero_queries() -> dict[str, int]:
    return {key: 0 for key in _QUERY_KEYS}


def _validate_manifest(manifest: object) -> dict[str, Any]:
    record = _mapping(manifest, name="manifest")
    if (
        record.get("schema_version") != "rl_attack.p4_v2de_unified_development_run.v1"
        or record.get("status") != "complete"
        or record.get("scope") != "development_in_sample_only"
        or record.get("conditions") != list(GOLDEN_CONDITIONS)
        or record.get("claims") != dict(GOLDEN_CLAIMS)
        or any(value is not False for value in record.get("claims", {}).values())
    ):
        raise InvalidP4V2FGolden("golden manifest authority differs")
    protocol = _mapping(record.get("common_seed_protocol"), name="common_seed_protocol")
    if (
        protocol.get("episode_seeds") != list(GOLDEN_EPISODE_SEEDS)
        or protocol.get("evaluation_episode_seeds") != list(GOLDEN_EPISODE_SEEDS)
        or protocol.get("same_scenarios_across_methods") is not True
        or protocol.get("train_evaluation_overlap_acknowledged") is not True
    ):
        raise InvalidP4V2FGolden("golden seed protocol differs")
    victim = _mapping(record.get("victim"), name="manifest.victim")
    if victim != {
        "checkpoint_sha256": GOLDEN_VICTIM_CHECKPOINT_SHA256,
        "policy_state_sha256": GOLDEN_VICTIM_POLICY_STATE_SHA256,
    }:
        raise InvalidP4V2FGolden("golden victim hashes differ")
    return record


def _victim_hashes(value: object, *, name: str) -> tuple[object, object]:
    record = _mapping(value, name=name)
    return record.get("checkpoint_sha256"), record.get("policy_state_sha256")


def _validate_cross_file_authority(documents: Mapping[str, Any]) -> None:
    for name in ("resolved_config.json", "summary.json", "comparison_table.json"):
        record = _mapping(documents[name], name=name)
        claims = record.get("claims")
        if claims != dict(GOLDEN_CLAIMS) or any(value is not False for value in claims.values()):
            raise InvalidP4V2FGolden(f"{name} claim authority differs")
    summary = _mapping(documents["summary.json"], name="summary.json")
    if (
        summary.get("episode_seeds") != list(GOLDEN_EPISODE_SEEDS)
        or summary.get("conditions") != list(GOLDEN_CONDITIONS)
    ):
        raise InvalidP4V2FGolden("summary seed or condition authority differs")
    for name in (
        "v2d_trajectory_dataset.npz.manifest.json",
        "v2e_signed_return_dataset.npz.manifest.json",
    ):
        victim = _mapping(documents[name], name=name).get("victim")
        if _victim_hashes(victim, name=f"{name}.victim") != (
            GOLDEN_VICTIM_CHECKPOINT_SHA256,
            GOLDEN_VICTIM_POLICY_STATE_SHA256,
        ):
            raise InvalidP4V2FGolden(f"{name} victim binding differs")
    for name in (
        "stfa_v2d_return_critic.pt.manifest.json",
        "stfa_v2e_signed_return_critic.pt.manifest.json",
    ):
        outer = _mapping(documents[name], name=name)
        inner = _mapping(outer.get("manifest"), name=f"{name}.manifest")
        if _victim_hashes(inner.get("victim"), name=f"{name}.victim") != (
            GOLDEN_VICTIM_CHECKPOINT_SHA256,
            GOLDEN_VICTIM_POLICY_STATE_SHA256,
        ):
            raise InvalidP4V2FGolden(f"{name} victim binding differs")


def _validate_schedules(value: object) -> dict[int, str]:
    schedules = _sequence(value, name="schedules.json")
    if len(schedules) != len(GOLDEN_EPISODE_SEEDS):
        raise InvalidP4V2FGolden("golden schedule count differs")
    schedule_hashes: dict[int, str] = {}
    for index, value in enumerate(schedules):
        schedule = _mapping(value, name=f"schedule[{index}]")
        expected_keys = {
            "candidate_count",
            "episode_seed",
            "physical_shared_queries",
            "schema_version",
            "selected",
            "selection_inputs",
            "selector_contract",
            "sha256",
            "shared_stfa_restart_plan",
            "shared_stfa_restart_plan_sha256",
        }
        if set(schedule) != expected_keys:
            raise InvalidP4V2FGolden(f"schedule[{index}] keys differ")
        seed = schedule["episode_seed"]
        if type(seed) is not int or seed != GOLDEN_EPISODE_SEEDS[index]:
            raise InvalidP4V2FGolden("golden schedule seed order differs")
        stored_sha = schedule["sha256"]
        if stored_sha != _SCHEDULE_SHA256_BY_SEED[seed]:
            raise InvalidP4V2FGolden(f"schedule hash authority differs: {seed}")
        payload = dict(schedule)
        payload.pop("sha256")
        if _canonical_sha256(payload) != stored_sha:
            raise InvalidP4V2FGolden(f"schedule canonical hash differs: {seed}")
        plan = _mapping(
            schedule["shared_stfa_restart_plan"],
            name=f"schedule[{seed}].shared_stfa_restart_plan",
        )
        if _canonical_sha256(plan) != schedule["shared_stfa_restart_plan_sha256"]:
            raise InvalidP4V2FGolden(f"schedule restart-plan hash differs: {seed}")
        selected = _sequence(schedule["selected"], name=f"schedule[{seed}].selected")
        inputs = _sequence(
            schedule["selection_inputs"], name=f"schedule[{seed}].selection_inputs"
        )
        if len(selected) != 2 or schedule["candidate_count"] != len(inputs):
            raise InvalidP4V2FGolden(f"schedule cardinality differs: {seed}")
        selected_steps = [row.get("step_index") for row in selected if isinstance(row, Mapping)]
        if (
            len(selected_steps) != 2
            or any(type(step) is not int or step < 0 for step in selected_steps)
            or len(set(selected_steps)) != 2
            or set(plan) != {str(step) for step in selected_steps}
        ):
            raise InvalidP4V2FGolden(f"schedule selected-step evidence differs: {seed}")
        _query_vector(
            schedule["physical_shared_queries"],
            name=f"schedule[{seed}].physical_shared_queries",
        )
        schedule_hashes[seed] = stored_sha
    return schedule_hashes


def _validate_outcome(value: object, *, name: str) -> dict[str, Any]:
    outcome = _mapping(value, name=name)
    if set(outcome) != _OUTCOME_KEYS:
        raise InvalidP4V2FGolden(f"{name} outcome keys differ")
    for key in _BOOL_OUTCOMES:
        if type(outcome[key]) is not bool:
            raise InvalidP4V2FGolden(f"{name}.{key} must be boolean")
    for key in _INT_OUTCOMES:
        if type(outcome[key]) is not int or outcome[key] < 0:
            raise InvalidP4V2FGolden(f"{name}.{key} must be a non-negative integer")
    numeric = _OUTCOME_KEYS - _BOOL_OUTCOMES - _INT_OUTCOMES - {"termination_reason"}
    if any(
        isinstance(outcome[key], bool)
        or not isinstance(outcome[key], (int, float))
        or not math.isfinite(float(outcome[key]))
        for key in numeric
    ):
        raise InvalidP4V2FGolden(f"{name} contains an invalid numeric outcome")
    if not isinstance(outcome["termination_reason"], str):
        raise InvalidP4V2FGolden(f"{name}.termination_reason must be a string")
    return outcome


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _validate_episodes_and_ledgers(
    episodes_value: object,
    steps_value: object,
    schedule_hashes: Mapping[int, str],
    schedules_value: object,
) -> None:
    episodes = _sequence(episodes_value, name="episodes.json")
    steps = _sequence(steps_value, name="steps.json")
    expected_pairs = [
        (condition, seed)
        for seed in GOLDEN_EPISODE_SEEDS
        for condition in GOLDEN_CONDITIONS
    ]
    if len(episodes) != len(expected_pairs):
        raise InvalidP4V2FGolden("golden episode matrix is not 8x5")
    physical_queries = {
        int(schedule["episode_seed"]): _query_vector(
            schedule["physical_shared_queries"], name="schedule physical queries"
        )
        for schedule in _sequence(schedules_value, name="schedules.json")
    }
    native_by_pair: dict[tuple[str, int], dict[str, int]] = defaultdict(_zero_queries)
    logical_by_pair: dict[tuple[str, int], dict[str, int]] = defaultdict(_zero_queries)
    environment_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    environment_steps: set[tuple[str, int, int]] = set()
    logical_steps: set[tuple[str, int, int]] = set()
    for index, value in enumerate(steps):
        row = _mapping(value, name=f"steps[{index}]")
        condition = row.get("condition")
        seed = row.get("episode_seed")
        step_index = row.get("step_index")
        if (
            condition not in GOLDEN_CONDITIONS
            or seed not in GOLDEN_EPISODE_SEEDS
            or type(step_index) is not int
            or step_index < 0
        ):
            raise InvalidP4V2FGolden(f"invalid step identity at row {index}")
        pair = (condition, seed)
        query = _query_vector(row.get("queries"), name=f"steps[{index}].queries")
        identity = (condition, seed, step_index)
        if row.get("row_kind") == "environment_step":
            if identity in environment_steps:
                raise InvalidP4V2FGolden(f"duplicate environment step: {identity!r}")
            environment_steps.add(identity)
            native_by_pair[pair] = _sum_queries(native_by_pair[pair], query)
            environment_rows[pair].append(row)
        elif row.get("row_kind") == "logical_schedule_charge":
            if condition == "clean" or identity in logical_steps:
                raise InvalidP4V2FGolden(f"invalid logical schedule step: {identity!r}")
            logical_steps.add(identity)
            logical_by_pair[pair] = _sum_queries(logical_by_pair[pair], query)
        else:
            raise InvalidP4V2FGolden(f"invalid step row kind at row {index}")

    seen_pairs: set[tuple[str, int]] = set()
    for index, value in enumerate(episodes):
        episode = _mapping(value, name=f"episodes[{index}]")
        condition = episode.get("condition")
        seed = episode.get("episode_seed")
        pair = (condition, seed)
        if pair != expected_pairs[index] or pair in seen_pairs:
            raise InvalidP4V2FGolden("golden episode order or identity differs")
        seen_pairs.add(pair)
        expected_keys = {
            "condition",
            "episode_seed",
            "logical_schedule_queries",
            "native_queries",
            "outcome",
            "queries",
        }
        if condition != "clean":
            expected_keys.add("schedule_sha256")
        if set(episode) != expected_keys:
            raise InvalidP4V2FGolden(f"episode keys differ: {pair!r}")
        outcome = _validate_outcome(episode["outcome"], name=f"episodes[{index}].outcome")
        native = _query_vector(episode["native_queries"], name=f"episodes[{index}].native")
        logical = _query_vector(
            episode["logical_schedule_queries"], name=f"episodes[{index}].logical"
        )
        total = _query_vector(episode["queries"], name=f"episodes[{index}].total")
        if native != native_by_pair[pair] or logical != logical_by_pair[pair]:
            raise InvalidP4V2FGolden(f"step-to-episode query ledger differs: {pair!r}")
        if total != _sum_queries(native, logical):
            raise InvalidP4V2FGolden(f"episode total query ledger differs: {pair!r}")
        if condition == "clean":
            if logical != _zero_queries():
                raise InvalidP4V2FGolden(f"clean logical query ledger differs: {seed}")
        elif (
            episode["schedule_sha256"] != schedule_hashes[seed]
            or logical != physical_queries[seed]
        ):
            raise InvalidP4V2FGolden(f"episode schedule binding differs: {pair!r}")
        rows = sorted(environment_rows[pair], key=lambda row: row["step_index"])
        if len(rows) != outcome["episode_length"] or [row["step_index"] for row in rows] != list(
            range(len(rows))
        ):
            raise InvalidP4V2FGolden(f"episode environment-step ledger differs: {pair!r}")
        if not rows:
            raise InvalidP4V2FGolden(f"episode contains no environment steps: {pair!r}")
        reconstructed = {
            "episode_return": sum(float(row["reward"]) for row in rows),
            "discounted_return": sum(
                (0.99**step) * float(row["reward"]) for step, row in enumerate(rows)
            ),
            "cumulative_safety_cost": sum(float(row["safety_cost"]) for row in rows),
            "discounted_safety_cost": sum(
                (0.99**step) * float(row["safety_cost"]) for step, row in enumerate(rows)
            ),
            "minimum_gap": min(float(row["min_gap"]) for row in rows),
            "minimum_ttc": min(float(row["minimum_ttc"]) for row in rows),
        }
        if any(not _close(float(outcome[key]), value) for key, value in reconstructed.items()):
            raise InvalidP4V2FGolden(f"episode numeric outcome ledger differs: {pair!r}")
        exact_outcomes = {
            "action_flips": sum(
                row["executed_action"] != row["local_clean_action"] for row in rows
            ),
            "collision": any(row["collision"] for row in rows),
            "merge_success": any(row["merge_success"] for row in rows),
            "missed_merge": any(row["missed_merge"] for row in rows),
            "near_miss": any(row["near_miss"] for row in rows),
            "near_miss_count": sum(bool(row["near_miss"]) for row in rows),
            "nonzero_steps": sum(bool(row["perturbation_nonzero"]) for row in rows),
            "selected_steps": sum(bool(row["selected"]) for row in rows),
            "terminated": rows[-1]["terminated"],
            "termination_reason": rows[-1]["termination_reason"],
            "truncated": rows[-1]["truncated"],
        }
        if any(outcome[key] != value for key, value in exact_outcomes.items()):
            raise InvalidP4V2FGolden(f"episode discrete outcome ledger differs: {pair!r}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def load_p4_v2f_golden(root: str | Path | None = None) -> P4V2FGoldenBundle:
    """Verify and load the frozen unified-development authority.

    ``root`` may point at a byte-identical mirror for verification tests or
    archival relocation.  It cannot authorize different content because the
    manifest and the four scientific payloads are pinned by exact SHA-256.
    """

    golden_root = _absolute(_repository_root() / GOLDEN_RELATIVE_ROOT if root is None else root)
    if not golden_root.is_dir() or _is_reparse(golden_root):
        raise InvalidP4V2FGolden("golden root must be a real directory")
    entries = {entry.name: entry for entry in golden_root.iterdir()}
    if set(entries) != _REQUIRED_FILES:
        raise InvalidP4V2FGolden("golden run must contain the exact 17-file set")
    for name, path in entries.items():
        if _is_reparse(path) or not path.is_file():
            raise InvalidP4V2FGolden(f"golden entry must be a regular file: {name}")

    actual_hashes = {name: _sha256_file(path) for name, path in entries.items()}
    for name, expected in _CORE_HASHES.items():
        if actual_hashes[name] != expected:
            raise InvalidP4V2FGolden(f"golden core SHA-256 differs: {name}")
    documents = {
        name: _strict_json_load_bytes(entries[name].read_bytes(), name=name)
        for name in _JSON_FILES
    }
    manifest = _validate_manifest(documents["manifest.json"])
    file_manifest = _mapping(manifest.get("files"), name="manifest.files")
    if set(file_manifest) != _REQUIRED_FILES - {"manifest.json"}:
        raise InvalidP4V2FGolden("golden file evidence set differs")
    evidence: dict[str, GoldenFileEvidence] = {
        "manifest.json": GoldenFileEvidence(
            sha256=GOLDEN_MANIFEST_SHA256,
            bytes=entries["manifest.json"].stat().st_size,
        )
    }
    for name, value in file_manifest.items():
        record = _mapping(value, name=f"manifest.files[{name!r}]")
        if set(record) != {"sha256", "bytes"}:
            raise InvalidP4V2FGolden(f"golden file evidence keys differ: {name}")
        if (
            record["sha256"] != actual_hashes[name]
            or type(record["bytes"]) is not int
            or record["bytes"] < 0
            or record["bytes"] != entries[name].stat().st_size
        ):
            raise InvalidP4V2FGolden(f"golden file evidence differs: {name}")
        evidence[name] = GoldenFileEvidence(
            sha256=record["sha256"], bytes=record["bytes"]
        )

    _validate_cross_file_authority(documents)
    schedule_hashes = _validate_schedules(documents["schedules.json"])
    _validate_episodes_and_ledgers(
        documents["episodes.json"],
        documents["steps.json"],
        schedule_hashes,
        documents["schedules.json"],
    )
    return P4V2FGoldenBundle(
        root=golden_root,
        manifest_sha256=GOLDEN_MANIFEST_SHA256,
        victim_checkpoint_sha256=GOLDEN_VICTIM_CHECKPOINT_SHA256,
        victim_policy_state_sha256=GOLDEN_VICTIM_POLICY_STATE_SHA256,
        episode_seeds=GOLDEN_EPISODE_SEEDS,
        conditions=GOLDEN_CONDITIONS,
        schedule_sha256_by_seed=MappingProxyType(dict(schedule_hashes)),
        file_evidence=MappingProxyType(dict(sorted(evidence.items()))),
        manifest=_freeze(manifest),
        schedules=_freeze(documents["schedules.json"]),
        steps=_freeze(documents["steps.json"]),
        episodes=_freeze(documents["episodes.json"]),
        summary=_freeze(documents["summary.json"]),
        comparison_table=_freeze(documents["comparison_table.json"]),
    )


__all__ = [
    "GOLDEN_CLAIMS",
    "GOLDEN_CONDITIONS",
    "GOLDEN_EPISODE_SEEDS",
    "GOLDEN_MANIFEST_SHA256",
    "GOLDEN_RELATIVE_ROOT",
    "GOLDEN_VICTIM_CHECKPOINT_SHA256",
    "GOLDEN_VICTIM_POLICY_STATE_SHA256",
    "GoldenFileEvidence",
    "InvalidP4V2FGolden",
    "P4V2FGoldenBundle",
    "load_p4_v2f_golden",
]
