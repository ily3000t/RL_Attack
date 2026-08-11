from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import PPO
from torch import Tensor, nn

from rl_attack.attacks.strong.stfa.action_factors import (
    ActionFactor,
    ActionFactorization,
    sumo_3x3_factorization,
)
from rl_attack.attacks.strong.stfa.contracts import (
    AttackStepContext,
    DirectorDecision,
    RNGNamespace,
)
from rl_attack.attacks.strong.stfa.sumo_v1 import (
    SumoMergeV1DiscretePlanner,
    SumoPhysicalBudgetsV1,
)
from rl_attack.cli.p4_audit import _parser as p4_audit_cli_parser
from rl_attack.experiments.p4_audit import (
    P4_AUDIT_SCHEMA_VERSION,
    P4_MERGELITE_ENVIRONMENT_REGISTRY,
    P4_PROJECTOR_GUARANTEE,
    P4_RNG_DERIVATION,
    P4_SUMO_DISCRETE_PLANNER,
    P4_SUMO_ENVIRONMENT_FACTORY,
    P4_SUMO_ENVIRONMENT_REGISTRY,
    P4_SUMO_ENVIRONMENT_TYPE,
    P4_SUMO_PROJECTOR_FACTORY,
    P4_SUMO_PROJECTOR_NAME,
    P4_SUMO_PROJECTOR_VERSION,
    AttackBuildContext,
    ClaimContext,
    InvalidP4Audit,
    OutputAliasError,
    ProjectorBuildContext,
    ScenarioAssetSpec,
    _configure_torch_threads,
    _execution_record,
    _parse_claim_context,
    _repository_provenance,
    _validate_claim_context,
    box_space_contract_sha256,
    build_stfa_attack,
    build_sumo_merge_v1_projector,
    discrete_space_contract_sha256,
    environment_contract_sha256,
    load_p4_audit_config,
    run_p4_audit,
    semantic_projector_contract_sha256,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_director import STFADirector, STFADirectorConfig


class TinyNineActionEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}
    action_labels = tuple(
        f"lat{lateral}_lon{longitudinal}"
        for lateral in (-1, 0, 1)
        for longitudinal in (-1, 0, 1)
    )

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(9)
        self._step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._step = 0
        offset = 0.0 if seed is None else (seed % 5) * 0.01
        return np.asarray([offset, 0.25], dtype=np.float32), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        self._step += 1
        observation = np.asarray(
            [0.05 * self._step, 0.25],
            dtype=np.float32,
        )
        reward = 1.0 - 0.01 * float(action)
        return observation, reward, self._step >= 2, False, {}


class NeverTerminatingTinyNineActionEnv(TinyNineActionEnv):
    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, _terminated, _truncated, info = super().step(action)
        return observation, reward, False, False, info


class UnboundedTinyNineActionEnv(TinyNineActionEnv):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _factorization() -> ActionFactorization:
    return ActionFactorization(
        name="toy_complete_3x3",
        actions=tuple(
            ActionFactor(
                index=index,
                lateral=lateral,
                longitudinal=longitudinal,
                label=f"lat{lateral}_lon{longitudinal}",
            )
            for index, (lateral, longitudinal) in enumerate(
                (lateral, longitudinal)
                for lateral in (-1, 0, 1)
                for longitudinal in (-1, 0, 1)
            )
        ),
    )


def _victim_record(checkpoint_sha: str, policy_sha: str) -> dict[str, Any]:
    return {
        "checkpoint_sha256": checkpoint_sha,
        "policy_state_sha256": policy_sha,
        "victim_action_mode": "deterministic",
        "frozen": True,
        "frozen_evidence": {
            "policy_training": False,
            "any_parameter_requires_grad": False,
            "policy_state_before_sha256": policy_sha,
            "policy_state_after_sha256": policy_sha,
        },
    }


def _write_sidecars(
    root: Path,
    *,
    factorization: ActionFactorization,
    victim_checkpoint_sha: str,
    victim_policy_sha: str,
    environment_contract_sha: str,
    normalization_contract_sha: str,
    cost_definition_sha: str,
    temporal_budget: dict[str, Any],
    horizon: int,
) -> dict[str, dict[str, str]]:
    critic = root / "critic.pt"
    critic.write_bytes(b"contract-test-critic")
    critic_sha = _sha256(critic)
    critic_state_sha = "a" * 64
    critic_space_sha = "1" * 64
    safety_dataset = {
        "schema_version": "p4-stfa-safety-dataset-binding-v1",
        "dataset_sha256": "2" * 64,
        "dataset_manifest_sha256": "3" * 64,
        "provenance_sha256": "4" * 64,
        "environment_contract_sha256": environment_contract_sha,
        "normalization_contract_sha256": normalization_contract_sha,
        "cost_definition_sha256": cost_definition_sha,
        "collector_contract_sha256": "5" * 64,
        "action_ontology_sha256": factorization.ontology_hash,
        "victim_checkpoint_sha256": victim_checkpoint_sha,
        "victim_policy_state_sha256": victim_policy_sha,
        "next_policy_probabilities_recomputed": True,
        "truncation_final_observation_declared": True,
    }
    critic_manifest = root / "critic.pt.manifest.json"
    critic_sidecar = {
        "schema_version": 1,
        "artifact_type": "stfa_safety_critic_checkpoint_manifest",
        "checkpoint": {"filename": critic.name, "sha256": critic_sha},
        "manifest": {
            "artifact_type": "stfa_safety_critic",
            "victim": _victim_record(victim_checkpoint_sha, victim_policy_sha),
            "critic": {"state_sha256": critic_state_sha},
            "space": {
                "observation_shape": [2],
                "observation_dtype": "float32",
                "n_actions": 9,
                "action_indexing": "zero_based_discrete",
                "action_ontology_sha256": factorization.ontology_hash,
                "sha256": critic_space_sha,
            },
            "dataset": safety_dataset,
        },
    }
    critic_manifest.write_text(
        json.dumps(critic_sidecar, sort_keys=True),
        encoding="utf-8",
    )

    director = root / "director.pt"
    director.write_bytes(b"contract-test-director")
    director_sha = _sha256(director)
    director_manifest = root / "director.pt.manifest.json"
    director_sidecar = {
        "schema_version": 1,
        "artifact_type": "stfa_director_checkpoint_manifest",
        "checkpoint": {"filename": director.name, "sha256": director_sha},
        "manifest": {
            "artifact_type": "stfa_learned_director",
            "victim": _victim_record(victim_checkpoint_sha, victim_policy_sha),
            "factorization": {
                "ontology_sha256": factorization.ontology_hash,
                "contract_sha256": factorization.contract_hash,
            },
            "safety_critic": {
                "artifact_type": "stfa_safety_critic",
                "checkpoint_sha256": critic_sha,
                "state_sha256": critic_state_sha,
                "space_sha256": critic_space_sha,
                "victim_checkpoint_sha256": victim_checkpoint_sha,
                "victim_policy_state_sha256": victim_policy_sha,
                "dataset_manifest_sha256": safety_dataset[
                    "dataset_manifest_sha256"
                ],
                "environment_contract_sha256": environment_contract_sha,
                "normalization_contract_sha256": normalization_contract_sha,
                "cost_definition_sha256": cost_definition_sha,
                "trained": True,
            },
            "dataset": {
                "schema_version": "p4-stfa-director-dataset-binding-v1",
                "dataset_sha256": "6" * 64,
                "dataset_manifest_sha256": "7" * 64,
                "provenance_sha256": "8" * 64,
                "environment_contract_sha256": environment_contract_sha,
                "normalization_contract_sha256": normalization_contract_sha,
                "collector_contract_sha256": "9" * 64,
                "action_ontology_sha256": factorization.ontology_hash,
                "victim_checkpoint_sha256": victim_checkpoint_sha,
                "victim_policy_state_sha256": victim_policy_sha,
                "safety_critic_checkpoint_sha256": critic_sha,
                "safety_critic_state_sha256": critic_state_sha,
                "safety_critic_space_sha256": critic_space_sha,
                "temporal_budget": temporal_budget,
                "horizon": horizon,
                "labeler_contract_sha256": "b" * 64,
                "victim_probabilities_recomputed": True,
                "safety_costs_recomputed": True,
            },
        },
    }
    director_manifest.write_text(
        json.dumps(director_sidecar, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "safety_critic": {
            "checkpoint": critic.name,
            "checkpoint_sha256": critic_sha,
            "manifest": critic_manifest.name,
            "manifest_sha256": _sha256(critic_manifest),
            "artifact_type": "stfa_safety_critic_checkpoint_manifest",
        },
        "director": {
            "checkpoint": director.name,
            "checkpoint_sha256": director_sha,
            "manifest": director_manifest.name,
            "manifest_sha256": _sha256(director_manifest),
            "artifact_type": "stfa_director_checkpoint_manifest",
        },
    }


def _factorization_record(factorization: ActionFactorization) -> dict[str, Any]:
    return {
        "name": factorization.name,
        "version": factorization.version,
        "actions": [
            {
                "index": action.index,
                "lateral": action.lateral,
                "longitudinal": action.longitudinal,
                "label": action.label,
                "available": action.available,
            }
            for action in factorization.actions
        ],
        "ontology_sha256": factorization.ontology_hash,
        "contract_sha256": factorization.contract_hash,
    }


def _make_config(
    tmp_path: Path,
    *,
    environment_type: type[TinyNineActionEnv] = TinyNineActionEnv,
) -> tuple[Path, ActionFactorization]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    factorization = _factorization()
    victim_env = environment_type()
    victim = PPO(
        "MlpPolicy",
        victim_env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        policy_kwargs={"net_arch": [8]},
        seed=19,
        device="cpu",
    )
    checkpoint = tmp_path / "victim.zip"
    victim.save(checkpoint)
    checkpoint_sha = _sha256(checkpoint)
    policy_sha = sb3_policy_state_sha256(victim)
    observation_low = np.asarray(
        victim_env.observation_space.low,
        dtype=np.float64,
    ).tolist()
    observation_high = np.asarray(
        victim_env.observation_space.high,
        dtype=np.float64,
    ).tolist()
    victim_env.close()

    projector_config = tmp_path / "projector.yaml"
    projector_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "rl_attack.p4_policy_input_projector.v1",
                "name": "toy_policy_input",
                "observation_shape": [2],
                "epsilon": [0.0, 0.0],
                "lower": [-1.0, -1.0],
                "upper": [1.0, 1.0],
                "mutable_mask": [False, False],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    observation_contract = box_space_contract_sha256(
        shape=(2,),
        dtype="float32",
        low=observation_low,
        high=observation_high,
    )
    action_contract = discrete_space_contract_sha256(
        n=9,
        start=0,
        dtype="int64",
        factorization_contract_sha256=factorization.contract_hash,
    )
    normalization_contract = "c" * 64
    cost_definition = "d" * 64
    temporal_budget = {
        "k": 1,
        "min_gap": 0,
        "window_size": None,
        "window_k": None,
    }
    runtime_type = (
        f"{environment_type.__module__}.{environment_type.__qualname__}"
    )
    environment_contract = environment_contract_sha256(
        environment_id="TinyNineAction-v0",
        max_episode_steps=2,
        registry_key="gymnasium_make_v1",
        factory="gymnasium:make",
        runtime_type=runtime_type,
        observation_space_contract_sha256=observation_contract,
        action_space_contract_sha256=action_contract,
        normalization_contract_sha256=normalization_contract,
        scenario_assets=[],
    )
    artifacts = _write_sidecars(
        tmp_path,
        factorization=factorization,
        victim_checkpoint_sha=checkpoint_sha,
        victim_policy_sha=policy_sha,
        environment_contract_sha=environment_contract,
        normalization_contract_sha=normalization_contract,
        cost_definition_sha=cost_definition,
        temporal_budget=temporal_budget,
        horizon=2,
    )
    projector_sha = _sha256(projector_config)
    projector_factory = (
        "rl_attack.experiments.p4_audit:"
        "build_policy_input_projector"
    )
    projector_contract = semantic_projector_contract_sha256(
        name="toy_policy_input",
        version="contract-test-v1",
        factory=projector_factory,
        factory_kwargs={},
        observation_shape=(2,),
        config_sha256=projector_sha,
        guarantee=P4_PROJECTOR_GUARANTEE,
    )
    config = {
        "schema_version": P4_AUDIT_SCHEMA_VERSION,
        "name": "tiny-nine-action-stfa-contract",
        "environment": {
            "id": "TinyNineAction-v0",
            "max_episode_steps": 2,
            "registry_key": "gymnasium_make_v1",
            "factory": "gymnasium:make",
            "runtime_type": runtime_type,
            "contract_sha256": environment_contract,
            "normalization_contract_sha256": normalization_contract,
            "scenario_assets": [],
            "observation_space": {
                "type": "Box",
                "shape": [2],
                "dtype": "float32",
                "low": observation_low,
                "high": observation_high,
                "contract_sha256": observation_contract,
            },
            "action_space": {
                "type": "Discrete",
                "n": 9,
                "start": 0,
                "dtype": "int64",
                "contract_sha256": action_contract,
            },
        },
        "victim": {
            "name": "tiny-real-sb3-ppo",
            "algorithm": "stable_baselines3.PPO",
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha,
            "policy_state_sha256": policy_sha,
        },
        "action_factorization": _factorization_record(factorization),
        "semantic_projector": {
            "name": "toy_policy_input",
            "version": "contract-test-v1",
            "factory": projector_factory,
            "factory_kwargs": {},
            "observation_shape": [2],
            "config": projector_config.name,
            "config_sha256": projector_sha,
            "contract_sha256": projector_contract,
            "guarantee": P4_PROJECTOR_GUARANTEE,
        },
        "safety": {
            "cost_definition_sha256": cost_definition,
        },
        "artifacts": artifacts,
        "attack": {
            "name": "stfa",
            "factory": "rl_attack.experiments.p4_audit:build_stfa_attack",
            "factory_kwargs": {
                "steps": 1,
                "restarts": 1,
                "random_start": False,
            },
            "temporal_budget": temporal_budget,
            "discrete_planner": {
                "registry_key": "disabled",
                "allowlist": [],
            },
        },
        "fairness": {
            "episode_seeds": [5, 7],
            "attack_base_seed": 301,
            "paired_clean_attacked": True,
            "victim_action_mode": "deterministic_argmax",
            "rng_derivation": P4_RNG_DERIVATION,
        },
        "evidence_scope": {
            "algorithm_contract": True,
            "sb3_9action_integration": True,
            "sumo_contract_integration": False,
            "sumo_empirical_effectiveness": False,
            "sumo_empirical_effectiveness_reason": (
                "No stable SUMO PPO victim checkpoint is available in P4."
            ),
        },
    }
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return config_path, factorization


class FakeSafetyCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("costs", torch.arange(9, dtype=torch.float32))
        self.eval()

    def forward(self, observation: Tensor) -> Tensor:
        return self.costs[None, :].expand(observation.shape[0], -1)


class DifferentActionDirector:
    def __init__(self, factorization: ActionFactorization) -> None:
        self.factorization = factorization

    def decide(
        self,
        context: AttackStepContext,
        **_kwargs: Any,
    ) -> DirectorDecision:
        target = (context.clean_action + 1) % self.factorization.n_actions
        factor = self.factorization.decode(target)
        return DirectorDecision(
            selected=True,
            target_action=target,
            target_lateral=factor.lateral,
            target_longitudinal=factor.longitudinal,
            score=1.0,
            available_action_mask=context.available_action_mask,
            metadata={"director": "different-action-contract-test"},
        )


def _artifact_loader(factorization: ActionFactorization):
    def load(_context: object) -> dict[str, object]:
        return {
            "safety_critic": FakeSafetyCritic(),
            "director": DifferentActionDirector(factorization),
        }

    return load


def _real_director_artifact_loader(factorization: ActionFactorization):
    def load(_context: object) -> dict[str, object]:
        director = STFADirector(
            STFADirectorConfig(
                observation_shape=(2,),
                n_actions=factorization.n_actions,
                hidden_sizes=(8,),
                selection_threshold=0.0,
            ),
            factorization,
        )
        director.eval()
        for parameter in director.parameters():
            parameter.requires_grad_(False)
        return {
            "safety_critic": FakeSafetyCritic(),
            "director": director,
        }

    return load


def _mutate_embedded_manifest(
    config_path: Path,
    role: str,
    mutation: Any,
) -> None:
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sidecar_path = config_path.parent / values["artifacts"][role]["manifest"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    mutation(sidecar["manifest"])
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True),
        encoding="utf-8",
    )
    values["artifacts"][role]["manifest_sha256"] = _sha256(sidecar_path)
    config_path.write_text(
        yaml.safe_dump(values, sort_keys=False),
        encoding="utf-8",
    )


def test_real_sb3_nine_action_paired_hard_k_audit_distinguishes_target(
    tmp_path: Path,
) -> None:
    config_path, factorization = _make_config(tmp_path)
    output = tmp_path / "output"
    manifest = run_p4_audit(
        config_path,
        output_directory=output,
        environment_factory=TinyNineActionEnv,
        artifact_loader=_artifact_loader(factorization),
    )

    assert manifest["status"] == "complete"
    assert manifest["test_scope"] is True
    assert manifest["robust_summary_eligible"] is False
    assert manifest["robust_summary_eligibility_meaning"] == (
        "bundle_integrity_only_not_formal_robustness"
    )
    assert manifest["claim_context"] == dataclasses.asdict(ClaimContext())
    assert manifest["execution"] == {
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    assert manifest["provenance"]["torch_num_threads"] == (
        manifest["execution"]["torch_num_threads"]
    )
    assert manifest["provenance"]["torch_num_interop_threads"] == (
        manifest["execution"]["torch_num_interop_threads"]
    )
    assert manifest["dependency_injection"] == [
        "environment_factory",
        "artifact_loader",
    ]
    assert manifest["audit"]["attack_probability_used"] is False
    assert manifest["audit"]["timing"] == {
        "mode": "director",
        "selection_rule": "learned_director_subject_to_hard_K_ledger",
        "bernoulli_selection_used": False,
        "random_selection_probability": None,
    }
    assert manifest["discrete_planner"] == {
        "enabled": False,
        "registry_key": "disabled",
        "allowlist": [],
        "discrete_budget": 0,
        "max_candidates": 0,
        "formal_sumo_evidence": False,
    }
    assert "summary" not in manifest
    assert manifest["evidence_scope"] == {
        "algorithm_contract": True,
        "sb3_9action_integration": True,
        "sumo_contract_integration": False,
        "sumo_empirical_effectiveness": False,
        "sumo_empirical_effectiveness_reason": (
            "No stable SUMO PPO victim checkpoint is available in P4."
        ),
    }
    totals = manifest["accounting"]
    assert totals["steps"] == 4
    assert totals["selected"] == 2
    assert totals["nonzero"] == 0
    assert totals["target_declared"] == 2
    assert totals["target_hit"] == 0
    assert totals["action_flip"] == 0
    assert totals["observation_queries"] == 2
    assert totals["critic_queries"] == 2
    assert totals["director_queries"] == 2
    assert totals["projection_queries"] == 2
    assert totals["discrete_edit_count"] == 0
    assert totals["discrete_cost"] == 0
    assert totals["discrete_candidates_planned"] == 0
    assert totals["discrete_candidates_evaluated"] == 0
    assert totals["discrete_candidate_selected"] == 0
    assert totals["discrete_common_random_number_steps"] == 0

    episodes = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
    assert [row["episode_seed"] for row in episodes["clean"]] == [5, 7]
    assert [row["episode_seed"] for row in episodes["attacked"]] == [5, 7]
    for episode in episodes["attacked"]:
        assert episode["temporal_ledger"]["selected_steps"] == [0]
        assert episode["temporal_ledger"]["consumed"] == 1
        selected = [step for step in episode["steps"] if step["selected"]]
        assert len(selected) == 1
        assert selected[0]["target_action"] != selected[0]["actual_adversarial_action"]
        assert selected[0]["target_hit"] is False
        assert selected[0]["action_flip"] is False
        assert all(
            step["discrete_search_scope"] == "disabled"
            and step["discrete_candidates_planned"] == 0
            and step["discrete_candidates_evaluated"] == 0
            for step in episode["steps"]
        )

    for name in (
        "resolved_config.json",
        "episodes.json",
        "integration_results.json",
        "manifest.json",
    ):
        parsed = json.loads(
            (output / name).read_text(encoding="utf-8"),
            parse_constant=lambda value: pytest.fail(f"non-finite JSON: {value}"),
        )
        assert parsed is not None
    assert manifest["victim"]["policy_state_sha256_before"] == manifest["victim"][
        "policy_state_sha256_after"
    ]


def test_official_audit_preserves_real_learned_director_signature(
    tmp_path: Path,
) -> None:
    config_path, factorization = _make_config(tmp_path)

    manifest = run_p4_audit(
        config_path,
        output_directory=tmp_path / "real_director_output",
        environment_factory=TinyNineActionEnv,
        artifact_loader=_real_director_artifact_loader(factorization),
    )

    assert manifest["status"] == "complete"
    assert manifest["accounting"]["director_queries"] == 2
    assert manifest["accounting"]["selected"] == 2


def test_invalid_attack_metadata_publishes_no_robust_summary(tmp_path: Path) -> None:
    config_path, factorization = _make_config(tmp_path)
    output = tmp_path / "invalid-output"

    def corrupting_factory(context: object) -> object:
        inner = build_stfa_attack(context)  # type: ignore[arg-type]

        class CorruptingAttack:
            temporal_ledger = inner.temporal_ledger

            def generate(self, *args: Any, **kwargs: Any) -> object:
                result = inner.generate(*args, **kwargs)
                metadata = dict(result.metadata)
                metadata.pop("ledger_nonzero_after")
                return dataclasses.replace(result, metadata=metadata)

        return CorruptingAttack()

    with pytest.raises(InvalidP4Audit) as raised:
        run_p4_audit(
            config_path,
            output_directory=output,
            environment_factory=TinyNineActionEnv,
            artifact_loader=_artifact_loader(factorization),
            attack_factory=corrupting_factory,
        )
    assert raised.value.code == "attack_metadata_invalid"
    assert raised.value.manifest is not None
    assert raised.value.manifest["status"] == "invalid"
    assert raised.value.manifest["robust_summary_eligible"] is False
    assert {path.name for path in output.iterdir()} == {
        "resolved_config.json",
        "manifest.json",
    }
    invalid = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert invalid["invalid_reason"]["code"] == "attack_metadata_invalid"
    assert invalid["claim_context"] == dataclasses.asdict(ClaimContext())
    assert invalid["execution"] == {
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    assert "summary" not in invalid


def test_artifact_hash_mismatch_fails_closed_without_summaries(
    tmp_path: Path,
) -> None:
    config_path, factorization = _make_config(tmp_path)
    (tmp_path / "critic.pt").write_bytes(b"tampered")
    output = tmp_path / "tampered-output"
    with pytest.raises(InvalidP4Audit, match="checkpoint SHA-256 mismatch"):
        run_p4_audit(
            config_path,
            output_directory=output,
            environment_factory=TinyNineActionEnv,
            artifact_loader=_artifact_loader(factorization),
        )
    assert {path.name for path in output.iterdir()} == {
        "resolved_config.json",
        "manifest.json",
    }
    invalid = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert invalid["robust_summary_eligible"] is False


def test_schema_rejects_attack_probability_and_duplicate_keys(tmp_path: Path) -> None:
    config_path, _ = _make_config(tmp_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["attack"]["attack_probability"] = 1.0
    config_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_p4_audit_config(config_path)

    config_path, _ = _make_config(tmp_path / "random-timing")
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["attack"]["factory_kwargs"]["timing_mode"] = "random"
    values["attack"]["factory_kwargs"]["random_selection_probability"] = 0.5
    config_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="timing_mode='director'"):
        load_p4_audit_config(config_path)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "schema_version: one\nschema_version: two\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate YAML"):
        load_p4_audit_config(duplicate)


def test_claim_context_defaults_conservatively_and_screening_is_strict() -> None:
    assert _parse_claim_context(None) == ClaimContext()
    screening = {
        "claim_tier": "screening",
        "task_scope": "synthetic_repository_owned",
        "formal_statistical_claim": False,
        "victim_training_seed_count": 1,
        "matched_baseline_comparison_completed": False,
        "sumo_evidence": False,
        "p5_authorized": False,
        "preparation_contract_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
    }
    parsed = _parse_claim_context(screening)
    assert parsed.claim_tier == "screening"
    assert parsed.formal_statistical_claim is False
    assert parsed.matched_baseline_comparison_completed is False
    assert parsed.sumo_evidence is False
    assert parsed.p5_authorized is False

    formal = dict(screening)
    formal["formal_statistical_claim"] = True
    with pytest.raises(ValueError, match="cannot assert formal"):
        _parse_claim_context(formal)
    unbound = dict(screening)
    unbound["preparation_contract_sha256"] = None
    with pytest.raises(ValueError, match="exact preparation/protocol hashes"):
        _parse_claim_context(unbound)
    unspecified = dict(screening)
    unspecified.update(
        {
            "claim_tier": "unspecified",
            "task_scope": "unspecified",
            "victim_training_seed_count": 0,
            "preparation_contract_sha256": None,
            "protocol_sha256": None,
        }
    )
    assert _parse_claim_context(unspecified) == ClaimContext()

    with pytest.raises(ValueError, match="cannot assert formal"):
        _validate_claim_context(
            dataclasses.replace(ClaimContext(), formal_statistical_claim=True)
        )


def test_repository_provenance_has_exact_git_schema_and_unavailable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_keys = {
        "python_implementation",
        "python_version",
        "platform",
        "packages",
        "repository_root",
        "git_commit",
        "git_dirty",
        "git_status_lines",
        "git_status",
        "git_error",
        "torch_num_threads",
        "torch_num_interop_threads",
    }
    calls: list[list[str]] = []

    def successful_run(command: list[str], **kwargs: Any) -> object:
        calls.append(command)
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": 10.0,
        }
        if command[-2:] == ["rev-parse", "--show-toplevel"]:
            stdout = str(tmp_path)
        elif command[-2:] == ["rev-parse", "HEAD"]:
            stdout = "a" * 40
        else:
            assert command[-3:] == [
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ]
            stdout = "?? zeta\n M alpha\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", successful_run)
    available = _repository_provenance()
    assert set(available) == expected_keys
    assert available["repository_root"] == str(tmp_path.resolve())
    assert available["git_commit"] == "a" * 40
    assert available["git_dirty"] is True
    assert available["git_status_lines"] == [" M alpha", "?? zeta"]
    assert available["git_status"] == "available"
    assert available["git_error"] is None
    assert available["torch_num_threads"] == torch.get_num_threads()
    assert available["torch_num_interop_threads"] == (
        torch.get_num_interop_threads()
    )
    assert len(calls) == 3
    assert all(command[:2] == ["git", "-C"] for command in calls)

    def unavailable_run(_command: list[str], **_kwargs: Any) -> object:
        raise OSError("git is unavailable")

    monkeypatch.setattr(subprocess, "run", unavailable_run)
    unavailable = _repository_provenance()
    assert set(unavailable) == expected_keys
    assert unavailable["repository_root"] is None
    assert unavailable["git_commit"] is None
    assert unavailable["git_dirty"] is None
    assert unavailable["git_status_lines"] == []
    assert unavailable["git_status"] == "unavailable"
    assert unavailable["git_error"] == "OSError: git is unavailable"


def test_torch_thread_contract_and_cli_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"intraop": 4, "interop": 3}
    monkeypatch.setattr(torch, "get_num_threads", lambda: state["intraop"])
    monkeypatch.setattr(
        torch,
        "get_num_interop_threads",
        lambda: state["interop"],
    )
    monkeypatch.setattr(
        torch,
        "set_num_threads",
        lambda value: state.__setitem__("intraop", value),
    )
    monkeypatch.setattr(
        torch,
        "set_num_interop_threads",
        lambda value: state.__setitem__("interop", value),
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "test-original")
    monkeypatch.setenv("MKL_NUM_THREADS", "test-original")

    _configure_torch_threads(2)
    assert _execution_record("cpu") == {
        "device": "cpu",
        "torch_num_threads": 2,
        "torch_num_interop_threads": 1,
    }
    with pytest.raises(TypeError, match="integer or None"):
        _configure_torch_threads(True)
    with pytest.raises(ValueError, match="positive"):
        _configure_torch_threads(0)

    parser = p4_audit_cli_parser()
    assert parser.parse_args(["config.yaml"]).torch_threads is None
    assert parser.parse_args(
        ["config.yaml", "--torch-threads", "1"]
    ).torch_threads == 1
    with pytest.raises(SystemExit):
        parser.parse_args(["config.yaml", "--torch-threads", "0"])


def test_direct_config_replacements_fail_before_any_output(tmp_path: Path) -> None:
    config_path, _ = _make_config(tmp_path)
    loaded = load_p4_audit_config(config_path)
    replacements = {
        "claim": dataclasses.replace(
            loaded,
            claim_context=dataclasses.replace(
                loaded.claim_context,
                formal_statistical_claim=True,
            ),
        ),
        "evidence": dataclasses.replace(
            loaded,
            evidence_scope=dataclasses.replace(
                loaded.evidence_scope,
                sumo_contract_integration=True,
            ),
        ),
        "fairness": dataclasses.replace(
            loaded,
            fairness=dataclasses.replace(
                loaded.fairness,
                paired_clean_attacked=False,
            ),
        ),
        "projector": dataclasses.replace(
            loaded,
            projector=dataclasses.replace(
                loaded.projector,
                factory="unregistered.module:projector",
            ),
        ),
    }
    for name, replaced in replacements.items():
        output = tmp_path / f"direct-{name}-output"
        with pytest.raises(ValueError):
            run_p4_audit(replaced, output_directory=output)
        assert not output.exists()


def test_loaded_config_rejects_source_rewrite_even_if_hash_is_replaced(
    tmp_path: Path,
) -> None:
    config_path, _ = _make_config(tmp_path)
    loaded = load_p4_audit_config(config_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["name"] = "rewritten_after_load"
    config_path.write_text(
        yaml.safe_dump(values, sort_keys=False),
        encoding="utf-8",
    )

    stale_output = tmp_path / "stale-source-output"
    with pytest.raises(ValueError, match="source config SHA-256 mismatch"):
        run_p4_audit(loaded, output_directory=stale_output)
    assert not stale_output.exists()

    hash_replaced = dataclasses.replace(
        loaded,
        config_sha256=_sha256(config_path),
    )
    replaced_output = tmp_path / "hash-replaced-output"
    with pytest.raises(ValueError, match="differs from its freshly parsed source"):
        run_p4_audit(hash_replaced, output_directory=replaced_output)
    assert not replaced_output.exists()


def test_source_config_mutation_during_execution_fails_pinned_rehash(
    tmp_path: Path,
) -> None:
    config_path, factorization = _make_config(tmp_path)
    mutated = False

    def mutating_factory(context: AttackBuildContext) -> object:
        nonlocal mutated
        if not mutated:
            config_path.write_text(
                config_path.read_text(encoding="utf-8") + "\n# runtime mutation\n",
                encoding="utf-8",
            )
            mutated = True
        return build_stfa_attack(context)

    output = tmp_path / "runtime-source-mutation-output"
    with pytest.raises(InvalidP4Audit, match="source config SHA-256 mismatch"):
        run_p4_audit(
            config_path,
            output_directory=output,
            environment_factory=TinyNineActionEnv,
            artifact_loader=_artifact_loader(factorization),
            attack_factory=mutating_factory,
        )
    invalid = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert invalid["status"] == "invalid"
    assert invalid["claim_context"] == dataclasses.asdict(ClaimContext())
    assert "summary" not in invalid


def test_output_overwrite_and_input_alias_guards(tmp_path: Path) -> None:
    config_path, factorization = _make_config(tmp_path)
    with pytest.raises(OutputAliasError):
        run_p4_audit(
            config_path,
            output_directory=tmp_path,
            overwrite=True,
            environment_factory=TinyNineActionEnv,
            artifact_loader=_artifact_loader(factorization),
        )

    output = tmp_path / "output"
    first = run_p4_audit(
        config_path,
        output_directory=output,
        environment_factory=TinyNineActionEnv,
        artifact_loader=_artifact_loader(factorization),
    )
    with pytest.raises(FileExistsError):
        run_p4_audit(
            config_path,
            output_directory=output,
            environment_factory=TinyNineActionEnv,
            artifact_loader=_artifact_loader(factorization),
        )
    second = run_p4_audit(
        config_path,
        output_directory=output,
        overwrite=True,
        environment_factory=TinyNineActionEnv,
        artifact_loader=_artifact_loader(factorization),
    )
    assert first["status"] == second["status"] == "complete"
    assert not any(".stage-" in path.name for path in tmp_path.iterdir())
    assert not any(".backup-" in path.name for path in tmp_path.iterdir())


def test_audit_uses_episode_context_named_streams_without_shared_generator(
    tmp_path: Path,
) -> None:
    config_path, factorization = _make_config(tmp_path)
    keyword_records: list[dict[str, Any]] = []

    def recording_factory(context: AttackBuildContext) -> object:
        inner = build_stfa_attack(context)

        class RecordingAttack:
            temporal_ledger = inner.temporal_ledger

            def generate(self, *args: Any, **kwargs: Any) -> object:
                keyword_records.append(dict(kwargs))
                return inner.generate(*args, **kwargs)

        return RecordingAttack()

    manifest = run_p4_audit(
        config_path,
        output_directory=tmp_path / "named-stream-output",
        environment_factory=TinyNineActionEnv,
        artifact_loader=_artifact_loader(factorization),
        attack_factory=recording_factory,
    )
    assert manifest["status"] == "complete"
    assert keyword_records == [{}, {}, {}, {}]


def test_audit_time_limit_is_visible_to_transition_callback(
    tmp_path: Path,
) -> None:
    config_path, factorization = _make_config(
        tmp_path,
        environment_type=NeverTerminatingTinyNineActionEnv,
    )
    transitions: list[dict[str, Any]] = []

    def tracking_factory(context: AttackBuildContext) -> object:
        inner = build_stfa_attack(context)

        class TrackingAttack:
            temporal_ledger = inner.temporal_ledger

            def generate(self, *args: Any, **kwargs: Any) -> object:
                return inner.generate(*args, **kwargs)

            def observe_transition(self, **kwargs: Any) -> None:
                transitions.append(dict(kwargs))

        return TrackingAttack()

    output = tmp_path / "time-limit-output"
    manifest = run_p4_audit(
        config_path,
        output_directory=output,
        environment_factory=NeverTerminatingTinyNineActionEnv,
        artifact_loader=_artifact_loader(factorization),
        attack_factory=tracking_factory,
    )
    assert manifest["status"] == "complete"
    assert len(transitions) == 4
    assert [item["truncated"] for item in transitions] == [
        False,
        True,
        False,
        True,
    ]
    for item in (transitions[1], transitions[3]):
        assert item["terminated"] is False
        assert item["info"]["audit_time_limit"] is True
        assert item["info"]["TimeLimit.truncated"] is True
    episodes = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
    assert all(
        row["terminated"] is False
        and row["truncated"] is True
        and row["audit_time_limit"] is True
        for row in episodes["attacked"]
    )


@pytest.mark.parametrize(
    "field",
    ["k", "min_gap", "window", "horizon"],
)
def test_director_dataset_temporal_binding_mismatch_fails_closed(
    tmp_path: Path,
    field: str,
) -> None:
    config_path, factorization = _make_config(tmp_path / field)

    def mutate(manifest: dict[str, Any]) -> None:
        dataset = manifest["dataset"]
        if field == "k":
            dataset["temporal_budget"]["k"] = 2
        elif field == "min_gap":
            dataset["temporal_budget"]["min_gap"] = 1
        elif field == "window":
            dataset["temporal_budget"]["window_size"] = 2
            dataset["temporal_budget"]["window_k"] = 1
        else:
            dataset["horizon"] = 3

    _mutate_embedded_manifest(config_path, "director", mutate)
    with pytest.raises(
        InvalidP4Audit,
        match="director dataset differs",
    ):
        run_p4_audit(
            config_path,
            output_directory=tmp_path / f"{field}-invalid",
            environment_factory=TinyNineActionEnv,
            artifact_loader=_artifact_loader(factorization),
        )


@pytest.mark.parametrize(
    ("role", "mutation"),
    [
        (
            "safety_critic",
            lambda manifest: manifest["dataset"].__setitem__(
                "normalization_contract_sha256",
                "e" * 64,
            ),
        ),
        (
            "safety_critic",
            lambda manifest: manifest["dataset"].__setitem__(
                "cost_definition_sha256",
                "e" * 64,
            ),
        ),
        (
            "director",
            lambda manifest: manifest["dataset"].__setitem__(
                "environment_contract_sha256",
                "e" * 64,
            ),
        ),
        (
            "director",
            lambda manifest: manifest["safety_critic"].__setitem__(
                "cost_definition_sha256",
                "e" * 64,
            ),
        ),
    ],
)
def test_artifact_contract_hash_binding_mismatch_fails_closed(
    tmp_path: Path,
    role: str,
    mutation: Any,
) -> None:
    config_path, factorization = _make_config(tmp_path / role)
    _mutate_embedded_manifest(config_path, role, mutation)
    with pytest.raises(InvalidP4Audit):
        run_p4_audit(
            config_path,
            output_directory=tmp_path / f"{role}-contract-invalid",
            environment_factory=TinyNineActionEnv,
            artifact_loader=_artifact_loader(factorization),
        )


def test_official_runtime_director_dataset_binding_is_revalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_attack.experiments.p4_audit as audit_module

    config_path, factorization = _make_config(tmp_path)

    def mismatched_loader(_context: object) -> dict[str, object]:
        director = DifferentActionDirector(factorization)
        director._dataset_binding = {"schema_version": "wrong"}  # type: ignore[attr-defined]
        return {
            "safety_critic": FakeSafetyCritic(),
            "director": director,
        }

    monkeypatch.setattr(
        audit_module,
        "_default_artifact_loader",
        mismatched_loader,
    )
    with pytest.raises(
        InvalidP4Audit,
        match="runtime director dataset binding differs",
    ):
        run_p4_audit(
            config_path,
            output_directory=tmp_path / "runtime-binding-invalid",
            environment_factory=TinyNineActionEnv,
        )


def test_infinite_box_bounds_are_stable_exact_and_nan_is_rejected(
    tmp_path: Path,
) -> None:
    config_path, factorization = _make_config(
        tmp_path,
        environment_type=UnboundedTinyNineActionEnv,
    )
    first_hash = box_space_contract_sha256(
        shape=(2,),
        dtype="float32",
        low=[-np.inf, -np.inf],
        high=[np.inf, np.inf],
    )
    second_hash = box_space_contract_sha256(
        shape=(2,),
        dtype="float32",
        low=np.asarray([-np.inf, -np.inf], dtype=np.float32),
        high=np.asarray([np.inf, np.inf], dtype=np.float32),
    )
    assert first_hash == second_hash
    manifest = run_p4_audit(
        config_path,
        output_directory=tmp_path / "unbounded-output",
        environment_factory=UnboundedTinyNineActionEnv,
        artifact_loader=_artifact_loader(factorization),
    )
    assert manifest["status"] == "complete"
    resolved = json.loads(
        (tmp_path / "unbounded-output" / "resolved_config.json").read_text(
            encoding="utf-8"
        )
    )
    observation = resolved["environment"]["observation_space"]
    assert observation["low"] == [
        "__negative_infinity__",
        "__negative_infinity__",
    ]
    assert observation["high"] == [
        "__positive_infinity__",
        "__positive_infinity__",
    ]
    with pytest.raises(ValueError, match="NaN"):
        box_space_contract_sha256(
            shape=(2,),
            dtype="float32",
            low=[np.nan, -np.inf],
            high=[np.inf, np.inf],
        )


def test_runtime_environment_exact_type_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    config_path, factorization = _make_config(tmp_path)
    with pytest.raises(InvalidP4Audit, match="exact type"):
        run_p4_audit(
            config_path,
            output_directory=tmp_path / "wrong-runtime-type",
            environment_factory=NeverTerminatingTinyNineActionEnv,
            artifact_loader=_artifact_loader(factorization),
        )


def test_non_sumo_positive_discrete_budget_is_rejected_by_schema(
    tmp_path: Path,
) -> None:
    config_path, _ = _make_config(tmp_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["attack"]["factory_kwargs"]["discrete_budget"] = 1
    values["attack"]["factory_kwargs"]["max_candidates"] = 4
    values["attack"]["discrete_planner"] = {
        "registry_key": P4_SUMO_DISCRETE_PLANNER,
        "allowlist": [2],
    }
    config_path.write_text(
        yaml.safe_dump(values, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact registered SUMO"):
        load_p4_audit_config(config_path)


def test_mergelite_registry_rejects_non_exact_factory_and_runtime(
    tmp_path: Path,
) -> None:
    config_path, _ = _make_config(tmp_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["environment"]["registry_key"] = P4_MERGELITE_ENVIRONMENT_REGISTRY
    config_path.write_text(
        yaml.safe_dump(values, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact registered"):
        load_p4_audit_config(config_path)


def test_production_factory_builds_only_registry_bound_sumo_planner(
    tmp_path: Path,
) -> None:
    base_path, _ = _make_config(tmp_path / "base")
    base = load_p4_audit_config(base_path)
    factorization = sumo_3x3_factorization()

    scenario_dir = base_path.parent / "scenario"
    scenario_dir.mkdir()
    assets: list[ScenarioAssetSpec] = []
    asset_records: list[dict[str, str]] = []
    for role, filename in (
        ("sumocfg", "highway_merge.sumocfg"),
        ("net", "highway_merge.net.xml"),
        ("route", "highway_merge.rou.xml"),
    ):
        path = scenario_dir / filename
        path.write_text(f"<{role}/>", encoding="utf-8")
        configured_path = f"scenario/{filename}"
        digest = _sha256(path)
        assets.append(
            ScenarioAssetSpec(
                role=role,
                configured_path=configured_path,
                path=path,
                sha256=digest,
            )
        )
        asset_records.append(
            {
                "role": role,
                "path": configured_path,
                "sha256": digest,
            }
        )

    observation_low = np.full((52,), -np.inf, dtype=np.float64)
    observation_high = np.full((52,), np.inf, dtype=np.float64)
    observation_contract = box_space_contract_sha256(
        shape=(52,),
        dtype="float32",
        low=observation_low,
        high=observation_high,
    )
    action_contract = discrete_space_contract_sha256(
        n=9,
        start=0,
        dtype="int64",
        factorization_contract_sha256=factorization.contract_hash,
    )
    observation_spec = dataclasses.replace(
        base.environment.observation_space,
        shape=(52,),
        low=observation_low,
        high=observation_high,
        contract_sha256=observation_contract,
    )
    action_spec = dataclasses.replace(
        base.environment.action_space,
        contract_sha256=action_contract,
    )
    environment_contract = environment_contract_sha256(
        environment_id=P4_SUMO_ENVIRONMENT_REGISTRY,
        max_episode_steps=base.environment.max_episode_steps,
        registry_key=P4_SUMO_ENVIRONMENT_REGISTRY,
        factory=P4_SUMO_ENVIRONMENT_FACTORY,
        runtime_type=P4_SUMO_ENVIRONMENT_TYPE,
        observation_space_contract_sha256=observation_contract,
        action_space_contract_sha256=action_contract,
        normalization_contract_sha256=(
            base.environment.normalization_contract_sha256
        ),
        scenario_assets=asset_records,
    )
    environment = dataclasses.replace(
        base.environment,
        id=P4_SUMO_ENVIRONMENT_REGISTRY,
        observation_space=observation_spec,
        action_space=action_spec,
        registry_key=P4_SUMO_ENVIRONMENT_REGISTRY,
        factory=P4_SUMO_ENVIRONMENT_FACTORY,
        runtime_type=P4_SUMO_ENVIRONMENT_TYPE,
        contract_sha256=environment_contract,
        scenario_assets=tuple(assets),
    )

    projector_config = base_path.parent / "sumo-projector.yaml"
    projector_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "rl_attack.p4_sumo_merge_v1_projector.v1",
                "name": P4_SUMO_PROJECTOR_NAME,
                "contract_version": P4_SUMO_PROJECTOR_VERSION,
                "observation_shape": [52],
                "physical_budgets": dataclasses.asdict(
                    SumoPhysicalBudgetsV1()
                ),
                "immutable_indices": [],
                "neighbor_order_tolerance_m": 1.0e-6,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    projector_config_sha = _sha256(projector_config)
    projector_contract = semantic_projector_contract_sha256(
        name=P4_SUMO_PROJECTOR_NAME,
        version=P4_SUMO_PROJECTOR_VERSION,
        factory=P4_SUMO_PROJECTOR_FACTORY,
        factory_kwargs={},
        observation_shape=(52,),
        config_sha256=projector_config_sha,
        guarantee=P4_PROJECTOR_GUARANTEE,
    )
    projector_spec = dataclasses.replace(
        base.projector,
        name=P4_SUMO_PROJECTOR_NAME,
        version=P4_SUMO_PROJECTOR_VERSION,
        factory=P4_SUMO_PROJECTOR_FACTORY,
        factory_kwargs={},
        observation_shape=(52,),
        config=projector_config,
        config_sha256=projector_config_sha,
        contract_sha256=projector_contract,
    )
    attack = dataclasses.replace(
        base.attack,
        factory_kwargs={
            **base.attack.factory_kwargs,
            "discrete_budget": 1,
            "max_candidates": 4,
        },
        discrete_planner=dataclasses.replace(
            base.attack.discrete_planner,
            registry_key=P4_SUMO_DISCRETE_PLANNER,
            allowlist=(2,),
        ),
    )
    config = dataclasses.replace(
        base,
        environment=environment,
        factorization=factorization,
        projector=projector_spec,
        attack=attack,
    )
    projector = build_sumo_merge_v1_projector(
        ProjectorBuildContext(
            config=config,
            observation_space=gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(52,),
                dtype=np.float32,
            ),
            config_path=projector_config,
            config_sha256=projector_config_sha,
        )
    )
    victim = PPO.load(base.victim.checkpoint, device="cpu")
    policy = SB3CategoricalPolicyAdapter(victim)
    context = AttackBuildContext(
        config=config,
        episode_index=0,
        episode_seed=5,
        victim=victim,
        policy=policy,
        factorization=factorization,
        projector=projector,
        runtime_artifacts={
            "safety_critic": FakeSafetyCritic(),
            "director": DifferentActionDirector(factorization),
        },
        verified_artifact_manifests={},
        temporal_budget=attack.temporal_budget,
        rng_namespace=RNGNamespace(
            base_seed=301,
            experiment_id="registry-bound-sumo-planner",
            episode_seed=5,
            attack_id="stfa",
            version=P4_RNG_DERIVATION,
        ),
        device=policy.device,
    )
    built = build_stfa_attack(context)
    assert isinstance(built.discrete_planner, SumoMergeV1DiscretePlanner)
    assert built.discrete_planner.allowlist == (2,)

    assets[1].path.write_text("<tampered/>", encoding="utf-8")
    with pytest.raises(ValueError, match="exact registered SUMO"):
        build_stfa_attack(context)
