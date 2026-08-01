from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from torch import nn

from rl_attack.cli.rapid_guard_training import main as training_main
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    state_dict_sha256,
    strict_json_load,
    strict_json_write,
)
from rl_attack.defenses.certification.ibp import actor_logit_bounds, clean_actor_logits
from rl_attack.defenses.rapid_guard.denoiser import (
    PROPOSAL_GUARANTEE_SCOPE,
    FrozenResidualDenoiser,
    ResidualDenoiserBatch,
    ResidualDenoiserConfig,
    ResidualDenoiserTrainConfig,
    ResidualDenoiserTrainingResult,
    train_residual_denoiser,
)
from rl_attack.defenses.rapid_guard.detector import FusionFitConfig
from rl_attack.defenses.rapid_guard.purifier import FrozenProposalTransform
from rl_attack.training.rapid_guard_pipeline import (
    RAPID_DATASET_FIELDS,
    RAPID_DATASET_SCHEMA,
    LoadedRapidGuardBundle,
    RapidGuardBundleTrainingResult,
    action_ontology_record,
    detector_preprocessing_record,
    hashed_contract,
    load_rapid_guard_bundle,
    load_rapid_guard_dataset,
    rapid_guard_bundle_manifest_path,
    rapid_guard_dataset_manifest_path,
    rapid_guard_dataset_sidecar,
    recompute_detector_data,
    save_rapid_guard_bundle,
    train_rapid_guard_from_npz,
)
from rl_attack.training.robust_sarsa import sb3_policy_state_sha256
from rl_attack.training.stfa_pipeline import (
    dataset_environment_contract,
    load_frozen_victim,
    normalization_contract,
)


class TinyEnv(gym.Env[np.ndarray, int]):
    observation_space = spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
    action_space = spaces.Discrete(3)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}


@dataclass
class PreparedData:
    root: Path
    victim_path: Path
    victim_sha256: str
    victim_policy_sha256: str
    fit_path: Path
    fit_sha256: str
    fit_manifest_sha256: str
    calibration_path: Path
    calibration_sha256: str
    calibration_manifest_sha256: str
    environment: dict[str, Any]
    environment_sha256: str
    normalization_sha256: str
    ontology: dict[str, Any]
    projector_sha256: str
    detector_preprocessing: dict[str, Any]
    history_bootstrap_contract: dict[str, Any]
    anchor_contract: dict[str, Any]
    purifier_config: dict[str, Any]
    fallback_config: dict[str, Any]
    shield_contract: dict[str, Any]
    test_episode_seeds: tuple[int, ...]
    test_scenario_seeds: tuple[int, ...]

    def expected_arguments(
        self,
        *,
        output_dir: Path,
        run_name: str,
    ) -> dict[str, Any]:
        return {
            "victim_checkpoint": self.victim_path,
            "expected_victim_checkpoint_sha256": self.victim_sha256,
            "fit_dataset_path": self.fit_path,
            "expected_fit_dataset_sha256": self.fit_sha256,
            "expected_fit_manifest_sha256": self.fit_manifest_sha256,
            "calibration_dataset_path": self.calibration_path,
            "expected_calibration_dataset_sha256": self.calibration_sha256,
            "expected_calibration_manifest_sha256": (
                self.calibration_manifest_sha256
            ),
            "expected_action_ontology_sha256": self.ontology["sha256"],
            "expected_projector_contract_sha256": self.projector_sha256,
            "expected_environment_contract_sha256": self.environment_sha256,
            "expected_normalization_contract_sha256": self.normalization_sha256,
            "expected_certificate_epsilon": 0.05,
            "expected_anchor_update_contract_sha256": self.anchor_contract[
                "sha256"
            ],
            "expected_purifier_config_sha256": self.purifier_config["sha256"],
            "expected_fallback_config_sha256": self.fallback_config["sha256"],
            "output_dir": output_dir,
            "run_name": run_name,
            "seed": 13,
            "alpha": 0.1,
            "device": "cpu",
            "active_channels": (
                "temporal_innovation",
                "categorical_js",
                "ibp_margin_deficit",
            ),
            "fusion_config": FusionFitConfig(
                gradient_steps=120,
                learning_rate=0.05,
                l2_penalty=0.001,
                scale_floor=1.0e-8,
            ),
            "denoiser_config": ResidualDenoiserConfig(
                observation_shape=(2,),
                hidden_sizes=(16,),
                activation="tanh",
            ),
            "denoiser_train_config": ResidualDenoiserTrainConfig(
                gradient_steps=120,
                learning_rate=0.01,
                mse_coefficient=1.0,
                policy_consistency_coefficient=0.05,
                max_gradient_norm=10.0,
                seed=13,
                device="cpu",
            ),
        }


def _raw_arrays(
    *,
    role: str,
    episode_seeds: np.ndarray,
    scenario_seeds: np.ndarray,
    families: tuple[str, ...],
) -> dict[str, np.ndarray]:
    count = len(families)
    index = np.arange(count, dtype=np.float32)
    clean = np.column_stack(
        (
            -0.3 + 0.02 * (index % 10),
            0.25 - 0.015 * (index % 10),
        )
    ).astype(np.float32)
    velocity = np.asarray([0.02, -0.01], dtype=np.float32)
    trusted = clean - velocity
    observations = clean.copy()
    attacked = np.asarray(
        [family.casefold() != "clean" for family in families],
        dtype=np.bool_,
    )
    observations[attacked] += np.asarray([0.45, -0.35], dtype=np.float32)
    history = np.stack(
        (
            clean - 2.0 * velocity,
            trusted,
            observations,
        ),
        axis=1,
    ).astype(np.float32)
    step_indices = np.full(count, 2, dtype=np.int64)
    return {
        "schema_version": np.asarray(RAPID_DATASET_SCHEMA),
        "role": np.asarray(role),
        "observations": observations,
        "clean_observations": clean,
        "trusted_observations": trusted,
        "reference_observations": trusted.copy(),
        "observation_history": history,
        "episode_seeds": np.asarray(episode_seeds, dtype=np.int64),
        "scenario_seeds": np.asarray(scenario_seeds, dtype=np.int64),
        "step_indices": step_indices,
        "history_episode_seeds": np.repeat(
            np.asarray(episode_seeds, dtype=np.int64)[:, None],
            3,
            axis=1,
        ),
        "history_scenario_seeds": np.repeat(
            np.asarray(scenario_seeds, dtype=np.int64)[:, None],
            3,
            axis=1,
        ),
        "history_step_indices": step_indices[:, None]
        + np.asarray([-2, -1, 0], dtype=np.int64),
        "attack_families": np.asarray(families),
    }


def _write_raw_dataset(
    prepared: PreparedData | SimpleNamespace,
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    role: str,
) -> tuple[str, str]:
    np.savez(path, **arrays)
    dataset_sha = sha256_file(path)
    sidecar = rapid_guard_dataset_sidecar(
        dataset_path=path,
        dataset_sha256=dataset_sha,
        role=role,
        environment=prepared.environment,
        action_ontology=prepared.ontology,
        victim_checkpoint_sha256=prepared.victim_sha256,
        victim_policy_state_sha256=prepared.victim_policy_sha256,
        projector_contract_sha256=prepared.projector_sha256,
        certificate_epsilon=0.05,
        detector_preprocessing=prepared.detector_preprocessing,
        history_bootstrap_contract=prepared.history_bootstrap_contract,
        anchor_update_contract=prepared.anchor_contract,
        purifier_config=prepared.purifier_config,
        fallback_config=prepared.fallback_config,
        shield_contract=prepared.shield_contract,
        reserved_test_episode_seeds=prepared.test_episode_seeds,
        reserved_test_scenario_seeds=prepared.test_scenario_seeds,
    )
    sidecar_path = rapid_guard_dataset_manifest_path(path)
    strict_json_write(sidecar_path, sidecar)
    return dataset_sha, sha256_file(sidecar_path)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> PreparedData:
    root = tmp_path_factory.mktemp("rapid_guard_training")
    env = TinyEnv()
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4,
        batch_size=2,
        seed=7,
        device="cpu",
        policy_kwargs={"net_arch": [8], "activation_fn": nn.Tanh},
    )
    victim_path = root / "victim.zip"
    model.save(victim_path)
    model.policy.set_training_mode(False)
    victim_sha = sha256_file(victim_path)
    frozen = load_frozen_victim(
        victim_path,
        expected_sha256=victim_sha,
        action_mode="stochastic",
        device="cpu",
    )
    normalization = normalization_contract()
    environment = dataset_environment_contract(
        env_id="TinyRapid-v0",
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalization=normalization,
    )
    ontology = action_ontology_record(("LEFT", "IDLE", "RIGHT"))
    projector = "a" * 64
    detector = detector_preprocessing_record(
        observation_shape=(2,),
        innovation_scale=np.asarray([0.2, 0.2], dtype=np.float32),
        required_margin=0.05,
    )
    history = hashed_contract(
        name="calibrated_trusted_history",
        version="v1",
        config={
            "schema_version": "p5-rapid-guard-history-bootstrap-v1",
            "mode": "strict_calibrated_v1",
            "window_frames": 3,
            "prior_trusted_frames": 2,
            "minimum_prefix_frames": 2,
            "bootstrap": "caller_attested_attack_free_trusted_prefix",
            "accepted_paths": [
                "pass_through",
                "certified_purification",
            ],
            "require_consecutive_steps": True,
            "on_gap": (
                "uncalibrated_warmup_fail_closed_until_explicit_rebootstrap"
            ),
            "episode_scope": "current_episode_only",
            "cross_episode_reuse": False,
            "first_step_attack_evaluation": (
                "fallback_cost_reported_separately"
            ),
        },
    )
    anchor = hashed_contract(
        name="trusted_anchor_update",
        version="v1",
        config={
            "commit": "accepted_pass_or_certified_purification_only",
            "continuity": "consecutive_guard_steps_only",
            "reset_on_fallback": True,
            "cross_episode_reuse": False,
        },
    )
    purifier = hashed_contract(
        name="semantic_temporal_purifier",
        version="v1",
        config={
            "temporal_radius": [0.5, 0.5],
            "line_search_points": 3,
            "projection_required": True,
            "envelope_atol": 2.0e-6,
        },
    )
    fallback = hashed_contract(
        name="legal_safety_cost_fallback",
        version="v1",
        config={"legal_mask_required": True, "static_order": [1, 0, 2]},
    )
    shield = hashed_contract(
        name="safety_shield",
        version="v1",
        config={"mode": "none"},
    )
    holder = SimpleNamespace(
        environment=environment,
        ontology=ontology,
        victim_sha256=victim_sha,
        victim_policy_sha256=frozen.policy_state_sha256,
        projector_sha256=projector,
        detector_preprocessing=detector,
        history_bootstrap_contract=history,
        anchor_contract=anchor,
        purifier_config=purifier,
        fallback_config=fallback,
        shield_contract=shield,
        test_episode_seeds=(200, 201),
        test_scenario_seeds=(1200, 1201),
    )
    fit_families = (
        *("clean" for _ in range(8)),
        *("p3_pa_ad" for _ in range(8)),
        *("p4_stfa" for _ in range(8)),
    )
    fit_episodes = np.repeat(np.asarray([1, 2, 3, 4], dtype=np.int64), 6)
    fit_scenarios = np.repeat(
        np.asarray([11, 12, 13, 14], dtype=np.int64),
        6,
    )
    fit_path = root / "fit.npz"
    fit_sha, fit_manifest_sha = _write_raw_dataset(
        holder,
        fit_path,
        _raw_arrays(
            role="fit",
            episode_seeds=fit_episodes,
            scenario_seeds=fit_scenarios,
            families=fit_families,
        ),
        role="fit",
    )
    calibration_families = tuple("clean" for _ in range(10))
    calibration_path = root / "calibration.npz"
    calibration_sha, calibration_manifest_sha = _write_raw_dataset(
        holder,
        calibration_path,
        _raw_arrays(
            role="calibration",
            episode_seeds=np.arange(100, 110, dtype=np.int64),
            scenario_seeds=np.arange(1100, 1110, dtype=np.int64),
            families=calibration_families,
        ),
        role="calibration",
    )
    env.close()
    return PreparedData(
        root=root,
        victim_path=victim_path,
        victim_sha256=victim_sha,
        victim_policy_sha256=frozen.policy_state_sha256,
        fit_path=fit_path,
        fit_sha256=fit_sha,
        fit_manifest_sha256=fit_manifest_sha,
        calibration_path=calibration_path,
        calibration_sha256=calibration_sha,
        calibration_manifest_sha256=calibration_manifest_sha,
        environment=environment,
        environment_sha256=canonical_json_sha256(environment),
        normalization_sha256=normalization["sha256"],
        ontology=ontology,
        projector_sha256=projector,
        detector_preprocessing=detector,
        history_bootstrap_contract=history,
        anchor_contract=anchor,
        purifier_config=purifier,
        fallback_config=fallback,
        shield_contract=shield,
        test_episode_seeds=(200, 201),
        test_scenario_seeds=(1200, 1201),
    )


@pytest.fixture(scope="module")
def trained(
    prepared: PreparedData,
) -> tuple[dict[str, Any], LoadedRapidGuardBundle]:
    run = train_rapid_guard_from_npz(
        **prepared.expected_arguments(
            output_dir=prepared.root / "outputs",
            run_name="primary",
        )
    )
    checkpoint = Path(run["checkpoint"]["path"])
    loaded = load_rapid_guard_bundle(
        checkpoint,
        expected_sha256=run["checkpoint"]["sha256"],
        device="cpu",
        expected_victim_checkpoint_sha256=prepared.victim_sha256,
        expected_victim_policy_state_sha256=prepared.victim_policy_sha256,
        expected_projector_contract_sha256=prepared.projector_sha256,
    )
    return run, loaded


def _archive_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def test_raw_dataset_is_strict_read_only_and_has_no_cached_features(
    prepared: PreparedData,
) -> None:
    dataset = load_rapid_guard_dataset(
        prepared.fit_path,
        expected_sha256=prepared.fit_sha256,
        expected_manifest_sha256=prepared.fit_manifest_sha256,
        expected_role="fit",
    )
    assert dataset.role == "fit"
    assert dataset.observation_shape == (2,)
    assert dataset.observations.flags.writeable is False
    assert dataset.episode_seed_set == (1, 2, 3, 4)
    assert set(_archive_arrays(prepared.fit_path)) == RAPID_DATASET_FIELDS
    forbidden = {
        "current_probabilities",
        "reference_probabilities",
        "clean_logits",
        "ibp_lower_logits",
        "ibp_upper_logits",
        "detector_channels",
    }
    assert not (set(_archive_arrays(prepared.fit_path)) & forbidden)


def test_real_sb3_probabilities_logits_and_ibp_are_recomputed(
    prepared: PreparedData,
) -> None:
    dataset = load_rapid_guard_dataset(
        prepared.fit_path,
        expected_sha256=prepared.fit_sha256,
        expected_manifest_sha256=prepared.fit_manifest_sha256,
        expected_role="fit",
    )
    victim = load_frozen_victim(
        prepared.victim_path,
        expected_sha256=prepared.victim_sha256,
        action_mode="stochastic",
        device="cpu",
    )
    before = sb3_policy_state_sha256(victim.model)
    result = recompute_detector_data(dataset, victim)
    observed = torch.from_numpy(np.array(dataset.observations, copy=True))
    expected_logits = clean_actor_logits(victim.model, observed).detach().numpy()
    expected_bounds = actor_logit_bounds(
        victim.model,
        observed,
        0.05,
        clip_to_observation_space=True,
    )
    np.testing.assert_allclose(result.clean_logits, expected_logits)
    np.testing.assert_allclose(
        result.lower_logits,
        expected_bounds.lower.detach().numpy(),
    )
    np.testing.assert_allclose(
        result.upper_logits,
        expected_bounds.upper.detach().numpy(),
    )
    expected_actions = np.argmax(expected_logits, axis=1)
    competitor_upper = expected_bounds.upper.detach().numpy().copy()
    competitor_upper[np.arange(expected_actions.size), expected_actions] = -np.inf
    expected_margin = (
        expected_bounds.lower.detach().numpy()[
            np.arange(expected_actions.size),
            expected_actions,
        ]
        - competitor_upper.max(axis=1)
    )
    np.testing.assert_allclose(result.certified_margin, expected_margin)
    np.testing.assert_allclose(result.current_probabilities.sum(axis=1), 1.0)
    assert result.evidence["cached_detector_features_consumed"] is False
    assert result.evidence["ibp_bounds_and_margin_recomputed"] is True
    assert sb3_policy_state_sha256(victim.model) == before
    assert victim.model.policy.training is False
    assert not any(
        parameter.requires_grad for parameter in victim.model.policy.parameters()
    )


def test_object_dtype_and_cached_extra_fields_are_rejected(
    prepared: PreparedData,
    tmp_path: Path,
) -> None:
    arrays = _archive_arrays(prepared.fit_path)
    arrays["observations"] = arrays["observations"].astype(object)
    object_path = tmp_path / "object.npz"
    np.savez(object_path, **arrays)
    with pytest.raises(ValueError, match="object|pickled"):
        load_rapid_guard_dataset(
            object_path,
            expected_sha256=sha256_file(object_path),
            expected_manifest_sha256="0" * 64,
            expected_role="fit",
        )
    arrays = _archive_arrays(prepared.fit_path)
    arrays["clean_logits"] = np.zeros((24, 3), dtype=np.float32)
    cached_path = tmp_path / "cached.npz"
    np.savez(cached_path, **arrays)
    with pytest.raises(ValueError, match="extra"):
        load_rapid_guard_dataset(
            cached_path,
            expected_sha256=sha256_file(cached_path),
            expected_manifest_sha256="0" * 64,
            expected_role="fit",
        )


def test_window_cross_episode_and_reserved_test_seed_fail_closed(
    prepared: PreparedData,
    tmp_path: Path,
) -> None:
    arrays = _archive_arrays(prepared.fit_path)
    arrays["history_episode_seeds"][0, 0] = 999
    crossing = tmp_path / "crossing.npz"
    crossing_sha, crossing_manifest = _write_raw_dataset(
        prepared,
        crossing,
        arrays,
        role="fit",
    )
    with pytest.raises(ValueError, match="crosses an episode"):
        load_rapid_guard_dataset(
            crossing,
            expected_sha256=crossing_sha,
            expected_manifest_sha256=crossing_manifest,
            expected_role="fit",
        )

    arrays = _archive_arrays(prepared.fit_path)
    arrays["episode_seeds"][0] = 200
    arrays["history_episode_seeds"][0, :] = 200
    leaked = tmp_path / "leaked.npz"
    leaked_sha, leaked_manifest = _write_raw_dataset(
        prepared,
        leaked,
        arrays,
        role="fit",
    )
    with pytest.raises(ValueError, match="test episode"):
        load_rapid_guard_dataset(
            leaked,
            expected_sha256=leaked_sha,
            expected_manifest_sha256=leaked_manifest,
            expected_role="fit",
        )


def test_fit_and_calibration_episode_and_scenario_overlap_is_rejected(
    prepared: PreparedData,
    tmp_path: Path,
) -> None:
    arrays = _archive_arrays(prepared.calibration_path)
    arrays["episode_seeds"][0] = 1
    arrays["history_episode_seeds"][0, :] = 1
    arrays["scenario_seeds"][0] = 11
    arrays["history_scenario_seeds"][0, :] = 11
    overlap = tmp_path / "calibration_overlap.npz"
    overlap_sha, overlap_manifest = _write_raw_dataset(
        prepared,
        overlap,
        arrays,
        role="calibration",
    )
    arguments = prepared.expected_arguments(
        output_dir=tmp_path / "outputs",
        run_name="overlap",
    )
    arguments["calibration_dataset_path"] = overlap
    arguments["expected_calibration_dataset_sha256"] = overlap_sha
    arguments["expected_calibration_manifest_sha256"] = overlap_manifest
    with pytest.raises(ValueError, match="episode seeds overlap|scenario seeds overlap"):
        train_rapid_guard_from_npz(**arguments)


def test_scenario_overlap_is_independently_rejected(
    prepared: PreparedData,
    tmp_path: Path,
) -> None:
    arrays = _archive_arrays(prepared.calibration_path)
    arrays["scenario_seeds"][0] = 11
    arrays["history_scenario_seeds"][0, :] = 11
    overlap = tmp_path / "scenario_overlap.npz"
    overlap_sha, overlap_manifest = _write_raw_dataset(
        prepared,
        overlap,
        arrays,
        role="calibration",
    )
    arguments = prepared.expected_arguments(
        output_dir=tmp_path / "scenario_outputs",
        run_name="scenario_overlap",
    )
    arguments["calibration_dataset_path"] = overlap
    arguments["expected_calibration_dataset_sha256"] = overlap_sha
    arguments["expected_calibration_manifest_sha256"] = overlap_manifest
    with pytest.raises(ValueError, match="scenario seeds overlap"):
        train_rapid_guard_from_npz(**arguments)


def test_fit_requires_declared_p3_and_p4_attack_families(
    prepared: PreparedData,
    tmp_path: Path,
) -> None:
    arrays = _archive_arrays(prepared.fit_path)
    families = arrays["attack_families"].copy()
    families[families == "p4_stfa"] = "p3_pa_ad"
    arrays["attack_families"] = families
    missing = tmp_path / "missing_p4.npz"
    missing_sha, missing_manifest = _write_raw_dataset(
        prepared,
        missing,
        arrays,
        role="fit",
    )
    arguments = prepared.expected_arguments(
        output_dir=tmp_path / "missing_outputs",
        run_name="missing_p4",
    )
    arguments["fit_dataset_path"] = missing
    arguments["expected_fit_dataset_sha256"] = missing_sha
    arguments["expected_fit_manifest_sha256"] = missing_manifest
    with pytest.raises(ValueError, match="P3 and one P4"):
        train_rapid_guard_from_npz(**arguments)


def test_dataset_sidecar_hash_schema_and_contract_tampering_are_rejected(
    prepared: PreparedData,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "copied.npz"
    shutil.copyfile(prepared.fit_path, copied)
    sidecar = copy.deepcopy(
        strict_json_load(rapid_guard_dataset_manifest_path(prepared.fit_path))
    )
    sidecar["dataset"]["filename"] = copied.name
    sidecar["dataset"]["sha256"] = sha256_file(copied)
    sidecar["unexpected"] = True
    copied_sidecar = rapid_guard_dataset_manifest_path(copied)
    strict_json_write(copied_sidecar, sidecar)
    with pytest.raises(ValueError, match="fields are invalid"):
        load_rapid_guard_dataset(
            copied,
            expected_sha256=sha256_file(copied),
            expected_manifest_sha256=sha256_file(copied_sidecar),
            expected_role="fit",
        )


def test_residual_denoiser_is_deterministic_frozen_and_unprojected_only() -> None:
    attacked = torch.tensor(
        [[0.8, -0.6], [0.7, -0.5], [0.9, -0.7]],
        dtype=torch.float32,
    )
    trusted = torch.tensor(
        [[0.2, -0.1], [0.1, 0.0], [0.3, -0.2]],
        dtype=torch.float32,
    )
    clean = attacked - torch.tensor([0.45, -0.35], dtype=torch.float32)
    batch = ResidualDenoiserBatch(attacked, trusted, clean)
    model_config = ResidualDenoiserConfig((2,), (8,), "tanh")
    train_config = ResidualDenoiserTrainConfig(
        gradient_steps=100,
        learning_rate=0.02,
        mse_coefficient=1.0,
        policy_consistency_coefficient=0.0,
        max_gradient_norm=10.0,
        seed=5,
        device="cpu",
    )
    first = train_residual_denoiser(
        batch,
        config=model_config,
        train_config=train_config,
    )
    second = train_residual_denoiser(
        batch,
        config=model_config,
        train_config=train_config,
    )
    assert state_dict_sha256(first.model.state_dict()) == state_dict_sha256(
        second.model.state_dict()
    )
    assert first.final_loss < first.initial_loss
    assert first.manifest["guarantee_scope"] == PROPOSAL_GUARANTEE_SCOPE
    assert first.manifest["requires_guard_projection"] is True
    assert first.manifest["physical_realizability_certified"] is False
    assert first.model.training is False
    assert not any(parameter.requires_grad for parameter in first.model.parameters())
    proposal = FrozenResidualDenoiser(first.model, binding_hash="b" * 64)
    assert isinstance(proposal, FrozenProposalTransform)
    output = proposal.propose(
        attacked[0].numpy(),
        trusted_observation=trusted[0].numpy(),
    )
    assert output.shape == (2,)
    assert output.dtype == np.float32
    assert output.flags.writeable is False
    with pytest.raises(ValueError, match="mse_coefficient"):
        ResidualDenoiserTrainConfig(mse_coefficient=0.0)


def test_bundle_round_trip_binds_all_inputs_and_exposes_guard_transform(
    prepared: PreparedData,
    trained: tuple[dict[str, Any], LoadedRapidGuardBundle],
) -> None:
    run, loaded = trained
    assert run["evidence_scope"] == "training_plumbing_not_formal_robustness_result"
    assert run["formal_robustness_result"] is False
    assert run["empirical_robustness_result"] is False
    manifest = loaded.manifest
    assert manifest["claims"] == {
        "formal_robustness": False,
        "empirical_robustness": False,
        "physical_realizability": False,
        "ibp_scope": "clean_greedy_action_invariance_only",
    }
    assert manifest["victim"]["checkpoint_sha256"] == prepared.victim_sha256
    assert (
        manifest["victim"]["policy_state_sha256"]
        == prepared.victim_policy_sha256
    )
    assert (
        manifest["contracts"]["projector_contract_sha256"]
        == prepared.projector_sha256
    )
    assert manifest["datasets"]["fit"]["sha256"] == prepared.fit_sha256
    assert (
        manifest["datasets"]["calibration"]["sha256"]
        == prepared.calibration_sha256
    )
    assert manifest["split"]["test_consumed_during_training"] is False
    assert manifest["recomputation"]["fit"]["raw_fields_only"] is True
    assert loaded.artifact.calibration_scores.shape == (10,)
    assert loaded.proposal_transform.frozen is True
    assert isinstance(loaded.proposal_transform, FrozenProposalTransform)
    assert loaded.proposal_transform_hash == manifest["denoiser"][
        "proposal_binding_sha256"
    ]
    fit = load_rapid_guard_dataset(
        prepared.fit_path,
        expected_sha256=prepared.fit_sha256,
        expected_manifest_sha256=prepared.fit_manifest_sha256,
        expected_role="fit",
    )
    proposal = loaded.proposal_transform.propose(
        np.array(fit.observations[8], copy=True),
        trusted_observation=np.array(fit.trusted_observations[8], copy=True),
    )
    assert proposal.shape == (2,)
    assert np.all(np.isfinite(proposal))
    with pytest.raises(ValueError, match="fallback_config_sha256"):
        load_rapid_guard_bundle(
            run["checkpoint"]["path"],
            expected_sha256=run["checkpoint"]["sha256"],
            expected_fallback_config_sha256="f" * 64,
        )


def test_training_is_deterministic_and_victim_remains_frozen(
    prepared: PreparedData,
    trained: tuple[dict[str, Any], LoadedRapidGuardBundle],
) -> None:
    _, first = trained
    second_run = train_rapid_guard_from_npz(
        **prepared.expected_arguments(
            output_dir=prepared.root / "outputs",
            run_name="deterministic_second",
        )
    )
    second = load_rapid_guard_bundle(
        second_run["checkpoint"]["path"],
        expected_sha256=second_run["checkpoint"]["sha256"],
        device="cpu",
    )
    assert first.artifact.head.state_sha256 == second.artifact.head.state_sha256
    assert first.artifact.threshold == second.artifact.threshold
    assert state_dict_sha256(
        first.proposal_transform.model.state_dict()
    ) == state_dict_sha256(second.proposal_transform.model.state_dict())
    assert canonical_json_sha256(first.manifest) == canonical_json_sha256(
        second.manifest
    )
    victim = load_frozen_victim(
        prepared.victim_path,
        expected_sha256=prepared.victim_sha256,
        action_mode="stochastic",
        device="cpu",
    )
    assert sb3_policy_state_sha256(victim.model) == prepared.victim_policy_sha256
    assert not any(
        parameter.requires_grad for parameter in victim.model.policy.parameters()
    )


def test_existing_output_and_input_output_alias_are_rejected(
    prepared: PreparedData,
    trained: tuple[dict[str, Any], LoadedRapidGuardBundle],
    tmp_path: Path,
) -> None:
    del trained
    with pytest.raises(FileExistsError, match="already exists"):
        train_rapid_guard_from_npz(
            **prepared.expected_arguments(
                output_dir=prepared.root / "outputs",
                run_name="primary",
            )
        )
    alias_root = tmp_path / "alias"
    alias_checkpoint = alias_root / "same" / "rapid_guard_bundle.pt"
    alias_checkpoint.parent.mkdir(parents=True)
    shutil.copyfile(prepared.fit_path, alias_checkpoint)
    arguments = prepared.expected_arguments(
        output_dir=alias_root,
        run_name="same",
    )
    arguments["fit_dataset_path"] = alias_checkpoint
    arguments["expected_fit_dataset_sha256"] = sha256_file(alias_checkpoint)
    with pytest.raises(ValueError, match="aliases an output"):
        train_rapid_guard_from_npz(**arguments)


def test_loader_rejects_sidecar_artifact_and_state_tampering(
    trained: tuple[dict[str, Any], LoadedRapidGuardBundle],
    tmp_path: Path,
) -> None:
    run, _ = trained
    checkpoint = Path(run["checkpoint"]["path"])
    sidecar_path = rapid_guard_bundle_manifest_path(checkpoint)
    original_sidecar = strict_json_load(sidecar_path)
    tampered_sidecar = copy.deepcopy(original_sidecar)
    tampered_sidecar["manifest"]["claims"]["physical_realizability"] = True
    strict_json_write(sidecar_path, tampered_sidecar)
    try:
        with pytest.raises(ValueError, match="claims|manifests differ"):
            load_rapid_guard_bundle(
                checkpoint,
                expected_sha256=run["checkpoint"]["sha256"],
            )
    finally:
        strict_json_write(sidecar_path, original_sidecar)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["artifact_payload"]["state"]["head"]["weights"][0] += 0.2
    artifact_tampered = tmp_path / "artifact_tampered.pt"
    torch.save(payload, artifact_tampered)
    artifact_sha = sha256_file(artifact_tampered)
    artifact_sidecar = copy.deepcopy(original_sidecar)
    artifact_sidecar["checkpoint"] = {
        "filename": artifact_tampered.name,
        "sha256": artifact_sha,
    }
    strict_json_write(
        rapid_guard_bundle_manifest_path(artifact_tampered),
        artifact_sidecar,
    )
    with pytest.raises(ValueError, match="tamper|artifact"):
        load_rapid_guard_bundle(
            artifact_tampered,
            expected_sha256=artifact_sha,
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    first_name = next(iter(payload["denoiser_state_dict"]))
    payload["denoiser_state_dict"][first_name] = (
        payload["denoiser_state_dict"][first_name] + 0.01
    )
    state_tampered = tmp_path / "state_tampered.pt"
    torch.save(payload, state_tampered)
    state_sha = sha256_file(state_tampered)
    state_sidecar = copy.deepcopy(original_sidecar)
    state_sidecar["checkpoint"] = {
        "filename": state_tampered.name,
        "sha256": state_sha,
    }
    strict_json_write(
        rapid_guard_bundle_manifest_path(state_tampered),
        state_sidecar,
    )
    with pytest.raises(ValueError, match="state hash"):
        load_rapid_guard_bundle(state_tampered, expected_sha256=state_sha)


def test_atomic_publish_failure_leaves_no_partial_bundle(
    trained: tuple[dict[str, Any], LoadedRapidGuardBundle],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, loaded = trained
    denoiser_manifest = loaded.manifest["denoiser"]["training"]
    denoiser = ResidualDenoiserTrainingResult(
        model=loaded.proposal_transform.model,
        manifest=denoiser_manifest,
        initial_loss=denoiser_manifest["initial_loss"],
        final_loss=denoiser_manifest["final_loss"],
    )
    result = RapidGuardBundleTrainingResult(
        artifact=loaded.artifact,
        denoiser=denoiser,
        proposal_binding_hash=loaded.proposal_transform_hash,
        manifest=loaded.manifest,
    )
    checkpoint = tmp_path / "atomic" / "bundle.pt"
    run_manifest = tmp_path / "atomic" / "manifest.json"

    def fail_publish(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected publish failure")

    monkeypatch.setattr(
        "rl_attack.training.rapid_guard_pipeline.publish_staged_files",
        fail_publish,
    )
    with pytest.raises(OSError, match="injected"):
        save_rapid_guard_bundle(checkpoint, run_manifest, result)
    assert not checkpoint.exists()
    assert not rapid_guard_bundle_manifest_path(checkpoint).exists()
    assert not run_manifest.exists()
    assert not list(checkpoint.parent.glob(".*.tmp"))


def test_verify_cli_reports_integrity_scope_only(
    trained: tuple[dict[str, Any], LoadedRapidGuardBundle],
    capsys: pytest.CaptureFixture[str],
) -> None:
    run, loaded = trained
    training_main(
        [
            "verify",
            "--checkpoint",
            run["checkpoint"]["path"],
            "--expected-checkpoint-sha256",
            run["checkpoint"]["sha256"],
            "--expected-victim-checkpoint-sha256",
            loaded.manifest["victim"]["checkpoint_sha256"],
            "--expected-victim-policy-state-sha256",
            loaded.manifest["victim"]["policy_state_sha256"],
            "--expected-environment-contract-sha256",
            loaded.manifest["contracts"]["environment_contract_sha256"],
            "--expected-observation-space-sha256",
            loaded.manifest["contracts"]["observation_space_sha256"],
            "--expected-action-space-sha256",
            loaded.manifest["contracts"]["action_space_sha256"],
            "--expected-normalization-contract-sha256",
            loaded.manifest["contracts"]["normalization_contract_sha256"],
            "--expected-action-ontology-sha256",
            loaded.manifest["contracts"]["action_ontology_sha256"],
            "--expected-projector-contract-sha256",
            loaded.manifest["contracts"]["projector_contract_sha256"],
            "--expected-certificate-epsilon",
            str(loaded.manifest["contracts"]["certificate_epsilon"]),
            "--expected-anchor-update-contract-sha256",
            loaded.manifest["contracts"]["anchor_update_contract_sha256"],
            "--expected-purifier-config-sha256",
            loaded.manifest["contracts"]["purifier_config_sha256"],
            "--expected-fallback-config-sha256",
            loaded.manifest["contracts"]["fallback_config_sha256"],
            "--expected-fit-dataset-sha256",
            loaded.manifest["datasets"]["fit"]["sha256"],
            "--expected-calibration-dataset-sha256",
            loaded.manifest["datasets"]["calibration"]["sha256"],
            "--expected-proposal-transform-sha256",
            loaded.proposal_transform_hash,
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["evidence_scope"] == "artifact_integrity_and_binding_verification_only"
    assert output["formal_robustness_result"] is False
    assert output["empirical_robustness_result"] is False
    assert output["proposal_transform_hash"] == loaded.proposal_transform_hash


def test_train_cli_executes_fixed_data_pipeline(
    prepared: PreparedData,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "cli_outputs"
    training_main(
        [
            "train",
            "--victim-checkpoint",
            str(prepared.victim_path),
            "--expected-victim-checkpoint-sha256",
            prepared.victim_sha256,
            "--fit-dataset",
            str(prepared.fit_path),
            "--expected-fit-dataset-sha256",
            prepared.fit_sha256,
            "--expected-fit-manifest-sha256",
            prepared.fit_manifest_sha256,
            "--calibration-dataset",
            str(prepared.calibration_path),
            "--expected-calibration-dataset-sha256",
            prepared.calibration_sha256,
            "--expected-calibration-manifest-sha256",
            prepared.calibration_manifest_sha256,
            "--expected-action-ontology-sha256",
            prepared.ontology["sha256"],
            "--expected-projector-contract-sha256",
            prepared.projector_sha256,
            "--expected-environment-contract-sha256",
            prepared.environment_sha256,
            "--expected-normalization-contract-sha256",
            prepared.normalization_sha256,
            "--expected-certificate-epsilon",
            "0.05",
            "--expected-anchor-update-contract-sha256",
            prepared.anchor_contract["sha256"],
            "--expected-purifier-config-sha256",
            prepared.purifier_config["sha256"],
            "--expected-fallback-config-sha256",
            prepared.fallback_config["sha256"],
            "--output-dir",
            str(output_dir),
            "--run-name",
            "cli",
            "--seed",
            "13",
            "--alpha",
            "0.1",
            "--fusion-gradient-steps",
            "30",
            "--denoiser-hidden-sizes",
            "8",
            "--denoiser-gradient-steps",
            "30",
            "--denoiser-learning-rate",
            "0.01",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == "rl_attack.p5_rapid_guard_training_run.v1"
    assert output["formal_robustness_result"] is False
    assert Path(output["checkpoint"]["path"]).is_file()
