from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor, nn

from rl_attack.attacks.strong.stfa.attack import DefenseTransform
from rl_attack.attacks.strong.stfa.projection import PolicyInputProjector
from rl_attack.core.artifacts import (
    canonical_json_sha256,
    sha256_file,
    strict_json_write,
)
from rl_attack.defenses.rapid_guard.denoiser import (
    ResidualDenoiserConfig,
    ResidualDenoiserTrainConfig,
)
from rl_attack.defenses.rapid_guard.detector import FusionFitConfig
from rl_attack.defenses.rapid_guard.guard import (
    BPDAIdentityPurifierAdapter,
    GuardPath,
    TrustedHistoryBootstrap,
)
from rl_attack.defenses.rapid_guard.sb3 import (
    SB3ActionInvarianceCertifier,
    SB3RapidBindingError,
    SB3RapidDetectorAdapter,
    SB3VictimMutationError,
    build_sb3_rapid_guard,
)
from rl_attack.policies.sb3 import SB3CategoricalPolicyAdapter
from rl_attack.training.rapid_guard_pipeline import (
    RAPID_DATASET_SCHEMA,
    LoadedRapidGuardBundle,
    action_ontology_record,
    detector_preprocessing_record,
    hashed_contract,
    load_rapid_guard_bundle,
    rapid_guard_dataset_manifest_path,
    rapid_guard_dataset_sidecar,
    train_rapid_guard_from_npz,
)
from rl_attack.training.robust_sarsa import (
    freeze_sb3_victim,
)
from rl_attack.training.stfa_pipeline import (
    dataset_environment_contract,
    load_frozen_victim,
    normalization_contract,
)


class TinyNineActionEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.full(2, -2.0, dtype=np.float32),
            high=np.full(2, 2.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(9)
        self._step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        self._step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self.action_space.contains(action)
        self._step += 1
        return (
            np.zeros(2, dtype=np.float32),
            0.0,
            False,
            self._step >= 2,
            {},
        )


class UnsupportedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 4) -> None:
        super().__init__(observation_space, features_dim)
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(observation_space.shape)), features_dim),
            nn.Sigmoid(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.network(observations)


@dataclass
class SB3Case:
    model: PPO
    checkpoint: Path
    environment: dict[str, object]
    bundle: LoadedRapidGuardBundle
    bundle_checkpoint: Path
    bundle_checkpoint_sha256: str
    projector: PolicyInputProjector
    history_contract_sha256: str


class BoundPolicyInputProjector(PolicyInputProjector):
    """Tiny semantic projector carrying its factory-produced contract digest."""

    def __init__(self, *args: Any, contract_sha256: str, **kwargs: Any) -> None:
        self.contract_sha256 = contract_sha256
        super().__init__(*args, **kwargs)


def _make_model(*, unsupported: bool = False) -> PPO:
    policy_kwargs: dict[str, object] = {"net_arch": [8]}
    if unsupported:
        policy_kwargs.update(
            {
                "features_extractor_class": UnsupportedExtractor,
                "features_extractor_kwargs": {"features_dim": 4},
            }
        )
    model = PPO(
        "MlpPolicy",
        TinyNineActionEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        policy_kwargs=policy_kwargs,
        seed=17,
        device="cpu",
        verbose=0,
    )
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.copy_(torch.linspace(-0.4, 0.4, 9))
    freeze_sb3_victim(model)
    return model


def _raw_dataset_arrays(
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
            -0.35 + 0.02 * (index % 10),
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
    observations[attacked] += np.asarray([0.8, -0.6], dtype=np.float32)
    history = np.stack(
        (clean - 2.0 * velocity, trusted, observations),
        axis=1,
    ).astype(np.float32)
    steps = np.full(count, 2, dtype=np.int64)
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
        "step_indices": steps,
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
        "history_step_indices": steps[:, None]
        + np.asarray([-2, -1, 0], dtype=np.int64),
        "attack_families": np.asarray(families),
    }


def _write_dataset(
    path: Path,
    *,
    role: str,
    arrays: dict[str, np.ndarray],
    environment: dict[str, object],
    ontology: dict[str, Any],
    victim_sha256: str,
    victim_policy_sha256: str,
    projector_sha256: str,
    preprocessing: dict[str, Any],
    history: dict[str, Any],
    anchor: dict[str, Any],
    purifier: dict[str, Any],
    fallback: dict[str, Any],
    shield: dict[str, Any],
) -> tuple[str, str]:
    np.savez(path, **arrays)
    dataset_sha256 = sha256_file(path)
    sidecar = rapid_guard_dataset_sidecar(
        dataset_path=path,
        dataset_sha256=dataset_sha256,
        role=role,
        environment=environment,
        action_ontology=ontology,
        victim_checkpoint_sha256=victim_sha256,
        victim_policy_state_sha256=victim_policy_sha256,
        projector_contract_sha256=projector_sha256,
        certificate_epsilon=0.0,
        detector_preprocessing=preprocessing,
        history_bootstrap_contract=history,
        anchor_update_contract=anchor,
        purifier_config=purifier,
        fallback_config=fallback,
        shield_contract=shield,
        reserved_test_episode_seeds=(200, 201),
        reserved_test_scenario_seeds=(1200, 1201),
    )
    sidecar_path = rapid_guard_dataset_manifest_path(path)
    strict_json_write(sidecar_path, sidecar)
    return dataset_sha256, sha256_file(sidecar_path)


def _case(tmp_path: Path, *, unsupported: bool = False) -> SB3Case:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = _make_model(unsupported=unsupported)
    checkpoint = tmp_path / "victim.zip"
    model.save(checkpoint)
    victim_sha256 = sha256_file(checkpoint)
    frozen = load_frozen_victim(
        checkpoint,
        expected_sha256=victim_sha256,
        action_mode="stochastic",
        device="cpu",
    )
    normalization = normalization_contract()
    environment = dataset_environment_contract(
        env_id="TinyNineAction-v0",
        observation_space=model.observation_space,
        action_space=model.action_space,
        normalization=normalization,
    )
    ontology = action_ontology_record(
        tuple(f"ACTION_{index}" for index in range(9))
    )
    projector_sha256 = canonical_json_sha256(
        {
            "schema_version": "tiny-nine-action-projector-v1",
            "observation_shape": [2],
            "epsilon": 2.0,
        }
    )
    preprocessing = detector_preprocessing_record(
        observation_shape=(2,),
        innovation_scale=np.full(2, 0.1, dtype=np.float32),
        required_margin=0.0,
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
    purifier_contract = hashed_contract(
        name="semantic_temporal_purifier",
        version="v1",
        config={
            "temporal_radius": [0.2, 0.2],
            "line_search_points": 3,
            "projection_required": True,
            "envelope_atol": 2.0e-6,
        },
    )
    fallback = hashed_contract(
        name="legal_safety_cost_fallback",
        version="v1",
        config={
            "legal_mask_required": True,
            "static_order": [4, 0, 1, 2, 3, 5, 6, 7, 8],
        },
    )
    shield = hashed_contract(
        name="safety_shield",
        version="v1",
        config={"mode": "none"},
    )
    fit_families = (
        *("clean" for _ in range(8)),
        *("p3_pa_ad" for _ in range(8)),
        *("p4_stfa" for _ in range(8)),
    )
    fit_path = tmp_path / "fit.npz"
    fit_sha256, fit_manifest_sha256 = _write_dataset(
        fit_path,
        role="fit",
        arrays=_raw_dataset_arrays(
            role="fit",
            episode_seeds=np.repeat(
                np.asarray([1, 2, 3, 4], dtype=np.int64),
                6,
            ),
            scenario_seeds=np.repeat(
                np.asarray([11, 12, 13, 14], dtype=np.int64),
                6,
            ),
            families=fit_families,
        ),
        environment=environment,
        ontology=ontology,
        victim_sha256=victim_sha256,
        victim_policy_sha256=frozen.policy_state_sha256,
        projector_sha256=projector_sha256,
        preprocessing=preprocessing,
        history=history,
        anchor=anchor,
        purifier=purifier_contract,
        fallback=fallback,
        shield=shield,
    )
    calibration_path = tmp_path / "calibration.npz"
    calibration_arrays = _raw_dataset_arrays(
        role="calibration",
        episode_seeds=np.arange(100, 110, dtype=np.int64),
        scenario_seeds=np.arange(1100, 1110, dtype=np.int64),
        families=tuple("clean" for _ in range(10)),
    )
    # Give the clean calibration cohort a non-zero, controlled innovation.
    # This makes the synthetic gate discriminate the minimum envelope repair
    # from the raw attack without hand-authoring a detector artifact.
    calibration_arrays["observation_history"][:, 0, :] += np.asarray(
        [0.2, 0.2],
        dtype=np.float32,
    )
    calibration_sha256, calibration_manifest_sha256 = _write_dataset(
        calibration_path,
        role="calibration",
        arrays=calibration_arrays,
        environment=environment,
        ontology=ontology,
        victim_sha256=victim_sha256,
        victim_policy_sha256=frozen.policy_state_sha256,
        projector_sha256=projector_sha256,
        preprocessing=preprocessing,
        history=history,
        anchor=anchor,
        purifier=purifier_contract,
        fallback=fallback,
        shield=shield,
    )
    run = train_rapid_guard_from_npz(
        victim_checkpoint=checkpoint,
        expected_victim_checkpoint_sha256=victim_sha256,
        fit_dataset_path=fit_path,
        expected_fit_dataset_sha256=fit_sha256,
        expected_fit_manifest_sha256=fit_manifest_sha256,
        calibration_dataset_path=calibration_path,
        expected_calibration_dataset_sha256=calibration_sha256,
        expected_calibration_manifest_sha256=calibration_manifest_sha256,
        expected_action_ontology_sha256=ontology["sha256"],
        expected_projector_contract_sha256=projector_sha256,
        expected_environment_contract_sha256=canonical_json_sha256(environment),
        expected_normalization_contract_sha256=normalization["sha256"],
        expected_certificate_epsilon=0.0,
        expected_anchor_update_contract_sha256=anchor["sha256"],
        expected_purifier_config_sha256=purifier_contract["sha256"],
        expected_fallback_config_sha256=fallback["sha256"],
        output_dir=tmp_path / "outputs",
        run_name="nine-action",
        seed=13,
        alpha=0.1,
        device="cpu",
        fusion_config=FusionFitConfig(
            gradient_steps=80,
            learning_rate=0.05,
            l2_penalty=0.001,
        ),
        denoiser_config=ResidualDenoiserConfig(
            observation_shape=(2,),
            hidden_sizes=(16,),
            activation="tanh",
        ),
        denoiser_train_config=ResidualDenoiserTrainConfig(
            gradient_steps=80,
            learning_rate=0.01,
            policy_consistency_coefficient=0.05,
            seed=13,
            device="cpu",
        ),
    )
    bundle_checkpoint = Path(run["checkpoint"]["path"])
    bundle = load_rapid_guard_bundle(
        bundle_checkpoint,
        expected_sha256=run["checkpoint"]["sha256"],
        device="cpu",
    )
    projector = BoundPolicyInputProjector(
        observation_shape=model.observation_space.shape,
        epsilon=2.0,
        lower=model.observation_space.low,
        upper=model.observation_space.high,
        mutable_mask=True,
        name="tiny_sb3_semantic_mock",
        contract_sha256=projector_sha256,
    )
    return SB3Case(
        model=model,
        checkpoint=checkpoint,
        environment=environment,
        bundle=bundle,
        bundle_checkpoint=bundle_checkpoint,
        bundle_checkpoint_sha256=run["checkpoint"]["sha256"],
        projector=projector,
        history_contract_sha256=history["sha256"],
    )


def _detector(case: SB3Case) -> SB3RapidDetectorAdapter:
    return SB3RapidDetectorAdapter(
        case.bundle,
        device="cpu",
    )


def _guard(case: SB3Case):
    return build_sb3_rapid_guard(
        case.bundle,
        case.projector,
        device="cpu",
    )


def _probabilities(model: PPO, observation: np.ndarray) -> np.ndarray:
    adapter = SB3CategoricalPolicyAdapter(model)
    with torch.no_grad():
        policy_input = np.array(observation, dtype=np.float32, copy=True)
        logits = adapter.logits(torch.as_tensor(policy_input).unsqueeze(0))[0]
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()


def _trusted_prefix(
    case: SB3Case,
    *,
    episode_id: str,
    frame_count: int = 2,
    first_step_index: int = 0,
    observations: tuple[np.ndarray, ...] | None = None,
) -> TrustedHistoryBootstrap:
    frames = (
        tuple(np.zeros(2, dtype=np.float32) for _ in range(frame_count))
        if observations is None
        else tuple(np.asarray(value, dtype=np.float32) for value in observations)
    )
    if observations is not None and frame_count != 2:
        raise ValueError("frame_count cannot be combined with explicit observations")
    return TrustedHistoryBootstrap(
        episode_id=episode_id,
        observations=frames,
        step_indices=tuple(
            range(first_step_index, first_step_index + len(frames))
        ),
        next_step_index=first_step_index + len(frames),
        contract_sha256=case.history_contract_sha256,
    )


def _begin_with_prefix(
    guard: object,
    case: SB3Case,
    *,
    episode_id: str = "tiny-episode",
    frame_count: int = 2,
    first_step_index: int = 0,
    observations: tuple[np.ndarray, ...] | None = None,
) -> None:
    prefix = _trusted_prefix(
        case,
        episode_id=episode_id,
        frame_count=frame_count,
        first_step_index=first_step_index,
        observations=observations,
    )
    anchor = prefix.observations[-1]
    guard.begin_episode(  # type: ignore[attr-defined]
        episode_id,
        trusted_observation=anchor,
        trusted_action_probabilities=_probabilities(case.model, anchor),
        trusted_history_bootstrap=prefix,
    )


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> SB3Case:
    return _case(tmp_path_factory.mktemp("p5_sb3_bundle"))


def test_real_tiny_sb3_nine_action_pass_and_exact_internal_accounting(
    case: SB3Case,
) -> None:
    assert case.model.action_space.n == 9
    guard = _guard(case)
    _begin_with_prefix(guard, case)

    result = guard.step(
        np.zeros(2, dtype=np.float32),
        legal_action_mask=(True,) * 9,
    )

    assert result.path is GuardPath.PASS_THROUGH
    assert result.observed_action == result.final_action == 8
    assert result.initial_detection.reason.endswith("strict_trusted_prefix_v1")
    assert result.accounting.policy_queries == 1
    assert result.accounting.detector_queries == 1
    assert result.accounting.detector_policy_queries == 1
    assert result.accounting.ibp_bound_queries == 1
    assert result.accounting.certificate_queries == 0
    assert result.accounting.total_queries == 4


def test_real_sb3_suspicious_observation_purifies_then_certifies(
    case: SB3Case,
) -> None:
    guard = _guard(case)
    _begin_with_prefix(guard, case)

    result = guard.step(
        np.asarray([1.0, 0.0], dtype=np.float32),
        legal_action_mask=(True,) * 9,
    )

    assert result.path is GuardPath.PURIFIED
    assert result.purification is not None
    assert result.purification.attempt_index == 0
    assert result.purified_observation is not None
    np.testing.assert_allclose(
        result.purified_observation,
        np.asarray([0.2, 0.0], dtype=np.float32),
        rtol=0.0,
        atol=2.0e-6,
    )
    assert result.certificate is not None and result.certificate.stable
    assert result.certificate.internal_policy_queries == 1
    assert result.certificate.ibp_bound_queries == 1
    assert result.accounting.policy_queries == 2
    assert result.accounting.detector_queries == 2
    assert result.accounting.detector_policy_queries == 2
    assert result.accounting.proposal_queries == 1
    assert result.accounting.projection_queries == 1
    assert result.accounting.purification_attempts == 1
    assert result.accounting.certificate_queries == 1
    assert result.accounting.certificate_policy_queries == 1
    assert result.accounting.ibp_bound_queries == 3
    assert result.accounting.fallback_queries == 0


def test_all_suspicious_candidates_use_static_legal_fallback_and_exact_counts(
    case: SB3Case,
) -> None:
    guard = _guard(case)
    _begin_with_prefix(
        guard,
        case,
        episode_id="all-suspicious",
        observations=(
            np.asarray([-2.0, 0.0], dtype=np.float32),
            np.asarray([-1.0, 0.0], dtype=np.float32),
        ),
    )

    result = guard.step(
        np.asarray([1.0, 0.0], dtype=np.float32),
        legal_action_mask=(True,) * 9,
    )

    assert result.path is GuardPath.FALLBACK
    assert result.reason == "purification_rejected:candidate_still_suspicious"
    assert result.fallback_action == result.final_action == 4
    assert result.fallback is not None and result.fallback.unverified
    assert result.accounting.policy_queries == 4
    assert result.accounting.detector_queries == 4
    assert result.accounting.detector_policy_queries == 4
    assert result.accounting.proposal_queries == 1
    assert result.accounting.projection_queries == 3
    assert result.accounting.purification_attempts == 3
    assert result.accounting.certificate_queries == 0
    assert result.accounting.certificate_policy_queries == 0
    assert result.accounting.ibp_bound_queries == 4
    assert result.accounting.fallback_queries == 1


def test_static_fallback_invalidates_history_until_explicit_rebootstrap(
    case: SB3Case,
) -> None:
    guard = _guard(case)
    _begin_with_prefix(guard, case)
    anchor = guard.trusted_observation

    result = guard.step(
        np.asarray([np.nan, 0.0], dtype=np.float32),
        legal_action_mask=(True,) * 9,
    )

    assert result.path is GuardPath.FALLBACK
    assert result.fallback_action == result.final_action == 4
    assert result.fallback is not None and result.fallback.unverified
    assert result.fallback.method == "static_legal_fallback"
    assert result.fallback.reason == "invalid_observation_for_cost_critic"
    assert result.accounting.policy_queries == 0
    assert result.accounting.detector_queries == 0
    assert result.accounting.detector_policy_queries == 0
    assert result.accounting.ibp_bound_queries == 0
    assert result.accounting.projection_queries == 0
    assert result.accounting.fallback_queries == 1
    np.testing.assert_array_equal(guard.trusted_observation, anchor)

    warmup = guard.step(
        np.zeros(2, dtype=np.float32),
        legal_action_mask=(True,) * 9,
    )
    assert warmup.path is GuardPath.FALLBACK
    assert warmup.initial_detection.reason.endswith(
        "insufficient_contiguous_trusted_history"
    )
    assert warmup.accounting.policy_queries == 0
    assert warmup.accounting.detector_policy_queries == 0
    assert warmup.accounting.ibp_bound_queries == 0

    prefix = _trusted_prefix(
        case,
        episode_id="tiny-episode",
        frame_count=2,
        first_step_index=2,
    )
    guard.rebootstrap_trusted_history(
        prefix,
        trusted_action_probabilities=_probabilities(
            case.model,
            prefix.observations[-1],
        ),
    )
    recovered = guard.step(
        np.zeros(2, dtype=np.float32),
        legal_action_mask=(True,) * 9,
    )
    assert recovered.path is GuardPath.PASS_THROUGH
    assert recovered.step_index == 4


def test_no_anchor_fails_closed_without_internal_policy_or_ibp_query(
    case: SB3Case,
) -> None:
    guard = _guard(case)
    guard.begin_episode("no-anchor")

    result = guard.step(
        np.zeros(2, dtype=np.float32),
        legal_action_mask=(True,) * 9,
    )

    assert result.path is GuardPath.FALLBACK
    assert result.initial_detection.reason == (
        "uncalibrated_warmup_fail_closed:no_trusted_history"
    )
    assert result.accounting.policy_queries == 0
    assert result.accounting.detector_queries == 1
    assert result.accounting.detector_policy_queries == 0
    assert result.accounting.ibp_bound_queries == 0
    assert result.accounting.fallback_queries == 1
    assert result.accounting.total_queries == 2


def test_single_anchor_fails_closed_before_policy_and_ibp_queries(
    case: SB3Case,
) -> None:
    guard = _guard(case)
    anchor = np.zeros(2, dtype=np.float32)
    guard.begin_episode(
        "single-anchor",
        trusted_observation=anchor,
        trusted_action_probabilities=_probabilities(case.model, anchor),
    )

    result = guard.step(
        np.zeros(2, dtype=np.float32),
        legal_action_mask=(True,) * 9,
    )

    assert result.path is GuardPath.FALLBACK
    assert result.initial_detection.reason == (
        "uncalibrated_warmup_fail_closed:insufficient_contiguous_trusted_history"
    )
    assert result.accounting.policy_queries == 0
    assert result.accounting.detector_queries == 1
    assert result.accounting.detector_policy_queries == 0
    assert result.accounting.ibp_bound_queries == 0
    assert result.accounting.fallback_queries == 1


def test_trusted_prefix_rejects_gaps_and_cross_episode_reuse(
    case: SB3Case,
) -> None:
    with pytest.raises(ValueError, match="strictly consecutive"):
        TrustedHistoryBootstrap(
            episode_id="gapped",
            observations=(
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            ),
            step_indices=(0, 2),
            next_step_index=3,
            contract_sha256=case.history_contract_sha256,
        )

    prefix = _trusted_prefix(case, episode_id="source-episode")
    guard = _guard(case)
    with pytest.raises(ValueError, match="episode boundary"):
        guard.begin_episode(
            "different-episode",
            trusted_observation=prefix.observations[-1],
            trusted_action_probabilities=_probabilities(
                case.model,
                prefix.observations[-1],
            ),
            trusted_history_bootstrap=prefix,
        )

    _begin_with_prefix(
        guard,
        case,
        episode_id="three-frame-prefix",
        observations=(
            np.asarray([-0.2, 0.0], dtype=np.float32),
            np.asarray([-0.1, 0.0], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        ),
    )
    assert guard.active
    np.testing.assert_array_equal(
        guard.trusted_observation,
        np.zeros(2, dtype=np.float32),
    )
    guard.end_episode()


def test_certifier_requires_the_exact_clean_greedy_action_and_counts_one_pass(
    case: SB3Case,
) -> None:
    certifier = SB3ActionInvarianceCertifier(
        case.bundle,
        device="cpu",
    )
    observation = np.zeros(2, dtype=np.float32)

    certificate = certifier.certify_action_invariance(
        observation,
        action=8,
        context=None,
    )

    assert certificate.action == 8
    assert certificate.stable
    assert certificate.internal_policy_queries == 1
    assert certificate.ibp_bound_queries == 1
    with pytest.raises(ValueError, match="clean greedy"):
        certifier.certify_action_invariance(
            observation,
            action=7,
            context=None,
        )


def test_builder_rejects_wrong_projector_hash_and_shape(
    case: SB3Case,
) -> None:
    wrong_hash = BoundPolicyInputProjector(
        observation_shape=(2,),
        epsilon=2.0,
        lower=np.full(2, -2.0, dtype=np.float32),
        upper=np.full(2, 2.0, dtype=np.float32),
        mutable_mask=True,
        name="wrong_hash",
        contract_sha256="a" * 64,
    )
    with pytest.raises(SB3RapidBindingError, match="projector contract"):
        build_sb3_rapid_guard(case.bundle, wrong_hash, device="cpu")

    wrong_shape = BoundPolicyInputProjector(
        observation_shape=(1,),
        epsilon=2.0,
        lower=np.full(1, -2.0, dtype=np.float32),
        upper=np.full(1, 2.0, dtype=np.float32),
        mutable_mask=True,
        name="wrong_shape",
        contract_sha256=case.projector.contract_sha256,
    )
    with pytest.raises(SB3RapidBindingError, match="projector shape"):
        build_sb3_rapid_guard(case.bundle, wrong_shape, device="cpu")


@pytest.mark.parametrize(
    "contract_name",
    ("fallback", "history_bootstrap", "anchor_update"),
)
def test_bundle_runtime_contract_tampering_fails_closed(
    case: SB3Case,
    contract_name: str,
) -> None:
    original_manifest = case.bundle.manifest
    tampered_manifest = copy.deepcopy(original_manifest)
    tampered_manifest["runtime_contracts"][contract_name]["config"][
        "test_only_tamper"
    ] = True
    object.__setattr__(case.bundle, "manifest", tampered_manifest)
    try:
        with pytest.raises(SB3RapidBindingError, match="integrity"):
            _detector(case)
    finally:
        object.__setattr__(case.bundle, "manifest", original_manifest)


def test_post_build_projector_mutation_fails_closed_during_purification(
    case: SB3Case,
) -> None:
    projector = BoundPolicyInputProjector(
        observation_shape=(2,),
        epsilon=2.0,
        lower=np.full(2, -2.0, dtype=np.float32),
        upper=np.full(2, 2.0, dtype=np.float32),
        mutable_mask=True,
        name="mutable_test_projector",
        contract_sha256=case.projector.contract_sha256,
    )
    guard = build_sb3_rapid_guard(case.bundle, projector, device="cpu")
    _begin_with_prefix(guard, case, episode_id="projector-mutation")
    projector.contract_sha256 = "a" * 64

    result = guard.step(
        np.asarray([1.0, 0.0], dtype=np.float32),
        legal_action_mask=(True,) * 9,
    )

    assert result.path is GuardPath.FALLBACK
    assert "semantic_projector_contract_changed" in result.reason
    assert result.accounting.projection_queries == 0
    assert result.accounting.fallback_queries == 1


def test_runtime_policy_or_checkpoint_mutation_fails_closed(
    case: SB3Case,
) -> None:
    detector = _detector(case)
    current = np.zeros(2, dtype=np.float32)
    probabilities = _probabilities(detector.victim, current)
    with torch.no_grad():
        detector.victim.policy.action_net.bias[0].add_(0.01)

    with pytest.raises(SB3VictimMutationError, match="policy state"):
        detector.assess(
            current,
            trusted_observation=current,
            current_action_probabilities=probabilities,
            trusted_action_probabilities=probabilities,
            trusted_history=(current, current),
            episode_id="mutation",
            step_index=2,
            context=None,
        )

    checkpoint_detector = _detector(case)
    checkpoint_probabilities = _probabilities(checkpoint_detector.victim, current)
    original_checkpoint = case.checkpoint.read_bytes()
    try:
        with case.checkpoint.open("ab") as stream:
            stream.write(b"tampered")
        with pytest.raises(SB3VictimMutationError, match="checkpoint"):
            checkpoint_detector.assess(
                current,
                trusted_observation=current,
                current_action_probabilities=checkpoint_probabilities,
                trusted_action_probabilities=checkpoint_probabilities,
                trusted_history=(current, current),
                episode_id="checkpoint-mutation",
                step_index=2,
                context=None,
            )
    finally:
        case.checkpoint.write_bytes(original_checkpoint)


def test_unsupported_ibp_actor_is_rejected_before_episode(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="feature extractor|unsupported"):
        _case(tmp_path, unsupported=True)


def test_bpda_adapter_matches_stfa_transform_shape_and_nonexact_declaration(
) -> None:
    adapter = BPDAIdentityPurifierAdapter(lambda value: value * 0.5)
    assert isinstance(adapter, DefenseTransform)
    observation = torch.ones((1, 2), dtype=torch.float32, requires_grad=True)

    transformed = adapter.transform(observation, sample_index=0)

    assert transformed.shape == observation.shape
    torch.testing.assert_close(transformed, torch.full_like(observation, 0.5))
    assert adapter.declaration.exact_end_to_end_gradient is False
    assert adapter.declaration.scope == "fixed_anchor_purifier_surrogate_only"
